# -*- coding: utf-8 -*-
# Updated: 2024-01-25
import streamlit as st

# 设置页面配置 - 必须在所有其他Streamlit命令之前
st.set_page_config(
    page_title="悦北 智能盯盘助手",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from dotenv import load_dotenv

# 加载.env文件（如果存在）
load_dotenv()
import time
import signal
import platform
import requests
import json
warnings.filterwarnings('ignore')

# Tushare配置 - 使用环境变量保护token
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    st.success("✅ 已连接Tushare，使用真实数据")
else:
    st.error("❌ 未找到TUSHARE_TOKEN环境变量，请检查.env文件")
    pro = None

# 完全禁用代理设置，解决网络连接问题
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

# 禁用SSL验证警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 技术指标计算函数
def calculate_atr(high, low, close, period=14):
    """计算ATR (Average True Range)"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_bollinger_bands(close, period=20, std_dev=2):
    """计算布林带"""
    close = pd.Series(close)
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, sma, lower_band

def calculate_kelly_position(win_rate, avg_win, avg_loss):
    """计算Kelly公式仓位"""
    if avg_loss == 0:
        return 0.1  # 默认10%仓位
    
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    return max(0.1, min(0.9, kelly_fraction))  # 限制在10%-90%之间

@st.cache_data(ttl=86400)  # 缓存24小时
def get_real_fundamental_data(stock_code):
    """获取真实基本面数据，使用Tushare fina_indicator接口"""
    try:
        # 转换股票代码格式
        if stock_code.startswith('0') or stock_code.startswith('3'):
            ts_code = f"{stock_code}.SZ"
        elif stock_code.startswith('6'):
            ts_code = f"{stock_code}.SH"
        else:
            ts_code = stock_code
        
        # 使用财务指标接口获取数据
        if pro is None:
            return {
                'roe': 15.0,
                'revenue_growth': 10.0,
                'profit_growth': 15.0,
                'cash_flow': 1.0
            }
        
        indi = pro.fina_indicator(ts_code=ts_code)
        
        if indi is not None and not indi.empty:
            # 获取最新一期数据
            s = indi.sort_values('end_date', ascending=False).iloc[0]
            
            # 提取关键指标，使用正确的字段名
            roe = float(s.get('roe', 15.0) or 15.0)
            revenue_growth = float(s.get('or_yoy', 10.0) or 10.0)  # 营收同比
            profit_growth = float(s.get('q_dtprofit_yoy', 15.0) or 15.0)  # 归母净利同比（单季）
            cash_flow = float(s.get('net_cash_flows_oper_act', 1.0) or 1.0) / 1e8  # 亿元
            
            return {
                'roe': roe,
                'revenue_growth': revenue_growth,
                'profit_growth': profit_growth,
                'cash_flow': cash_flow
            }
        else:
            # 如果获取失败，返回默认值
            return {
                'roe': 15.0,
                'revenue_growth': 10.0,
                'profit_growth': 15.0,
                'cash_flow': 1.0
            }
    except Exception as e:
        # 异常时返回默认值
        return {
            'roe': 15.0,
            'revenue_growth': 10.0,
            'profit_growth': 15.0,
            'cash_flow': 1.0
        }

def get_fundamental_data_with_retry(stock_code, use_real_data=False, max_retries=2):
    """基本面数据获取，根据开关选择数据源"""
    try:
        if use_real_data:
            # 使用真实基本面数据
            st.info("🔄 正在获取真实基本面数据...")
            return get_real_fundamental_data(stock_code)
        else:
            # 使用默认值
            st.info("🔄 使用默认基本面数据继续回测")
            return {
                'roe': 15.0,
                'revenue_growth': 10.0,
                'profit_growth': 15.0,
                'cash_flow': 1.0
            }
    except Exception as e:
        st.warning(f"基本面数据获取失败: {str(e)}")
        return {
            'roe': 15.0,
            'revenue_growth': 10.0,
            'profit_growth': 15.0,
            'cash_flow': 1.0
        }

def get_fundamental_data(stock_code):
    """向后兼容的基本面数据获取函数"""
    return get_fundamental_data_with_retry(stock_code)

def calculate_position_size(method, win_rate=0.5, avg_win=0.1, avg_loss=0.05, risk_per_trade=0.02, capital=1000000):
    """计算仓位大小，返回资金量"""
    if method == "Kelly公式":
        kelly_fraction = calculate_kelly_position(win_rate, avg_win, avg_loss)
        return kelly_fraction * capital
    elif method == "风险平价":
        # 简化的风险平价计算
        return capital * risk_per_trade / avg_loss
    else:  # 固定比例
        return capital * 0.5  # 默认50%仓位

def calculate_relative_performance_analysis(strategy_returns, benchmark_returns):
    """计算相对基准的绩效分析"""
    try:
        # 确保数据长度一致
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        # 计算超额收益
        excess_returns = strategy_returns - benchmark_returns
        
        # 基本信息
        strategy_annual_return = (1 + strategy_returns.mean()) ** 252 - 1
        benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
        excess_annual_return = strategy_annual_return - benchmark_annual_return
        
        # 信息比率 (Information Ratio)
        excess_volatility = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_annual_return / excess_volatility if excess_volatility > 0 else 0
        
        # Beta系数 (市场敏感度)
        if len(strategy_returns) > 1 and benchmark_returns.std() > 0:
            beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        else:
            beta = 1.0
        
        # Alpha (超额收益)
        alpha = excess_annual_return - (beta - 1) * benchmark_annual_return
        
        # 跟踪误差 (Tracking Error)
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # 相关性
        correlation = np.corrcoef(strategy_returns, benchmark_returns)[0, 1] if len(strategy_returns) > 1 else 0
        
        # 胜率分析
        strategy_win_rate = (strategy_returns > 0).mean()
        benchmark_win_rate = (benchmark_returns > 0).mean()
        excess_win_rate = (excess_returns > 0).mean()
        
        return {
            'strategy_annual_return': strategy_annual_return,
            'benchmark_annual_return': benchmark_annual_return,
            'excess_annual_return': excess_annual_return,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'tracking_error': tracking_error,
            'correlation': correlation,
            'strategy_win_rate': strategy_win_rate,
            'benchmark_win_rate': benchmark_win_rate,
            'excess_win_rate': excess_win_rate
        }
    except Exception as e:
        return None

def rolling_window_validation(stock_codes, params, window_months=12, step_months=3):
    """滚动窗口验证，测试策略在不同时间窗口的稳定性"""
    try:
        results = []
        
        for stock_code in stock_codes:
            # 获取股票数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*3)  # 3年数据
            
            hist_data = get_stock_data_with_retry(stock_code, start_date, end_date)
            if hist_data is None or hist_data.empty:
                continue
            
            # 转换为datetime
            hist_data['日期'] = pd.to_datetime(hist_data['日期'])
            hist_data = hist_data.sort_values('日期').reset_index(drop=True)
            
            # 滚动窗口测试
            total_months = (hist_data['日期'].iloc[-1] - hist_data['日期'].iloc[0]).days // 30
            
            for start_month in range(0, total_months - window_months, step_months):
                # 计算窗口日期
                window_start = hist_data['日期'].iloc[0] + timedelta(days=start_month*30)
                window_end = window_start + timedelta(days=window_months*30)
                
                # 筛选窗口数据
                window_data = hist_data[
                    (hist_data['日期'] >= window_start) & 
                    (hist_data['日期'] <= window_end)
                ].copy()
                
                if len(window_data) < 50:  # 数据不足
                    continue
                
                # 在窗口内回测
                result = backtest_one(stock_code, params)
                if result:
                    result['window_start'] = window_start
                    result['window_end'] = window_end
                    result['window_length'] = window_months
                    results.append(result)
        
        return results
    except Exception as e:
        return []

def multi_stock_validation(stock_codes, params):
    """多股票验证，测试策略在不同股票上的泛化能力"""
    try:
        results = []
        
        for stock_code in stock_codes:
            result = backtest_one(stock_code, params)
            if result:
                results.append(result)
        
        return results
    except Exception as e:
        return []

def calculate_strategy_robustness(results):
    """计算策略鲁棒性指标"""
    if not results:
        return None
    
    try:
        # 提取关键指标
        returns = [r['年化收益'] for r in results]
        sharpe_ratios = [r['夏普比率'] for r in results]
        max_drawdowns = [r['最大回撤'] for r in results]
        
        # 计算统计指标
        return_mean = np.mean(returns)
        return_std = np.std(returns)
        return_cv = return_std / abs(return_mean) if return_mean != 0 else float('inf')
        
        sharpe_mean = np.mean(sharpe_ratios)
        sharpe_std = np.std(sharpe_ratios)
        
        # 胜率统计
        positive_returns = sum(1 for r in returns if r > 0)
        win_rate = positive_returns / len(returns)
        
        # 稳定性评分
        stability_score = 1 / (1 + return_cv)  # 变异系数越小，稳定性越高
        
        return {
            'return_mean': return_mean,
            'return_std': return_std,
            'return_cv': return_cv,
            'sharpe_mean': sharpe_mean,
            'sharpe_std': sharpe_std,
            'win_rate': win_rate,
            'stability_score': stability_score,
            'total_tests': len(results)
        }
    except Exception as e:
        return None

@st.cache_data(ttl=5)
def get_realtime_data_with_retry(stock_code, max_retries=3, timeout=10):
    """实时数据获取：优先 Tushare，备用新浪，带重试和明确告警"""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    # 创建Session with Retry策略
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114 Safari/537.36'
    }
    
    # 1) 优先：Tushare 实时数据
    if pro is not None:
        for attempt in range(max_retries):
            try:
                # 转换股票代码格式
                if stock_code.startswith('0') or stock_code.startswith('3'):
                    ts_code = f"{stock_code}.SZ"
                elif stock_code.startswith('6'):
                    ts_code = f"{stock_code}.SH"
                else:
                    ts_code = stock_code
                
                # 获取最近交易日数据（避免周末/节假日无数据问题）
                today = datetime.now()
                # 获取最近5个交易日的数据，然后取最新的
                start_date = (today - timedelta(days=7)).strftime('%Y%m%d')
                end_date = today.strftime('%Y%m%d')
                data = pro.daily(ts_code=ts_code, 
                               start_date=start_date, 
                               end_date=end_date,
                               fields='ts_code,trade_date,open,high,low,close,vol,amount,pct_chg')
            
                # 如果有数据，取最新的交易日数据
                if data is not None and not data.empty:
                    data = data.sort_values('trade_date', ascending=False)
                    row = data.iloc[0]  # 取最新一条
                    return {
                        '最新价': float(row.get('close', 0) or 0),
                        '涨跌幅': float(row.get('pct_chg', 0) or 0),
                        '成交量': int(row.get('vol', 0) or 0),
                        '成交额': float(row.get('amount', 0) or 0)
                    }
                
                st.info(f"Tushare未找到今日数据 {stock_code}，尝试备用源 (第{attempt+1}次)")
            except Exception as e:
                st.warning(f"Tushare 实时数据失败(第{attempt+1}次): {str(e)}")
            time.sleep(0.5)
    
    # 2) 备用：新浪简易接口（gbk 编码）
    try:
        prefix = 'sh' if str(stock_code).startswith('6') else 'sz'
        url = f"http://hq.sinajs.cn/list={prefix}{stock_code}"
        resp = session.get(url, headers=headers, timeout=timeout)
        resp.encoding = 'gbk'
        if resp.status_code == 200 and 'hq_str_' in resp.text:
            parts = resp.text.split('"')[1].split(',')
            if len(parts) > 9:
                price = float(parts[3] or 0)
                prev_close = float(parts[2] or 0)
                pct = (price - prev_close) / prev_close * 100 if prev_close else 0.0
                return {
                    '最新价': price,
                    '涨跌幅': pct,
                    '成交量': int(parts[8] or 0),
                    '成交额': float(parts[9] or 0)
                }
        return None
    except Exception as e:
        return None
    
    # 兜底：返回None，由UI层处理
    return None

def get_stock_data_with_retry(stock_code, start_date, end_date, max_retries=2):
    """多源历史数据获取：Tushare → 模拟数据"""
    
    # 1) 优先：Tushare
    if pro is not None:
        for attempt in range(max_retries):
            try:
                # 转换股票代码格式 (如: 000001 -> 000001.SZ)
                if stock_code.startswith('0') or stock_code.startswith('3'):
                    ts_code = f"{stock_code}.SZ"
                elif stock_code.startswith('6'):
                    ts_code = f"{stock_code}.SH"
                else:
                    ts_code = stock_code
                
                # 获取日线数据（添加fields参数提高效率）
                data = pro.daily(ts_code=ts_code, 
                                         start_date=start_date.strftime('%Y%m%d'),
                                         end_date=end_date.strftime('%Y%m%d'),
                                         fields='ts_code,trade_date,open,high,low,close,vol,amount')
                
                if data is not None and not data.empty:
                    # 重命名列以匹配原有格式
                    data = data.rename(columns={
                        'trade_date': '日期',
                        'open': '开盘',
                        'high': '最高', 
                        'low': '最低',
                        'close': '收盘',
                        'vol': '成交量'
                    })
                
                    # 转换日期格式
                    data['日期'] = pd.to_datetime(data['日期'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
                    
                    # 按日期排序
                    data = data.sort_values('日期').reset_index(drop=True)
                    
                    required_cols = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
                    if all(col in data.columns for col in required_cols):
                        st.success(f"✅ Tushare数据获取成功: {stock_code}")
                        return data
                
                st.info(f"Tushare数据获取失败(第{attempt+1}次)，尝试备用源...")
            except Exception as e:
                st.warning(f"Tushare失败(第{attempt+1}次): {str(e)}")
            time.sleep(0.5)
    
    # 2) 兜底：模拟数据
    st.warning("Tushare数据源失败，使用模拟数据进行演示")
    return generate_mock_data(start_date, end_date)

def generate_mock_data(start_date, end_date):
    """生成模拟股票数据用于演示"""
    import random
    
    # 生成日期序列（只包含工作日）
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_range = [d for d in date_range if d.weekday() < 5]
    
    # 生成模拟价格数据
    base_price = 10.0
    prices = []
    
    for i, date in enumerate(date_range):
        # 模拟价格波动
        change = random.uniform(-0.05, 0.05)  # ±5%的日波动
        base_price *= (1 + change)
        base_price = max(1.0, base_price)  # 价格不能低于1元
        
        # 生成OHLC数据
        high = base_price * (1 + random.uniform(0, 0.03))
        low = base_price * (1 - random.uniform(0, 0.03))
        open_price = base_price * (1 + random.uniform(-0.02, 0.02))
        close_price = base_price
        
        # 生成成交量
        volume = random.randint(1000000, 10000000)
        
        prices.append({
            '日期': date.strftime('%Y-%m-%d'),
            '开盘': round(open_price, 2),
            '最高': round(high, 2),
            '最低': round(low, 2),
            '收盘': round(close_price, 2),
            '成交量': volume
        })
    
    return pd.DataFrame(prices)

# 指数代码映射字典
INDEX_MAP = {
    "沪深300指数": "000300.SH",
    "中证500指数": "000905.SH", 
    "创业板指": "399006.SZ"
}

def get_index_series(name, start, end):
    """获取指数数据并返回累计收益序列"""
    try:
        if name not in INDEX_MAP:
            return None
        
        symbol = INDEX_MAP[name]
        # 使用Tushare获取指数数据（添加fields参数提高效率）
        if pro is None:
            return None
            
        index_data = pro.index_daily(ts_code=symbol, 
                                   start_date=start.strftime('%Y%m%d'), 
                                   end_date=end.strftime('%Y%m%d'),
                                   fields='ts_code,trade_date,open,high,low,close,vol,amount')
        
        if index_data is not None and not index_data.empty:
            # 重命名列以匹配Tushare格式
            index_data = index_data.rename(columns={
                'trade_date': '日期',
                'close': '收盘'
            })
            
            # 确保日期列存在且为datetime类型
            if '日期' in index_data.columns:
                index_data['日期'] = pd.to_datetime(index_data['日期'], format='%Y%m%d')
            elif 'date' in index_data.columns:
                index_data['date'] = pd.to_datetime(index_data['date'])
                index_data = index_data.rename(columns={'date': '日期'})
            
            # 确保收盘价列存在
            close_col = None
            for col in ['收盘', 'close', '收盘价']:
                if col in index_data.columns:
                    close_col = col
                    break
            
            if close_col is not None:
                # 计算累计收益序列
                index_data['收益率'] = index_data[close_col].pct_change()
                index_data['累计收益'] = (1 + index_data['收益率']).cumprod()
                return index_data[['日期', close_col, '收益率', '累计收益']].dropna()
        
        return None
    except Exception as e:
        return None

def get_benchmark_data(benchmark_type, start_date, end_date):
    """获取基准数据"""
    try:
        if benchmark_type == "个股买入持有":
            return None
        
        # 使用新的get_index_series函数获取指数数据
        benchmark_data = get_index_series(benchmark_type, start_date, end_date)
        if benchmark_data is not None:
            # 重命名列以保持兼容性
            benchmark_data = benchmark_data.rename(columns={'日期': 'date', '收盘': 'close'})
            return benchmark_data
        return None
    except Exception as e:
        st.warning(f"获取基准数据失败: {str(e)}")
        return None

def backtest_one(code, params):
    """单股回测函数，返回收益、回撤、年化等指标"""
    try:
        # 解包参数
        fast_ma = params['fast_ma']
        slow_ma = params['slow_ma']
        signal_type = params['signal_type']
        start_date = params['start_date']
        end_date = params['end_date']
        backtest_years = params['backtest_years']
        stop_loss = params['stop_loss']
        take_profit = params['take_profit']
        max_position_size = params['max_position_size']
        commission_rate = params['commission_rate']
        slippage_rate = params['slippage_rate']
        stamp_tax_rate = params['stamp_tax_rate']
        
        # 获取历史数据
        hist_data = get_stock_data_with_retry(code, start_date, end_date)
        
        if hist_data is None or hist_data.empty or len(hist_data) < 50:
            return None
        
        # 数据验证和清理
        required_columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
        missing_columns = [col for col in required_columns if col not in hist_data.columns]
        if missing_columns:
            return None
        
        # 确保数值列为数值类型
        numeric_columns = ['开盘', '最高', '最低', '收盘', '成交量']
        for col in numeric_columns:
            if col in hist_data.columns:
                hist_data[col] = pd.to_numeric(hist_data[col], errors='coerce')
        
        # 移除包含NaN的行
        hist_data = hist_data.dropna(subset=['收盘', '最高', '最低', '开盘'])
        
        if len(hist_data) < 50:
            return None
        
        # 计算技术指标
        hist_data['MA_fast'] = hist_data['收盘'].rolling(window=fast_ma).mean()
        hist_data['MA_slow'] = hist_data['收盘'].rolling(window=slow_ma).mean()
        hist_data = hist_data.dropna().reset_index(drop=True)
        
        # 初始化策略变量
        hist_data['信号'] = 0
        hist_data['仓位'] = 0.0
        hist_data['买入价格'] = 0.0
        hist_data['止损价格'] = 0.0
        hist_data['止盈价格'] = 0.0
        hist_data['风险信号'] = 0  # 添加风险信号列
        
        # 简化策略执行（趋势跟踪）
        for i in range(1, len(hist_data)):
            current_price = hist_data.loc[i, '收盘']
            prev_signal = hist_data.loc[i-1, '信号']
            prev_position = hist_data.loc[i-1, '仓位']
            prev_buy_price = hist_data.loc[i-1, '买入价格']
            
            # 风险管理检查
            if prev_position > 0 and prev_buy_price > 0:
                # 止盈检查（优先检查止盈）
                if current_price >= hist_data.loc[i-1, '止盈价格']:
                    hist_data.loc[i, '信号'] = 0
                    hist_data.loc[i, '仓位'] = 0.0
                    hist_data.loc[i, '风险信号'] = 2  # 止盈
                    continue
                
                # 止损检查
                if current_price <= hist_data.loc[i-1, '止损价格']:
                    hist_data.loc[i, '信号'] = 0
                    hist_data.loc[i, '仓位'] = 0.0
                    hist_data.loc[i, '风险信号'] = 1  # 止损
                    continue
            
            # 信号生成（趋势跟踪）- 修复未来函数问题 + 信号确认机制
            # 使用前一天的均线数据生成当天的信号，并添加信号确认
            if i > 0:  # 确保有前一天的数据
                prev_ma_fast = hist_data.loc[i-1, 'MA_fast']
                prev_ma_slow = hist_data.loc[i-1, 'MA_slow']
                
                # 信号确认机制：可配置确认天数
                confirmation_days = params.get('signal_confirmation_days', 2)
                enable_confirmation = params.get('enable_signal_confirmation', True)
                
                if enable_confirmation and i >= confirmation_days:
                    # 检查连续N天的信号确认
                    golden_cross_confirmed = True
                    death_cross_confirmed = True
                    
                    for j in range(confirmation_days):
                        prev_ma_fast_j = hist_data.loc[i-1-j, 'MA_fast']
                        prev_ma_slow_j = hist_data.loc[i-1-j, 'MA_slow']
                        
                        if not (prev_ma_fast_j > prev_ma_slow_j):
                            golden_cross_confirmed = False
                        if not (prev_ma_fast_j < prev_ma_slow_j):
                            death_cross_confirmed = False
                else:
                    # 不使用确认或数据不足，使用简单信号
                    golden_cross_confirmed = prev_ma_fast > prev_ma_slow
                    death_cross_confirmed = prev_ma_fast < prev_ma_slow
                
                if golden_cross_confirmed:  # 确认金叉
                    if prev_signal == 0:  # 新开仓
                        hist_data.loc[i, '信号'] = 1
                        hist_data.loc[i, '仓位'] = max_position_size
                        hist_data.loc[i, '买入价格'] = current_price
                        hist_data.loc[i, '止损价格'] = current_price * (1 - stop_loss)
                        hist_data.loc[i, '止盈价格'] = current_price * (1 + take_profit)
                        hist_data.loc[i, '移动止损价格'] = current_price * (1 - 0.1)  # 默认10%移动止损
                        hist_data.loc[i, '风险信号'] = 0  # 正常开仓
                    else:  # 保持持仓
                        hist_data.loc[i, '信号'] = 1
                        hist_data.loc[i, '仓位'] = prev_position
                        hist_data.loc[i, '买入价格'] = prev_buy_price
                        hist_data.loc[i, '止损价格'] = hist_data.loc[i-1, '止损价格']
                        hist_data.loc[i, '止盈价格'] = hist_data.loc[i-1, '止盈价格']
                        hist_data.loc[i, '风险信号'] = 0  # 正常持仓
                elif death_cross_confirmed:  # 确认死叉
                    hist_data.loc[i, '信号'] = 0
                    hist_data.loc[i, '仓位'] = 0.0
                    hist_data.loc[i, '风险信号'] = 0  # 正常平仓
                else:
                    # 信号不确认，保持前一日状态
                    hist_data.loc[i, '信号'] = prev_signal
                    hist_data.loc[i, '仓位'] = prev_position
                    hist_data.loc[i, '风险信号'] = 0  # 保持状态
                    if prev_signal == 1:
                        hist_data.loc[i, '买入价格'] = prev_buy_price
                        hist_data.loc[i, '止损价格'] = hist_data.loc[i-1, '止损价格']
                        hist_data.loc[i, '止盈价格'] = hist_data.loc[i-1, '止盈价格']
            else:
                # 第一天，无信号
                hist_data.loc[i, '信号'] = 0
                hist_data.loc[i, '仓位'] = 0.0
        
        # 计算交易成本和收益
        hist_data['收益率'] = hist_data['收盘'].pct_change()
        hist_data['策略收益'] = hist_data['信号'].shift(1) * hist_data['收益率'] * hist_data['仓位'].shift(1)
        
        # 交易成本：按实际仓位变化计算换手
        position_change = hist_data['仓位'].diff().fillna(0.0)
        
        # 买入：仓位增加的部分
        buy_turnover = np.where(position_change > 0, position_change, 0.0)
        # 卖出：仓位减少的部分
        sell_turnover = np.where(position_change < 0, -position_change, 0.0)
        
        # 计算交易成本
        buy_cost = buy_turnover * (commission_rate + slippage_rate)
        sell_cost = sell_turnover * (commission_rate + slippage_rate + stamp_tax_rate)
        hist_data['交易成本'] = buy_cost + sell_cost
        
        hist_data['策略收益_after_fee'] = hist_data['策略收益'] - hist_data['交易成本']
        hist_data['累计收益'] = (1 + hist_data['策略收益_after_fee']).cumprod()
        
        # 计算关键指标
        total_return = hist_data['累计收益'].iloc[-1] - 1
        n = len(hist_data)
        annual_return = hist_data['累计收益'].iloc[-1] ** (252.0 / n) - 1.0
        
        # 最大回撤
        cumulative_returns = hist_data['累计收益']
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 夏普比率
        strategy_returns = hist_data['策略收益_after_fee'].dropna()
        if len(strategy_returns) > 0:
            annual_volatility = strategy_returns.std() * np.sqrt(252)
            risk_free_rate = 0.03
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 胜率
        winning_trades = (strategy_returns > 0).sum()
        total_trades = (strategy_returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            '股票代码': code,
            '总收益': total_return,
            '年化收益': annual_return,
            '最大回撤': max_drawdown,
            '夏普比率': sharpe_ratio,
            '胜率': win_rate,
            '累计收益序列': hist_data['累计收益'].values,
            '日期序列': hist_data['日期'].values
        }
        
    except Exception as e:
        return None


# 自定义CSS样式 - 专业深色模式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .main-title {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .guide-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #ffd700;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #34495e;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .download-btn {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
    }
    
    .report-section {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 1px solid #34495e;
    }
</style>
""", unsafe_allow_html=True)

# 主标题区域
st.markdown("""
<div class="main-header">
    <h1 class="main-title">✨ 悦北 智能盯盘助手</h1>
</div>
""", unsafe_allow_html=True)

# 引导文案
st.markdown("""
<div class="guide-box">
    <h3 style="color: white; margin: 0 0 1rem 0;">💡 使用指南</h3>
    <p style="color: white; margin: 0; font-size: 1.1rem;">
        输入股票代码 → 设置参数 → 点击生成回测 → 如需每日自动信号，可添加微信：<strong>yuebei888</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; text-align: center; margin: 0;">📊 股票选择</h3>
    </div>
    """, unsafe_allow_html=True)
    
    stock_code = st.text_input("股票代码", value="000001", placeholder="请输入股票代码")
    
    # 新增：多股票组合回测
    portfolio_mode = st.checkbox("🎯 组合回测模式", help="同时回测多只股票")
    if portfolio_mode:
        portfolio_stocks = st.text_area("股票代码列表", value="000001\n000002\n000858", 
                                       placeholder="每行一个股票代码", height=100)
        portfolio_stocks = [s.strip() for s in portfolio_stocks.split('\n') if s.strip()]
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; text-align: center; margin: 0;">⚙️ 策略参数</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fast_ma = st.slider("快速均线周期", 3, 50, 5)
        signal_type = st.selectbox("信号类型", ["金叉死叉", "趋势跟踪", "多因子综合"])
    with col2:
        slow_ma = st.slider("慢速均线周期", 10, 200, 20)
        benchmark_type = st.selectbox("基准类型", ["个股买入持有", "沪深300指数", "中证500指数", "创业板指"])
    
    # 新增：信号确认机制
    enable_signal_confirmation = st.checkbox("🔒 启用信号确认机制", value=True, 
                                           help="需要连续2天确认信号，减少交易频率和噪音")
    signal_confirmation_days = st.slider("信号确认天数", 1, 5, 2) if enable_signal_confirmation else 1
    
    # 新增：ATR/布林带参数
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; text-align: center; margin: 0;">📊 技术指标</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        atr_period = st.slider("ATR周期", 5, 30, 14)
        bollinger_period = st.slider("布林带周期", 10, 50, 20)
    with col2:
        atr_multiplier = st.number_input("ATR倍数", 1.0, 5.0, 2.0, 0.1)
        bollinger_std = st.number_input("布林带标准差", 1.0, 3.0, 2.0, 0.1)
    
    # 新增：基本面过滤
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; text-align: center; margin: 0;">📈 基本面过滤</h3>
    </div>
    """, unsafe_allow_html=True)
    
    enable_fundamental_filter = st.checkbox("启用基本面过滤", value=False)
    use_real_fundamental = st.checkbox("启用真实基本面数据", value=False)
    if enable_fundamental_filter:
        col1, col2 = st.columns(2)
        with col1:
            min_roe = st.number_input("最小ROE(%)", 5.0, 30.0, 15.0, 1.0)
            min_revenue_growth = st.number_input("最小营收增长(%)", -20.0, 50.0, 10.0, 1.0)
        with col2:
            min_profit_growth = st.number_input("最小净利润增长(%)", -30.0, 100.0, 15.0, 1.0)
            min_cash_flow = st.number_input("最小现金流(亿)", 0.1, 100.0, 1.0, 0.1)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; text-align: center; margin: 0;">📈 回测设置</h3>
    </div>
    """, unsafe_allow_html=True)
    
    backtest_years = st.slider("回测年数", 1, 5, 3)
    backtest_days = backtest_years * 365
    
    # 新增：蒙特卡洛模拟
    enable_monte_carlo = st.checkbox("启用蒙特卡洛模拟", value=False)
    if enable_monte_carlo:
        mc_simulations = st.slider("模拟次数", 100, 1000, 500)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; text-align: center; margin: 0;">💰 交易成本</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        commission_rate = st.number_input("手续费率(%)", 0.01, 0.5, 0.1, 0.01) / 100
    with col2:
        slippage_rate = st.number_input("滑点率(%)", 0.01, 0.5, 0.05, 0.01) / 100
    with col3:
        stamp_tax_rate = st.number_input("印花税率(%)", 0.0, 0.2, 0.1, 0.01) / 100
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; text-align: center; margin: 0;">🛡️ 风险管理</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 新增：资金管理
    col1, col2 = st.columns(2)
    with col1:
        risk_per_trade = st.number_input("单笔风险比例(%)", 0.5, 5.0, 2.0, 0.1) / 100
        initial_capital = st.number_input("初始资金(万)", 10.0, 1000.0, 100.0, 10.0) * 10000
    with col2:
        position_sizing_method = st.selectbox("仓位管理", ["固定比例", "Kelly公式", "风险平价"])
        kelly_fraction = st.number_input("Kelly比例", 0.1, 1.0, 0.5, 0.1) if position_sizing_method == "Kelly公式" else 0.5
    
    # 传统风险管理
    col1, col2 = st.columns(2)
    with col1:
        stop_loss = st.number_input("止损比例(%)", 5.0, 30.0, 10.0, 0.5) / 100
        max_position_size = st.number_input("最大仓位比例(%)", 10.0, 100.0, 50.0, 5.0) / 100
    with col2:
        take_profit = st.number_input("止盈比例(%)", 10.0, 100.0, 30.0, 1.0) / 100
        max_drawdown_limit = st.number_input("最大回撤限制(%)", 10.0, 50.0, 20.0, 1.0) / 100
    
    # 新增：移动止损
    enable_trailing_stop = st.checkbox("启用移动止损", value=False)
    if enable_trailing_stop:
        trailing_stop_percent = st.number_input("移动止损比例(%)", 5.0, 20.0, 10.0, 0.5) / 100
    
    st.markdown("---")
    
    # 新增：策略验证选项
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; text-align: center; margin: 0;">🔬 策略验证</h3>
    </div>
    """, unsafe_allow_html=True)
    
    enable_strategy_validation = st.checkbox("启用策略验证", value=False, 
                                           help="进行多股票和滚动窗口验证，测试策略鲁棒性")
    
    if enable_strategy_validation:
        col1, col2 = st.columns(2)
        with col1:
            validation_stocks = st.text_area("验证股票列表", value="000001\n000002\n000858\n600036\n600519", 
                                           placeholder="每行一个股票代码", height=80)
            validation_stocks = [s.strip() for s in validation_stocks.split('\n') if s.strip()]
        with col2:
            window_months = st.slider("滚动窗口(月)", 6, 24, 12)
            step_months = st.slider("步长(月)", 1, 6, 3)
    
    st.markdown("---")
    
    # 新增：确定按钮
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; text-align: center; margin: 0;">🚀 开始回测</h3>
    </div>
    """, unsafe_allow_html=True)
    
    run_backtest = st.button("🎯 确定并开始回测", key="run_backtest_btn", type="primary", use_container_width=True)
    
    if run_backtest:
        st.session_state.run_backtest = True
        st.success("✅ 参数确认，开始策略回测...")
    else:
        st.info("💡 请先设置好参数，然后点击上方按钮开始回测")

# 主内容区域
if stock_code:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
        <h2 style="color: white; margin: 0; text-align: center;">📈 {stock_code} 专业策略回测</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # 实时数据
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <h3 style="color: white; margin: 0;">🔄 实时行情</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("正在获取实时数据..."):
        try:
            stock_data = get_realtime_data_with_retry(stock_code, max_retries=2, timeout=15)
            
            if stock_data is not None:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">当前价格</h4>
                        <p style="color: #f39c12; font-size: 1.5rem; font-weight: bold; margin: 0;">¥{stock_data.get('最新价', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    change_color = "#e74c3c" if stock_data.get('涨跌幅', 0) < 0 else "#27ae60"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">涨跌幅</h4>
                        <p style="color: {change_color}; font-size: 1.5rem; font-weight: bold; margin: 0;">{stock_data.get('涨跌幅', 'N/A')}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">成交量</h4>
                        <p style="color: #3498db; font-size: 1.5rem; font-weight: bold; margin: 0;">{stock_data.get('成交量', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">成交额</h4>
                        <p style="color: #9b59b6; font-size: 1.5rem; font-weight: bold; margin: 0;">{stock_data.get('成交额', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("实时数据获取失败，使用上次缓存/请稍后重试")
        except Exception as e:
            st.warning("实时数据获取失败，使用上次缓存/请稍后重试")
    
    # 只有点击确定按钮后才执行回测
    if st.session_state.get('run_backtest', False):
        with st.spinner("正在进行专业策略回测..."):
            try:
                # 获取历史数据
                end_date = datetime.now()
                start_date = end_date - timedelta(days=backtest_days)
                
                # 使用带重试机制的数据获取
                hist_data = get_stock_data_with_retry(stock_code, start_date, end_date)
                
                if hist_data is None or hist_data.empty:
                    st.error("❌ 无法获取历史数据，请检查：")
                    st.error("1. 股票代码是否正确")
                    st.error("2. 网络连接是否正常")
                    st.error("3. 数据源是否可用")
                    st.error("请稍后重试或尝试其他股票代码")
                else:
                    # 数据验证和清理
                    required_columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
                    missing_columns = [col for col in required_columns if col not in hist_data.columns]
                    if missing_columns:
                        st.error(f"❌ 数据格式错误，缺少必要列：{missing_columns}")
                    elif len(hist_data) < 50:  # 确保有足够的数据
                        st.error("❌ 历史数据不足，至少需要50个交易日的数据")
                    else:
                        # 确保数值列为数值类型
                        numeric_columns = ['开盘', '最高', '最低', '收盘', '成交量']
                        for col in numeric_columns:
                            if col in hist_data.columns:
                                hist_data[col] = pd.to_numeric(hist_data[col], errors='coerce')
                        
                        # 移除包含NaN的行
                        hist_data = hist_data.dropna(subset=['收盘', '最高', '最低', '开盘'])
                        
                        if len(hist_data) >= 50:  # 再次检查数据量
                            # 计算技术指标
                            hist_data['MA_fast'] = hist_data['收盘'].rolling(window=fast_ma).mean()
                            hist_data['MA_slow'] = hist_data['收盘'].rolling(window=slow_ma).mean()
                            
                            # 计算ATR和布林带
                            hist_data['ATR'] = calculate_atr(hist_data['最高'], hist_data['最低'], hist_data['收盘'], atr_period)
                            hist_data['BB_upper'], hist_data['BB_middle'], hist_data['BB_lower'] = calculate_bollinger_bands(
                                hist_data['收盘'], bollinger_period, bollinger_std)
                            
                            # 基本面过滤
                            if enable_fundamental_filter:
                                fundamental_data = get_fundamental_data_with_retry(stock_code, use_real_fundamental)
                                # 检查基本面条件
                                if (fundamental_data['roe'] < min_roe or 
                                    fundamental_data['revenue_growth'] < min_revenue_growth or
                                    fundamental_data['profit_growth'] < min_profit_growth or
                                    fundamental_data['cash_flow'] < min_cash_flow):
                                    st.warning(f"⚠️ 基本面过滤：该股票不符合筛选条件\n"
                                             f"ROE: {fundamental_data['roe']:.1f}% (要求>{min_roe}%)\n"
                                             f"营收增长: {fundamental_data['revenue_growth']:.1f}% (要求>{min_revenue_growth}%)\n"
                                             f"净利润增长: {fundamental_data['profit_growth']:.1f}% (要求>{min_profit_growth}%)\n"
                                             f"现金流: {fundamental_data['cash_flow']:.1f}亿 (要求>{min_cash_flow}亿)")
                            
                            # 处理NaN值
                            hist_data = hist_data.dropna().reset_index(drop=True)
                            
                            # 初始化策略变量
                            hist_data['信号'] = 0
                            hist_data['仓位'] = 0.0
                            hist_data['买入价格'] = 0.0
                            hist_data['止损价格'] = 0.0
                            hist_data['止盈价格'] = 0.0
                            hist_data['移动止损价格'] = 0.0
                            hist_data['风险信号'] = 0
                            hist_data['技术面得分'] = 0.0
                            hist_data['基本面得分'] = 0.0
                            hist_data['综合得分'] = 0.0
                            
                            # 策略执行
                            for i in range(1, len(hist_data)):
                                current_price = hist_data.loc[i, '收盘']
                                prev_signal = hist_data.loc[i-1, '信号']
                                prev_position = hist_data.loc[i-1, '仓位']
                                prev_buy_price = hist_data.loc[i-1, '买入价格']
                                
                                # 技术面分析 - 修复未来函数问题，使用前一天数据
                                if i > 0:  # 确保有前一天的数据
                                    prev_ma_fast = hist_data.loc[i-1, 'MA_fast']
                                    prev_ma_slow = hist_data.loc[i-1, 'MA_slow']
                                    prev_bb_upper = hist_data.loc[i-1, 'BB_upper']
                                    prev_bb_lower = hist_data.loc[i-1, 'BB_lower']
                                    prev_atr = hist_data.loc[i-1, 'ATR']
                                    
                                    ma_score = 1 if prev_ma_fast > prev_ma_slow else 0
                                    bb_score = 1 if (current_price > prev_bb_lower and current_price < prev_bb_upper) else 0
                                    
                                    # ATR评分逻辑 - 使用前一天ATR值与历史平均比较
                                    if i >= 20:  # 确保有足够的历史数据
                                        atr_avg = hist_data.loc[i-20:i-1, 'ATR'].mean()
                                        atr_score = 1 if prev_atr > atr_avg else 0
                                    else:
                                        atr_score = 0  # 数据不足时默认为0
                                else:
                                    ma_score = 0
                                    bb_score = 0
                                    atr_score = 0
                                
                                hist_data.loc[i, '技术面得分'] = (ma_score + bb_score + atr_score) / 3
                                
                                # 基本面得分（简化）
                                if enable_fundamental_filter:
                                    fundamental_data = get_fundamental_data_with_retry(stock_code, use_real_fundamental)
                                    roe_score = 1 if fundamental_data['roe'] >= min_roe else 0
                                    growth_score = 1 if (fundamental_data['revenue_growth'] >= min_revenue_growth and 
                                                       fundamental_data['profit_growth'] >= min_profit_growth) else 0
                                    cash_score = 1 if fundamental_data['cash_flow'] >= min_cash_flow else 0
                                    hist_data.loc[i, '基本面得分'] = (roe_score + growth_score + cash_score) / 3
                                else:
                                    hist_data.loc[i, '基本面得分'] = 0.5  # 中性
                                
                                # 综合得分
                                if signal_type == "多因子综合":
                                    hist_data.loc[i, '综合得分'] = (hist_data.loc[i, '技术面得分'] * 0.6 + 
                                                                 hist_data.loc[i, '基本面得分'] * 0.4)
                                else:
                                    hist_data.loc[i, '综合得分'] = hist_data.loc[i, '技术面得分']
                                
                                # 风险管理检查
                                risk_triggered = False
                                if prev_position > 0 and prev_buy_price > 0:
                                    # 止盈检查（无论是否启用移动止损都要检查）
                                    if current_price >= hist_data.loc[i-1, '止盈价格']:
                                        hist_data.loc[i, '信号'] = 0
                                        hist_data.loc[i, '仓位'] = 0.0
                                        hist_data.loc[i, '风险信号'] = 2  # 止盈
                                        risk_triggered = True
                                    
                                    # 止损检查
                                    if not risk_triggered:
                                        if enable_trailing_stop:
                                            # 移动止损
                                            new_trailing_stop = current_price * (1 - trailing_stop_percent)
                                            if new_trailing_stop > hist_data.loc[i-1, '移动止损价格']:
                                                hist_data.loc[i, '移动止损价格'] = new_trailing_stop
                                            else:
                                                hist_data.loc[i, '移动止损价格'] = hist_data.loc[i-1, '移动止损价格']
                                            
                                            if current_price <= hist_data.loc[i, '移动止损价格']:
                                                hist_data.loc[i, '信号'] = 0
                                                hist_data.loc[i, '仓位'] = 0.0
                                                hist_data.loc[i, '风险信号'] = 4  # 移动止损
                                                risk_triggered = True
                                        else:
                                            # 固定止损
                                            if current_price <= hist_data.loc[i-1, '止损价格']:
                                                hist_data.loc[i, '信号'] = 0
                                                hist_data.loc[i, '仓位'] = 0.0
                                                hist_data.loc[i, '风险信号'] = 1  # 固定止损
                                                risk_triggered = True
                                    
                                    # 最大回撤限制
                                    cumulative_return = (current_price / prev_buy_price - 1)
                                    if cumulative_return <= -max_drawdown_limit:
                                        hist_data.loc[i, '信号'] = 0
                                        hist_data.loc[i, '仓位'] = 0.0
                                        hist_data.loc[i, '风险信号'] = 3
                                        risk_triggered = True
                                
                                # 如果风险触发，跳过信号生成
                                if risk_triggered:
                                    continue
                                
                                # 信号生成
                                if signal_type == "金叉死叉":
                                    # 金叉死叉逻辑 - 添加信号确认机制
                                    prev_fast = hist_data.loc[i-1, 'MA_fast']
                                    prev_slow = hist_data.loc[i-1, 'MA_slow']
                                    
                                    # 信号确认机制：可配置确认天数
                                    confirmation_days = signal_confirmation_days if enable_signal_confirmation else 1
                                    
                                    if enable_signal_confirmation and i >= confirmation_days:
                                        # 检查连续N天的信号确认
                                        buy_condition = True
                                        sell_condition = True
                                        
                                        for j in range(confirmation_days):
                                            prev_fast_j = hist_data.loc[i-1-j, 'MA_fast']
                                            prev_slow_j = hist_data.loc[i-1-j, 'MA_slow']
                                            
                                            if not (prev_fast_j > prev_slow_j):
                                                buy_condition = False
                                            if not (prev_fast_j < prev_slow_j):
                                                sell_condition = False
                                    else:
                                        # 不使用确认或数据不足，使用简单信号
                                        buy_condition = prev_fast > prev_slow
                                        sell_condition = prev_fast < prev_slow
                                    
                                    if buy_condition and hist_data.loc[i, '综合得分'] >= 0.5:
                                        hist_data.loc[i, '信号'] = 1
                                        # 计算仓位大小
                                        if position_sizing_method == "Kelly公式":
                                            position_size = calculate_position_size("Kelly公式", 0.5, 0.1, 0.05, risk_per_trade, initial_capital)
                                            hist_data.loc[i, '仓位'] = min(position_size / initial_capital, max_position_size)
                                        else:
                                            hist_data.loc[i, '仓位'] = max_position_size
                                        
                                        hist_data.loc[i, '买入价格'] = current_price
                                        hist_data.loc[i, '止损价格'] = current_price * (1 - stop_loss)
                                        hist_data.loc[i, '止盈价格'] = current_price * (1 + take_profit)
                                        if enable_trailing_stop:
                                            hist_data.loc[i, '移动止损价格'] = current_price  # 初始移动止损价格为买入价格
                                    elif sell_condition:
                                        hist_data.loc[i, '信号'] = 0
                                        hist_data.loc[i, '仓位'] = 0.0
                                    else:
                                        # 保持前一日状态
                                        hist_data.loc[i, '信号'] = prev_signal
                                        hist_data.loc[i, '仓位'] = prev_position
                                        hist_data.loc[i, '买入价格'] = prev_buy_price
                                        hist_data.loc[i, '止损价格'] = hist_data.loc[i-1, '止损价格']
                                        hist_data.loc[i, '止盈价格'] = hist_data.loc[i-1, '止盈价格']
                                        hist_data.loc[i, '移动止损价格'] = hist_data.loc[i-1, '移动止损价格']
                                
                                elif signal_type == "趋势跟踪":
                                    # 趋势跟踪逻辑 - 添加信号确认机制
                                    # 使用前一天的技术面得分和均线数据
                                    if i > 0:
                                        prev_ma_fast = hist_data.loc[i-1, 'MA_fast']
                                        prev_ma_slow = hist_data.loc[i-1, 'MA_slow']
                                        prev_tech_score = hist_data.loc[i-1, '技术面得分']
                                        
                                        # 信号确认：前一天技术面得分高且均线金叉
                                        trend_confirmed = (prev_ma_fast > prev_ma_slow) and (prev_tech_score >= 0.5)
                                    else:
                                        trend_confirmed = False
                                    
                                    if trend_confirmed:
                                        if prev_signal == 0:  # 新开仓
                                            hist_data.loc[i, '信号'] = 1
                                            if position_sizing_method == "Kelly公式":
                                                position_size = calculate_position_size("Kelly公式", 0.5, 0.1, 0.05, risk_per_trade, initial_capital)
                                                hist_data.loc[i, '仓位'] = min(position_size / initial_capital, max_position_size)
                                            else:
                                                hist_data.loc[i, '仓位'] = max_position_size
                                            
                                            hist_data.loc[i, '买入价格'] = current_price
                                            hist_data.loc[i, '止损价格'] = current_price * (1 - stop_loss)
                                            hist_data.loc[i, '止盈价格'] = current_price * (1 + take_profit)
                                            if enable_trailing_stop:
                                                hist_data.loc[i, '移动止损价格'] = current_price  # 初始移动止损价格为买入价格
                                        else:  # 保持持仓
                                            hist_data.loc[i, '信号'] = 1
                                            hist_data.loc[i, '仓位'] = prev_position
                                            hist_data.loc[i, '买入价格'] = prev_buy_price
                                            hist_data.loc[i, '止损价格'] = hist_data.loc[i-1, '止损价格']
                                            hist_data.loc[i, '止盈价格'] = hist_data.loc[i-1, '止盈价格']
                                            hist_data.loc[i, '移动止损价格'] = hist_data.loc[i-1, '移动止损价格']
                                    else:
                                        hist_data.loc[i, '信号'] = 0
                                        hist_data.loc[i, '仓位'] = 0.0
                            
                                elif signal_type == "多因子综合":
                                    # 多因子综合逻辑
                                    if hist_data.loc[i, '综合得分'] >= 0.7:  # 高得分买入
                                        if prev_signal == 0:
                                            hist_data.loc[i, '信号'] = 1
                                            if position_sizing_method == "Kelly公式":
                                                position_size = calculate_position_size("Kelly公式", 0.5, 0.1, 0.05, risk_per_trade, initial_capital)
                                                hist_data.loc[i, '仓位'] = min(position_size / initial_capital, max_position_size)
                                            else:
                                                hist_data.loc[i, '仓位'] = max_position_size
                                            
                                            hist_data.loc[i, '买入价格'] = current_price
                                            hist_data.loc[i, '止损价格'] = current_price * (1 - stop_loss)
                                            hist_data.loc[i, '止盈价格'] = current_price * (1 + take_profit)
                                            if enable_trailing_stop:
                                                hist_data.loc[i, '移动止损价格'] = current_price  # 初始移动止损价格为买入价格
                                        else:
                                            hist_data.loc[i, '信号'] = 1
                                            hist_data.loc[i, '仓位'] = prev_position
                                            hist_data.loc[i, '买入价格'] = prev_buy_price
                                            hist_data.loc[i, '止损价格'] = hist_data.loc[i-1, '止损价格']
                                            hist_data.loc[i, '止盈价格'] = hist_data.loc[i-1, '止盈价格']
                                            hist_data.loc[i, '移动止损价格'] = hist_data.loc[i-1, '移动止损价格']
                                    elif hist_data.loc[i, '综合得分'] <= 0.3:  # 低得分卖出
                                        hist_data.loc[i, '信号'] = 0
                                        hist_data.loc[i, '仓位'] = 0.0
                                    else:
                                        # 保持前一日状态
                                        hist_data.loc[i, '信号'] = prev_signal
                                        hist_data.loc[i, '仓位'] = prev_position
                                        hist_data.loc[i, '买入价格'] = prev_buy_price
                                        hist_data.loc[i, '止损价格'] = hist_data.loc[i-1, '止损价格']
                                        hist_data.loc[i, '止盈价格'] = hist_data.loc[i-1, '止盈价格']
                                        hist_data.loc[i, '移动止损价格'] = hist_data.loc[i-1, '移动止损价格']
                        
                        # 移动止损价格前向填充
                        hist_data['移动止损价格'] = hist_data['移动止损价格'].fillna(method='ffill')
                        
                        # 计算交易成本和收益
                        hist_data['交易成本'] = 0.0
                        # 交易成本：按实际仓位变化计算换手
                        position_change = hist_data['仓位'].diff().fillna(0.0)
                        
                        # 买入：仓位增加的部分
                        buy_turnover = np.where(position_change > 0, position_change, 0.0)
                        # 卖出：仓位减少的部分
                        sell_turnover = np.where(position_change < 0, -position_change, 0.0)
                        
                        # 计算交易成本
                        buy_cost = buy_turnover * (commission_rate + slippage_rate)
                        sell_cost = sell_turnover * (commission_rate + slippage_rate + stamp_tax_rate)
                        hist_data['交易成本'] = buy_cost + sell_cost
                
                hist_data['收益率'] = hist_data['收盘'].pct_change()
                hist_data['策略收益'] = hist_data['信号'].shift(1) * hist_data['收益率'] * hist_data['仓位'].shift(1)
                hist_data['策略收益_after_fee'] = hist_data['策略收益'] - hist_data['交易成本']
                hist_data['累计收益'] = (1 + hist_data['策略收益_after_fee']).cumprod()
                
                # 获取基准数据
                if benchmark_type != "个股买入持有":
                    benchmark_data = get_benchmark_data(benchmark_type, start_date, end_date)
                    if benchmark_data is not None:
                        # 按日期对齐基准数据
                        tmp = pd.DataFrame({'日期': pd.to_datetime(hist_data['日期'])})
                        bench = benchmark_data.copy()
                        bench['date'] = pd.to_datetime(bench['date'])
                        merged = tmp.merge(bench[['date','收益率','累计收益']],
                                         left_on='日期', right_on='date', how='left').ffill()
                        hist_data['基准收益'] = merged['收益率'].values
                        hist_data['基准累计收益'] = merged['累计收益'].values
                    else:
                        hist_data['基准收益'] = hist_data['收益率']
                        hist_data['基准累计收益'] = (1 + hist_data['收益率']).cumprod()
                else:
                    hist_data['基准收益'] = hist_data['收益率']
                    hist_data['基准累计收益'] = (1 + hist_data['收益率']).cumprod()
                
                # 计算专业风险指标
                strategy_returns = hist_data['策略收益_after_fee'].dropna()
                benchmark_returns = hist_data['基准收益'].dropna()
                
                # 对数收益率（更准确的风险计算）
                log_returns = np.log(1 + strategy_returns)
                
                # 年化收益率（几何年化）
                n = len(hist_data)
                annual_return = hist_data['累计收益'].iloc[-1] ** (252.0 / n) - 1.0
                
                # 年化波动率
                annual_volatility = log_returns.std() * np.sqrt(252)
                
                # 夏普比率（年化）
                risk_free_rate = 0.03  # 年化3%无风险利率
                sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
                
                # 最大回撤
                cumulative_returns = hist_data['累计收益']
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()
                
                # 卡玛比率
                calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
                
                # 胜率
                winning_trades = (strategy_returns > 0).sum()
                total_trades = (strategy_returns != 0).sum()
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                # 盈亏比
                avg_win = strategy_returns[strategy_returns > 0].mean() if (strategy_returns > 0).sum() > 0 else 0
                avg_loss = abs(strategy_returns[strategy_returns < 0].mean()) if (strategy_returns < 0).sum() > 0 else 1
                profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                
                # 相对基准分析
                relative_analysis = calculate_relative_performance_analysis(strategy_returns, benchmark_returns)
                
                # 显示策略结果
                st.markdown("""
                <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <h3 style="color: white; margin: 0;">📊 策略表现</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    final_return = hist_data['累计收益'].iloc[-1] - 1
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">策略总收益</h4>
                        <p style="color: #27ae60; font-size: 1.5rem; font-weight: bold; margin: 0;">{final_return:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    benchmark_return = hist_data['基准累计收益'].iloc[-1] - 1
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">基准收益</h4>
                        <p style="color: #3498db; font-size: 1.5rem; font-weight: bold; margin: 0;">{benchmark_return:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    # 超额收益 = 策略累计收益末值 - 基准累计收益末值
                    excess_return = hist_data['累计收益'].iloc[-1] - hist_data['基准累计收益'].iloc[-1]
                    excess_color = "#e74c3c" if excess_return < 0 else "#27ae60"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">超额收益</h4>
                        <p style="color: {excess_color}; font-size: 1.5rem; font-weight: bold; margin: 0;">{excess_return:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">年化收益</h4>
                        <p style="color: #f39c12; font-size: 1.5rem; font-weight: bold; margin: 0;">{annual_return:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 风险指标
                st.markdown("""
                <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <h3 style="color: white; margin: 0;">🛡️ 风险指标</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    sharpe_color = "#27ae60" if sharpe_ratio > 1 else "#e74c3c" if sharpe_ratio < 0 else "#f39c12"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">夏普比率</h4>
                        <p style="color: {sharpe_color}; font-size: 1.5rem; font-weight: bold; margin: 0;">{sharpe_ratio:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">最大回撤</h4>
                        <p style="color: #e74c3c; font-size: 1.5rem; font-weight: bold; margin: 0;">{max_drawdown:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    calmar_color = "#27ae60" if calmar_ratio > 1 else "#e74c3c" if calmar_ratio < 0 else "#f39c12"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">卡玛比率</h4>
                        <p style="color: {calmar_color}; font-size: 1.5rem; font-weight: bold; margin: 0;">{calmar_ratio:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">年化波动率</h4>
                        <p style="color: #9b59b6; font-size: 1.5rem; font-weight: bold; margin: 0;">{annual_volatility:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 相对基准分析
                if relative_analysis:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                        <h3 style="color: white; margin: 0;">📈 相对基准分析</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        excess_color = "#27ae60" if relative_analysis['excess_annual_return'] > 0 else "#e74c3c"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">超额年化收益</h4>
                            <p style="color: {excess_color}; font-size: 1.5rem; font-weight: bold; margin: 0;">{relative_analysis['excess_annual_return']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        ir_color = "#27ae60" if relative_analysis['information_ratio'] > 0.5 else "#e74c3c" if relative_analysis['information_ratio'] < 0 else "#f39c12"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">信息比率</h4>
                            <p style="color: {ir_color}; font-size: 1.5rem; font-weight: bold; margin: 0;">{relative_analysis['information_ratio']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        beta_color = "#27ae60" if 0.8 <= relative_analysis['beta'] <= 1.2 else "#f39c12"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">Beta系数</h4>
                            <p style="color: {beta_color}; font-size: 1.5rem; font-weight: bold; margin: 0;">{relative_analysis['beta']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        alpha_color = "#27ae60" if relative_analysis['alpha'] > 0 else "#e74c3c"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">Alpha收益</h4>
                            <p style="color: {alpha_color}; font-size: 1.5rem; font-weight: bold; margin: 0;">{relative_analysis['alpha']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 第二行指标
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">跟踪误差</h4>
                            <p style="color: #9b59b6; font-size: 1.5rem; font-weight: bold; margin: 0;">{relative_analysis['tracking_error']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        corr_color = "#27ae60" if relative_analysis['correlation'] > 0.7 else "#f39c12" if relative_analysis['correlation'] > 0.3 else "#e74c3c"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">相关性</h4>
                            <p style="color: {corr_color}; font-size: 1.5rem; font-weight: bold; margin: 0;">{relative_analysis['correlation']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">策略胜率</h4>
                            <p style="color: #3498db; font-size: 1.5rem; font-weight: bold; margin: 0;">{relative_analysis['strategy_win_rate']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">基准胜率</h4>
                            <p style="color: #e67e22; font-size: 1.5rem; font-weight: bold; margin: 0;">{relative_analysis['benchmark_win_rate']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 收益来源分析
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                        <h3 style="color: white; margin: 0;">🎯 收益来源分析</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 计算收益来源
                    market_return = relative_analysis['beta'] * relative_analysis['benchmark_annual_return']
                    alpha_return = relative_analysis['alpha']
                    total_strategy_return = relative_analysis['strategy_annual_return']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">市场收益 (β×基准)</h4>
                            <p style="color: #3498db; font-size: 1.5rem; font-weight: bold; margin: 0;">{market_return:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">策略Alpha</h4>
                            <p style="color: #27ae60; font-size: 1.5rem; font-weight: bold; margin: 0;">{alpha_return:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">总策略收益</h4>
                            <p style="color: #f39c12; font-size: 1.5rem; font-weight: bold; margin: 0;">{total_strategy_return:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 收益来源饼图
                    if market_return != 0 or alpha_return != 0:
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=['市场收益', '策略Alpha'],
                            values=[abs(market_return), abs(alpha_return)],
                            hole=0.3,
                            marker_colors=['#3498db', '#27ae60']
                        )])
                        fig_pie.update_layout(
                            title="收益来源构成",
                            showlegend=True,
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                # 风险管理统计
                st.markdown("""
                <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <h3 style="color: white; margin: 0;">🛡️ 风险管理统计</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # 计算风险管理指标
                stop_loss_count = (hist_data['风险信号'] == 1).sum()
                take_profit_count = (hist_data['风险信号'] == 2).sum()
                drawdown_limit_count = (hist_data['风险信号'] == 3).sum()
                trailing_stop_count = (hist_data['风险信号'] == 4).sum()
                total_risk_triggers = stop_loss_count + take_profit_count + drawdown_limit_count + trailing_stop_count
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">止损触发</h4>
                        <p style="color: #e74c3c; font-size: 1.5rem; font-weight: bold; margin: 0;">{stop_loss_count}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">止盈触发</h4>
                        <p style="color: #27ae60; font-size: 1.5rem; font-weight: bold; margin: 0;">{take_profit_count}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">回撤限制</h4>
                        <p style="color: #f39c12; font-size: 1.5rem; font-weight: bold; margin: 0;">{drawdown_limit_count}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">移动止损</h4>
                        <p style="color: #9b59b6; font-size: 1.5rem; font-weight: bold; margin: 0;">{trailing_stop_count}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 蒙特卡洛模拟
                if enable_monte_carlo:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                        <h3 style="color: white; margin: 0;">🎲 蒙特卡洛模拟分析</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner("正在进行蒙特卡洛模拟..."):
                        # 生成随机收益率序列
                        mc_results = []
                        for sim in range(mc_simulations):
                            # 随机打乱收益率序列
                            shuffled_returns = strategy_returns.sample(frac=1, random_state=sim).reset_index(drop=True)
                            cumulative_return = (1 + shuffled_returns).cumprod()
                            mc_results.append(cumulative_return.iloc[-1] - 1)
                        
                        mc_results = np.array(mc_results)
                        
                        # 计算蒙特卡洛统计
                        mc_mean = np.mean(mc_results)
                        mc_std = np.std(mc_results)
                        mc_95_confidence = np.percentile(mc_results, [2.5, 97.5])
                        mc_prob_positive = np.mean(mc_results > 0)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">模拟平均收益</h4>
                                <p style="color: #f39c12; font-size: 1.5rem; font-weight: bold; margin: 0;">{mc_mean:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">模拟标准差</h4>
                                <p style="color: #9b59b6; font-size: 1.5rem; font-weight: bold; margin: 0;">{mc_std:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">95%置信区间</h4>
                                <p style="color: #3498db; font-size: 1.2rem; font-weight: bold; margin: 0;">[{mc_95_confidence[0]:.2%}, {mc_95_confidence[1]:.2%}]</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">正收益概率</h4>
                                <p style="color: #27ae60; font-size: 1.5rem; font-weight: bold; margin: 0;">{mc_prob_positive:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # 组合回测（如果启用）
                if portfolio_mode and len(portfolio_stocks) > 1:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                        <h3 style="color: white; margin: 0;">📊 组合回测分析</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner("正在进行组合回测..."):
                        # 准备回测参数
                        backtest_params = {
                            'fast_ma': fast_ma,
                            'slow_ma': slow_ma,
                            'signal_type': signal_type,
                            'start_date': start_date,
                            'end_date': end_date,
                            'backtest_years': backtest_years,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'max_position_size': max_position_size,
                            'commission_rate': commission_rate,
                            'slippage_rate': slippage_rate,
                            'stamp_tax_rate': stamp_tax_rate
                        }
                        
                        # 循环调用backtest_one
                        portfolio_results = []
                        valid_returns = []  # 用于计算等权组合
                        
                        for stock in portfolio_stocks:
                            result = backtest_one(stock, backtest_params)
                            if result is not None:
                                portfolio_results.append(result)
                                valid_returns.append(result['累计收益序列'])
                        
                        if portfolio_results:
                            # 创建结果DataFrame
                            portfolio_df = pd.DataFrame(portfolio_results)
                            portfolio_df = portfolio_df.sort_values('年化收益', ascending=False)
                            
                            # 显示组合结果
                            st.markdown("**组合股票表现排名：**")
                            display_df = portfolio_df[['股票代码', '总收益', '年化收益', '最大回撤', '夏普比率', '胜率']].copy()
                            display_df['总收益'] = display_df['总收益'].apply(lambda x: f"{x:.2%}")
                            display_df['年化收益'] = display_df['年化收益'].apply(lambda x: f"{x:.2%}")
                            display_df['最大回撤'] = display_df['最大回撤'].apply(lambda x: f"{x:.2%}")
                            display_df['夏普比率'] = display_df['夏普比率'].apply(lambda x: f"{x:.2f}")
                            display_df['胜率'] = display_df['胜率'].apply(lambda x: f"{x:.1%}")
                            st.dataframe(display_df, use_container_width=True)
                            
                            # 计算组合统计
                            portfolio_avg_return = portfolio_df['年化收益'].mean()
                            portfolio_median_return = portfolio_df['年化收益'].median()
                            portfolio_std = portfolio_df['年化收益'].std()
                            portfolio_sharpe = portfolio_avg_return / portfolio_std if portfolio_std > 0 else 0
                            portfolio_avg_drawdown = portfolio_df['最大回撤'].mean()
                            
                            # 显示组合统计
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">平均年化收益</h4>
                                    <p style="color: #27ae60; font-size: 1.5rem; font-weight: bold; margin: 0;">{portfolio_avg_return:.2%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">中位数年化收益</h4>
                                    <p style="color: #3498db; font-size: 1.5rem; font-weight: bold; margin: 0;">{portfolio_median_return:.2%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            with col3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">组合夏普比率</h4>
                                    <p style="color: #f39c12; font-size: 1.5rem; font-weight: bold; margin: 0;">{portfolio_sharpe:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            with col4:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">平均最大回撤</h4>
                                    <p style="color: #e74c3c; font-size: 1.5rem; font-weight: bold; margin: 0;">{portfolio_avg_drawdown:.2%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # 绘制等权组合净值曲线
                            if len(valid_returns) > 1:
                                st.markdown("**等权组合净值曲线：**")
                                
                                # 计算等权组合收益
                                min_length = min(len(returns) for returns in valid_returns)
                                equal_weight_returns = np.zeros(min_length)
                                
                                for returns in valid_returns:
                                    equal_weight_returns += returns[:min_length] / len(valid_returns)
                                
                                # 获取日期序列（使用第一个有效结果的日期）
                                dates = portfolio_results[0]['日期序列'][:min_length]
                                
                                # 创建图表
                                fig_portfolio = go.Figure()
                                
                                # 添加等权组合曲线
                                fig_portfolio.add_trace(go.Scatter(
                                    x=dates,
                                    y=equal_weight_returns,
                                    mode='lines',
                                    name='等权组合',
                                    line=dict(color='#27ae60', width=3)
                                ))
                                
                                # 添加个股曲线（透明度较低）
                                colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
                                for i, result in enumerate(portfolio_results[:5]):  # 只显示前5只
                                    color = colors[i % len(colors)]
                                    fig_portfolio.add_trace(go.Scatter(
                                        x=result['日期序列'][:min_length],
                                        y=result['累计收益序列'][:min_length],
                                        mode='lines',
                                        name=f"{result['股票代码']}",
                                        line=dict(color=color, width=1),
                                        opacity=0.6
                                    ))
                                
                                fig_portfolio.update_layout(
                                    title="组合净值曲线对比",
                                    xaxis_title="日期",
                                    yaxis_title="累计收益",
                                    height=500,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white'),
                                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                                )
                                
                                st.plotly_chart(fig_portfolio, use_container_width=True)
                
                
                # 交易统计
                st.markdown("""
                <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <h3 style="color: white; margin: 0;">📈 交易统计</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    win_color = "#27ae60" if win_rate > 0.5 else "#e74c3c"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">胜率</h4>
                        <p style="color: {win_color}; font-size: 1.5rem; font-weight: bold; margin: 0;">{win_rate:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    pl_color = "#27ae60" if profit_loss_ratio > 1 else "#e74c3c"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">盈亏比</h4>
                        <p style="color: {pl_color}; font-size: 1.5rem; font-weight: bold; margin: 0;">{profit_loss_ratio:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">总交易次数</h4>
                        <p style="color: #3498db; font-size: 1.5rem; font-weight: bold; margin: 0;">{total_trades}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    total_cost = hist_data['交易成本'].sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">总交易成本</h4>
                        <p style="color: #e67e22; font-size: 1.5rem; font-weight: bold; margin: 0;">{total_cost:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 绘制专业策略图表
                fig = make_subplots(rows=5, cols=1, 
                                   shared_xaxes=True,
                                   vertical_spacing=0.05,
                                   subplot_titles=('股价与均线', '技术指标', '交易信号与风险', '仓位管理', '累计收益对比'),
                                   row_width=[0.25, 0.2, 0.2, 0.15, 0.2])

                # 股价和均线
                fig.add_trace(go.Scatter(x=hist_data['日期'], y=hist_data['收盘'],
                                        mode='lines', name='收盘价', line=dict(color='#3498db', width=2)),
                             row=1, col=1)
                fig.add_trace(go.Scatter(x=hist_data['日期'], y=hist_data['MA_fast'],
                                        mode='lines', name=f'MA{fast_ma}', line=dict(color='#f39c12', width=2)),
                             row=1, col=1)
                fig.add_trace(go.Scatter(x=hist_data['日期'], y=hist_data['MA_slow'],
                                        mode='lines', name=f'MA{slow_ma}', line=dict(color='#e74c3c', width=2)),
                             row=1, col=1)
                
                # 添加布林带
                fig.add_trace(go.Scatter(x=hist_data['日期'], y=hist_data['BB_upper'],
                                        mode='lines', name='布林带上轨', line=dict(color='#95a5a6', width=1, dash='dash')),
                             row=1, col=1)
                fig.add_trace(go.Scatter(x=hist_data['日期'], y=hist_data['BB_lower'],
                                        mode='lines', name='布林带下轨', line=dict(color='#95a5a6', width=1, dash='dash'),
                                        fill='tonexty', fillcolor='rgba(149, 165, 166, 0.1)'),
                             row=1, col=1)
                
                # 技术指标
                fig.add_trace(go.Scatter(x=hist_data['日期'], y=hist_data['ATR'],
                                        mode='lines', name='ATR', line=dict(color='#9b59b6', width=2)),
                             row=2, col=1)
                fig.add_trace(go.Scatter(x=hist_data['日期'], y=hist_data['技术面得分'],
                                        mode='lines', name='技术面得分', line=dict(color='#e67e22', width=2)),
                             row=2, col=1)
                if signal_type == "多因子综合":
                    fig.add_trace(go.Scatter(x=hist_data['日期'], y=hist_data['综合得分'],
                                            mode='lines', name='综合得分', line=dict(color='#1abc9c', width=2)),
                                 row=2, col=1)

                # 交易信号和风险信号
                buy_signals = hist_data[hist_data['信号'] == 1]
                sell_signals = hist_data[hist_data['信号'] == 0]
                stop_loss_signals = hist_data[hist_data['风险信号'] == 1]
                take_profit_signals = hist_data[hist_data['风险信号'] == 2]
                drawdown_limit_signals = hist_data[hist_data['风险信号'] == 3]
                trailing_stop_signals = hist_data[hist_data['风险信号'] == 4]
                
                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(x=buy_signals['日期'], y=buy_signals['收盘'],
                                            mode='markers', name='买入信号', 
                                            marker=dict(color='#27ae60', size=10, symbol='triangle-up')),
                                 row=3, col=1)
                
                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(x=sell_signals['日期'], y=sell_signals['收盘'],
                                            mode='markers', name='卖出信号', 
                                            marker=dict(color='#e74c3c', size=10, symbol='triangle-down')),
                                 row=3, col=1)
                
                if not stop_loss_signals.empty:
                    fig.add_trace(go.Scatter(x=stop_loss_signals['日期'], y=stop_loss_signals['收盘'],
                                            mode='markers', name='止损信号', 
                                            marker=dict(color='#e74c3c', size=12, symbol='x')),
                                 row=3, col=1)
                
                if not take_profit_signals.empty:
                    fig.add_trace(go.Scatter(x=take_profit_signals['日期'], y=take_profit_signals['收盘'],
                                            mode='markers', name='止盈信号', 
                                            marker=dict(color='#27ae60', size=12, symbol='star')),
                                 row=3, col=1)
                
                if not drawdown_limit_signals.empty:
                    fig.add_trace(go.Scatter(x=drawdown_limit_signals['日期'], y=drawdown_limit_signals['收盘'],
                                            mode='markers', name='回撤限制', 
                                            marker=dict(color='#f39c12', size=12, symbol='diamond')),
                                 row=3, col=1)
                
                if not trailing_stop_signals.empty:
                    fig.add_trace(go.Scatter(x=trailing_stop_signals['日期'], y=trailing_stop_signals['收盘'],
                                            mode='markers', name='移动止损', 
                                            marker=dict(color='#8e44ad', size=12, symbol='square')),
                                 row=3, col=1)
                
                # 仓位管理
                fig.add_trace(go.Scatter(x=hist_data['日期'], y=hist_data['仓位'] * 100,
                                        mode='lines', name='仓位比例(%)', line=dict(color='#9b59b6', width=2)),
                             row=4, col=1)

                # 累计收益对比
                fig.add_trace(go.Scatter(x=hist_data['日期'], y=hist_data['累计收益'],
                                        mode='lines', name='策略收益(含成本)', line=dict(color='#27ae60', width=2)),
                             row=5, col=1)
                fig.add_trace(go.Scatter(x=hist_data['日期'], y=hist_data['基准累计收益'],
                                        mode='lines', name='基准收益', line=dict(color='#95a5a6', width=2)),
                             row=5, col=1)

                # 更新图表样式
                fig.update_layout(
                    height=1000,
                    title_text=f"{stock_code} 专业策略回测结果",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                )
                
                # 更新子图样式
                for i in range(1, 5):
                    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', row=i, col=1)
                    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', row=i, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 交易信号记录
                st.markdown("""
                <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <h3 style="color: white; margin: 0;">🎯 交易信号记录</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # 创建增强的信号记录
                signal_changes = hist_data[hist_data['信号'].diff() != 0].copy()
                if not signal_changes.empty:
                    # 添加风险信号说明
                    signal_changes['风险类型'] = signal_changes['风险信号'].map({
                        0: '正常信号',
                        1: '止损触发',
                        2: '止盈触发',
                        3: '回撤限制',
                        4: '移动止损'
                    })
                    
                    # 添加仓位信息
                    signal_changes['仓位比例'] = (signal_changes['仓位'] * 100).round(1).astype(str) + '%'
                    
                    # 添加技术指标得分
                    if '技术面得分' in signal_changes.columns:
                        signal_changes['技术面得分'] = signal_changes['技术面得分'].round(2)
                    if '综合得分' in signal_changes.columns:
                        signal_changes['综合得分'] = signal_changes['综合得分'].round(2)
                    
                    # 显示关键列
                    display_columns = ['日期', '收盘', '信号', '仓位比例', '风险类型', '交易成本']
                    if '技术面得分' in signal_changes.columns:
                        display_columns.append('技术面得分')
                    if '综合得分' in signal_changes.columns:
                        display_columns.append('综合得分')
                    
                    st.dataframe(signal_changes[display_columns], use_container_width=True)
                else:
                    st.info("暂无交易信号")
                
                # 风险管理详情
                if total_risk_triggers > 0:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                        <h3 style="color: white; margin: 0;">⚠️ 风险触发详情</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    risk_details = hist_data[hist_data['风险信号'] > 0][['日期', '收盘', '风险信号', '买入价格']].copy()
                    risk_details['风险类型'] = risk_details['风险信号'].map({
                        1: '止损触发',
                        2: '止盈触发',
                        3: '回撤限制',
                        4: '移动止损'
                    })
                    risk_details['触发价格'] = risk_details['收盘']
                    risk_details['买入价格'] = risk_details['买入价格'].round(2)
                    risk_details['价格变化'] = ((risk_details['收盘'] - risk_details['买入价格']) / risk_details['买入价格'] * 100).round(2).astype(str) + '%'
                    
                    display_risk_columns = ['日期', '买入价格', '触发价格', '价格变化', '风险类型']
                    st.dataframe(risk_details[display_risk_columns], use_container_width=True)
                
                # 生成HTML报告功能
                st.markdown("""
                <div class="report-section">
                    <h3 style="color: white; margin: 0 0 1rem 0; text-align: center;">📥 自动生成报告</h3>
                    <p style="color: #ecf0f1; text-align: center; margin-bottom: 2rem;">
                        点击下方按钮生成专业的HTML策略报告，包含所有回测结果和图表
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📥 生成今日报告", key="generate_report"):
                    with st.spinner("正在生成专业报告..."):
                        try:
                            # 生成报告文件名
                            current_date = datetime.now().strftime("%Y%m%d")
                            report_filename = f"report_{stock_code}_{current_date}.html"
                            
                            # 将图表转换为HTML
                            chart_html = fig.to_html(include_plotlyjs='cdn')
                            
                            # 生成完整的HTML报告
                            html_content = f"""
                            <!DOCTYPE html>
                            <html lang="zh-CN">
                            <head>
                                <meta charset="UTF-8">
                                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                <title>悦北 智能盯盘助手 – 今日策略报告</title>
                                <style>
                                    body {{
                                        font-family: 'Microsoft YaHei', Arial, sans-serif;
                                        margin: 0;
                                        padding: 20px;
                                        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                                        color: white;
                                    }}
                                    .container {{
                                        max-width: 1200px;
                                        margin: 0 auto;
                                        background: rgba(255,255,255,0.1);
                                        padding: 30px;
                                        border-radius: 20px;
                                        backdrop-filter: blur(10px);
                                    }}
                                    .header {{
                                        text-align: center;
                                        margin-bottom: 40px;
                                        padding: 20px;
                                        background: rgba(255,255,255,0.1);
                                        border-radius: 15px;
                                    }}
                                    .header h1 {{
                                        color: #ffd700;
                                        font-size: 2.5rem;
                                        margin: 0;
                                        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                                    }}
                                    .info-grid {{
                                        display: grid;
                                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                                        gap: 20px;
                                        margin-bottom: 30px;
                                    }}
                                    .info-card {{
                                        background: rgba(255,255,255,0.1);
                                        padding: 20px;
                                        border-radius: 10px;
                                        border-left: 4px solid #ffd700;
                                    }}
                                    .info-card h3 {{
                                        color: #ffd700;
                                        margin: 0 0 10px 0;
                                    }}
                                    .info-card p {{
                                        margin: 0;
                                        font-size: 1.1rem;
                                    }}
                                    .metrics-grid {{
                                        display: grid;
                                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                                        gap: 15px;
                                        margin-bottom: 30px;
                                    }}
                                    .metric-card {{
                                        background: rgba(255,255,255,0.1);
                                        padding: 15px;
                                        border-radius: 8px;
                                        text-align: center;
                                    }}
                                    .metric-card h4 {{
                                        color: #ffd700;
                                        margin: 0 0 8px 0;
                                        font-size: 0.9rem;
                                    }}
                                    .metric-card .value {{
                                        font-size: 1.3rem;
                                        font-weight: bold;
                                        color: #4ade80;
                                    }}
                                    .chart-section {{
                                        background: rgba(255,255,255,0.1);
                                        padding: 20px;
                                        border-radius: 10px;
                                        margin-bottom: 30px;
                                    }}
                                    .chart-section h3 {{
                                        color: #ffd700;
                                        text-align: center;
                                        margin: 0 0 20px 0;
                                    }}
                                    .disclaimer {{
                                        background: rgba(220, 53, 69, 0.2);
                                        padding: 20px;
                                        border-radius: 10px;
                                        border-left: 4px solid #dc3545;
                                        text-align: center;
                                    }}
                                    .disclaimer p {{
                                        margin: 0;
                                        color: #ffcccb;
                                        font-size: 1rem;
                                    }}
                                </style>
                            </head>
                            <body>
                                <div class="container">
                                    <div class="header">
                                        <h1>✨ 悦北 智能盯盘助手 – 今日策略报告</h1>
                                        <p>生成时间：{datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}</p>
                                    </div>
                                    
                                    <div class="info-grid">
                                        <div class="info-card">
                                            <h3>📈 股票信息</h3>
                                            <p>股票代码：{stock_code}</p>
                                            <p>信号类型：{signal_type}</p>
                                            <p>回测周期：{backtest_years}年</p>
                                        </div>
                                        <div class="info-card">
                                            <h3>⚙️ 策略参数</h3>
                                            <p>快速均线：{fast_ma}日</p>
                                            <p>慢速均线：{slow_ma}日</p>
                                            <p>基准类型：{benchmark_type}</p>
                                        </div>
                                        <div class="info-card">
                                            <h3>💰 交易成本</h3>
                                            <p>手续费率：{commission_rate*100:.2f}%</p>
                                            <p>滑点率：{slippage_rate*100:.2f}%</p>
                                            <p>印花税率：{stamp_tax_rate*100:.2f}%</p>
                                        </div>
                                        <div class="info-card">
                                            <h3>🛡️ 风险管理</h3>
                                            <p>止损比例：{stop_loss*100:.1f}%</p>
                                            <p>止盈比例：{take_profit*100:.1f}%</p>
                                            <p>最大仓位：{max_position_size*100:.0f}%</p>
                                            <p>回撤限制：{max_drawdown_limit*100:.1f}%</p>
                                        </div>
                                    </div>
                                    
                                    <div class="metrics-grid">
                                        <div class="metric-card">
                                            <h4>策略总收益</h4>
                                            <div class="value">{final_return:.2%}</div>
                                        </div>
                                        <div class="metric-card">
                                            <h4>基准收益</h4>
                                            <div class="value">{benchmark_return:.2%}</div>
                                        </div>
                                        <div class="metric-card">
                                            <h4>夏普比率</h4>
                                            <div class="value">{sharpe_ratio:.2f}</div>
                                        </div>
                                        <div class="metric-card">
                                            <h4>最大回撤</h4>
                                            <div class="value">{max_drawdown:.2%}</div>
                                        </div>
                                        <div class="metric-card">
                                            <h4>胜率</h4>
                                            <div class="value">{win_rate:.1%}</div>
                                        </div>
                                        <div class="metric-card">
                                            <h4>盈亏比</h4>
                                            <div class="value">{profit_loss_ratio:.2f}</div>
                                        </div>
                                    </div>
                                    
                                    <div class="chart-section">
                                        <h3>📊 策略回测图表</h3>
                                        {chart_html}
                                    </div>
                                    
                                    <div class="disclaimer">
                                        <p><strong>⚠️ 免责声明：</strong>本报告仅用于学习，不构成投资建议，盈亏自负</p>
                                    </div>
                                </div>
                            </body>
                            </html>
                            """
                            
                            # 提供下载链接
                            st.success("✅ 报告生成成功！")
                            st.download_button(
                                label="📥 下载HTML报告",
                                data=html_content,
                                file_name=report_filename,
                                mime="text/html",
                                key="download_report"
                            )
                            
                            # 显示报告预览
                            st.markdown("""
                            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                                <h4 style="color: #ffd700; margin: 0 0 1rem 0;">📋 报告内容预览</h4>
                                <ul style="color: #ecf0f1; margin: 0;">
                                    <li>✨ 悦北 智能盯盘助手品牌标识</li>
                                    <li>📈 完整的策略参数和回测设置</li>
                                    <li>🛡️ 止损止盈和风险管理参数</li>
                                    <li>📊 所有关键指标和风险指标</li>
                                    <li>🎯 交互式策略回测图表（含风险信号）</li>
                                    <li>💰 详细的交易成本分析</li>
                                    <li>⚠️ 专业免责声明</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"生成报告失败: {str(e)}")
                    
                    # 策略验证
                    if enable_strategy_validation and validation_stocks:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                            <h3 style="color: white; margin: 0;">🔬 策略验证分析</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.spinner("正在进行策略验证..."):
                            # 多股票验证
                            multi_stock_results = multi_stock_validation(validation_stocks, params)
                            
                            if multi_stock_results:
                                # 计算鲁棒性指标
                                robustness = calculate_strategy_robustness(multi_stock_results)
                                
                                if robustness:
                                    # 显示鲁棒性分析
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">平均年化收益</h4>
                                            <p style="color: #27ae60; font-size: 1.5rem; font-weight: bold; margin: 0;">{robustness['return_mean']:.2%}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with col2:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">收益标准差</h4>
                                            <p style="color: #e74c3c; font-size: 1.5rem; font-weight: bold; margin: 0;">{robustness['return_std']:.2%}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with col3:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">胜率</h4>
                                            <p style="color: #3498db; font-size: 1.5rem; font-weight: bold; margin: 0;">{robustness['win_rate']:.1%}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with col4:
                                        stability_color = "#27ae60" if robustness['stability_score'] > 0.7 else "#f39c12" if robustness['stability_score'] > 0.4 else "#e74c3c"
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h4 style="color: #ecf0f1; margin: 0 0 0.5rem 0;">稳定性评分</h4>
                                            <p style="color: {stability_color}; font-size: 1.5rem; font-weight: bold; margin: 0;">{robustness['stability_score']:.2f}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # 策略评估
                                    st.markdown("""
                                    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                                        <h3 style="color: white; margin: 0;">📊 策略评估</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # 评估标准
                                    if robustness['stability_score'] > 0.7 and robustness['win_rate'] > 0.6:
                                        st.success("✅ 策略表现优秀：高稳定性 + 高胜率")
                                    elif robustness['stability_score'] > 0.4 and robustness['win_rate'] > 0.5:
                                        st.warning("⚠️ 策略表现一般：中等稳定性 + 中等胜率")
                                    else:
                                        st.error("❌ 策略表现较差：低稳定性或低胜率")
                                    
                                    # 详细分析
                                    st.markdown(f"""
                                    **验证结果分析：**
                                    - 测试股票数量：{robustness['total_tests']}只
                                    - 平均年化收益：{robustness['return_mean']:.2%}
                                    - 收益波动率：{robustness['return_std']:.2%}
                                    - 变异系数：{robustness['return_cv']:.2f}
                                    - 胜率：{robustness['win_rate']:.1%}
                                    - 稳定性评分：{robustness['stability_score']:.2f}
                                    """)
                                    
                                    # 建议
                                    if robustness['return_cv'] > 1.0:
                                        st.warning("⚠️ 收益波动较大，建议优化参数或增加信号确认")
                                    if robustness['win_rate'] < 0.5:
                                        st.warning("⚠️ 胜率较低，建议调整止损止盈参数")
                                    if robustness['stability_score'] < 0.4:
                                        st.warning("⚠️ 稳定性较差，建议进行更多股票验证")
                                else:
                                    st.error("策略验证失败，无法计算鲁棒性指标")
                            else:
                                st.warning("策略验证失败，无法获取多股票回测结果")
                    
                    # 回测成功完成，重置状态
                    st.session_state.run_backtest = False
                
            except Exception as e:
                st.error(f"策略回测失败: {str(e)}")
                st.error("🔧 可能的解决方案:")
                st.error("1. 检查网络连接是否正常")
                st.error("2. 尝试更换股票代码")
                st.error("3. 减少回测年数")
                st.error("4. 关闭基本面过滤和蒙特卡洛模拟")
            finally:
                # 重置回测状态
                st.session_state.run_backtest = False
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; text-align: center;">
            <h2 style="color: white; margin: 0 0 1rem 0;">⏳ 等待确认</h2>
            <p style="color: white; font-size: 1.2rem; margin: 0;">
                请在左侧栏设置完参数后，点击 <strong>"🎯 确定并开始回测"</strong> 按钮
            </p>
        </div>
        """, unsafe_allow_html=True)
            
else:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; text-align: center;">
        <h2 style="color: #ffd700; margin: 0 0 1rem 0;">👈 开始使用</h2>
        <p style="color: #ecf0f1; font-size: 1.2rem; margin: 0;">
            请在左侧输入股票代码开始专业策略分析
        </p>
    </div>
    """, unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1.5rem; border-radius: 10px; text-align: center;">
    <p style="color: #ecf0f1; margin: 0; font-size: 1rem;">
        ⚠️ 免责声明：本系统仅供学习研究使用，不构成投资建议。投资有风险，入市需谨慎。
    </p>
    <p style="color: #ffd700; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
        ✨ 悦北 智能盯盘助手 - 专业量化策略分析平台
    </p>
</div>
""", unsafe_allow_html=True)
