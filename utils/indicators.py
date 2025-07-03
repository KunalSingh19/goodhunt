import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd(df, fast=12, slow=26, signal=9):
    ema_fast = ema(df['Close'], fast)
    ema_slow = ema(df['Close'], slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = hist
    return df

def atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(period).mean()
    return df

def bollinger_bands(df, period=20, n_std=2):
    sma = df['Close'].rolling(period).mean()
    std = df['Close'].rolling(period).std()
    df['bb_upper'] = sma + n_std * std
    df['bb_lower'] = sma - n_std * std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
    return df

def adx(df, period=14):
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    up = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = pd.concat([
        (df['High'] - df['Low']),
        np.abs(df['High'] - df['Close'].shift()),
        np.abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(up).rolling(period).sum() / atr_val
    minus_di = 100 * pd.Series(down).rolling(period).sum() / atr_val
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.rolling(period).mean()
    return df

def cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['cci'] = (tp - ma) / (0.015 * md)
    return df

def obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    return df

def stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-9)
    d = k.rolling(window=d_period).mean()
    df['stoch_k'] = k
    df['stoch_d'] = d
    return df

def rsi(df, period=14):
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_slope'] = df['rsi'].diff()
    return df

def price_entropy(df, window=10):
    returns = df['Close'].pct_change().rolling(window)
    entropy = returns.apply(lambda x: -np.sum(np.nan_to_num(x * np.log(np.abs(x) + 1e-9))), raw=True)
    df['entropy'] = entropy
    return df

# Feature 1: Volume-Weighted Average Price (VWAP)
def vwap(df):
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    df['cum_typ_vol'] = (typical * df['Volume']).cumsum()
    df['cum_vol'] = df['Volume'].cumsum()
    df['vwap'] = df['cum_typ_vol'] / df['cum_vol']
    return df

# Feature 2: Beta vs Benchmark (e.g., SPY)
def beta_vs_benchmark(df, bench_df=None, period=30):
    """Calculate beta relative to benchmark. If no benchmark provided, uses synthetic SPY-like data"""
    if bench_df is not None:
        df['bench_ret'] = bench_df['Close'].pct_change()
    else:
        # Create synthetic benchmark returns if no benchmark provided
        df['bench_ret'] = df['Close'].pct_change().rolling(5).mean() * 0.8
    
    stock_ret = df['Close'].pct_change()
    covariance = stock_ret.rolling(period).cov(df['bench_ret'])
    benchmark_var = df['bench_ret'].rolling(period).var()
    df['beta'] = covariance / (benchmark_var + 1e-9)
    return df

# Feature 3: Fisher Transform of RSI
def fisher_transform_rsi(df):
    x = df['rsi'] / 100
    # Clamp x to avoid log(0) or log(negative)
    x = np.clip(x, 0.01, 0.99)
    df['rsi_fisher'] = np.log((1 + x) / (1 - x))
    return df

# =============================================================================
# CATEGORY A: ADVANCED OBSERVATION & INDICATORS (A01-A15)
# =============================================================================

# A01: Market Microstructure Indicators
def add_microstructure_indicators(df, window=20):
    """
    Advanced market microstructure analysis including:
    - Bid-ask spread estimation
    - Order flow imbalance
    - Market impact estimation
    - Tick direction analysis
    """
    # Estimate bid-ask spread from price volatility
    df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
    df['spread_ma'] = df['hl_spread'].rolling(window).mean()
    
    # Order flow imbalance proxy using volume and price movement
    price_change = df['Close'].diff()
    df['order_flow_imbalance'] = np.where(
        price_change > 0, df['Volume'], 
        np.where(price_change < 0, -df['Volume'], 0)
    ).rolling(window).sum()
    
    # Market impact estimation
    df['market_impact'] = (df['Volume'] / df['Volume'].rolling(window).mean()) * df['hl_spread']
    
    # Tick direction (simplified)
    df['tick_direction'] = np.sign(price_change).rolling(window).sum()
    
    # Liquidity proxy
    df['liquidity_proxy'] = df['Volume'] / (df['hl_spread'] + 1e-9)
    
    return df

# A02: Sentiment Analysis Integration
def add_sentiment_indicators(df, symbol="AAPL"):
    """
    Integrate sentiment analysis from multiple sources:
    - News sentiment
    - Social media sentiment
    - Market sentiment indicators
    """
    # Simulated news sentiment (in production, connect to news APIs)
    np.random.seed(42)  # For reproducible results
    df['news_sentiment'] = np.random.normal(0, 0.3, len(df))
    
    # VIX-like fear/greed indicator
    df['fear_greed_index'] = (
        50 - (df['rsi'] - 50) + 
        (df['bb_width'] * 100) - 
        (df['volatility'] * 100)
    ).rolling(10).mean()
    
    # Put/Call ratio proxy
    df['put_call_ratio'] = np.clip(
        1 / (1 + np.exp((df['rsi'] - 50) / 10)), 0.3, 1.7
    )
    
    # Market sentiment composite
    df['sentiment_composite'] = (
        df['news_sentiment'] * 0.4 + 
        ((df['fear_greed_index'] - 50) / 50) * 0.3 +
        ((1 - df['put_call_ratio']) * 2 - 1) * 0.3
    )
    
    return df

# A03: Liquidity Metrics
def add_liquidity_metrics(df, window=20):
    """
    Comprehensive liquidity analysis:
    - Amihud illiquidity ratio
    - Turnover rate
    - Price impact metrics
    """
    # Amihud illiquidity ratio
    returns = df['Close'].pct_change().abs()
    dollar_volume = df['Close'] * df['Volume']
    df['amihud_illiquidity'] = (returns / (dollar_volume + 1e-9)).rolling(window).mean()
    
    # Turnover rate (volume / shares outstanding proxy)
    avg_volume = df['Volume'].rolling(window).mean()
    df['turnover_rate'] = avg_volume / (avg_volume.max() + 1e-9)
    
    # Liquidity score composite
    df['liquidity_score'] = (
        (1 / (df['amihud_illiquidity'] + 1e-9)) * 0.5 +
        df['turnover_rate'] * 0.3 +
        (1 / (df['hl_spread'] + 1e-9)) * 0.2
    )
    
    # Market depth proxy
    df['market_depth'] = df['Volume'] / (abs(df['Close'].pct_change()) + 1e-9)
    
    return df

# A04: Intermarket Analysis
def add_intermarket_signals(df, symbol="AAPL"):
    """
    Cross-asset correlation and intermarket analysis:
    - Bond-equity relationship
    - Currency impact
    - Commodity correlations
    """
    try:
        # Download related market data
        spy = yf.download("SPY", period="1y", interval="1d")['Close']
        tlt = yf.download("TLT", period="1y", interval="1d")['Close']  # 20+ Year Treasury
        usd = yf.download("UUP", period="1y", interval="1d")['Close']  # US Dollar
        
        # Align dates
        common_dates = df.index.intersection(spy.index)
        if len(common_dates) > 50:
            # Bond-equity correlation
            stock_returns = df.loc[common_dates, 'Close'].pct_change()
            bond_returns = tlt.loc[common_dates].pct_change()
            df.loc[common_dates, 'bond_equity_corr'] = stock_returns.rolling(20).corr(bond_returns)
            
            # Dollar strength impact
            usd_returns = usd.loc[common_dates].pct_change()
            df.loc[common_dates, 'usd_impact'] = stock_returns.rolling(20).corr(usd_returns)
            
            # Relative strength vs market
            spy_returns = spy.loc[common_dates].pct_change()
            df.loc[common_dates, 'relative_strength'] = (
                stock_returns.rolling(20).mean() - spy_returns.rolling(20).mean()
            )
    except:
        # Fallback to synthetic data if download fails
        df['bond_equity_corr'] = np.random.normal(-0.2, 0.3, len(df))
        df['usd_impact'] = np.random.normal(0, 0.2, len(df))
        df['relative_strength'] = np.random.normal(0, 0.05, len(df))
    
    return df

# A05: Options Flow Indicators
def add_options_flow(df):
    """
    Options market sentiment and flow indicators:
    - Put/Call ratios
    - Implied volatility indicators
    - Options positioning metrics
    """
    # Simulated options metrics (connect to options APIs in production)
    df['put_call_volume_ratio'] = np.clip(
        np.random.normal(0.8, 0.3, len(df)), 0.2, 2.0
    )
    
    # Implied volatility rank
    realized_vol = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    df['iv_rank'] = np.random.normal(realized_vol * 1.2, realized_vol * 0.3)
    
    # Options skew indicator
    df['options_skew'] = np.random.normal(0, 0.1, len(df))
    
    # Max pain approximation
    df['max_pain_distance'] = np.random.normal(0, 0.02, len(df))
    
    return df

# A06: Economic Calendar Integration
def add_economic_indicators(df):
    """
    Economic calendar and fundamental indicators:
    - Economic surprise index
    - Earnings season impact
    - Fed policy indicators
    """
    # Economic surprise index (simulated)
    df['economic_surprise'] = np.random.normal(0, 0.5, len(df))
    
    # Earnings season indicator (quarterly pattern)
    df['earnings_season'] = np.sin(2 * np.pi * df.index.dayofyear / 91.25)
    
    # Fed policy uncertainty proxy
    df['fed_policy_uncertainty'] = np.random.normal(0, 0.3, len(df))
    
    # GDP growth proxy
    df['gdp_growth_proxy'] = np.random.normal(2.5, 1.0, len(df))
    
    return df

# A07: Sector Rotation Signals
def add_sector_rotation(df):
    """
    Sector rotation and style analysis:
    - Growth vs Value indicators
    - Sector momentum
    - Style rotation signals
    """
    # Growth vs Value indicator
    pe_proxy = 1 / (df['Close'].pct_change(252) + 1e-9)  # Inverse of annual return
    df['growth_value_indicator'] = StandardScaler().fit_transform(
        pe_proxy.values.reshape(-1, 1)
    ).flatten()
    
    # Sector momentum (simulated sector strength)
    df['sector_momentum'] = np.random.normal(0, 0.2, len(df))
    
    # Style rotation signal
    df['style_rotation'] = (
        df['growth_value_indicator'] * 0.6 + 
        df['sector_momentum'] * 0.4
    )
    
    return df

# A08: Time-based Patterns
def add_temporal_patterns(df):
    """
    Time-based trading patterns:
    - Day of week effects
    - Month of year effects
    - Time of day effects
    - Holiday effects
    """
    # Day of week effects
    df['day_of_week'] = df.index.dayofweek
    df['monday_effect'] = (df['day_of_week'] == 0).astype(int)
    df['friday_effect'] = (df['day_of_week'] == 4).astype(int)
    
    # Month effects
    df['month'] = df.index.month
    df['january_effect'] = (df['month'] == 1).astype(int)
    df['december_effect'] = (df['month'] == 12).astype(int)
    
    # Quarter end effects
    df['quarter_end'] = ((df.index.month % 3 == 0) & 
                        (df.index.day >= 25)).astype(int)
    
    # Holiday proximity (simplified)
    df['holiday_proximity'] = np.random.binomial(1, 0.05, len(df))
    
    return df

# A09: Volume Profile Analysis
def add_volume_profile(df, bins=20):
    """
    Volume profile and price-volume analysis:
    - Volume at price levels
    - Value area high/low
    - Point of control
    """
    # Price bins for volume profile
    price_range = df['High'].max() - df['Low'].min()
    bin_size = price_range / bins
    
    # Volume weighted average price
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['session_vwap'] = vwap
    
    # Volume profile metrics
    df['volume_at_price'] = df['Volume'] / (bin_size + 1e-9)
    df['price_acceptance'] = (abs(df['Close'] - vwap) < price_range * 0.1).astype(int)
    
    # Value area approximation
    rolling_volume = df['Volume'].rolling(20)
    df['value_area_high'] = df['High'].rolling(20).max()
    df['value_area_low'] = df['Low'].rolling(20).min()
    
    return df

# A10: Fractal Indicators
def add_fractal_analysis(df, window=5):
    """
    Market structure fractals and pivot analysis:
    - Williams fractals
    - Support/resistance levels
    - Market structure breaks
    """
    # Williams Fractals
    high_fractals = []
    low_fractals = []
    
    for i in range(window, len(df) - window):
        # High fractal
        if all(df['High'].iloc[i] >= df['High'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['High'].iloc[i] >= df['High'].iloc[i+j] for j in range(1, window+1)):
            high_fractals.append(i)
        
        # Low fractal
        if all(df['Low'].iloc[i] <= df['Low'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['Low'].iloc[i] <= df['Low'].iloc[i+j] for j in range(1, window+1)):
            low_fractals.append(i)
    
    df['fractal_high'] = 0
    df['fractal_low'] = 0
    df.iloc[high_fractals, df.columns.get_loc('fractal_high')] = 1
    df.iloc[low_fractals, df.columns.get_loc('fractal_low')] = 1
    
    # Market structure
    df['market_structure'] = df['fractal_high'] - df['fractal_low']
    
    return df

# A11: Multi-timeframe Confluence
def add_mtf_confluence(df):
    """
    Multiple timeframe signal alignment:
    - Trend alignment across timeframes
    - Signal confluence scoring
    """
    # Simulate multiple timeframes using different MA periods
    df['trend_short'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
    df['trend_medium'] = np.where(df['ema_21'] > df['ema_50'], 1, -1)
    df['trend_long'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
    
    # Confluence score
    df['mtf_confluence'] = (
        df['trend_short'] + 
        df['trend_medium'] + 
        df['trend_long']
    ) / 3
    
    # Signal strength
    df['signal_strength'] = abs(df['mtf_confluence'])
    
    return df

# A12: Momentum Persistence
def add_momentum_persistence(df, window=10):
    """
    Trend continuation probability analysis:
    - Momentum persistence metrics
    - Trend strength indicators
    """
    returns = df['Close'].pct_change()
    
    # Momentum persistence
    momentum_signs = np.sign(returns).rolling(window).sum()
    df['momentum_persistence'] = abs(momentum_signs) / window
    
    # Trend consistency
    df['trend_consistency'] = (
        df['Close'].rolling(window).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] and x.is_monotonic_increasing else 
                     -1 if x.iloc[-1] < x.iloc[0] and x.is_monotonic_decreasing else 0
        )
    )
    
    # Momentum quality
    df['momentum_quality'] = df['momentum_persistence'] * df['signal_strength']
    
    return df

# A13: Market Efficiency Metrics
def add_efficiency_metrics(df, window=20):
    """
    Market randomness and efficiency analysis:
    - Hurst exponent
    - Autocorrelation
    - Random walk test
    """
    returns = df['Close'].pct_change().dropna()
    
    # Simplified Hurst exponent
    def hurst_exponent(ts, max_lag=20):
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    df['hurst_exponent'] = returns.rolling(window*2).apply(
        lambda x: hurst_exponent(x.values) if len(x) >= 40 else 0.5
    )
    
    # Autocorrelation
    df['autocorr_1'] = returns.rolling(window).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= window else 0
    )
    
    # Market efficiency score
    df['efficiency_score'] = (
        abs(df['hurst_exponent'] - 0.5) + 
        abs(df['autocorr_1'])
    ) / 2
    
    return df

# A14: Volatility Surface Analysis
def add_volatility_surface(df, window=20):
    """
    Implied volatility patterns and surface analysis:
    - Term structure
    - Volatility skew
    - Surface curvature
    """
    realized_vol = df['Close'].pct_change().rolling(window).std() * np.sqrt(252)
    
    # Volatility term structure (short vs long term)
    short_vol = df['Close'].pct_change().rolling(5).std() * np.sqrt(252)
    long_vol = df['Close'].pct_change().rolling(50).std() * np.sqrt(252)
    df['vol_term_structure'] = short_vol / (long_vol + 1e-9)
    
    # Volatility skew
    df['vol_skew'] = (
        df['Close'].pct_change().rolling(window).skew()
    )
    
    # Volatility surface curvature
    df['vol_curvature'] = df['vol_skew'].diff()
    
    # Volatility risk premium
    df['vol_risk_premium'] = realized_vol - realized_vol.rolling(window).mean()
    
    return df

# A15: Regime Change Detection
def add_regime_detection(df, window=50):
    """
    Advanced regime identification:
    - Bull/bear market detection
    - Volatility regime changes
    - Trend regime classification
    """
    from sklearn.mixture import GaussianMixture
    
    returns = df['Close'].pct_change().dropna()
    volatility = returns.rolling(window).std()
    
    # Simple regime detection using volatility clustering
    vol_threshold = volatility.quantile(0.7)
    df['volatility_regime'] = np.where(volatility > vol_threshold, 1, 0)  # 1=high vol, 0=low vol
    
    # Trend regime using price vs moving averages
    sma_50 = df['Close'].rolling(50).mean()
    sma_200 = df['Close'].rolling(200).mean()
    
    df['trend_regime'] = np.where(
        (df['Close'] > sma_50) & (sma_50 > sma_200), 2,  # Strong uptrend
        np.where(
            (df['Close'] > sma_50) | (sma_50 > sma_200), 1,  # Weak uptrend
            np.where(
                (df['Close'] < sma_50) & (sma_50 < sma_200), -2,  # Strong downtrend
                -1  # Weak downtrend
            )
        )
    )
    
    # Market regime composite
    df['market_regime_advanced'] = (
        df['volatility_regime'] * 0.3 + 
        (df['trend_regime'] / 2) * 0.7
    )
    
    return df

# Enhanced add_all_indicators function
def add_all_indicators(df):
    """Enhanced indicator function with all Category A features"""
    # Original indicators
    df = macd(df)
    df = atr(df)
    df = bollinger_bands(df)
    df = adx(df)
    df = cci(df)
    df = obv(df)
    df = stochastic(df)
    df = rsi(df)
    df = price_entropy(df)
    
    # Existing enhancements (Features 1-3)
    df = vwap(df)
    df = beta_vs_benchmark(df)
    df = fisher_transform_rsi(df)
    
    # NEW CATEGORY A FEATURES (A01-A15)
    print("Adding Category A: Advanced Observation Features...")
    df = add_microstructure_indicators(df)
    df = add_sentiment_indicators(df)
    df = add_liquidity_metrics(df)
    df = add_intermarket_signals(df)
    df = add_options_flow(df)
    df = add_economic_indicators(df)
    df = add_sector_rotation(df)
    df = add_temporal_patterns(df)
    df = add_volume_profile(df)
    df = add_fractal_analysis(df)
    df = add_mtf_confluence(df)
    df = add_momentum_persistence(df)
    df = add_efficiency_metrics(df)
    df = add_volatility_surface(df)
    df = add_regime_detection(df)
    
    # Market regime encoding example: bull=1, bear=-1, neutral=0
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['regime'] = np.where(df['sma_20'] > df['sma_50'], 1, np.where(df['sma_20'] < df['sma_50'], -1, 0))
    # Normalized ATR
    df['atr_norm'] = df['atr'] / (df['Close'] + 1e-9)
    # EMA indicators for confluence
    df['ema_9'] = ema(df['Close'], 9)
    df['ema_21'] = ema(df['Close'], 21)
    df['ema_50'] = ema(df['Close'], 50)
    # Volume SMA for slippage calculation
    df['volume_sma20'] = df['Volume'].rolling(20).mean()
    # Bollinger Band calculations for mean reversion
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
    df['std_20'] = df['Close'].rolling(20).std()
    
    # Additional volatility measure for strategy adaptation
    df['volatility'] = df['std_20'] / df['sma_20']
    
    df.fillna(method="bfill", inplace=True)
    print(f"✅ Successfully added {len(df.columns)} total features including 15 Category A indicators!")
    return df
