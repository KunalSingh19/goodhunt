import pandas as pd
import numpy as np

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
    entropy = returns.apply(lambda x: -np.sum(np.nan_to_num(x * np.log(np.abs(x) + 1e-9)))), raw=True)
    df['entropy'] = entropy
    return df

def add_all_indicators(df):
    df = macd(df)
    df = atr(df)
    df = bollinger_bands(df)
    df = adx(df)
    df = cci(df)
    df = obv(df)
    df = stochastic(df)
    df = rsi(df)
    df = price_entropy(df)
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
    df.fillna(method="bfill", inplace=True)
    return df
