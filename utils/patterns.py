import pandas as pd
import numpy as np

def is_doji(df, thresh=0.1):
    body = (df['Close'] - df['Open']).abs()
    candle_range = (df['High'] - df['Low'])
    return (body / (candle_range + 1e-9) < thresh).astype(int)

def is_hammer(df, ratio=2.0):
    body = (df['Close'] - df['Open']).abs()
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    return ((lower_shadow > ratio * body) & (upper_shadow < body)).astype(int)

def is_engulfing(df):
    prev_open = df['Open'].shift()
    prev_close = df['Close'].shift()
    cond_bull = (df['Close'] > df['Open']) & (prev_close < prev_open) & (df['Close'] > prev_open) & (df['Open'] < prev_close)
    cond_bear = (df['Close'] < df['Open']) & (prev_close > prev_open) & (df['Open'] > prev_close) & (df['Close'] < prev_open)
    return (cond_bull | cond_bear).astype(int)

def is_morning_star(df, thresh=0.3):
    # Very basic: check for gap down, small candle, gap up (bullish reversal)
    prev_close = df['Close'].shift()
    prev_open = df['Open'].shift()
    curr_body = (df['Close'] - df['Open']).abs()
    gap_down = (df['Open'] < prev_close)
    small_candle = (curr_body / (df['High'] - df['Low'] + 1e-9) < thresh)
    gap_up = (df['Close'] > df['Open']) & (df['Close'] > prev_open)
    return (gap_down & small_candle & gap_up).astype(int)

def add_patterns(df):
    df['doji'] = is_doji(df)
    df['hammer'] = is_hammer(df)
    df['engulfing'] = is_engulfing(df)
    df['morning_star'] = is_morning_star(df)
    return df
