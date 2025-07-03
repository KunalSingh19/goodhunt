import numpy as np

def check_ema_confluence(row):
    """
    📊 Boost signal confidence if all EMAs are aligned
    Returns confluence score (0-1)
    """
    confluence_score = 0
    
    # Check if EMAs are in proper order for bullish alignment
    if row['ema_9'] > row['ema_21'] > row['ema_50']:
        confluence_score += 1
        
    return confluence_score

def check_mean_reversion_signal(row, threshold=0.02):
    """
    📉 Mean reversion: if price far below mean and BB tight
    Returns True if mean reversion signal is detected
    """
    mean_revert_signal = False
    
    # Check if BB is tight and price is far below mean
    if (row['bb_width'] < threshold and 
        row['Close'] < row['sma_20'] - 2 * row['std_20']):
        mean_revert_signal = True
        
    return mean_revert_signal

def check_trade_conflict(row):
    """
    ❗Avoid trades where indicators conflict
    Returns True if there's a conflict (should avoid trade)
    """
    # Conflicting signals: oversold but negative MACD, or overbought but positive MACD
    if ((row['rsi'] < 30 and row['macd_hist'] < 0) or 
        (row['rsi'] > 70 and row['macd_hist'] > 0)):
        return True
    return False