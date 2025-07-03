import numpy as np

def compute_slippage_cost(price, volume, volatility, model="dynamic", min_slip=0.0001, max_slip=0.005):
    """
    Dynamic slippage: higher when volume is low or volatility is high.
    model: "fixed", "dynamic"
    """
    if model == "fixed":
        return price * min_slip
    elif model == "dynamic":
        slip = min_slip + (max_slip - min_slip) * (1 / (volume + 1e-6)) + (volatility * 0.5)
        return price * np.clip(slip, min_slip, max_slip)
    else:
        return price * min_slip
