import numpy as np

def compute_reward(
    env,
    action,
    trade_executed,
    trade_pnl,
    trade_reason,
    regime,
    config=None
):
    """
    Modular reward: supports Sharpe, R:R, smoothing, win-boost, decay, etc.
    """
    cfg = config or {}
    base_reward = 0.0
    risk_penalty = -0.1
    reward_info = {}

    # Sharpe/Sortino based (risk-adjusted)
    returns = np.diff(env.equity_curve[-10:]) if len(env.equity_curve) > 10 else [0]
    volatility = np.std(returns) + 1e-9
    sharpe = np.mean(returns) / volatility if volatility > 0 else 0
    reward_info['sharpe'] = sharpe

    if cfg.get("mode", "sharpe") == "sharpe":
        base_reward = sharpe
    elif cfg.get("mode") == "rr":
        risk = abs(env.avg_entry_price - env.trail_sl) if env.trail_sl else 1
        base_reward = (trade_pnl / risk) if trade_pnl > 0 else risk_penalty
    else:
        base_reward = trade_pnl / (env.initial_balance + 1e-9)

    # Win-boost
    if trade_executed and trade_pnl > 0:
        base_reward += cfg.get("win_boost", 0.01)
    # Hold penalty
    if env.position_size > 0 and env.hold_steps > 0:
        base_reward -= env.hold_steps * cfg.get("hold_penalty", 0.01) / 100.0
    # Chaos bonus
    if regime == 0:  # neutral, flat market
        base_reward += cfg.get("chaos_flat_bonus", 0.0)
    # Reward decay
    base_reward *= cfg.get("decay", 0.995)
    # Reward clipping
    base_reward = np.clip(base_reward, -cfg.get("reward_clip", 2.0), cfg.get("reward_clip", 2.0))

    reward_info['reward'] = base_reward
    return base_reward, reward_info
