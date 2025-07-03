import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from utils.indicators import add_all_indicators
from utils.patterns import add_patterns
from utils.regime import detect_regime
from env.reward_engine import compute_reward
from env.slippage import compute_slippage_cost
from utils.decision import check_ema_confluence, check_mean_reversion_signal, check_trade_conflict

class TradingEnv(gym.Env):
    """
    GoodHunt v3+ RL Environment:
    - Multi-action: Buy, Sell, Hold, Scale Up, Scale Down, Flat
    - Position sizing, volatility/ATR-based scaling
    - Hedging, exposure caps, regime, trailing TP/SL, drawdown-aware
    - Pluggable reward and slippage models
    - Works with multi-asset if needed (extend self.df)
    """
    ACTIONS = {
        0: "HOLD",
        1: "BUY",
        2: "SELL",
        3: "SCALE_UP",
        4: "SCALE_DOWN",
        5: "FLAT"
    }
    def __init__(
        self,
        df,
        window_size=50,
        initial_balance=100.0,
        max_exposure=0.7,
        min_position=0.01,
        fee_pct=0.001,
        slippage_model="dynamic",
        multi_asset=False,
        config=None
    ):
        super().__init__()
        # Preprocess dataframe with all technicals/patterns/regimes
        df = add_all_indicators(df)
        df = add_patterns(df)
        df = detect_regime(df)
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.max_exposure = max_exposure
        self.min_position = min_position
        self.fee_pct = fee_pct
        self.slippage_model = slippage_model
        self.multi_asset = multi_asset
        self.config = config or {}

        self.features = [
            # Price/volume/technicals
            "Open", "High", "Low", "Close", "Volume",
            "macd", "macd_signal", "macd_hist", "obv", "stoch_k", "stoch_d",
            "cci", "adx", "rsi", "rsi_slope", "entropy", "atr_norm",
            "bb_upper", "bb_lower", "bb_width", "regime",
            # Patterns
            "doji", "hammer", "engulfing", "morning_star",
            # Market regime
            "market_regime",
            # Existing indicators
            "ema_9", "ema_21", "ema_50", "volume_sma20", "sma_20", "std_20",
            # NEW OBSERVATION ENHANCEMENTS
            "vwap", "beta", "rsi_fisher", "volatility"
        ]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, len(self.features) + 3),  # +3 for holding, position size, exposure
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self._reset_state_vars()

    def _reset_state_vars(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.peak_net = self.initial_balance
        self.position_size = 0.0
        self.avg_entry_price = 0.0
        self.trades = []
        self.current_step = self.window_size
        self.drawdown = 0.0
        self.hold_steps = 0
        self.trail_tp = 0.0
        self.trail_sl = 0.0
        self.equity_curve = [self.net_worth]
        self.exposure = 0.0
        self.last_action = 0  # HOLD
        
        # Enhanced state variables for new features
        self.entry_price = 0.0
        self.risk = 1.0  # Feature 19: starts at 1.0, can be adjusted
        self.stop_loss = 0.0
        self.trades_today = 0
        self.current_date = None
        self.position = 0  # 0=flat, 1=long, -1=short
        self.days_in_trade = 0
        self.shares = [0.0]  # Feature 4: for multi-asset tracking
        self.prices = [0.0]  # Feature 4: for multi-asset tracking
        self.recent_profits = []  # Feature 10: for profit streak tracking
        self.net_worth_history = [self.net_worth]  # Feature 13: for multi-step averaging
        self.prev_avg_worth = self.net_worth
        self.std_vol = 0.0  # Feature 19: standard volatility reference
        self.ent_coef = 1.0  # Feature 20: exploration coefficient

    def reset(self, seed=None, options=None):
        self._reset_state_vars()
        # Calculate standard volatility for regime switching
        if len(self.df) > 100:
            self.std_vol = self.df['volatility'].rolling(100).mean().iloc[-1]
        return self._get_observation(), {}

    def _get_observation(self):
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        # state vector = features + [position_flag, position_size, curr_exposure]
        obs = window[self.features].values.astype(np.float32)
        holding_flag = np.ones((self.window_size, 1), dtype=np.float32) * int(self.position_size > 0)
        pos_size_vec = np.ones((self.window_size, 1), dtype=np.float32) * self.position_size
        exposure_vec = np.ones((self.window_size, 1), dtype=np.float32) * self._get_exposure()
        obs = np.hstack([obs, holding_flag, pos_size_vec, exposure_vec])
        return obs

    def _get_exposure(self):
        return (self.position_size * self.avg_entry_price) / (self.balance + 1e-9)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        price = row["Close"]
        volume = row["Volume"]
        volatility = row["atr_norm"]
        regime = row["market_regime"]
        reward_info = {}

        # Feature 4: Multi-Asset Exposure Control
        self.shares = [self.position_size]  # For simplicity, treating as single asset
        self.prices = [price]
        total_value = self.balance + sum([a*p for a, p in zip(self.shares, self.prices)])
        if total_value > self.initial_balance * 1.2:
            action = 0  # prevent over-exposure

        # Feature 5: Anti-Chasing Logic
        if self.entry_price > 0 and price / self.entry_price > 1.03 and self.days_in_trade < 2:
            action = 0  # avoid chasing fast spikes

        # Feature 19: Volatility Regime Switcher
        current_vol = self.df['volatility'].iloc[self.current_step] if 'volatility' in self.df.columns else volatility
        if current_vol > self.std_vol * 1.5:
            self.risk *= 0.5

        # Dynamic position sizing with Feature 11: Fractional Trading
        max_affordable = self.balance / (price * (1 + self.fee_pct))
        vol_factor = 1.0 / (1.0 + volatility)
        base_size = np.clip(vol_factor * self.risk, self.min_position, self.max_exposure)
        size = min(base_size, max_affordable, self.max_exposure)
        # Feature 11: Fractional trading - round to 4 decimal places
        size = round(size, 4)

        # Restrict trading in dead zones (volume filter)
        avg_volume = self.df["Volume"].rolling(30).mean().iloc[self.current_step]
        trade_allowed = volume > (avg_volume * 0.5 if avg_volume > 0 else 1)

        # Drawdown/risk-off handling
        risk_off = self.drawdown > (self.config.get("max_drawdown") or 0.05)
        if risk_off:
            size *= 0.5  # reduce size

        # Capital exposure cap handling
        if (self.position_size * price) / (self.balance + 1e-9) > self.max_exposure:
            size = 0  # can't buy more!

        # Feature 6: Spread Simulation (Bid-Ask)
        spread = price * 0.0005
        buy_price = price + spread/2
        sell_price = price - spread/2

        # Slippage and transaction cost
        slippage = compute_slippage_cost(price, volume, volatility, model=self.slippage_model)
        fee = price * size * self.fee_pct

        # Feature 14: End-Of-Day Forced Close (assuming 390 minutes in trading day)
        if self.current_step % 390 == 0 and self.position_size > 0:
            action = 2  # Force sell at end of day

        # Multi-action logic
        done = False
        info = {}
        trade_executed = False
        trade_reason = ""
        pnl = 0.0

        # Smart entry: only act if >=2 signals agree (e.g., rsi, sma, volume spike)
        entry_signals = 0
        if row["rsi"] < 30: entry_signals += 1
        if row["sma_20"] > row["sma_50"]: entry_signals += 1
        if volume > avg_volume: entry_signals += 1

        # Action execution with enhanced features
        if action == 1 and trade_allowed and entry_signals >= 2:  # BUY
            buy_amount = size if self.position_size == 0 else max(0, size - self.position_size)
            if buy_amount > 0:
                cost = buy_price * buy_amount + fee  # Feature 6: use bid-ask spread
                if self.balance >= cost:
                    self.balance -= cost
                    self.avg_entry_price = (self.avg_entry_price * self.position_size + buy_price * buy_amount) / (self.position_size + buy_amount + 1e-9)
                    self.position_size += buy_amount
                    self.entry_price = buy_price  # Feature 5: track entry price
                    self.days_in_trade = 0  # Reset days counter
                    trade_executed = True
                    trade_reason = "ENTRY"
                    
        elif action == 2 and self.position_size > 0:  # SELL
            # Only sell if not in strong up momentum (MACD+, RSI rising)
            if not (row['macd'] > row['macd_signal'] and row['rsi_slope'] > 0):
                sell_amount = size if self.position_size <= size else self.position_size
                proceeds = sell_price * sell_amount - fee  # Feature 6: use bid-ask spread
                self.balance += proceeds
                pnl = (sell_price - self.avg_entry_price) * sell_amount
                self.position_size -= sell_amount
                if self.position_size < self.min_position:
                    self.position_size = 0
                    self.avg_entry_price = 0.0
                    self.entry_price = 0.0
                trade_executed = True
                trade_reason = "EXIT"
                
        elif action == 3 and self.position_size > 0:  # SCALE UP
            # Feature 12: Partial Position Scaling
            add_size = min(size * 0.5, max_affordable)
            if add_size > 0:
                cost = buy_price * add_size + fee
                if self.balance >= cost:
                    self.balance -= cost
                    self.avg_entry_price = (self.avg_entry_price * self.position_size + buy_price * add_size) / (self.position_size + add_size + 1e-9)
                    self.position_size += add_size
                    trade_executed = True
                    trade_reason = "SCALEUP"
                    
        elif action == 4 and self.position_size > 0.01:  # SCALE DOWN
            # Feature 12: Partial Position Scaling
            reduce_size = min(size * 0.5, self.position_size)
            proceeds = sell_price * reduce_size - fee
            self.balance += proceeds
            pnl = (sell_price - self.avg_entry_price) * reduce_size
            self.position_size -= reduce_size
            trade_executed = True
            trade_reason = "SCALEDOWN"
            
        elif action == 5 and self.position_size > 0:  # FLAT (close all)
            proceeds = sell_price * self.position_size - fee
            self.balance += proceeds
            pnl = (sell_price - self.avg_entry_price) * self.position_size
            self.position_size = 0
            self.avg_entry_price = 0.0
            self.entry_price = 0.0
            trade_executed = True
            trade_reason = "FORCE_FLAT"

        # Feature 9: Stop-Loss Breach Penalty
        if self.stop_loss > 0 and price < self.stop_loss:
            # Force close position
            if self.position_size > 0:
                proceeds = sell_price * self.position_size - fee
                self.balance += proceeds
                pnl = (sell_price - self.avg_entry_price) * self.position_size
                self.position_size = 0
                self.avg_entry_price = 0.0
                self.entry_price = 0.0
                trade_executed = True
                trade_reason = "STOP_LOSS"

        # Trailing TP/SL logic
        if self.position_size > 0:
            if self.trail_tp == 0.0 or price > self.trail_tp:
                self.trail_tp = price - 0.01
            if self.trail_sl == 0.0 or price < self.trail_sl:
                self.trail_sl = price + 0.01
            # Trailing stop logic (close if price below trail_sl or above trail_tp)
            if price < self.trail_sl or price > self.trail_tp:
                proceeds = sell_price * self.position_size - fee
                self.balance += proceeds
                pnl = (sell_price - self.avg_entry_price) * self.position_size
                self.position_size = 0
                self.avg_entry_price = 0.0
                self.entry_price = 0.0
                trade_executed = True
                trade_reason = "TRAIL_TP_SL"

        # Update days in trade counter
        if self.position_size > 0:
            self.days_in_trade += 1
        else:
            self.days_in_trade = 0

        # Drawdown calculation
        self.net_worth = self.balance + self.position_size * price
        self.peak_net = max(self.peak_net, self.net_worth)
        self.drawdown = (self.peak_net - self.net_worth) / (self.peak_net + 1e-9)
        self.equity_curve.append(self.net_worth)
        self.net_worth_history.append(self.net_worth)

        # Feature 10: Track recent profits for streak bonus
        if trade_executed and pnl != 0:
            self.recent_profits.append(pnl)
            if len(self.recent_profits) > 5:
                self.recent_profits.pop(0)

        # Logging trade
        if trade_executed:
            self.trades.append({
                "step": self.current_step,
                "action": self.ACTIONS[action],
                "price": price,
                "size": self.position_size,
                "pnl": pnl,
                "drawdown": self.drawdown,
                "exposure": self._get_exposure(),
                "reason": trade_reason,
                "regime": regime
            })

        # Enhanced reward calculation with new features
        reward = self._compute_enhanced_reward(action, trade_executed, pnl, trade_reason, slippage)

        # Feature 20: On-the-Fly Hyperparameter Adjuster
        if reward < -0.05:
            self.ent_coef *= 1.1  # encourage exploration

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1 or self.net_worth < self.initial_balance * 0.5

        return self._get_observation(), reward, done, False, reward_info

    def _compute_enhanced_reward(self, action, trade_executed, pnl, trade_reason, slippage):
        """Enhanced reward computation with all new features"""
        
        # Base reward
        reward = pnl / self.initial_balance if trade_executed else 0.0
        
        # Feature 7: Reward Decay on Time
        time_decay = 1 - (self.current_step / len(self.df)) * 0.1
        reward *= time_decay
        
        # Feature 8: Slippage Penalty
        reward -= abs(slippage) * 100
        
        # Feature 9: Stop-Loss Breach Penalty
        if trade_reason == "STOP_LOSS":
            reward -= 0.2
        
        # Feature 10: Profit Streak Bonus
        if len(self.recent_profits) >= 3 and all(p > 0 for p in self.recent_profits[-3:]):
            reward += 0.05
        
        # Feature 13: Multi-Step Reward Averaging
        n = min(5, len(self.net_worth_history))
        if n > 0:
            avg_worth = sum(self.net_worth_history[-n:]) / n
            reward += (avg_worth - self.prev_avg_worth) / self.initial_balance
            self.prev_avg_worth = avg_worth
        
        return reward

    def render(self):
        print(f"Step {self.current_step} | Net: ${self.net_worth:.2f} | Position: {self.position_size:.2f} | Drawdown: {self.drawdown:.2%}")

    def save_trades(self, file="backtest/trades.csv"):
        df = pd.DataFrame(self.trades)
        df.to_csv(file, index=False)

    def save_equity_curve(self, file="backtest/equity_curve.csv"):
        pd.DataFrame({"net_worth": self.equity_curve}).to_csv(file, index=False)
