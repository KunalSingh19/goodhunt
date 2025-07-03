import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from utils.indicators import add_all_indicators
from utils.patterns import add_patterns
from utils.regime import detect_regime
from env.reward_engine import compute_reward
from env.slippage import compute_slippage_cost

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
            "market_regime"
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

    def reset(self, seed=None, options=None):
        self._reset_state_vars()
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

        # Dynamic position sizing
        max_affordable = self.balance / (price * (1 + self.fee_pct))
        vol_factor = 1.0 / (1.0 + volatility)
        base_size = np.clip(vol_factor, self.min_position, self.max_exposure)
        size = min(base_size, max_affordable, self.max_exposure)

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

        # Slippage and transaction cost
        slippage = compute_slippage_cost(price, volume, volatility, model=self.slippage_model)
        fee = price * size * self.fee_pct

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

        # Action execution
        if action == 1 and trade_allowed and entry_signals >= 2:  # BUY
            buy_amount = size if self.position_size == 0 else max(0, size - self.position_size)
            if buy_amount > 0:
                cost = (price + slippage) * buy_amount + fee
                if self.balance >= cost:
                    self.balance -= cost
                    self.avg_entry_price = (self.avg_entry_price * self.position_size + price * buy_amount) / (self.position_size + buy_amount + 1e-9)
                    self.position_size += buy_amount
                    trade_executed = True
                    trade_reason = "ENTRY"
        elif action == 2 and self.position_size > 0:  # SELL
            # Only sell if not in strong up momentum (MACD+, RSI rising)
            if not (row['macd'] > row['macd_signal'] and row['rsi_slope'] > 0):
                sell_amount = size if self.position_size <= size else self.position_size
                proceeds = (price - slippage) * sell_amount - fee
                self.balance += proceeds
                pnl = (price - self.avg_entry_price) * sell_amount
                self.position_size -= sell_amount
                if self.position_size < self.min_position:
                    self.position_size = 0
                    self.avg_entry_price = 0.0
                trade_executed = True
                trade_reason = "EXIT"
        elif action == 3 and self.position_size > 0:  # SCALE UP
            add_size = min(size, max_affordable)
            if add_size > 0:
                cost = (price + slippage) * add_size + fee
                if self.balance >= cost:
                    self.balance -= cost
                    self.avg_entry_price = (self.avg_entry_price * self.position_size + price * add_size) / (self.position_size + add_size + 1e-9)
                    self.position_size += add_size
                    trade_executed = True
                    trade_reason = "SCALEUP"
        elif action == 4 and self.position_size > 0.01:  # SCALE DOWN
            reduce_size = min(size, self.position_size)
            proceeds = (price - slippage) * reduce_size - fee
            self.balance += proceeds
            pnl = (price - self.avg_entry_price) * reduce_size
            self.position_size -= reduce_size
            trade_executed = True
            trade_reason = "SCALEDOWN"
        elif action == 5 and self.position_size > 0:  # FLAT (close all)
            proceeds = (price - slippage) * self.position_size - fee
            self.balance += proceeds
            pnl = (price - self.avg_entry_price) * self.position_size
            self.position_size = 0
            self.avg_entry_price = 0.0
            trade_executed = True
            trade_reason = "FORCE_FLAT"

        # Trailing TP/SL logic
        if self.position_size > 0:
            if self.trail_tp == 0.0 or price > self.trail_tp:
                self.trail_tp = price - 0.01
            if self.trail_sl == 0.0 or price < self.trail_sl:
                self.trail_sl = price + 0.01
            # Trailing stop logic (close if price below trail_sl or above trail_tp)
            if price < self.trail_sl or price > self.trail_tp:
                proceeds = (price - slippage) * self.position_size - fee
                self.balance += proceeds
                pnl = (price - self.avg_entry_price) * self.position_size
                self.position_size = 0
                self.avg_entry_price = 0.0
                trade_executed = True
                trade_reason = "TRAIL_TP_SL"

        # Drawdown calculation
        self.net_worth = self.balance + self.position_size * price
        self.peak_net = max(self.peak_net, self.net_worth)
        self.drawdown = (self.peak_net - self.net_worth) / (self.peak_net + 1e-9)
        self.equity_curve.append(self.net_worth)

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

        # Reward calculation (pluggable)
        reward, reward_info = compute_reward(
            env=self,
            action=action,
            trade_executed=trade_executed,
            trade_pnl=pnl,
            trade_reason=trade_reason,
            regime=regime
        )

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1 or self.net_worth < self.initial_balance * 0.5

        return self._get_observation(), reward, done, False, reward_info

    def render(self):
        print(f"Step {self.current_step} | Net: ${self.net_worth:.2f} | Position: {self.position_size:.2f} | Drawdown: {self.drawdown:.2%}")

    def save_trades(self, file="backtest/trades.csv"):
        df = pd.DataFrame(self.trades)
        df.to_csv(file, index=False)

    def save_equity_curve(self, file="backtest/equity_curve.csv"):
        pd.DataFrame({"net_worth": self.equity_curve}).to_csv(file, index=False)
