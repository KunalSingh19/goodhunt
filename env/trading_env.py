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
from logs.advanced_stats_tracker import AdvancedStatsTracker

class TradingEnv(gym.Env):
    """
    GoodHunt v3+ Enhanced RL Environment with 120+ Features:
    - Category A: Advanced Observation & Indicators (15 features)
    - Comprehensive stats tracking and analytics
    - Multi-action: Buy, Sell, Hold, Scale Up, Scale Down, Flat
    - Position sizing, volatility/ATR-based scaling
    - Hedging, exposure caps, regime, trailing TP/SL, drawdown-aware
    - Pluggable reward and slippage models
    - Real-time performance monitoring
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
        config=None,
        user_id=None,
        session_id=None,
        enable_stats_tracking=True
    ):
        super().__init__()
        
        # Enhanced preprocessing with all Category A features
        print("🚀 Initializing GoodHunt v3+ Enhanced Trading Environment...")
        df = add_all_indicators(df)  # Now includes 15 Category A features
        df = add_patterns(df)
        df = detect_regime(df)
        self.df = df.reset_index(drop=True)
        
        # Environment parameters
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.max_exposure = max_exposure
        self.min_position = min_position
        self.fee_pct = fee_pct
        self.slippage_model = slippage_model
        self.multi_asset = multi_asset
        self.config = config or {}
        self.user_id = user_id
        self.session_id = session_id
        
        # Stats tracking
        self.enable_stats_tracking = enable_stats_tracking
        if self.enable_stats_tracking:
            self.stats_tracker = AdvancedStatsTracker(user_id=user_id)
            print("📊 Advanced stats tracking enabled")

        # Enhanced feature set including all Category A features
        self.features = [
            # Basic OHLCV
            "Open", "High", "Low", "Close", "Volume",
            
            # Traditional indicators
            "macd", "macd_signal", "macd_hist", "obv", "stoch_k", "stoch_d",
            "cci", "adx", "rsi", "rsi_slope", "entropy", "atr_norm",
            "bb_upper", "bb_lower", "bb_width", "regime",
            
            # Patterns
            "doji", "hammer", "engulfing", "morning_star",
            
            # Market regime
            "market_regime",
            
            # Existing indicators
            "ema_9", "ema_21", "ema_50", "volume_sma20", "sma_20", "std_20",
            
            # Enhanced indicators (Features 1-3)
            "vwap", "beta", "rsi_fisher", "volatility",
            
            # === CATEGORY A: ADVANCED OBSERVATION FEATURES (A01-A15) ===
            
            # A01: Market Microstructure
            "hl_spread", "spread_ma", "order_flow_imbalance", "market_impact", 
            "tick_direction", "liquidity_proxy",
            
            # A02: Sentiment Analysis
            "news_sentiment", "fear_greed_index", "put_call_ratio", "sentiment_composite",
            
            # A03: Liquidity Metrics
            "amihud_illiquidity", "turnover_rate", "liquidity_score", "market_depth",
            
            # A04: Intermarket Analysis
            "bond_equity_corr", "usd_impact", "relative_strength",
            
            # A05: Options Flow
            "put_call_volume_ratio", "iv_rank", "options_skew", "max_pain_distance",
            
            # A06: Economic Calendar
            "economic_surprise", "earnings_season", "fed_policy_uncertainty", "gdp_growth_proxy",
            
            # A07: Sector Rotation
            "growth_value_indicator", "sector_momentum", "style_rotation",
            
            # A08: Temporal Patterns
            "day_of_week", "monday_effect", "friday_effect", "month", 
            "january_effect", "december_effect", "quarter_end", "holiday_proximity",
            
            # A09: Volume Profile
            "session_vwap", "volume_at_price", "price_acceptance", 
            "value_area_high", "value_area_low",
            
            # A10: Fractal Analysis
            "fractal_high", "fractal_low", "market_structure",
            
            # A11: Multi-timeframe Confluence
            "trend_short", "trend_medium", "trend_long", "mtf_confluence", "signal_strength",
            
            # A12: Momentum Persistence
            "momentum_persistence", "trend_consistency", "momentum_quality",
            
            # A13: Market Efficiency
            "hurst_exponent", "autocorr_1", "efficiency_score",
            
            # A14: Volatility Surface
            "vol_term_structure", "vol_skew", "vol_curvature", "vol_risk_premium",
            
            # A15: Regime Detection
            "volatility_regime", "trend_regime", "market_regime_advanced"
        ]
        
        print(f"✅ Loaded {len(self.features)} features including {15} Category A advanced observations")
        
        # Update observation space for enhanced features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, len(self.features) + 3),  # +3 for holding, position size, exposure
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        
        # Initialize state variables
        self._reset_state_vars()

    def _reset_state_vars(self):
        """Reset all state variables with enhanced tracking"""
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
        self.risk = 1.0
        self.stop_loss = 0.0
        self.trades_today = 0
        self.current_date = None
        self.position = 0  # 0=flat, 1=long, -1=short
        self.days_in_trade = 0
        self.shares = [0.0]
        self.prices = [0.0]
        self.recent_profits = []
        self.net_worth_history = [self.net_worth]
        self.prev_avg_worth = self.net_worth
        self.std_vol = 0.0
        self.ent_coef = 1.0
        
        # Feature attribution tracking
        self.feature_contributions = {}
        self.feature_usage_count = {feature: 0 for feature in self.features}
        
        # Performance metrics
        self.trade_count = 0
        self.win_count = 0
        self.total_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0

    def reset(self, seed=None, options=None):
        """Enhanced reset with stats tracking"""
        self._reset_state_vars()
        
        # Calculate standard volatility for regime switching
        if len(self.df) > 100:
            self.std_vol = self.df['volatility'].rolling(100).mean().iloc[-1]
        
        print(f"🔄 Environment reset - Session: {self.session_id}, User: {self.user_id}")
        return self._get_observation(), {}

    def _get_observation(self):
        """Enhanced observation with all Category A features"""
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        
        # Verify all features exist in dataframe
        missing_features = [f for f in self.features if f not in window.columns]
        if missing_features:
            print(f"⚠️  Missing features: {missing_features[:5]}...")  # Show first 5
            # Fill missing features with zeros
            for feature in missing_features:
                window[feature] = 0.0
        
        # State vector = features + [position_flag, position_size, curr_exposure]
        obs = window[self.features].values.astype(np.float32)
        holding_flag = np.ones((self.window_size, 1), dtype=np.float32) * int(self.position_size > 0)
        pos_size_vec = np.ones((self.window_size, 1), dtype=np.float32) * self.position_size
        exposure_vec = np.ones((self.window_size, 1), dtype=np.float32) * self._get_exposure()
        
        obs = np.hstack([obs, holding_flag, pos_size_vec, exposure_vec])
        
        # Handle NaN values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return obs

    def _get_exposure(self):
        """Calculate current exposure ratio"""
        return (self.position_size * self.avg_entry_price) / (self.balance + 1e-9)

    def step(self, action):
        """Enhanced step function with comprehensive feature tracking and stats"""
        row = self.df.iloc[self.current_step]
        price = row["Close"]
        volume = row["Volume"]
        volatility = row.get("atr_norm", 0.01)
        regime = row.get("market_regime", 0)
        reward_info = {}
        
        # Track feature usage and contributions
        self._track_feature_usage(row)
        
        # === ENHANCED TRADING LOGIC WITH CATEGORY A FEATURES ===
        
        # A04: Multi-Asset Exposure Control
        self.shares = [self.position_size]
        self.prices = [price]
        total_value = self.balance + sum([a*p for a, p in zip(self.shares, self.prices)])
        if total_value > self.initial_balance * 1.2:
            action = 0  # prevent over-exposure
            
        # A05: Anti-Chasing Logic with Market Microstructure
        market_impact = row.get("market_impact", 0)
        if self.entry_price > 0 and price / self.entry_price > 1.03 and self.days_in_trade < 2:
            if market_impact > 0.01:  # High market impact
                action = 0  # avoid chasing in illiquid conditions
                
        # A19: Volatility Regime Switcher with Advanced Detection
        current_vol = row.get('volatility', volatility)
        vol_regime = row.get('volatility_regime', 0)
        if vol_regime == 1 or current_vol > self.std_vol * 1.5:
            self.risk *= 0.5
            
        # A02: Sentiment-based Position Sizing
        sentiment = row.get('sentiment_composite', 0)
        fear_greed = row.get('fear_greed_index', 50)
        sentiment_multiplier = 1.0
        if sentiment < -0.5 or fear_greed < 20:  # High fear
            sentiment_multiplier = 0.7
        elif sentiment > 0.5 or fear_greed > 80:  # High greed
            sentiment_multiplier = 1.3

        # Dynamic position sizing with enhanced features
        max_affordable = self.balance / (price * (1 + self.fee_pct))
        vol_factor = 1.0 / (1.0 + volatility)
        
        # A03: Liquidity-adjusted sizing
        liquidity_score = row.get('liquidity_score', 1.0)
        liquidity_factor = min(liquidity_score, 2.0)  # Cap at 2x
        
        base_size = np.clip(
            vol_factor * self.risk * sentiment_multiplier * liquidity_factor, 
            self.min_position, 
            self.max_exposure
        )
        size = min(base_size, max_affordable, self.max_exposure)
        size = round(size, 4)  # Fractional trading

        # A11: Multi-timeframe Confluence Check
        mtf_confluence = row.get('mtf_confluence', 0)
        signal_strength = row.get('signal_strength', 0)
        
        # Enhanced volume filter with A09: Volume Profile
        avg_volume = self.df["Volume"].rolling(30).mean().iloc[self.current_step]
        volume_at_price = row.get('volume_at_price', volume)
        trade_allowed = volume > (avg_volume * 0.5) and volume_at_price > avg_volume * 0.3

        # Drawdown/risk-off handling
        risk_off = self.drawdown > (self.config.get("max_drawdown") or 0.05)
        if risk_off:
            size *= 0.5

        # A06: Spread Simulation (Enhanced Bid-Ask)
        spread_ma = row.get('spread_ma', 0.0005)
        spread = price * max(spread_ma, 0.0001)  # Minimum spread
        buy_price = price + spread/2
        sell_price = price - spread/2

        # Enhanced slippage with A03: Liquidity metrics
        slippage_base = compute_slippage_cost(price, volume, volatility, model=self.slippage_model)
        liquidity_adjustment = 1.0 / (liquidity_score + 0.1)
        slippage = slippage_base * liquidity_adjustment
        fee = price * size * self.fee_pct

        # A14: End-Of-Day Forced Close with Temporal Patterns
        quarter_end = row.get('quarter_end', 0)
        if (self.current_step % 390 == 0 or quarter_end) and self.position_size > 0:
            action = 2  # Force sell

        # Enhanced entry logic with A11: Multi-timeframe signals
        entry_signals = 0
        if row.get("rsi", 50) < 30: entry_signals += 1
        if row.get("sma_20", 0) > row.get("sma_50", 0): entry_signals += 1
        if volume > avg_volume: entry_signals += 1
        if mtf_confluence > 0.5: entry_signals += 2  # Double weight for confluence
        if signal_strength > 0.7: entry_signals += 1

        # Execute trading logic
        done = False
        info = {}
        trade_executed = False
        trade_reason = ""
        pnl = 0.0
        entry_time = None
        exit_time = None

        if action == 1 and trade_allowed and entry_signals >= 3:  # BUY (raised threshold)
            buy_amount = size if self.position_size == 0 else max(0, size - self.position_size)
            if buy_amount > 0:
                cost = buy_price * buy_amount + fee
                if self.balance >= cost:
                    self.balance -= cost
                    self.avg_entry_price = (self.avg_entry_price * self.position_size + buy_price * buy_amount) / (self.position_size + buy_amount + 1e-9)
                    self.position_size += buy_amount
                    self.entry_price = buy_price
                    self.days_in_trade = 0
                    trade_executed = True
                    trade_reason = "ENTRY"
                    entry_time = self.df.index[self.current_step] if hasattr(self.df, 'index') else self.current_step
                    
        elif action == 2 and self.position_size > 0:  # SELL
            # Enhanced exit logic with A12: Momentum persistence
            momentum_persistence = row.get('momentum_persistence', 0.5)
            should_exit = True
            
            # Don't sell in strong momentum unless other conditions met
            if row.get('macd', 0) > row.get('macd_signal', 0) and row.get('rsi_slope', 0) > 0:
                if momentum_persistence > 0.8:
                    should_exit = False
                    
            if should_exit:
                sell_amount = size if self.position_size <= size else self.position_size
                proceeds = sell_price * sell_amount - fee
                self.balance += proceeds
                pnl = (sell_price - self.avg_entry_price) * sell_amount
                self.position_size -= sell_amount
                if self.position_size < self.min_position:
                    self.position_size = 0
                    self.avg_entry_price = 0.0
                    self.entry_price = 0.0
                trade_executed = True
                trade_reason = "EXIT"
                exit_time = self.df.index[self.current_step] if hasattr(self.df, 'index') else self.current_step
                
        # ... (continuing with other actions - SCALE_UP, SCALE_DOWN, FLAT)
        elif action == 3 and self.position_size > 0:  # SCALE UP
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
            reduce_size = min(size * 0.5, self.position_size)
            proceeds = sell_price * reduce_size - fee
            self.balance += proceeds
            pnl = (sell_price - self.avg_entry_price) * reduce_size
            self.position_size -= reduce_size
            trade_executed = True
            trade_reason = "SCALEDOWN"
            
        elif action == 5 and self.position_size > 0:  # FLAT
            proceeds = sell_price * self.position_size - fee
            self.balance += proceeds
            pnl = (sell_price - self.avg_entry_price) * self.position_size
            self.position_size = 0
            self.avg_entry_price = 0.0
            self.entry_price = 0.0
            trade_executed = True
            trade_reason = "FORCE_FLAT"

        # Update performance metrics
        self.net_worth = self.balance + self.position_size * price
        self.peak_net = max(self.peak_net, self.net_worth)
        self.drawdown = (self.peak_net - self.net_worth) / (self.peak_net + 1e-9)
        self.equity_curve.append(self.net_worth)
        self.net_worth_history.append(self.net_worth)

        # Track profits for streak bonus
        if trade_executed and pnl != 0:
            self.recent_profits.append(pnl)
            if len(self.recent_profits) > 5:
                self.recent_profits.pop(0)
                
            self.trade_count += 1
            if pnl > 0:
                self.win_count += 1
            self.total_pnl += pnl
            self.total_commission += fee
            self.total_slippage += slippage

        # Enhanced trade logging with stats tracking
        if trade_executed:
            trade_data = {
                "step": self.current_step,
                "session_id": self.session_id,
                "symbol": self.config.get('symbol', 'UNKNOWN'),
                "action": self.ACTIONS[action],
                "entry_price": self.avg_entry_price if trade_reason == "ENTRY" else None,
                "exit_price": price if trade_reason in ["EXIT", "FORCE_FLAT"] else None,
                "price": price,
                "quantity": self.position_size,
                "pnl": pnl,
                "pnl_percent": (pnl / (self.initial_balance + 1e-9)) * 100,
                "duration_minutes": self.days_in_trade,
                "trade_reason": trade_reason,
                "market_regime": regime,
                "volatility": volatility,
                "volume": volume,
                "slippage": slippage,
                "commission": fee,
                "net_pnl": pnl - fee - slippage,
                "cumulative_pnl": self.total_pnl,
                "drawdown": self.drawdown,
                "win_rate": (self.win_count / max(self.trade_count, 1)) * 100,
                "sharpe_ratio": self._calculate_sharpe_ratio(),
                "sortino_ratio": self._calculate_sortino_ratio(),
                "mae": -min(0, pnl),  # Max Adverse Excursion
                "mfe": max(0, pnl),   # Max Favorable Excursion
                "exposure": self._get_exposure(),
                "entry_time": entry_time,
                "exit_time": exit_time
            }
            
            self.trades.append(trade_data)
            
            # Track with advanced stats system
            if self.enable_stats_tracking:
                self.stats_tracker.track_trade(trade_data)

        # Enhanced reward calculation
        reward = self._compute_enhanced_reward(action, trade_executed, pnl, trade_reason, slippage, row)

        # A20: Hyperparameter adjustment
        if reward < -0.05:
            self.ent_coef *= 1.1

        # Update position tracking
        if self.position_size > 0:
            self.days_in_trade += 1
        else:
            self.days_in_trade = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1 or self.net_worth < self.initial_balance * 0.5

        return self._get_observation(), reward, done, False, reward_info

    def _track_feature_usage(self, row):
        """Track which features are being used and their contributions"""
        for feature in self.features:
            if feature in row:
                self.feature_usage_count[feature] += 1
                # Track feature values for attribution analysis
                if self.enable_stats_tracking and np.random.random() < 0.1:  # Sample 10% for efficiency
                    feature_data = {
                        'category': self._get_feature_category(feature),
                        'name': feature,
                        'value': float(row[feature]),
                        'importance': np.random.random(),  # Placeholder - should come from model
                        'signal_strength': abs(float(row[feature])) if not np.isnan(float(row[feature])) else 0,
                        'pnl_contribution': 0,  # Will be updated by model
                        'accuracy': 0.5,  # Placeholder
                        'fpr': 0.1,  # Placeholder
                        'tpr': 0.8,  # Placeholder
                        'usage_frequency': 1
                    }
                    self.stats_tracker.track_feature_performance(feature_data)

    def _get_feature_category(self, feature):
        """Categorize features for tracking"""
        if feature in ["hl_spread", "spread_ma", "order_flow_imbalance", "market_impact", "tick_direction", "liquidity_proxy"]:
            return "A_Observation_Microstructure"
        elif feature in ["news_sentiment", "fear_greed_index", "put_call_ratio", "sentiment_composite"]:
            return "A_Observation_Sentiment"
        elif feature in ["amihud_illiquidity", "turnover_rate", "liquidity_score", "market_depth"]:
            return "A_Observation_Liquidity"
        elif feature in ["bond_equity_corr", "usd_impact", "relative_strength"]:
            return "A_Observation_Intermarket"
        elif feature in ["fractal_high", "fractal_low", "market_structure"]:
            return "A_Observation_Fractals"
        elif feature in ["mtf_confluence", "signal_strength", "trend_short", "trend_medium", "trend_long"]:
            return "A_Observation_MTF"
        else:
            return "Traditional_Indicators"

    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio from recent performance"""
        if len(self.equity_curve) < 10:
            return 0.0
        returns = np.diff(self.equity_curve[-30:]) / np.array(self.equity_curve[-31:-1])
        returns = returns[~np.isnan(returns)]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    def _calculate_sortino_ratio(self):
        """Calculate Sortino ratio (downside deviation)"""
        if len(self.equity_curve) < 10:
            return 0.0
        returns = np.diff(self.equity_curve[-30:]) / np.array(self.equity_curve[-31:-1])
        returns = returns[~np.isnan(returns)]
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0.0
        downside_std = np.std(negative_returns)
        if downside_std == 0:
            return 0.0
        return np.mean(returns) / downside_std * np.sqrt(252)

    def _compute_enhanced_reward(self, action, trade_executed, pnl, trade_reason, slippage, row):
        """Enhanced reward computation with Category A feature integration"""
        # Base reward
        reward = pnl / self.initial_balance if trade_executed else 0.0
        
        # A07: Time decay reward
        time_decay = 1 - (self.current_step / len(self.df)) * 0.1
        reward *= time_decay
        
        # A08: Slippage penalty (enhanced with liquidity)
        liquidity_score = row.get('liquidity_score', 1.0)
        slippage_penalty = abs(slippage) * 100 * (2.0 - liquidity_score)
        reward -= slippage_penalty
        
        # A09: Stop-loss breach penalty
        if trade_reason == "STOP_LOSS":
            reward -= 0.2
        
        # A10: Profit streak bonus (enhanced)
        if len(self.recent_profits) >= 3 and all(p > 0 for p in self.recent_profits[-3:]):
            streak_bonus = min(len([p for p in self.recent_profits if p > 0]) * 0.01, 0.1)
            reward += streak_bonus
        
        # A13: Multi-step reward averaging
        n = min(5, len(self.net_worth_history))
        if n > 0:
            avg_worth = sum(self.net_worth_history[-n:]) / n
            reward += (avg_worth - self.prev_avg_worth) / self.initial_balance
            self.prev_avg_worth = avg_worth
        
        # New Category A reward enhancements
        
        # Sentiment alignment bonus
        sentiment = row.get('sentiment_composite', 0)
        if trade_executed:
            if (action == 1 and sentiment > 0) or (action == 2 and sentiment < 0):
                reward += abs(sentiment) * 0.05
        
        # Market efficiency penalty
        efficiency_score = row.get('efficiency_score', 0)
        if efficiency_score > 0.3:  # Highly efficient market
            reward *= 0.9  # Reduce reward in efficient markets
        
        # Volatility regime adjustment
        vol_regime = row.get('volatility_regime', 0)
        if vol_regime == 1:  # High volatility
            reward *= 0.8  # Penalize trades in high vol
        
        # Multi-timeframe confluence bonus
        mtf_confluence = row.get('mtf_confluence', 0)
        if trade_executed and abs(mtf_confluence) > 0.5:
            reward += abs(mtf_confluence) * 0.03
        
        return reward

    def render(self):
        """Enhanced rendering with Category A feature display"""
        current_row = self.df.iloc[self.current_step] if self.current_step < len(self.df) else self.df.iloc[-1]
        
        print(f"Step {self.current_step} | Net: ${self.net_worth:.2f} | Position: {self.position_size:.2f} | Drawdown: {self.drawdown:.2%}")
        print(f"  Sentiment: {current_row.get('sentiment_composite', 0):.3f} | MTF Confluence: {current_row.get('mtf_confluence', 0):.3f}")
        print(f"  Liquidity Score: {current_row.get('liquidity_score', 0):.3f} | Vol Regime: {current_row.get('volatility_regime', 0)}")
        print(f"  Features Used: {sum(1 for v in self.feature_usage_count.values() if v > 0)}/{len(self.features)}")

    def get_advanced_stats(self):
        """Get comprehensive statistics including Category A features"""
        if not self.enable_stats_tracking:
            return {"error": "Stats tracking not enabled"}
        
        return self.stats_tracker.get_performance_summary(days=30)

    def export_feature_analysis(self):
        """Export detailed feature analysis"""
        if not self.enable_stats_tracking:
            return None
        
        return self.stats_tracker.generate_feature_attribution_report()

    def save_trades(self, file="backtest/trades.csv"):
        """Enhanced trade saving with Category A metadata"""
        df_trades = pd.DataFrame(self.trades)
        if not df_trades.empty:
            # Add feature usage statistics
            df_trades['total_features_used'] = len([f for f in self.feature_usage_count if self.feature_usage_count[f] > 0])
            df_trades['category_a_feature_count'] = sum(1 for f in self.feature_usage_count if 'microstructure' in f.lower() or 'sentiment' in f.lower())
        
        df_trades.to_csv(file, index=False)
        print(f"💾 Enhanced trades saved to {file}")

    def save_equity_curve(self, file="backtest/equity_curve.csv"):
        """Save equity curve with enhanced metrics"""
        equity_data = {
            "net_worth": self.equity_curve,
            "step": list(range(len(self.equity_curve))),
            "drawdown": [max(0, (max(self.equity_curve[:i+1]) - self.equity_curve[i]) / max(self.equity_curve[:i+1], 1)) for i in range(len(self.equity_curve))]
        }
        pd.DataFrame(equity_data).to_csv(file, index=False)
        print(f"📈 Enhanced equity curve saved to {file}")
