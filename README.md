# 🦊 GoodHunt v3+ — Profit-Optimized RL Trading Framework

**The next-gen, research-ready, profit-first RL system for stock, crypto, and multi-asset trading.**

---

## 🚀 Features

- **Elite Observations:**  
  30+ real-world indicators — MACD, ADX, Stochastic, Bollinger Bands, entropy, market regime, volume triggers, and more.

- **Smarter Execution:**  
  Dynamic position sizing, multi-action scaling, regime-aware strategies, drawdown/risk-off behavior, trailing take-profits, and advanced entry/exit logic.

- **True-to-Reality Simulation:**  
  Dynamic slippage, bid-ask spread, liquidity/volume filters, fractional trading, capital exposure caps.

- **Reward Engineering:**  
  Sharpe/Sortino, risk/reward, decay, stability boosts, chaos bonuses, and volatility-aware reward clipping.

- **Analytics & Research:**  
  Trade logs, ROI/duration, top/worst triggers, confusion matrix, drawdown, rolling equity, per-pattern PnL, and live TensorBoard.

- **Multi-Agent & Multi-Asset:**  
  Compare PPO, A2C, DQN, and rule-based agents across stocks, crypto, and indices.

- **Configurable & Modular:**  
  YAML-powered experiment configs. Plug in new rewards, indicators, assets, or agent types.

- **Colab & CLI Ready:**  
  Quickstart notebook, CLI for training, testing, sweeps, and plots.

---

## 📦 Project Structure

```
goodhunt_v3/
├── data/            # Data download & normalization
├── utils/           # Indicators, patterns, regime, config
├── env/             # TradingEnv, reward, slippage, risk logic
├── agent/           # Training, evaluation, grid search
├── backtest/        # Analytics, plots, logs
├── notebooks/       # Colab/E2E demo
├── models/          # Saved RL agents
├── main.py          # CLI entry point (train/test/grid)
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Download Data & Add Features

Edit `utils/config.yaml` for your assets/settings, or use Colab:

```python
from data.fetch_data import get_data
from utils.indicators import add_all_indicators
from utils.patterns import add_patterns
from utils.regime import detect_regime

df = get_data(symbol="AAPL", start="2023-01-01", end="2024-06-30")
df = add_all_indicators(df)
df = add_patterns(df)
df = detect_regime(df)
```

### 3. Train

```bash
python main.py --train --config utils/config.yaml
```

### 4. Backtest

```bash
python main.py --test --config utils/config.yaml
```

### 5. Plot & Analyze

See `backtest/` for equity curves, trade ROI, drawdown, and pattern analytics.

---

## 🧠 Key Concepts

- **Multi-Action:** Buy, Sell, Hold, Scale Up/Down, Flat, with volatility-aware sizing.
- **Risk Management:** Exposure caps, trailing stops, dynamic ATR stops, drawdown-aware throttling.
- **Realism:** Slippage based on volume/volatility, volume filters, bid-ask spread simulation.
- **Reward:** Modular, risk-adjusted, smooth, and robust to market chaos.

---

## 🛠️ Customization

- **Assets:**  
  Edit `utils/config.yaml`, e.g.  
  `assets: ["AAPL", "BTC-USD", "NIFTY"]`

- **Agent/Algo:**  
  Switch PPO/A2C/DQN in config.

- **Indicators & Patterns:**  
  Add your own in `utils/indicators.py` or `patterns.py`.

- **Reward:**  
  Plug new reward functions in `env/reward_engine.py`.

---

## 📊 Analytics

- **Equity Curve & Drawdown**
- **Per-Trade ROI% & Duration**
- **Pattern/Trigger Profitability**
- **Confusion Matrix (Decisions)**
- **Max Drawdown & Capital Exposure**
- **TensorBoard Logging**

---

## 🧪 Research & Scaling

- **Multi-Agent Compare:**  
  `python agent/evaluate.py --multi`

- **Grid Search:**  
  `python main.py --grid`

- **Colab Demo:**  
  See [`notebooks/demo_colab.ipynb`](notebooks/demo_colab.ipynb)

---

## ❤️ Why GoodHunt v3+?

- Built for real-world profit, not just “smart” trading.
- Handles volatility, drawdown, and regime shifts—out of the box.
- Modular for research, robust enough for live trading simulation.

---

## 📜 License

MIT — Use, remix, and profit!

---

## ✨ Credits

Made by [KunalSingh19](https://github.com/KunalSingh19) and contributors.

---

**Good hunting! 🦊**
