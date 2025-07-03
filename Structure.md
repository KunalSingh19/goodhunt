# GoodHunt v3+ — Profit-Optimized RL Trading System

## 📁 Project Structure

```
goodhunt_v3/
├── data/
│   └── fetch_data.py                # Download & preprocess raw data (multi-asset)
├── utils/
│   ├── indicators.py                # All technicals: EMA, MACD, ADX, ATR, Bollinger, etc.
│   ├── patterns.py                  # Candlestick pattern recognition
│   ├── regime.py                    # Market regime (trend/range/volatility) detection
│   └── config.yaml                  # Modular config (env, agent, reward, assets)
├── env/
│   ├── trading_env.py               # Core RL env: multi-action, hedging, scaling, regime
│   ├── reward_engine.py             # Modular reward shaping (Sharpe, R:R, smooth, etc)
│   └── slippage.py                  # Spread, slippage, liquidity, fractional trading
├── agent/
│   ├── train.py                     # RL agent training (PPO/A2C/DQN) + tuning
│   ├── evaluate.py                  # Backtest, multi-agent comparison, replay
│   └── grid_search.py               # Hyperparameter sweep runner
├── backtest/
│   ├── test_agent.py                # Out-of-sample eval, profit analytics
│   ├── plots.py                     # Equity curve, confusion matrix, PnL breakdown
│   └── logs/
│       └── trades.csv               # Per-trade log (ROI, reason, regime, etc)
│       └── equity_curve.csv
│       └── decisions.csv            # Action classification/confusion matrix
├── notebooks/
│   └── demo_colab.ipynb             # Colab: E2E demo, config, visualization
├── models/
│   └── best_model.zip               # Saved RL agent(s)
├── main.py                          # CLI: train/test/grid/all (single entry point)
├── requirements.txt
└── README.md
```

---

## 🧱 Layers & Modules

- **data/**         — Data acquisition and normalization
- **utils/**        — Indicators, patterns, regime, config
- **env/**          — RL environment, reward, slippage, risk logic
- **agent/**        — Training, hyperparam search, evaluation
- **backtest/**     — Analytics, plots, logs
- **notebooks/**    — Colab demo & quickstart
- **models/**       — Agent checkpoints
- **main.py**       — Command-line (train/test/grid/plot)
- **requirements.txt** — Full dependency list

---

## 📦 Modular & Configurable

- **Config-driven**: All features toggled in `utils/config.yaml`
- **Multi-agent**: PPO, A2C, DQN, rule-based
- **Multi-asset**: BTC, AAPL, NIFTY, TSLA, etc.
- **Extensible**: Plug in new indicators, regimes, reward modules, agent types

---

**Ready to evolve GoodHunt v3+ with this structure.**