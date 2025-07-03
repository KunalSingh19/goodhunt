import argparse
import yaml
import pandas as pd
from stable_baselines3 import PPO, A2C, DQN

from data.fetch_data import get_data
from utils.indicators import add_all_indicators
from utils.patterns import add_patterns
from utils.regime import detect_regime
from env.trading_env import TradingEnv
from backtest.plots import plot_equity_curve, plot_trade_roi, plot_drawdown_curve, plot_top_patterns

def load_config(config_path="utils/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_agent(agent_path, algo="PPO"):
    if algo == "PPO":
        return PPO.load(agent_path)
    elif algo == "A2C":
        return A2C.load(agent_path)
    elif algo == "DQN":
        return DQN.load(agent_path)
    else:
        raise ValueError(f"Unsupported agent: {algo}")

def backtest_single(config, asset, agent_path, save_dir="backtest"):
    print(f"\n--- Backtesting {agent_path} on {asset} ---")
    df = get_data(
        symbol=asset,
        start=config["data"]["start"],
        end=config["data"]["end"],
        interval=config["data"]["interval"],
        preprocess=False,
        save=False
    )
    df = add_all_indicators(df)
    df = add_patterns(df)
    df = detect_regime(df)
    env = TradingEnv(
        df,
        window_size=config["env"]["window_size"],
        initial_balance=config["env"]["initial_balance"],
        max_exposure=config["env"]["exposure_cap"],
        fee_pct=config["env"]["fee_pct"],
        config=config["reward"]
    )
    model = load_agent(agent_path, algo=config["agent"]["algo"])
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        # env.render()  # Uncomment for stepwise output

    trades_csv = f"{save_dir}/{asset}_trades.csv"
    equity_csv = f"{save_dir}/{asset}_equity_curve.csv"
    env.save_trades(trades_csv)
    env.save_equity_curve(equity_csv)

    print(f"Trade log saved: {trades_csv}")
    print(f"Equity curve saved: {equity_csv}")

    # Analytics/plots
    plot_equity_curve(equity_csv, asset, save_dir)
    plot_trade_roi(trades_csv, asset, save_dir)
    plot_drawdown_curve(equity_csv, asset, save_dir)
    plot_top_patterns(trades_csv, asset, save_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="utils/config.yaml")
    parser.add_argument("--asset", type=str, default=None)
    parser.add_argument("--multi", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    assets = config["data"].get("assets", ["AAPL"]) if args.asset is None else [args.asset]
    algo = config["agent"]["algo"]

    if args.multi or len(assets) > 1:
        for asset in assets:
            agent_path = f"models/{asset}_{algo}_model"
            backtest_single(config, asset, agent_path)
    else:
        asset = assets[0]
        agent_path = f"models/{asset}_{algo}_model"
        backtest_single(config, asset, agent_path)

if __name__ == "__main__":
    main()
