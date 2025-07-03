import argparse
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN

from data.fetch_data import get_data
from utils.indicators import add_all_indicators
from utils.patterns import add_patterns
from utils.regime import detect_regime
from env.trading_env import TradingEnv

def load_agent(agent_path, algo="PPO"):
    if algo == "PPO":
        return PPO.load(agent_path)
    elif algo == "A2C":
        return A2C.load(agent_path)
    elif algo == "DQN":
        return DQN.load(agent_path)
    else:
        raise ValueError(f"Unsupported agent: {algo}")

def eval_agent(asset, model_path, config, plot=True):
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
    model = load_agent(model_path, algo=config["agent"]["algo"])
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        env.render()

    env.save_trades(f"backtest/{asset}_trades.csv")
    env.save_equity_curve(f"backtest/{asset}_equity_curve.csv")

    if plot:
        eq = pd.read_csv(f"backtest/{asset}_equity_curve.csv")
        plt.figure()
        plt.plot(eq['net_worth'])
        plt.title(f"Equity Curve: {asset}")
        plt.xlabel("Step")
        plt.ylabel("Net Worth")
        plt.tight_layout()
        plt.savefig(f"backtest/{asset}_equity_curve.png")
        print(f"Equity curve plot saved for {asset}")

def multi_agent_compare(config_path="utils/config.yaml"):
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    assets = config["data"].get("assets", ["AAPL"])
    algos = ["PPO", "A2C", "DQN"]
    for algo in algos:
        for asset in assets:
            print(f"\n--- {algo} on {asset} ---")
            model_path = f"models/{asset}_{algo}_model"
            if not pd.io.common.file_exists(model_path + ".zip"):
                print(f"Model not trained: {model_path}")
                continue
            config["agent"]["algo"] = algo
            eval_agent(asset, model_path, config, plot=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="utils/config.yaml")
    parser.add_argument("--multi", action="store_true")
    args = parser.parse_args()
    if args.multi:
        multi_agent_compare(config_path=args.config)
    else:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        asset = config["data"].get("assets", ["AAPL"])[0]
        model_path = f"models/{asset}_{config['agent']['algo']}_model"
        eval_agent(asset, model_path, config, plot=True)
