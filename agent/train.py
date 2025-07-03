import os
import yaml
import argparse
import pandas as pd
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from data.fetch_data import get_data
from utils.indicators import add_all_indicators
from utils.patterns import add_patterns
from utils.regime import detect_regime
from env.trading_env import TradingEnv

def load_config(config_path="utils/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_env(config, asset="AAPL"):
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
    return DummyVecEnv([lambda: Monitor(env)])

def get_agent(algo, env, config):
    if algo == "PPO":
        return PPO(config["agent"]["policy"], env, verbose=1, tensorboard_log=config["agent"]["tensorboard_log"])
    elif algo == "A2C":
        return A2C(config["agent"]["policy"], env, verbose=1, tensorboard_log=config["agent"]["tensorboard_log"])
    elif algo == "DQN":
        return DQN(config["agent"]["policy"], env, verbose=1, tensorboard_log=config["agent"]["tensorboard_log"])
    else:
        raise ValueError(f"Unsupported agent: {algo}")

def train(config_path="utils/config.yaml", asset=None):
    config = load_config(config_path)
    if asset is None:
        assets = config["data"].get("assets", ["AAPL"])
    else:
        assets = [asset]
    algo = config["agent"]["algo"]

    for sym in assets:
        print(f"\n=== Training on {sym} with {algo} ===")
        env = get_env(config, asset=sym)
        model = get_agent(algo, env, config)
        model.learn(total_timesteps=config["agent"]["total_timesteps"])
        os.makedirs("models", exist_ok=True)
        model.save(f"models/{sym}_{algo}_model")
        print(f"Model saved: models/{sym}_{algo}_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="utils/config.yaml")
    parser.add_argument("--asset", type=str, default=None)
    args = parser.parse_args()
    train(config_path=args.config, asset=args.asset)
