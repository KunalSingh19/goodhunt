import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_equity_curve(equity_csv, asset, save_dir="backtest"):
    df = pd.read_csv(equity_csv)
    plt.figure(figsize=(10, 4))
    plt.plot(df['net_worth'], label="Equity Curve", color='navy')
    plt.title(f"{asset} — Equity Curve")
    plt.xlabel("Step")
    plt.ylabel("Net Worth")
    plt.legend()
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/{asset}_equity_curve.png"
    plt.savefig(path)
    plt.close()
    print(f"[Plot] Saved equity curve: {path}")

def plot_trade_roi(trades_csv, asset, save_dir="backtest"):
    df = pd.read_csv(trades_csv)
    df = df[df['action'].isin(["SELL", "FORCE_FLAT"])]
    if "pnl" in df.columns and "size" in df.columns and "price" in df.columns:
        df["roi_%"] = 100 * df["pnl"] / (df["size"] * df["price"] + 1e-9)
        plt.figure(figsize=(10, 4))
        plt.bar(df["step"], df["roi_%"], color=np.where(df["roi_%"]>0, "green", "red"))
        plt.title(f"{asset} — Trade ROI%")
        plt.xlabel("Step")
        plt.ylabel("ROI (%)")
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir}/{asset}_trade_roi.png"
        plt.savefig(path)
        plt.close()
        print(f"[Plot] Saved trade ROI: {path}")

def plot_drawdown_curve(equity_csv, asset, save_dir="backtest"):
    df = pd.read_csv(equity_csv)
    running_max = df['net_worth'].cummax()
    drawdown = (running_max - df['net_worth']) / (running_max + 1e-9)
    plt.figure(figsize=(10, 3))
    plt.plot(drawdown, color="crimson")
    plt.title(f"{asset} — Drawdown Curve")
    plt.xlabel("Step")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/{asset}_drawdown.png"
    plt.savefig(path)
    plt.close()
    print(f"[Plot] Saved drawdown curve: {path}")

def plot_top_patterns(trades_csv, asset, save_dir="backtest", top_n=5):
    df = pd.read_csv(trades_csv)
    if "reason" in df.columns and "pnl" in df.columns:
        pattern_pnl = df.groupby("reason")["pnl"].sum().sort_values(ascending=False)
        top_patterns = pattern_pnl.head(top_n)
        plt.figure(figsize=(8, 4))
        top_patterns.plot(kind="bar", color="teal")
        plt.title(f"{asset} — Top {top_n} Patterns by PnL")
        plt.xlabel("Pattern/Trigger")
        plt.ylabel("Total PnL")
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir}/{asset}_top_patterns.png"
        plt.savefig(path)
        plt.close()
        print(f"[Plot] Saved top patterns bar: {path}")
