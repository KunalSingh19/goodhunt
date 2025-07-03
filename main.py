import argparse

def main():
    parser = argparse.ArgumentParser(description="GoodHunt v3+ CLI")
    parser.add_argument("--train", action="store_true", help="Train the RL agent(s)")
    parser.add_argument("--test", action="store_true", help="Backtest the RL agent(s)")
    parser.add_argument("--grid", action="store_true", help="Run grid search hyperparameter tuning")
    parser.add_argument("--multi", action="store_true", help="Multi-asset train/test")
    parser.add_argument("--config", type=str, default="utils/config.yaml", help="Config file path")
    args = parser.parse_args()

    if args.train:
        from agent.train import train
        train(config_path=args.config)
    if args.test:
        from backtest.test_agent import main as test_main
        test_main()
    if args.grid:
        from agent.grid_search import run_grid_search
        # Example grid: {"agent.algo": ["PPO", "A2C"], "env.window_size": [30, 60]}
        import ast
        grid = input("Enter grid dict (e.g. {'agent.algo': ['PPO', 'A2C'], 'env.window_size': [30,60]}): ")
        grid_dict = ast.literal_eval(grid)
        run_grid_search(args.config, grid_dict)
    if not args.train and not args.test and not args.grid:
        print("No action specified. Use --train, --test, or --grid.")

if __name__ == "__main__":
    main()
