import itertools
import yaml
import argparse
from agent.train import train

def run_grid_search(config_path, param_grid):
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    keys, values = zip(*param_grid.items())
    for param_set in itertools.product(*values):
        config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
        update = dict(zip(keys, param_set))
        # Recursively update config
        def recursive_update(cfg, upd):
            for k, v in upd.items():
                if isinstance(v, dict) and k in cfg:
                    recursive_update(cfg[k], v)
                else:
                    keysplit = k.split('.')
                    target = cfg
                    for subk in keysplit[:-1]:
                        target = target[subk]
                    target[keysplit[-1]] = v
        recursive_update(config, update)
        # Save temporary config for this run
        tmp_config_path = "utils/tmp_grid_config.yaml"
        with open(tmp_config_path, "w") as f:
            yaml.dump(config, f)
        print(f"\n>>> Running with config: {update}")
        train(config_path=tmp_config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="utils/config.yaml")
    parser.add_argument("--grid", type=str, required=True,
                        help="Grid params as yaml dict, e.g. 'agent.algo: [PPO, A2C], env.window_size: [30, 60]'")
    args = parser.parse_args()
    import ast
    grid_dict = ast.literal_eval(args.grid)
    run_grid_search(args.config, grid_dict)
