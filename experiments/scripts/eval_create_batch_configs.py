import argparse
import os

import yaml
from lm_eval import config

CHECKPOINT_DIR = "checkpoint-final"


def run(
    config_path: str,
    root_models_path: str,
):
    raw_config = config.load_config(config_path)

    model_names = [
        name
        for name in os.listdir(root_models_path)
        if os.path.isdir(os.path.join(root_models_path, name))
    ]

    for model_name in model_names:
        raw_config["model_args"] = (
            f"pretrained={root_models_path}/{model_name}/{CHECKPOINT_DIR}"
        )
        raw_config["wandb_run_name"] = model_name

        with open(
            f"{root_models_path}/{model_name}/eval_config.yml", "w"
        ) as new_config:
            yaml.dump(raw_config, new_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", help="The full path to the example YAML config file."
    )
    parser.add_argument(
        "root_models_path",
        help="The full path to the parent directory containing models.",
    )
    args = parser.parse_args()
    run(args.config_path, args.root_models_path)
