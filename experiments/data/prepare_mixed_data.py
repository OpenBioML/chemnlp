import argparse
import json

import datasets

from chemnlp.data_val.config import DataMixingConfig
from chemnlp.utils import load_config

RANDOM_SEED = 1234


def run(config_path: str) -> None:
    """Create a mixed dataset for a given YAML defined configuration"""
    raw_config = load_config(config_path)
    config = DataMixingConfig(**raw_config)

    mixed_data = datasets.interleave_datasets(
        datasets=[
            datasets.load_from_disk(data_path) for data_path in config.data_paths
        ],
        probabilities=config.data_proportions,
        seed=RANDOM_SEED,
        stopping_strategy="first_exhausted",
    )

    mixed_data.save_to_disk(config.save_path)

    summary_stats = {
        "random_state": RANDOM_SEED,
        "component_datasets": config.data_paths,
        "component_proportions": config.data_proportions,
        "interleave_stopping_strategy": "first_exhausted",
        "dataset_size": len(mixed_data),
    }

    with open(f"{config.save_path}/summary_statistics.json", "w") as f:
        f.write(json.dumps(summary_stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="The full path to the YAML config file.")
    args = parser.parse_args()
    run(args.config_path)
