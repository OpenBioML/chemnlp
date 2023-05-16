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

    if len(config.data_paths) != len(config.num_tokens):
        raise ValueError("Must specify num_token for each data_path")

    dataset_slices = []
    for path, num_token in zip(config.data_paths, config.num_tokens):
        print(path)
        dataset = datasets.load_from_disk(path)
        num_rows = int(num_token / config.context_length)

        if num_rows > len(dataset):
            raise ValueError(
                f"Dataset at {path} is smaller than requested number of tokens"
            )

        elif num_rows == len(dataset):
            dataset_slices.append(dataset)

        else:
            dataset_slices.append(
                dataset.train_test_split(train_size=num_rows)["train"]
            )

    mixed_data = datasets.concatenate_datasets(dataset_slices)
    mixed_data = mixed_data.shuffle(seed=RANDOM_SEED)

    mixed_data.save_to_disk(config.save_path)

    summary_stats = {
        "random_state": RANDOM_SEED,
        "component_datasets": config.data_paths,
        "component_num_tokens": config.num_tokens,
        "context_length": config.context_length,
        "dataset_size_samples": len(mixed_data),
        "dataset_size_tokens_Billions": (len(mixed_data) * config.context_length)
        // 1e9,
    }

    print(summary_stats)

    with open(f"{config.save_path}/summary_statistics.json", "w") as f:
        f.write(json.dumps(summary_stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="The full path to the YAML config file.")
    args = parser.parse_args()
    run(args.config_path)
