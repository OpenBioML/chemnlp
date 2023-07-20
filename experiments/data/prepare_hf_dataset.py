"""
Preparing chemrxiv dataset as per HF guidelines on the Stability AI cluster
Make sure to install optional dependencies; pip install chemnlp[tokenisation]

Example Usage:
    python prepare_hf_dataset.py full_path/config.yml
"""
import argparse
import json
import os

import datasets
from transformers import AutoTokenizer

from chemnlp.data.utils import tokenise
from chemnlp.data_val.config import HFDatasetConfig
from chemnlp.utils import load_config


def run(config_path: str):
    """Download, tokenise and save a HF dataset"""

    raw_config = load_config(config_path)
    config = HFDatasetConfig(**raw_config)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        model_max_length=config.context_length,
    )
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    dataset = datasets.load_dataset(
        config.dataset_name, **config.dataset_args, num_proc=os.cpu_count()
    )
    filtered_ds = dataset.filter(
        lambda x: isinstance(x[config.string_key], str), num_proc=os.cpu_count()
    )
    print(
        f"Removed {dataset.num_rows-filtered_ds.num_rows} samples due as not str type."
    )

    tokenised_data = filtered_ds.map(
        lambda batch: tokenise(
            batch, tokenizer, config.context_length, config.string_key
        ),
        batched=True,
        batch_size=config.batch_size,
        remove_columns=filtered_ds.column_names,
        num_proc=os.cpu_count(),
        load_from_cache_file=False,
    )

    summary_stats = {
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "total_samples": filtered_ds.num_rows,
        "dataset_args": config.dataset_args,
        "max_context_length": config.context_length,
        "total_tokens_in_billions": round(
            config.context_length * tokenised_data.num_rows / 1e9, 4
        ),
        "string_key": config.string_key,
    }
    print(summary_stats)

    tokenised_data.save_to_disk(config.save_path, num_proc=os.cpu_count())

    with open(f"{config.save_path}/summary_statistics.json", "w") as f:
        f.write(json.dumps(summary_stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="The full path to the YAML config file.")
    args = parser.parse_args()
    run(args.config_path)
