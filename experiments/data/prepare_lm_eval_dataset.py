"""
Create training dataset from subset of lm-eval datasets"

Example Usage:
    python prepare_lm_eval_dataset.py full/config_path.yml
"""

import argparse
import json

import datasets
import lm_eval.tasks
from transformers import AutoTokenizer

from chemnlp.data.utils import get_tokenised_data_minimum_padding
from chemnlp.data_val.config import LMEvalDataConfig
from chemnlp.utils import load_config

STRING_KEY = "TEXT"
RANDOM_SEED = 1234
EOS_TOKEN = "\n\n"


def make_dataset(tasks, data_split):
    task_dict = lm_eval.tasks.get_task_dict(tasks)

    docs = []
    task_sizes = {}
    for task_name, task in task_dict.items():
        task_condition = f"has_{data_split}_docs"
        task_func = f"{data_split}_docs"
        if getattr(task, task_condition):
            task_doc_func = getattr(task, task_func)
        else:
            raise ValueError(f"{task_name} does not have {data_split} split")

        task_docs = list(task_doc_func())
        task_sizes[task_name] = len(task_docs)

        for doc in task_docs:
            docs.append(f'{doc["query"]} {doc["choices"][doc["gold"]]}{EOS_TOKEN}')

    return docs, task_sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="The full path to the YAML config file.")
    args = parser.parse_args()
    raw_config = load_config(args.config_path)
    config = LMEvalDataConfig(**raw_config)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        model_max_length=config.context_length,
    )
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    dataset, task_sizes = make_dataset(
        tasks=config.tasks,
        data_split=config.data_split,
    )

    processed_data = get_tokenised_data_minimum_padding(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=config.context_length,
        eos_string=EOS_TOKEN,
    )
    processed_data = datasets.Dataset.from_dict(processed_data)

    summary_stats = {
        "model_name": config.model_name,
        "total_samples": processed_data.num_rows,
        "samples_per_task": task_sizes,
        "max_context_length": config.context_length,
        "total_tokens": config.context_length * processed_data.num_rows,
        "total_padded_tokens_in_billions": round(
            config.context_length * processed_data.num_rows / 1e9, 4
        ),
        "data_split_collected": config.data_split,
    }
    print(summary_stats)

    save_path = (
        f"{config.out_dir}/{config.model_name}/{config.save_name}_{config.data_split}"
    )
    processed_data.save_to_disk(save_path)
    if "s3" in save_path:
        # yet to be implemented
        pass
    else:
        with open(f"{save_path}/summary_statistics.json", "w") as f:
            f.write(json.dumps(summary_stats))
