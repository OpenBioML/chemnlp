"""
Convert hendrycks eval dataset into a training dataset as a sanity check"

Example Usage:
    python prepare_lmeval_dataset.py EleutherAI/pythia-1b 2048 # all Pythia models are 2048
"""

import argparse
import json
import os
import random

import datasets
import lm_eval.tasks
from transformers import AutoTokenizer

from chemnlp.data.utils import tokenise

TASKS = [
    "hendrycksTest-college_biology",
    "hendrycksTest-college_chemistry",
    "hendrycksTest-college_mathematics",
    "hendrycksTest-college_physics",
    "hendrycksTest-high_school_mathematics",
    "hendrycksTest-high_school_biology",
    "hendrycksTest-high_school_chemistry",
    "hendrycksTest-high_school_physics",
]

STRING_KEY = "TEXT"
OUT_DIR = "/fsx/proj-chemnlp/data"
NAME = "hendrycks_STEM"


def make_test_dataset(tasks):
    task_dict = lm_eval.tasks.get_task_dict(tasks)
    task_dict_items = [
        (name, task) for name, task in task_dict.items() if task.has_test_docs()
    ]

    docs = []
    for _, task in task_dict_items:
        task_doc_func = task.test_docs

        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)
        for doc in task_docs:
            docs.append(doc["query"] + doc["choices"][doc["gold"]])

    return datasets.Dataset.from_dict({STRING_KEY: docs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Which HF tokeniser model class to use.")
    parser.add_argument(
        "max_length", help="Maximum context length of the model.", type=int
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        model_max_length=args.max_length,
    )
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    dataset = make_test_dataset(TASKS)
    words_per_sample = [len(x[STRING_KEY].split(" ")) for x in dataset]

    tokenised_data = dataset.map(
        lambda batch: tokenise(batch, tokenizer, args.max_length, STRING_KEY),
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=1,
        num_proc=os.cpu_count(),
        load_from_cache_file=False,
    )

    summary_stats = {
        "total_raw_samples": dataset.num_rows,
        "average_words_per_sample": round(sum(words_per_sample) / dataset.num_rows, 0),
        "max_words_per_sample": max(words_per_sample),
        "min_words_per_sample": min(words_per_sample),
        "total_tokenised_samples": tokenised_data.num_rows,
        "max_context_length": args.max_length,
        "total_tokens_in_billions": round(
            args.max_length * tokenised_data.num_rows / 1e9, 4
        ),
    }
    print(summary_stats)

    save_path = f"{OUT_DIR}/{args.model_name}/{NAME}"
    tokenised_data.save_to_disk(save_path)
    with open(f"{save_path}/summary_statistics.json", "w") as f:
        f.write(json.dumps(summary_stats))
