"""
Preparing chemrxiv dataset as per HF guidelines on the Stability AI cluster

Example Usage:
    python prepare_hf_chemrxiv.py EleutherAI/pythia-160m 768
"""
import argparse
import json
import os

import datasets
from transformers import AutoTokenizer

from chemnlp.data.utils import tokenise

DATASET = "marianna13/chemrxiv"
STRING_KEY = "TEXT"  # only taking research article body (not abstract, etc)
OUT_DIR = "/fsx/proj-chemnlp/data"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Which HF tokeniser model class to use.")
    parser.add_argument(
        "max_length", help="Maximum context length of the model.", type=int
    )
    args = parser.parse_args()

    # load tokeniser
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        model_max_length=args.max_length,
    )
    if not tokenizer.pad_token:
        # GPT NeoX has no provided pad token
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    # load data (only has a single split)
    chem_data = datasets.load_dataset(DATASET, split="train")
    words_per_sample = [len(x[STRING_KEY].split(" ")) for x in chem_data]

    # process data
    tokenised_data = chem_data.map(
        lambda batch: tokenise(batch, tokenizer, args.max_length, STRING_KEY),
        batched=True,
        remove_columns=chem_data.column_names,
        num_proc=os.cpu_count(),
        load_from_cache_file=False,
    )
    summary_stats = {
        "total_raw_samples": chem_data.num_rows,
        "average_words_per_sample": round(
            sum(words_per_sample) / chem_data.num_rows, 0
        ),
        "max_words_per_sample": max(words_per_sample),
        "min_words_per_sample": min(words_per_sample),
        "total_tokenised_samples": tokenised_data.num_rows,
        "max_context_length": args.max_length,
        "total_tokens_in_billions": round(
            args.max_length * tokenised_data.num_rows / 1e9, 4
        ),
    }
    print(summary_stats)

    # save to disk
    save_path = f"{OUT_DIR}/{args.model_name}/{DATASET}"
    tokenised_data.save_to_disk(save_path)
    with open(f"{save_path}/summary_statistics.json", "w") as f:
        f.write(json.dumps(summary_stats))
