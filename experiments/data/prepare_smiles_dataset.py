"""
Create training dataset from subset of coconut smiles"

Example Usage:
    python prepare_smiles_dataset.py full/config_path.yml
"""

import argparse
import itertools
import json

import datasets
from numpy import random
from rdkit import Chem, RDLogger
from transformers import AutoTokenizer

from chemnlp.data.utils import pad_sequence
from chemnlp.data_val.config import LMEvalDataConfig
from chemnlp.utils import load_config

RDLogger.DisableLog("rdApp.*")


DATASET_PATH = "OpenBioML/coconut_molecules"
VALID_PREFIX = "The following is a valid molecule:"
INVALID_PREFIX = "The following is not a valid molecule:"
LIFT_Q = "Question: Is the following a valid molecule:"
VALID_LIFT_A = "Yes"
INVALID_LIFT_A = "No"
DATA_TYPE = "text"
EOS_TOKEN = "\n\n"
SEED = 1234


def process_docs(docs):
    valid = map(_get_smiles_string, docs)
    invalid = map(_process_invalid_smiles, docs)
    mixed_data = list(valid) + list(invalid)
    mixed_data = [string for pair in mixed_data for string in pair]
    return random.choice(mixed_data, len(mixed_data)).tolist()


def _process_invalid_smiles(doc):
    invalid_smiles = doc[: random.randint(1, len(doc))]
    return _get_smiles_string(invalid_smiles)


def _get_smiles_string(doc):
    is_valid = Chem.MolFromsmiles(doc)
    lift_answer = INVALID_LIFT_A if is_valid is None else VALID_LIFT_A
    prefix = INVALID_PREFIX if is_valid is None else VALID_PREFIX
    return (
        f"{prefix} {doc}{EOS_TOKEN}",
        f"{LIFT_Q} {doc}? Answer: {lift_answer}{EOS_TOKEN}",
    )


def concatenate_samples_without_splitting(dataset, tokenizer, max_length):
    """concatenate samples into batches upto max_length without
    splitting any of the individual samples between batches"""

    tok_articles = [tokenizer(x)["input_ids"] for x in dataset]
    tok_articles = [sample for sample in tok_articles if len(sample) <= max_length]
    tok_articles = list(itertools.chain.from_iterable(tok_articles))
    eos_token = tokenizer.encode(EOS_TOKEN)[0]

    concatenated_articles = []
    p0, p1, last_eos = 0, 1, 0
    while p1 < len(tok_articles):
        if tok_articles[p1] == eos_token:
            if (p1 - p0 + 1) < max_length:
                # keep track of most recent eos index, continue exploring
                last_eos = p1

            elif (p1 - p0 + 1) == max_length:
                # collect whole pointer window
                concatenated_articles.append(tok_articles[p0 : p1 + 1])
                last_eos = p1
                p0 = p1 + 1
                p1 = p0
            else:
                # max_length exceeded, collect only up to last eos
                concatenated_articles.append(tok_articles[p0 : last_eos + 1])
                p0 = last_eos + 1
                p1 = p0
        p1 += 1

    # collect final batch
    concatenated_articles.append(tok_articles[p0:])
    return concatenated_articles


def pad_batched_data(dataset, tokenizer, max_length):
    padded_sequences_all = []
    attention_masks_all = []

    for article in dataset:
        if len(article) < max_length:
            article, attention_masks = pad_sequence(
                article, max_length, tokenizer.pad_token_id
            )
        else:
            attention_masks = [1] * max_length
        padded_sequences_all.append(article)
        attention_masks_all.append(attention_masks)

    return {
        "input_ids": padded_sequences_all,
        "token_type_ids": [[0] * max_length] * len(padded_sequences_all),
        "attention_mask": attention_masks_all,
    }


def run(config):
    random.seed(seed=SEED)

    dataset = datasets.load_dataset(
        path=DATASET_PATH,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
    )
    dataset = dataset[config.data_split][DATA_TYPE]
    mixed_data = process_docs(dataset)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        model_max_length=config.context_length,
    )
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    batched_data = concatenate_samples_without_splitting(
        mixed_data, tokenizer, config.context_length
    )
    total_tokens = sum([len(batch) for batch in batched_data])
    processed_data = pad_batched_data(batched_data, tokenizer, config.context_length)
    processed_data = datasets.Dataset.from_dict(processed_data)

    summary_stats = {
        "model_name": config.model_name,
        "total_samples": processed_data.num_rows,
        "max_context_length": config.context_length,
        "total_padded_tokens_in_billions": round(
            config.context_length * processed_data.num_rows / 1e9, 4
        ),
        "total_tokens_in_billions": round(total_tokens / 1e9, 4),
        "data_split_collected": config.data_split,
    }

    save_path = f"{config.out_dir}/{config.save_name}_{config.data_split}"
    processed_data.save_to_disk(save_path)

    print(summary_stats)
    if "s3" in save_path:
        # yet to be implemented
        pass
    else:
        with open(f"{save_path}/summary_statistics.json", "w") as f:
            f.write(json.dumps(summary_stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="The full path to the YAML config file.")
    args = parser.parse_args()
    raw_config = load_config(args.config_path)
    config = LMEvalDataConfig(**raw_config)
    run(config)
