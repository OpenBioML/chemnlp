import itertools
import random
from typing import Dict, List, Optional

from datasets import concatenate_datasets
from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizer

import chemnlp.data.hf_datasets as hf_datasets

import yaml
from typing import Any


def sample_dataset(dataset, num_samples):
    n = len(dataset)
    num_samples = min(num_samples, n)
    return dataset.select(random.sample(range(n), k=num_samples))


def get_datasets(config, tokenizer):
    train_datasets, val_datasets = [], []
    for dataset_name in config.data.datasets:
        dataset_fn_ref = getattr(hf_datasets, dataset_name)
        train_tokenized, val_tokenized = dataset_fn_ref(tokenizer)
        train_datasets.append(train_tokenized)
        val_datasets.append(val_tokenized)
    return concatenate_datasets(train_datasets), concatenate_datasets(val_datasets)


def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    NOTE Hugging face truncates any large samples -> 1 sample
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n - 1]


def pad_sequence(sequence, max_len, pad_token_id):
    """Pad a input sequence"""
    num_pad_tokens = max_len - len(sequence)
    attention_mask = [1] * len(sequence) + [0] * num_pad_tokens
    sequence += [pad_token_id] * num_pad_tokens
    return sequence, attention_mask


def tokenise(
    batch: LazyBatch,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    string_key: str,
    keep_columns: Optional[List[str]] = None,
) -> Dict[str, List]:
    """Tokenise a batch of data using sample chunking"""
    tok_articles = [tokenizer(x)["input_ids"] for x in batch[string_key]]
    flattened_tokens = list(itertools.chain.from_iterable(tok_articles))
    chunked_tokens = list(chunks(flattened_tokens, max_length))
    padded_sample_tokens = _pad_batched_data(
        dataset=chunked_tokens,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    if keep_columns:
        # augment with token-level metadata
        sample_metadata = [
            {meta: batch[meta][i] or "" for meta in keep_columns}
            for i, _ in enumerate(batch[string_key])
        ]
        tok_metadata = [
            [sample_meta] * len(x)
            for sample_meta, x in zip(sample_metadata, tok_articles)
        ]
        tok_metadata = list(itertools.chain.from_iterable(tok_metadata))
        tok_metadata = list(chunks(tok_metadata, max_length))
        padded_sample_tokens["metadata"] = tok_metadata

    return padded_sample_tokens


def get_tokenised_data_minimum_padding(
    dataset: List,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    eos_string: str,
) -> Dict[str, List]:
    batched_data = _concatenate_samples_without_splitting(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        eos_string=eos_string,
    )

    return _pad_batched_data(
        dataset=batched_data,
        tokenizer=tokenizer,
        max_length=max_length,
    )


def _concatenate_samples_without_splitting(
    dataset: List,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    eos_string: str,
):
    """concatenate samples into batches upto max_length without
    splitting any of the individual samples between batches"""

    tok_articles = [tokenizer(x)["input_ids"] for x in dataset]
    tok_articles = [sample for sample in tok_articles if len(sample) <= max_length]
    tok_articles = list(itertools.chain.from_iterable(tok_articles))
    eos_token = tokenizer.encode(eos_string)[0]

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


def _pad_batched_data(
    dataset: List,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
):
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


def oxford_comma_join(items: List[str]) -> str:
    """Join a list of items with Oxford comma"""
    if len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return ", ".join(items[:-1]) + f", and {items[-1]}"


def load_yaml(file_path: str) -> Any:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(data: Any, file_path: str) -> None:
    with open(file_path, "w") as file:
        yaml.dump(data, file, sort_keys=False)


def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter)
