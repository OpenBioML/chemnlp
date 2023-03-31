import random

from datasets import concatenate_datasets

import chemnlp.data.hf_datasets as hf_datasets


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


def pad_sequence(sequence, seq_len):
    """Pad a input sequence"""
    if len(sequence) != seq_len:
        num_pad_tokens = seq_len - len(sequence)
        attention_mask = [1] * len(sequence) + [0] * num_pad_tokens
        sequence += [0] * num_pad_tokens
    else:
        attention_mask = [1] * seq_len

    return sequence, attention_mask