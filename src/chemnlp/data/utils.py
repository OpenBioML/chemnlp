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
