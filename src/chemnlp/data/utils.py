import random
import itertools

from datasets import concatenate_datasets
from datasets.formatting.formatting import LazyBatch

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

def tokenise(batch: LazyBatch, tokenizer, max_length: int, string_key: str):
        """Tokenise a batch of data using sample chunking"""
        tok_articles = [tokenizer(x)["input_ids"][1:] for x in batch[string_key]]
        tok_articles = list(itertools.chain.from_iterable(tok_articles))
        tok_articles = list(chunks(tok_articles, max_length))

        padded_sequences_all = []
        attention_masks_all = []

        # Since articles are stitched together at the batch level
        # we might need to pad the last article
        for article in tok_articles:
            padded_sequences, attention_masks = pad_sequence(
                article, seq_len=max_length
            )
            padded_sequences_all.append(padded_sequences)
            attention_masks_all.append(attention_masks)

        token_type_ids = [0] * max_length
        output = {
            "input_ids": padded_sequences_all,
            "token_type_ids": [token_type_ids] * len(padded_sequences_all),
            "attention_mask": attention_masks_all,
        }
        return output