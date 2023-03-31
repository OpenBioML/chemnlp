"""
Preparing chemrxiv dataset as per HF guidelines on the Stability AI cluster

Example Usage:
    python prepare_hf_chemrxiv.py EleutherAI/pythia-160m 768 <save-dir>
"""
import argparse
import itertools
import os

import datasets
from datasets.formatting.formatting import LazyBatch
from transformers import AutoTokenizer

DATASET = "marianna13/chemrxiv"
STRING_KEY = "TEXT"  # only taking research article body (not abstract, etc)
OUT_DIR = "/fsx/proj-chemnlp/data"


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

    def tokenise(batch: LazyBatch):
        """Tokenise a batch of data using sample chunking"""
        tok_articles = [tokenizer(x)["input_ids"][1:] for x in batch[STRING_KEY]]
        tok_articles = list(itertools.chain.from_iterable(tok_articles))
        tok_articles = list(chunks(tok_articles, args.max_length))

        padded_sequences_all = []
        attention_masks_all = []

        # Since articles are stitched together at the batch level
        # we might need to pad the last article
        for article in tok_articles:
            padded_sequences, attention_masks = pad_sequence(
                article, seq_len=args.max_length
            )
            padded_sequences_all.append(padded_sequences)
            attention_masks_all.append(attention_masks)

        token_type_ids = [0] * args.max_length
        output = {
            "input_ids": padded_sequences_all,
            "token_type_ids": [token_type_ids] * len(padded_sequences_all),
            "attention_mask": attention_masks_all,
        }
        return output

    # load data (only has a single split)
    chem_data = datasets.load_dataset(DATASET, split="train")
    words_per_sample = [len(x[STRING_KEY].split(" ")) for x in chem_data]

    # process data
    tokenised_data = chem_data.map(
        tokenise,
        batched=True,
        remove_columns=chem_data.column_names,
        num_proc=os.cpu_count(),
        load_from_cache_file=False,
    )
    print(
        f"""
        Found {chem_data.num_rows} raw text samples.
        Average words {round(sum(words_per_sample) / chem_data.num_rows,0)}.
        Max words {max(words_per_sample)}
        Min words {min(words_per_sample)}
        Properties {chem_data.column_names}

        Found {tokenised_data.num_rows} tokenised samples
        Context length {args.max_length}
        Tokens {round(args.max_length * tokenised_data.num_rows / 1e9, 4)}B
        """
    )

    # save to disk
    tokenised_data.save_to_disk(f"{OUT_DIR}/{args.model_name}/{DATASET}")
