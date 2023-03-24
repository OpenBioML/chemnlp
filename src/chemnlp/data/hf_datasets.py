from datasets import load_dataset


def boolq(tokenizer):
    dataset = load_dataset("boolq")

    def _tokenize_function(example, tokenizer):
        all_text = f"Passage:\n{example['passage']} \nQuestion:\n{example['question']}\nAnswer:\n{example['answer']}"
        return tokenizer(all_text)

    tokenized = dataset.map(
        _tokenize_function,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["question", "answer", "passage"],
    )

    return tokenized["train"], tokenized["validation"]


def rotten_tomatoes(tokenizer):
    dataset = load_dataset("rotten_tomatoes")

    def _tokenize_function(example, tokenizer):
        return tokenizer(example["text"])

    tokenized = dataset.map(
        _tokenize_function,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["text", "label"],
    )

    return tokenized["train"], tokenized["validation"]
