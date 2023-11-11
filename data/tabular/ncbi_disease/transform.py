import pandas as pd
from datasets import load_dataset

from chemnlp.data.ner import group_tokens_by_labels, punctuation_joiner
from chemnlp.data.utils import oxford_comma_join


def process():
    # tokenized at whitespaces and punctuations
    dataset = load_dataset("bigbio/blurb", "ncbi_disease")
    dfs = []
    for split in ["train", "validation", "test"]:
        df_ = dataset[split].to_pandas()
        df_["split"] = split
        dfs.append(df_)
    df = pd.concat(dfs)
    ner_labels = df["ner_tags"]

    matched_words = []
    for tokens, ner_label in zip(df["tokens"], ner_labels):
        words = group_tokens_by_labels(tokens, ner_label)
        if len(words) == 0:
            matched_words.append("no match")
        else:
            matched_words.append(oxford_comma_join(words))

    df["matched_words"] = matched_words
    df["sentence"] = df["tokens"].apply(punctuation_joiner)

    df = df[["sentence", "matched_words"]]

    # ensure we have at least 5 words in a sentence
    df = df[df["sentence"].apply(lambda x: len(x.split()) >= 5)]

    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
