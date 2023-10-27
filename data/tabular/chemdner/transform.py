from datasets import load_dataset
from chemnlp.data.utils import oxford_comma_join


def process():
    dataset = load_dataset("kjappelbaum/chemnlp-chemdner")
    df = dataset["train"].to_pandas()

    matched_words = []
    for ent in df["entities"]:
        if len(ent) == 0:
            matched_words.append("no match")
        else:
            matched_words.append(oxford_comma_join(ent))

    df["matched_words"] = matched_words
    df["sentence"] = df["text"]

    print(len(df))

    df = df[["sentence", "matched_words"]]
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
