from datasets import load_dataset

from chemnlp.data.convert import is_longer_than_allowed


def process():
    dataset = load_dataset("kjappelbaum/chemnlp-qm9-file-translation")
    df = dataset["train"].to_pandas()
    df.replace(to_replace="RDKit", value="ChemNLP", inplace=True)
    df["is_longer_than_allowed"] = df["mol2000"].apply(is_longer_than_allowed)
    df = df[~df["is_longer_than_allowed"]]
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
