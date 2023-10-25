from datasets import load_dataset


def process():
    dataset = load_dataset("kjappelbaum/chemnlp-ord")
    df = dataset["train"].to_pandas()
    print(df.head())


if __name__ == "__main__":
    process()
