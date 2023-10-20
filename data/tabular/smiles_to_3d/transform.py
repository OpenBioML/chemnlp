from datasets import load_dataset


def process():
    dataset = load_dataset("kjappelbaum/chemnlp-qm9-file-translation")
    df = dataset["train"].to_pandas()
    df.replace(to_replace="RDKit", value="ChemNLP", inplace=True)
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()