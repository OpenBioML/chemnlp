from datasets import load_dataset


def process():
    dataset = load_dataset("kjappelbaum/chemnlp-orbnet-denali")
    df = dataset["train"].to_pandas()
    df = df.dropna()
    print(len(df))
    df.rename(columns={"smiles": "SMILES"}, inplace=True)
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
