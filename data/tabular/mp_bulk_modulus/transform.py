from datasets import load_dataset


def transform():
    dataset = load_dataset("kjappelbaum/chemnlp-mp-bulk-modulus")["train"]
    df = dataset.to_pandas()
    print(len(df))
    df[["formula", "bulk_modulus", "split"]].to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    transform()
