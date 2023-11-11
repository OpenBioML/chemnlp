from datasets import load_dataset


def process():
    dataset = load_dataset("kjappelbaum/chemnlp-robocrys")
    df = dataset["train"].to_pandas()
    df.dropna(
        subset=["cifstr", "description", "description_w_bondlengths"], inplace=True
    )
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
