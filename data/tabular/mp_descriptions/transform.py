from datasets import load_dataset

from chemnlp.data.convert import remove_composition_rows


def process():
    dataset = load_dataset("kjappelbaum/chemnlp-robocrys")
    df = dataset["train"].to_pandas()
    df.dropna(
        subset=["cifstr", "description", "description_w_bondlengths"], inplace=True
    )
    df["cifstr"] = df["cifstr"].apply(remove_composition_rows)
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
