import pandas as pd


def process():
    # get the smarts config
    df = pd.read_parquet(
        "https://huggingface.co/datasets/kjappelbaum/chemnlp-chem-caption/resolve/main/smarts/train-00000-of-00001-71cef18c6383b463.parquet"  # noqa
    )
    df["completion_labels"] = df["completion_labels"].astype(str)
    df["completion_labels"] = df["completion_labels"].str.replace(
        "_count", "", regex=True
    )
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
