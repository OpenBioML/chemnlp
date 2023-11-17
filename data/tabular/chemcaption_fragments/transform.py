import pandas as pd


def process():
    df = pd.read_parquet(
        "https://huggingface.co/datasets/kjappelbaum/chemnlp-chem-caption/resolve/main/smarts/train-00000-of-00001-71cef18c6383b463.parquet?download=true"  # noqa
    )
    df.dropna(inplace=True)
    print(len(df))
    df["fragment"] = df["completion_labels"].str.replace("_count", "")
    df["presence"] = df["completion"] > 0
    df["molecule"] = df["representation"]
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
