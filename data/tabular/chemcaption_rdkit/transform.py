import fire
import pandas as pd


def process():
    df = pd.read_parquet(
        "https://huggingface.co/datasets/kjappelbaum/chemnlp-chem-caption/resolve/main/rdkit_feat/train-00000-of-00001-27355e7935aa33a9.parquet"  # noqa
    )
    print(df.columns)
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    fire.Fire(process)
