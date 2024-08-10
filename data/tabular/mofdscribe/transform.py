import pandas as pd
from huggingface_hub import hf_hub_download

from chemnlp.data.convert import remove_composition_rows


def process():
    file = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-text-mofdscribe",
        filename="data/train-00000-of-00001-ccae794e6d461778.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(file)
    print(len(df))
    df["cif"] = df["cif"].apply(remove_composition_rows)
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
