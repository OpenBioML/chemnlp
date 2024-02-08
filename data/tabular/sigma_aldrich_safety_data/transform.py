import pandas as pd
from huggingface_hub import hf_hub_download


def transform_data():
    file = hf_hub_download(
        repo_id="chemNLP/msds_sigma_aldrich",
        filename="msds.csv",
        repo_type="dataset",
    )

    df = pd.read_csv(file)
    df = df.drop(columns=["h_statements"])
    df.to_csv("data_clean.csv")


if __name__ == "__main__":
    transform_data()
