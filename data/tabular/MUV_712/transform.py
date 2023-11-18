import pandas as pd
from huggingface_hub import hf_hub_download


def transform_data():
    file = hf_hub_download(
        repo_id="chemNLP/MUV", filename="MUV_712/data_clean.csv", repo_type="dataset"
    )
    df = pd.read_csv(file)
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    transform_data()
