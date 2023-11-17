import pandas as pd


def transform_data():
    df = pd.read_csv(
        "https://huggingface.co/datasets/chemNLP/MUV/raw/main/MUV_846/data_clean.csv"
    )
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    transform_data()
