import pandas as pd


def transform_data():
    df = pd.read_csv(
        "https://huggingface.co/datasets/AdrianM0/MUV/raw/main/MUV_733/data_clean.csv"
    )
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    transform_data()
