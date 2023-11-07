import pandas as pd


def extract_data():
    aminoacids = pd.read_csv(
        "https://huggingface.co/datasets/chemNLP/uniprot/raw/main/aminoacid_seq.csv"
    )
    aminoacids.to_csv("data_clean.csv", index=False)
    return aminoacids


if __name__ == "__main__":
    extract_data()
