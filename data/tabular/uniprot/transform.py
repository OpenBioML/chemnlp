import pandas as pd


# https://huggingface.co/datasets/chemNLP/uniprot/resolve/main/reactions_sentences.csv
def load_dataset() -> pd.DataFrame:
    uniprot = pd.read_csv("reactions_sentences.csv")
    uniprot.to_csv("data_clean.csv", index=False)
    return uniprot


if __name__ == "__main__":
    print(len(load_dataset()))
