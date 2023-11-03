import pandas as pd

FILENAME = "uniprot_organisms"


def load_dataset() -> pd.DataFrame:
    uniprot = pd.read_csv(
        f"https://huggingface.co/datasets/chemNLP/uniprot/resolve/main/{FILENAME}/data_clean.csv"  # noqa: E501
    )
    uniprot.rename(columns={"sequence": "other"}, inplace=True)
    uniprot.to_csv("data_clean.csv", index=False)
    print(f"Successfully loaded {FILENAME}!")
    return uniprot


if __name__ == "__main__":
    load_dataset()
