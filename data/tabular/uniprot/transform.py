import pandas as pd


def load_dataset() -> pd.DataFrame:
    uniprot = pd.read_csv(
        "https://huggingface.co/datasets/chemNLP/uniprot/resolve/main/reactions_sentences_domains_organisms_binding_sites.csv" # noqa
    )
    uniprot.rename(columns={"sequence": "other"}, inplace=True)
    uniprot.to_csv("data_clean.csv", index=False)
    print("Successfully loaded dataset!")
    return uniprot


if __name__ == "__main__":
    load_dataset()
