import pandas as pd
import regex as re

FILENAME = "uniprot_sentences"


def remove_text_from_column(sentence: str) -> str:
    # Replace "(By similarity)" with empty string and remove extra spaces
    updated_text = re.sub(r"\s*\(By similarity\)", "", sentence)
    return updated_text


def load_dataset() -> pd.DataFrame:
    uniprot = pd.read_csv(
        f"https://huggingface.co/datasets/chemNLP/uniprot/resolve/main/{FILENAME}/data_clean.csv"  # noqa: E501
    )

    uniprot.rename(columns={"sequence": "other"}, inplace=True)
    uniprot["sentences"] = uniprot["sentences"].apply(remove_text_from_column)
    uniprot.to_csv("data_clean.csv", index=False)
    print(f"Successfully loaded {FILENAME}!")
    return uniprot


if __name__ == "__main__":
    load_dataset()
