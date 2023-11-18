import pandas as pd
from huggingface_hub import hf_hub_download

DATA = "uniprot_organisms"


def load_dataset() -> pd.DataFrame:
    uniprot = hf_hub_download(
        repo_id="chemnlp/uniprot",
        filename=f"{DATA}/data_clean.csv",
        repo_type="dataset",
    )
    uniprot = pd.read_csv(uniprot)
    uniprot.rename(columns={"sequence": "other"}, inplace=True)
    uniprot.drop_duplicates(
        inplace=True,
    )
    print(f"Successfully loaded {DATA}! {len(uniprot)} rows")
    uniprot.to_csv("data_clean.csv", index=False)
    print(f"Successfully loaded {DATA}!")
    return uniprot


if __name__ == "__main__":
    load_dataset()
