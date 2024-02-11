import pandas as pd
from huggingface_hub import hf_hub_download


def process():
    file = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-ld50catmos",
        filename="cleaned_ld50.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(file)
    print(len(df))
    df[
        [
            "num_ghose_violations",
            "num_lead_likeness_violations",
            "num_lipinski_violations",
            "num_carbon_atoms",
            "num_oxygen_atoms",
        ]
    ] = df[
        [
            "num_ghose_violations",
            "num_lead_likeness_violations",
            "num_lipinski_violations",
            "num_carbon_atoms",
            "num_oxygen_atoms",
        ]
    ].astype(
        int
    )
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
