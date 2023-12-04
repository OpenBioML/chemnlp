import pandas as pd
from huggingface_hub import hf_hub_download


def extract_data():
    file = hf_hub_download(
        repo_id="chemNLP/uniprot", filename="aminoacid_seq.csv", repo_type="dataset"
    )
    aminoacids = pd.read_csv(file)
    aminoacids.to_csv("data_clean.csv", index=False)
    return aminoacids


if __name__ == "__main__":
    extract_data()
