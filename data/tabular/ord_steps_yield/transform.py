import pandas as pd
from huggingface_hub import hf_hub_download


def process():
    file = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-ord",
        filename="ord_data_compiled.json",
        repo_type="dataset",
    )
    df = pd.read_json(file)
    df = df.dropna(subset=["non_yield_steps_string", "yield"])
    df = df.query("non_yield_steps_string != 'None'")
    df = df[["non_yield_steps_string", "yield"]]
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
