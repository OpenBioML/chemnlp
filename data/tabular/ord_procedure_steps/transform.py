import pandas as pd
from huggingface_hub import hf_hub_download


def process():
    file = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-ord",
        filename="ord_data_compiled.json",
        repo_type="dataset",
    )
    df = pd.read_json(file)
    df = df.dropna(subset=["steps_string", "procedure"])
    df.query("steps_string != 'None'", inplace=True)
    df.query("procedure != 'None'", inplace=True)
    df = df[["steps_string", "procedure"]]
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
