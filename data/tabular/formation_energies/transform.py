import pandas as pd


def process():
    df = pd.read_json(
        "https://raw.githubusercontent.com/CJBartel/TestStabilityML/master/mlstabilitytest/mp_data/data/hullout.json"
    )

    rows = []

    df = df.T.reset_index().rename({"index": "composition"})
    df["rxn"] = df["rxn"].apply(lambda x: x.replace("_ ", ""))
    