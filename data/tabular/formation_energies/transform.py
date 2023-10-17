import pandas as pd


def process():
    df = pd.read_json(
        "https://raw.githubusercontent.com/CJBartel/TestStabilityML/master/mlstabilitytest/mp_data/data/hullout.json"
    )

    df = df.T.reset_index().rename(columns={"index": "composition"})
    df["rxn"] = df["rxn"].str.replace("_", " ")
    df.dropna(subset=["rxn", "Ef", "Ed"], inplace=True)
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
