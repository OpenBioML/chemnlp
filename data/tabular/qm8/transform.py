import pandas as pd


def process():
    df = pd.read_json(
        "https://huggingface.co/datasets/kjappelbaum/chemnlp-qm8/resolve/main/qm8.json"
    )

    df = df.replace("RDKit", "ChemNLP", regex=True)
    df.dropna(inplace=True)
    df = df.rename(columns={"smiles": "SMILES"})
    df = df.query("is_longer_than_allowed==False")
    columns = [
        "E1-CC2",
        "E2-CC2",
        "f1-CC2",
        "f2-CC2",
        "E1-PBE0",
        "E2-PBE0",
        "f1-PBE0",
        "f2-PBE0",
        "E1-CAM",
        "E2-CAM",
        "f1-CAM",
        "f2-CAM",
    ]
    # filter out rows in which one of the columns is not a float. Filter explicitly for the row in which
    # the values for all those columns are floats.
    df = df[df[columns].apply(lambda x: x.apply(lambda y: isinstance(y, float))).all(1)]
    df[columns] = df[columns].astype(float)
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
