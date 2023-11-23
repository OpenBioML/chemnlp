import pandas as pd


def remove_composition_from_cif(cif):
    # in the second line of cif, split at _ and then take the first element and join it with "cif"
    parts = cif.split("\n")
    parts[1] = "data_cif"
    return "\n".join(parts)


def process():
    df = pd.read_json(
        "https://huggingface.co/datasets/kjappelbaum/chemnlp-mp-cifs/resolve/main/mpid.json"
    )
    df = df.query("is_longer_than_allowed==False").dropna()
    df["cif"] = df["cif"].apply(remove_composition_from_cif)
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
