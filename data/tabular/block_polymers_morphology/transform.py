import pandas as pd

columns_to_keep = ["phase1", "T", "BigSMILES", "Mn", "f1", "Mw", "D"]


def process():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/olsenlabmit/BCDB/main/data/diblock.csv"
    )
    df = df[df["phase2"].isna()]  # remove multiple phases
    mw_clean = []
    dispersity_clean = []

    for mw, dispersity in zip(df["Mw"], df["D"]):
        # if nan, make empty string
        # else, add the units
        if pd.isna(mw) or "nan" in str(mw):
            mw_clean.append("REPLACENULL")
        else:
            mw_clean.append(f", average molecular mass of {mw:.1f} g/mol")

        if pd.isna(dispersity) or "nan" in str(dispersity):
            # empty character that will still appear in the csv
            dispersity_clean.append("REPLACENULL")
        else:
            dispersity_clean.append(f", and dispersity of {dispersity:.1f}")

    df["Mw"] = mw_clean
    df["D"] = dispersity_clean
    df.dropna(subset=columns_to_keep, inplace=True)
    print(len(df))
    df[columns_to_keep].to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
