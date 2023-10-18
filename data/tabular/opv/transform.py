from datasets import load_dataset


def process():
    dataset = load_dataset("kjappelbaum/chemnlp-opv")["train"]
    df = dataset.to_pandas()
    df["LUMO"] = -1 * df["-LUMO (eV)"]
    df["HOMO"] = -1 * df["-HOMO (eV)"]
    df.rename(
        columns={
            "PCE_ave(%)": "PCE_ave",
            "Voc (V)": "Voc",
            "Jsc (mA cm^2)": "Jsc",
            "Mw (kg mol^-1)": "Mw",
            "Mn (kg mol^-1)": "Mn",
            "PDI (=Mw/Mn)": "PDI",
        },
        inplace=True,
    )

    df = df.dropna(
        subset=[
            "HOMO",
            "LUMO",
            "Mw",
            "PDI",
            "FF",
            "Jsc",
            "Voc",
            "PCE_ave",
            "bandgap(eV)",
        ]
    )
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
