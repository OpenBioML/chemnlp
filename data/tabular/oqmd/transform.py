from datasets import load_dataset


def process():
    dataset = load_dataset("kjappelbaum/chemnlp-oqmd")["train"]
    df = dataset.to_pandas()

    df.dropna(
        subset=[
            "name",
            "formula",
            "spacegroup",
            "nelements",
            "nsites",
            "energy_per_atom",
            "formation_energy_per_atom",
            "band_gap",
            "volume_per_atom",
            "magnetization_per_atom",
            "atomic_volume_per_atom",
        ],
        inplace=True,
    )
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
