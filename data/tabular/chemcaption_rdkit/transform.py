import fire
import pandas as pd


def process():
    df = pd.read_parquet(
        "https://huggingface.co/datasets/kjappelbaum/chemnlp-chem-caption/resolve/main/rdkit_feat/train-00000-of-00001-7cea16ab26bf74cf.parquet?download=true"  # noqa
    )
    df["num_bonds_simple"] = df[
        [
            "num_single_bonds",
            "num_double_bonds",
            "num_triple_bonds",
            "num_quadruple_bonds",
            "num_quintuple_bonds",
            "num_aromatic_bonds",
        ]
    ].sum(axis=1)

    df = df[df["num_bonds_simple"].astype(int) == df["num_bonds"].astype(int)]

    df[
        [
            "num_valence_electrons",
            "num_single_bonds",
            "num_double_bonds",
            "num_triple_bonds",
            "num_quadruple_bonds",
            "num_quintuple_bonds",
            "num_aromatic_bonds",
            "num_bonds",
            "num_carbon_atoms",
            "num_hydrogen_atoms",
            "num_nitrogen_atoms",
            "num_oxygen_atoms",
            "num_hydrogen_bond_acceptors",
            "num_hydrogen_bond_donors",
            "num_lipinski_violations",
            "num_chiral_centers",
        ]
    ] = df[
        [
            "num_valence_electrons",
            "num_single_bonds",
            "num_double_bonds",
            "num_triple_bonds",
            "num_quadruple_bonds",
            "num_quintuple_bonds",
            "num_aromatic_bonds",
            "num_bonds",
            "num_carbon_atoms",
            "num_hydrogen_atoms",
            "num_nitrogen_atoms",
            "num_oxygen_atoms",
            "num_hydrogen_bond_acceptors",
            "num_hydrogen_bond_donors",
            "num_lipinski_violations",
            "num_chiral_centers",
        ]
    ].astype(
        int
    )
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    fire.Fire(process)
