import ast

import pandas as pd


def process():
    df = pd.read_json(
        "https://huggingface.co/datasets/kjappelbaum/chemnlp-qmof-data/resolve/main/qmof_data.json"
    )

    df.dropna(
        subset=[
            "outputs.pbe.bandgap",
            "outputs.pbe.cbm",
            "outputs.pbe.vbm",
            "outputs.hle17.bandgap",
            "outputs.hle17.cbm",
            "outputs.hle17.vbm",
            "outputs.hse06.bandgap",
            "outputs.hse06.cbm",
            "outputs.hse06.vbm",
            "info.pld",
            "info.lcd",
            "info.density",
            "info.mofid.mofid",
            "info.mofid.smiles_nodes",
            "info.mofid.smiles_linkers",
            "info.mofid.topology",
            "info.symmetry.spacegroup_number",
        ],
        inplace=True,
    )

    df["info.mofid.smiles_nodes"] = df["info.mofid.smiles_nodes"].apply(
        lambda x: ", ".join(ast.literal_eval(x))
    )

    df["info.mofid.smiles_linkers"] = df["info.mofid.smiles_linkers"].apply(
        lambda x: ", ".join(ast.literal_eval(x))
    )

    print(len(df))

    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
