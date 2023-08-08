import glob
import os

import pandas as pd

if __name__ == "__main__":
    path_base = __file__.replace("text_sampling/extend_tabular_processed.py", "")
    path_data_dir = sorted(glob.glob(path_base + "tabular/**/data_clean.csv"))
    path_processed_smiles = path_base + "text_sampling/extend_tabular_processed.csv"

    cols = [
        "SMILES",
        "selfies",
        "deepsmiles",
        "canonical",
        "inchi",
        "iupac_name",
    ]

    if not os.path.isfile(path_processed_smiles):
        pd.DataFrame({c: [] for c in cols}).to_csv(path_processed_smiles, index=False)
        print("Created empty extend_tabular_processed.csv file.")

    for path in path_data_dir:
        print(f"\n###### {path}")

        if not os.path.isfile(path):
            print("No data_clean.csv file in the dataset directory.")
            continue

        # check cols are there
        df = pd.read_csv(path, index_col=False, nrows=0)  # only get columns
        if set(cols).issubset(df.columns):
            df = pd.read_csv(path, low_memory=False)
            df_append = df[cols].copy()
            df_append.to_csv(path_processed_smiles, mode="a", header=False, index=False)
            print("Added processed SMILES to extend_tabular_processed.csv file.")
        else:
            print("No processed columns in the extend_tabular_processed.csv file.")

    # deduplicate processed entries
    df_processed = pd.read_csv(path_processed_smiles)
    df_processed.drop_duplicates(inplace=True)
    df_processed.to_csv(path_processed_smiles, index=False)
    print("Deduplicated extend_tabular_processed.csv file.")
