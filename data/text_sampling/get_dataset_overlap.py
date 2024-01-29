import glob
import pandas as pd

skip_ds = [
        "rdkit_features",
        "iupac_smiles",
        "orbnet_denali",
        "qmof_gcmc",
        "qmof_quantum",
        "zinc",
        ]

if __name__ == "__main__":
    path_base = __file__.replace("text_sampling/get_dataset_overlap.py", "")
    fns = sorted(glob.glob(path_base + "tabular/**/data_clean.csv"))
    for i in range(len(fns)):
        for j in range(i + 1, len(fns)):
            fn1 = fns[i]
            fn2 = fns[j]
            ds1 = fn1.split("/")[-2]
            ds2 = fn2.split("/")[-2]
            if (ds1 in skip_ds) or (ds2 in skip_ds):
                continue
            df1 = pd.read_csv(fn1, index_col=False, low_memory=False, nrows=0)  # only get columns
            df2 = pd.read_csv(fn2, index_col=False, low_memory=False, nrows=0)  # only get columns
            if ("SMILES" in df1.columns) and ("SMILES" in df2.columns):
                df1 = pd.read_csv(fn1, index_col=False, low_memory=False, usecols=["SMILES"])
                df2 = pd.read_csv(fn2, index_col=False, low_memory=False, usecols=["SMILES"])
                print(fn1.split("/")[-2], fn2.split("/")[-2], len(set(df1.SMILES) & set(df2.SMILES)))
