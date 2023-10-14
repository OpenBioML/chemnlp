import tdc
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import sys
from tqdm import tqdm
from random import Random
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
from glob import glob

RDLogger.DisableLog("rdApp.*")

def print_sys(s):
    """system print

    Args:
        s (str): the string to print
    """
    print(s, flush=True, file=sys.stderr)


def create_scaffold_split(df, seed, frac, entity):
    """create scaffold split. it first generates molecular scaffold for each molecule and then split based on scaffolds
    adapted from: https://github.com/mims-harvard/TDC/tdc/utils/split.py

    Args:
        df (pd.DataFrame): dataset dataframex   
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions
        entity (str): the column name for where molecule stores

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    random = Random(seed)

    s = df[entity].values
    scaffolds = defaultdict(set)
    idx2mol = dict(zip(list(range(len(s))), s))

    error_smiles = 0
    for i, smiles in tqdm(enumerate(s), total=len(s)):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(smiles), includeChirality=False
            )
            scaffolds[scaffold].add(i)
        except:
            print_sys(smiles + " returns RDKit error and is thus omitted...")
            error_smiles += 1

    train, val, test = [], [], []
    train_size = int((len(df) - error_smiles) * frac[0])
    val_size = int((len(df) - error_smiles) * frac[1])
    test_size = (len(df) - error_smiles) - train_size - val_size
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # index_sets = sorted(list(scaffolds.values()), key=lambda i: len(i), reverse=True)
    index_sets = list(scaffolds.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    if frac[2] == 0:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            else:
                val += index_set
                val_scaffold_count += 1
    else:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1

    return {
        "train": df.iloc[train].reset_index(drop=True),
        "valid": df.iloc[val].reset_index(drop=True),
        "test": df.iloc[test].reset_index(drop=True),
    }


def rewrite_data_with_splits(csv_paths: List[str], train_test_df: pd.DataFrame):

    for path in csv_paths:
        read_dataset = pd.read_csv(path)
        try:
            read_dataset = read_dataset.drop("split", axis=1)
        except:
            print("No split column")
        
        col_to_merge = "SMILES"
        train_test_df = train_test_df.rename(columns={"REPR" : col_to_merge})
        merged_data = pd.merge(read_dataset, train_test_df, on=col_to_merge, how="left")
        merged_data = merged_data.dropna()        
        merged_data.to_csv(path, index=False)


if __name__ == '__main__':
        
    paths_to_data = glob('*/data_clean.csv')

    REPRESENTATION_LIST = []
    
    for path in paths_to_data:
        df = pd.read_csv(path)
        if "SMILES" in df.columns:
            REPRESENTATION_LIST.extend(df["SMILES"].to_list())
    
    REPR_DF = pd.DataFrame()
    REPR_DF["REPR"] = list(set(REPRESENTATION_LIST))
    
    scaffold_split = create_scaffold_split(REPR_DF, seed=42, frac=[0.8, 0, 0.2], entity='REPR')
    
    train_df = scaffold_split['train']
    train_df["split"] = len(train_df) * ["train"]
    test_df = scaffold_split['test']
    test_df["split"] = len(test_df) * ["test"]
    
    merge = pd.concat([train_df, test_df], axis=0)

    rewrite_data_with_splits(paths_to_data, merge)