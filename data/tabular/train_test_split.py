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


REPRESENTATIONS = ['SMILES',
                   'SELFIES',
                   'PSMILES',
                   'DeepSMILES',
                   'canonical SMILES',
                   'InChI',
                   'TUCAN',
                   'IUPAC name'
                   ]


def print_sys(s):
    """system print

    Args:
        s (str): the string to print
    """
    print(s, flush=True, file=sys.stderr)


def create_scaffold_split(df, seed, frac, entity):
    """create scaffold split. it first generates molecular scaffold for each molecule and then split based on scaffolds
    reference: https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaffold.py

    Args:
        df (pd.DataFrame): dataset dataframe
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


def combine_representations(csv_paths: List[str]):

    return


if __name__ == '__main__':
    
    data = pd.read_csv('ames_mutagenicity/data_clean.csv')
    
    paths_to_data = glob('*/data_clean.csv')[:5]

    REPRESENTATION_LIST = []
    
    for path in paths_to_data:
        df = pd.read_csv(path)
        for col in df.columns:
            if col in REPRESENTATIONS:
                REPRESENTATION_LIST.extend(df[col].to_list())
    
    REPR_DF = pd.DataFrame()
    REPR_DF["REPR"] = REPRESENTATION_LIST
    
        
    scaffold_split = create_scaffold_split(REPR_DF, seed=0, frac=[0.8, 0.1, 0.1], entity='REPR')
