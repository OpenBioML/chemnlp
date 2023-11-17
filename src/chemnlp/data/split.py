from collections import defaultdict
from random import Random
from typing import Dict, Iterable, List

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


def create_scaffold_split(
    df: pd.DataFrame, seed: int, frac: List[float], entity: str = "SMILES"
):
    """create scaffold split. it first generates molecular scaffold for each molecule
    and then split based on scaffolds
    adapted from: https://github.com/mims-harvard/TDC/tdc/utils/split.py

    Args:
        df (pd.DataFrame): dataset dataframe
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions
        entity (str): the column name for where molecule stores

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test
        and values correspond to each dataframe
    """
    return _create_scaffold_split(df[entity], seed, frac)


def _create_scaffold_split(
    smiles: Iterable[str], seed: int, frac: List[float]
) -> Dict[str, pd.DataFrame]:
    """create scaffold split. it first generates molecular scaffold for each molecule
    and then split based on scaffolds
    adapted from: https://github.com/mims-harvard/TDC/tdc/utils/split.py

    Args:
        smiles (Iterable[str]): dataset smiles
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions

    Returns:
        dict: a dictionary of indices for splitted data, where keys are train/valid/test
    """
    random = Random(seed)

    s = smiles
    scaffolds = defaultdict(set)

    error_smiles = 0
    for i, smiles in tqdm(enumerate(s), total=len(s)):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(smiles), includeChirality=False
            )
            scaffolds[scaffold].add(i)
        except Exception:
            print(smiles + " returns RDKit error and is thus omitted...")
            error_smiles += 1

    train, val, test = [], [], []
    train_size = int((len(s) - error_smiles) * frac[0])
    val_size = int((len(s) - error_smiles) * frac[1])
    test_size = (len(s) - error_smiles) - train_size - val_size
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
        "train": train,
        "valid": val,
        "test": test,
    }
