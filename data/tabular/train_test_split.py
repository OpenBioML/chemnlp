"""Perform train/test split on all tabular and knowledge graph datasets.

First, a scaffold split is run on selected SMILES datasets, than a random split is run on all other datasets.
For this second step, we ensure if there are SMILES in the dataset, that the there is no SMILES
that is in the validation or test set of the scaffold split that is in the training set of the random split.

If there are multiple SMILES columns, we move to test/val if any of the SMILES
is in the test/val set of the scaffold split.

The meta.yaml files are used to determine the if there are SMILES in the dataset.
For this reason, certain scripts (e.g. preprocess_kg.py, several transform.py) need to be run before this script
as they sometimes update or create the meta.yaml files.

Prior to the SMILES split and the split of all other files, we perform a random split on the files with
amino acid sequences.

Warning:
    - Note that the logic assumes that the SMILES columns only contain valid SMILES.
    - The current script does not set up a dask client. If distributed computing is needed, please set up.
"""
import os
import random
import subprocess
from functools import partial
from glob import glob
from pathlib import Path
from typing import List, Literal, Union

import dask.array as da
import dask.dataframe as dd
import fire
import numpy as np
import pandas as pd
import yaml
from pandarallel import pandarallel
from tqdm import tqdm

from chemnlp.data.split import _create_scaffold_split

pandarallel.initialize(progress_bar=True)

# see issue https://github.com/OpenBioML/chemnlp/issues/498 for the list of datasets
# mostly those from TDC/moleculenet that have scaffold split as recommended method
to_scaffold_split = [
    "blood_brain_barrier_martins_et_al",
    "serine_threonine_kinase_33_butkiewicz",
    "ld50_zhu",
    "herg_blockers",
    "clearance_astrazeneca",
    "half_life_obach",
    "drug_induced_liver_injury",
    "kcnq2_potassium_channel_butkiewicz",
    "sarscov2_3clpro_diamond",
    "volume_of_distribution_at_steady_state_lombardo_et_al",
    "sr_hse_tox21",
    "nr_er_tox21",
    "cyp2c9_substrate_carbonmangels",
    "nr_aromatase_tox21",
    "cyp_p450_2d6_inhibition_veith_et_al",
    "cyp_p450_1a2_inhibition_veith_et_al",
    "cyp_p450_2c9_inhibition_veith_et_al",
    "m1_muscarinic_receptor_antagonists_butkiewicz",
    "nr_ar_tox21",
    "sr_atad5_tox21",
    "tyrosyl-dna_phosphodiesterase_butkiewicz",
    "cav3_t-type_calcium_channels_butkiewicz",
    "clintox",
    "sr_p53_tox21",
    "nr_er_lbd_tox21",
    "pampa_ncats",
    "sr_mmp_tox21",
    "caco2_wang",
    "sarscov2_vitro_touret",
    "choline_transporter_butkiewicz",
    "orexin1_receptor_butkiewicz",
    "human_intestinal_absorption",
    "nr_ahr_tox21",
    "cyp3a4_substrate_carbonmangels",
    "herg_karim_et_al",
    "hiv",
    "carcinogens",
    "sr_are_tox21",
    "nr_ppar_gamma_tox21",
    "solubility_aqsoldb",
    "m1_muscarinic_receptor_agonists_butkiewicz",
    "ames_mutagenicity",
    "potassium_ion_channel_kir2_1_butkiewicz",
    "cyp_p450_3a4_inhibition_veith_et_al",
    "skin_reaction",
    "cyp2d6_substrate_carbonmangels",
    "cyp_p450_2c19_inhibition_veith_et_al",
    "nr_ar_lbd_tox21",
    "p_glycoprotein_inhibition_broccatelli_et_al",
    "bioavailability_ma_et_al",
    "BACE",
    "lipophilicity",
    "thermosol",
    "MUV_466",
    "MUV_548",
    "MUV_600",
    "MUV_644",
    "MUV_652",
    "MUV_689",
    "MUV_692",
    "MUV_712",
    "MUV_713",
    "MUV_733",
    "MUV_737",
    "MUV_810",
    "MUV_832",
    "MUV_846",
    "MUV_852",
    "MUV_858",
    "MUV_859",
]


def split_for_smiles(
    smiles: str, train_smiles: List[str], val_smiles: List[str]
) -> str:
    """Returns the split for a SMILES based on the train and val smiles."""
    if smiles in train_smiles:
        return "train"
    elif smiles in val_smiles:
        return "valid"
    else:
        return "test"


def yaml_file_has_column_of_type(
    yaml_file: Union[str, Path], data_type: Literal["AS_SEQUENCE", "SMILES"]
) -> bool:
    """Returns True if the yaml file has a SMILES column.

    Those columns are in either the targets or identifiers keys
    (which are both lists). In there, we have dicts and it is a SMILES
    if the key type is SMILES.
    """
    with open(yaml_file, "r") as f:
        meta = yaml.safe_load(f)

    if "targets" in meta:
        for target in meta["targets"]:
            if target["type"] == data_type:
                return True
    if "identifiers" in meta:
        for identifier in meta["identifiers"]:
            if identifier["type"] == data_type:
                return True

    return False


def is_in_scaffold_split_list(yaml_file: Union[str, Path]) -> bool:
    """Returns True if the yaml file is in the to_scaffold_split list."""
    with open(yaml_file, "r") as f:
        meta = yaml.safe_load(f)

    return meta["name"] in to_scaffold_split


def get_meta_yaml_files(data_dir: Union[str, Path]) -> List[str]:
    """Returns all yaml files in the data_dir directory."""
    return glob(os.path.join(data_dir, "**", "**", "meta.yaml"), recursive=True)


def run_transform(file: Union[str, Path]) -> None:
    if not os.path.exists(os.path.join(os.path.dirname(file), "data_clean.csv")):
        subprocess.run(
            ["python", "transform.py"],
            cwd=os.path.dirname(file),
        )


def get_columns_of_type(
    yaml_file: Union[str, Path],
    column_type: Literal["SMILES", "AS_SEQUENCE"] = "SMILES",
) -> List[str]:
    """Returns the id for all columns with type SMILES"""
    with open(yaml_file, "r") as f:
        meta = yaml.safe_load(f)

    smiles_columns = []
    if "targets" in meta:
        for target in meta["targets"]:
            if target["type"] == column_type:
                smiles_columns.append(target["id"])
    if "identifiers" in meta:
        for identifier in meta["identifiers"]:
            if identifier["type"] == column_type:
                smiles_columns.append(identifier["id"])

    return smiles_columns


def remaining_split(
    data_dir,
    override: bool = False,
    seed: int = 42,
    debug: bool = False,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    run_transform_py: bool = False,
) -> None:
    """Run a random split on all datasets in the data_dir directory that do not have a AS_SEQUENCE column
    and do not have a SMILES column.

    Args:
        data_dir ([type]): [description]
        override (bool, optional): [description]. Defaults to False.
        seed (int, optional): [description]. Defaults to 42.
        debug (bool, optional): [description]. Defaults to False.
        train_frac (float, optional): [description]. Defaults to 0.7.
        val_frac (float, optional): [description]. Defaults to 0.15.
        test_frac (float, optional): [description]. Defaults to 0.15.
        run_transform_py (bool, optional): [description]. Defaults to False.

    Returns:
        None
    """
    # make deterministic
    np.random.seed(seed)
    random.seed(seed)
    dask_random_state = da.random.RandomState(seed)

    yaml_files = get_meta_yaml_files(data_dir)
    non_smiles_yaml_files = [
        file
        for file in yaml_files
        if not (
            yaml_file_has_column_of_type(file, "SMILES")
            or yaml_file_has_column_of_type(file, "AS_SEQUENCE")
        )
    ]

    # if we debug, we only run split on the first 5 datasets
    if debug:
        non_smiles_yaml_files = non_smiles_yaml_files[:5]

    def assign_random_split(ddf, train_frac, test_frac, dask_random_state):
        # Create a random array of the same length as the DataFrame
        index_size = ddf.index.size.compute()
        random_values = dask_random_state.random(index_size)

        # Determine the split based on random values and fractions
        ddf["split"] = da.where(
            random_values < train_frac,
            "train",
            da.where(random_values < train_frac + test_frac, "test", "valid"),
        )

        return ddf

    # we run random splitting for each dataset
    for file in tqdm(non_smiles_yaml_files):
        print(f"Processing {file}")
        if run_transform_py:
            run_transform(file)

        ddf = dd.read_csv(os.path.join(os.path.dirname(file), "data_clean.csv"))
        ddf = assign_random_split(ddf, train_frac, test_frac, dask_random_state)
        split_counts = ddf["split"].value_counts().compute()

        print(f"Dataset {file} has {len(ddf)} datapoints. Split sizes: {split_counts}")

        # we then write the new data_clean.csv file
        # if the override option is true, we write this new file to `data_clean.csv` in the same directory
        # otherwise, we copy the old `data_clean.csv` to `data_clean_old.csv`
        # and write the new file to `data_clean.csv`
        if not override:
            os.rename(
                os.path.join(os.path.dirname(file), "data_clean.csv"),
                os.path.join(os.path.dirname(file), "data_clean_old.csv"),
            )
        ddf.to_csv(
            os.path.join(os.path.dirname(file), "data_clean.csv"),
            index=False,
            single_file=True,
        )


def as_sequence_split(
    data_dir,
    override: bool = False,
    seed: int = 42,
    debug: bool = False,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    run_transform_py: bool = False,
) -> None:
    """Run a random split on all datasets in the data_dir directory that have a AS_SEQUENCE column.

    Ensures that, overall, a AS_SEQUENCE is not split across train/val/test.

    Args:
        data_dir ([type]): [description]
        override (bool, optional): [description]. Defaults to False.
        seed (int, optional): [description]. Defaults to 42.
        debug (bool, optional): [description]. Defaults to False.
        train_frac (float, optional): [description]. Defaults to 0.7.
        val_frac (float, optional): [description]. Defaults to 0.15.
        test_frac (float, optional): [description]. Defaults to 0.15.
        run_transform_py (bool, optional): [description]. Defaults to False.

    Returns:
        None
    """
    all_yaml_files = get_meta_yaml_files(data_dir)
    as_sequence_yaml_files = [
        file
        for file in all_yaml_files
        if yaml_file_has_column_of_type(file, "AS_SEQUENCE")
    ]

    # if we debug, we only run split on the first 5 datasets
    if debug:
        as_sequence_yaml_files = as_sequence_yaml_files[:5]

    # make deterministic
    np.random.seed(seed)
    random.seed(seed)

    all_as_sequence = set()

    for file in tqdm(as_sequence_yaml_files):
        print(f"Processing {file}")
        if run_transform_py:
            run_transform(file)

        df = pd.read_csv(
            os.path.join(os.path.dirname(file), "data_clean.csv"), low_memory=False
        )
        for as_seq_col in get_columns_of_type(file, "AS_SEQUENCE"):
            all_as_sequence.update(df[as_seq_col].tolist())

    all_as_sequence = list(all_as_sequence)
    # random split into train/val/test using numpy
    # randomly take train_frac of the data for training set, val_frac for validation set
    # and the rest for test set
    train_size = int(len(all_as_sequence) * train_frac)
    val_size = int(len(all_as_sequence) * val_frac)

    # shuffle the data
    np.random.shuffle(all_as_sequence)
    # split into train/val/test
    train = all_as_sequence[:train_size]
    val = all_as_sequence[train_size : train_size + val_size]
    test = all_as_sequence[train_size + val_size :]
    print(
        f"In total, there are {len(all_as_sequence)} AS_SEQUENCEs. Split sizes: {len(train)} train, {len(val)} valid, {len(test)} test."  # noqa: E501
    )

    with open("val_as_sequences.txt", "w") as f:
        f.write("\n".join(val))

    with open("test_as_sequences.txt", "w") as f:
        f.write("\n".join(test))

    def assign_split(ddf, as_sequence_columns, test_sequences, val_sequences):
        test_mask = ddf[as_sequence_columns].isin(test_sequences).any(axis=1)
        val_mask = ddf[as_sequence_columns].isin(val_sequences).any(axis=1)

        # Assign the 'split' based on the masks
        ddf["split"] = "train"  # Default assignment
        ddf["split"] = ddf["split"].mask(test_mask, "test")
        ddf["split"] = ddf["split"].mask(val_mask, "valid")
        return ddf

    for file in tqdm(as_sequence_yaml_files):
        print(f"Processing {file}")
        as_seq_cols = get_columns_of_type(file, "AS_SEQUENCE")
        ddf = dd.read_csv(os.path.join(os.path.dirname(file), "data_clean.csv"))

        ddf = assign_split(ddf, as_seq_cols, test, val)

        split_counts = ddf["split"].value_counts().compute()

        print(f"Dataset {file} has {len(ddf)} datapoints. Split sizes: {split_counts}")

        # we then write the new data_clean.csv file
        # if the override option is true, we write this new file to `data_clean.csv` in the same directory
        # otherwise, we copy the old `data_clean.csv` to `data_clean_old.csv`
        # and write the new file to `data_clean.csv`
        if not override:
            os.rename(
                os.path.join(os.path.dirname(file), "data_clean.csv"),
                os.path.join(os.path.dirname(file), "data_clean_old.csv"),
            )
        ddf.to_csv(
            os.path.join(os.path.dirname(file), "data_clean.csv"),
            index=False,
            single_file=True,
        )


def smiles_split(
    data_dir: Union[str, Path],
    override: bool = False,
    seed: int = 42,
    debug: bool = False,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    val_smiles_path: Union[str, Path] = "val_smiles.txt",
    test_smiles_path: Union[str, Path] = "test_smiles.txt",
    run_transform_py: bool = False,
) -> None:
    """Runs a random split on all datasets in the data_dir directory that have a SMILES column.

    The validation and test sets are ensured to not contain any SMILES that are in the validation and test sets
    of the scaffold split.

    Args:
        data_dir (Union[str, Path]): The directory containing the directories with the datasets.
        override (bool, optional): Whether to override the existing data_clean.csv files with the new ones.
            Defaults to False. In this case, the old data_clean.csv files will be renamed to data_clean_old.csv.
        seed (int, optional): The random seed. Defaults to 42.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
            In this case, only the first 5 datasets will be processed.
        train_frac (float, optional): The fraction of the data to be used for training. Defaults to 0.7.
        val_frac (float, optional): The fraction of the data to be used for validation. Defaults to 0.15.
        test_frac (float, optional): The fraction of the data to be used for testing. Defaults to 0.15.
        val_smiles_path (Union[str, Path], optional): The path to the validation smiles file.
            Defaults to "val_smiles.txt".
        test_smiles_path (Union[str, Path], optional): The path to the test smiles file.
            Defaults to "test_smiles.txt".
        run_transform_py (bool, optional): Whether to run the transform.py script in the directory of the dataset.
            Defaults to False.

    Returns:
        None
    """
    # make deterministic
    np.random.seed(seed)
    random.seed(seed)
    dask_random_state = da.random.RandomState(seed)

    def assign_splits(
        ddf, smiles_columns, test_smiles, val_smiles, train_frac, test_frac, seed
    ):
        # Create masks for test and validation based on SMILES columns
        test_mask = da.isin(ddf[smiles_columns].values, test_smiles).any(axis=1).compute()
        val_mask = da.isin(ddf[smiles_columns].values, val_smiles).any(axis=1).compute()

        # Generate random splits for the rest
        index_size = ddf.index.size.compute()
        random_values = dask_random_state.random(index_size)
        train_mask = (random_values < train_frac) & ~test_mask & ~val_mask
        test_mask = (
            (random_values >= train_frac)
            & (random_values < train_frac + test_frac)
            & ~val_mask
        )

        # Combine masks into a single column
        ddf["split"] = da.where(
            test_mask,
            "test",
            da.where(val_mask, "valid", da.where(train_mask, "train", "valid")),
        )

        return ddf

    # we err toward doing more I/O but having simpler code to ensure we don't make anything stupid
    all_yaml_files = get_meta_yaml_files(data_dir)
    smiles_yaml_files = [
        file
        for file in all_yaml_files
        if (
            yaml_file_has_column_of_type(file, "SMILES")
            and not yaml_file_has_column_of_type(file, "AS_SEQUENCE")
        )  # noqa: E501
    ]
    # we filter those out that are in the to_scaffold_split list
    not_scaffold_split_yaml_files = [
        file for file in smiles_yaml_files if not is_in_scaffold_split_list(file)
    ]

    # if we debug, we only run split on the first 5 datasets
    if debug:
        not_scaffold_split_yaml_files = not_scaffold_split_yaml_files[:5]

    # we run random splitting for each dataset but ensure that the validation and test sets
    # do not contain any SMILES that are in the validation and test sets of the scaffold split

    with open(val_smiles_path, "r") as f:
        val_smiles = f.read().splitlines()

    with open(test_smiles_path, "r") as f:
        test_smiles = f.read().splitlines()

    # some datasets are hundreds of GB, so we have to use dask
    # we will first do a random split on the whole dataset
    # then we flip the values based on whether a SMILES
    # is in the validation or test set of the scaffold split
    # we will then write the new data_clean.csv file

    # if the is no data_clean.csv file, we run the transform.py script
    # in the directory of the yaml file
    # we will then read the data_clean.csv file and do the split
    for file in tqdm(not_scaffold_split_yaml_files):
        print(f"Processing {file}")
        if run_transform_py:
            run_transform(file)
        ddf = dd.read_csv(os.path.join(os.path.dirname(file), "data_clean.csv"))
        smiles_columns = get_columns_of_type(file, "SMILES")

        # Assign splits using the revised function
        ddf = assign_splits(
            ddf, smiles_columns, test_smiles, val_smiles, train_frac, test_frac, seed
        )
        split_counts = ddf["split"].value_counts().compute()

        print(
            f"Dataset {file} has {len(ddf)} datapoints. Split sizes: {split_counts['train']} train, {split_counts['valid']} valid, {split_counts['test']} test."  # noqa: E501
        )

        # we then write the new data_clean.csv file
        # if the override option is true, we write this new file to `data_clean.csv` in the same directory
        # otherwise, we copy the old `data_clean.csv` to `data_clean_old.csv`
        # and write the new file to `data_clean.csv`
        if not override:
            os.rename(
                os.path.join(os.path.dirname(file), "data_clean.csv"),
                os.path.join(os.path.dirname(file), "data_clean_old.csv"),
            )
        ddf.to_csv(
            os.path.join(os.path.dirname(file), "data_clean.csv"),
            index=False,
            single_file=True,
        )


def scaffold_split(
    data_dir: Union[str, Path],
    override: bool = False,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    debug: bool = False,
    run_transform_py: bool = False,
):
    """Performs scaffold splitting on all datasets in the data_dir directory
    that are in the to_scaffold_split list.

    It serializes the validation and test smiles into `val_smiles.txt` and `test_smiles.txt`
    in the current directory.

    It is the first step of the splitting procedure as we will then perform a random split
    on the remaining datasets on all other datasets. However, if they have smiles,
    we will set all smiles that are in the validation and test sets to be in the validation and test sets.

    Args:
        data_dir (Union[str, Path]): The directory containing the directories with the datasets.
        override (bool, optional): Whether to override the existing data_clean.csv files with the new ones.
            Defaults to False. In this case, the old data_clean.csv files will be renamed to data_clean_old.csv.
        train_frac (float, optional): The fraction of the data to be used for training. Defaults to 0.7.
        val_frac (float, optional): The fraction of the data to be used for validation. Defaults to 0.15.
        test_frac (float, optional): The fraction of the data to be used for testing. Defaults to 0.15.
        seed (int, optional): The random seed. Defaults to 42.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
            In this case, only the first 5 datasets will be processed.
        run_transform_py (bool, optional): Whether to run the transform.py script in the directory of the dataset.
    """
    # make deterministic
    np.random.seed(seed)
    random.seed(seed)

    all_yaml_files = get_meta_yaml_files(data_dir)
    if debug:
        all_yaml_files = all_yaml_files[:5]
    transformed_files = []
    for file in tqdm(all_yaml_files):
        print(f"Processing {file}")
        with open(file, "r") as f:
            meta = yaml.safe_load(f)

        if meta["name"] in to_scaffold_split:
            # run transform.py script in this directory if `data_clean.csv` does not exist
            # use this dir as cwd
            if run_transform_py:
                run_transform(file)

            transformed_files.append(
                os.path.join(os.path.dirname(file), "data_clean.csv")
            )

    all_smiles = set()
    for file in transformed_files:
        df = pd.read_csv(file, low_memory=False)
        all_smiles.update(df["SMILES"].tolist())
        del df  # ensure memory is freed

    all_smiles = list(all_smiles)
    splits = _create_scaffold_split(
        all_smiles, frac=[train_frac, val_frac, test_frac], seed=seed
    )

    # select the right indices for each split
    train_smiles = [all_smiles[i] for i in splits["train"]]
    val_smiles = [all_smiles[i] for i in splits["valid"]]
    test_smiles = [all_smiles[i] for i in splits["test"]]

    # write the validation and test smiles to files
    with open("val_smiles.txt", "w") as f:
        f.write("\n".join(val_smiles))
    with open("test_smiles.txt", "w") as f:
        f.write("\n".join(test_smiles))

    print(
        "Train smiles:",
        len(train_smiles),
        "Val smiles:",
        len(val_smiles),
        "Test smiles:",
        len(test_smiles),
    )

    # now for each dataframe, add a split column based on in which list in the `splits` dict the smiles is
    # smiles are in the SMILES column
    # if the override option is true, we write this new file to `data_clean.csv` in the same directory
    # otherwise, we copy the old `data_clean.csv` to `data_clean_old.csv` and write the new file to `data_clean.csv`

    split_for_smiles_curried = partial(
        split_for_smiles, train_smiles=train_smiles, val_smiles=val_smiles
    )
    for file in tqdm(transformed_files):
        df = pd.read_csv(file, low_memory=False)
        df["split"] = df["SMILES"].parallel_apply(split_for_smiles_curried)

        # to ensure overall scaffold splitting does not distort train/val/test split sizes for each dataset
        print(
            f"Dataset {file} has {len(df)} datapoints. Split sizes: {len(df[df['split'] == 'train'])} train, {len(df[df['split'] == 'valid'])} valid, {len(df[df['split'] == 'test'])} test."  # noqa: E501
        )
        print(
            f"Dataset {file} has {len(df)} datapoints. Split fractions: {len(df[df['split'] == 'train']) / len(df)} train, {len(df[df['split'] == 'valid']) / len(df)} valid, {len(df[df['split'] == 'test']) / len(df)} test."  # noqa: E501
        )
        if override:
            # write the new data_clean.csv file
            df.to_csv(file, index=False)
        else:
            # copy the old data_clean.csv to data_clean_old.csv
            os.rename(file, file.replace(".csv", "_old.csv"))
            # write the new data_clean.csv file
            df.to_csv(file, index=False)


def run_all_split(
    data_dir: Union[str, Path],
    override: bool = False,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    debug: bool = False,
    run_transform_py: bool = False,
):
    """Runs all splitting steps on the datasets in the data_dir directory."""
    # print('Running "as_sequence" split...')
    # as_sequence_split(
    #     data_dir,
    #     override,
    #     seed,
    #     debug,
    #     train_frac,
    #     val_frac,
    #     test_frac,
    #     run_transform_py,
    # )
    print("Running scaffold split...")
    scaffold_split(
        data_dir,
        override,
        train_frac,
        val_frac,
        test_frac,
        seed,
        debug,
        run_transform_py,
    )
    print("Running SMILES split...")
    smiles_split(
        data_dir,
        override,
        seed,
        debug,
        train_frac,
        val_frac,
        test_frac,
        run_transform_py=run_transform_py,
    )
    print("Running remaining split...")
    remaining_split(
        data_dir,
        override,
        seed,
        debug,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        run_transform_py=run_transform_py,
    )


if __name__ == "__main__":
    fire.Fire(run_all_split)
