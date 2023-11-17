import os
import subprocess
from functools import partial
from glob import glob
from typing import List

import fire
import pandas as pd
import yaml
from pandarallel import pandarallel

from chemnlp.data.split import _create_scaffold_split

pandarallel.initialize(progress_bar=True)

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
    "clintox" "sr_p53_tox21",
    "nr_er_lbd_tox21" "pampa_ncats",
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
    "hiv",
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


def split_for_smiles(smiles: str, train_smiles: List[str], val_smiles: List[str]):
    if smiles in train_smiles:
        return "train"
    elif smiles in val_smiles:
        return "valid"
    else:
        return "test"


def main(
    data_dir,
    override: bool = False,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    debug: bool = False,
):
    all_yaml_files = glob(os.path.join(data_dir, "**", "*.yaml"), recursive=True)
    if debug:
        all_yaml_files = all_yaml_files[:5]
    transformed_files = []
    for file in all_yaml_files:
        with open(file, "r") as f:
            meta = yaml.safe_load(f)

        if meta["name"] in to_scaffold_split:
            # run transform.py script in this directory if `data_clean.csv` does not exist
            # use this dir as cwd
            if not os.path.exists(
                os.path.join(os.path.dirname(file), "data_clean.csv")
            ):
                subprocess.run(
                    ["python", "transform.py"],
                    cwd=os.path.dirname(file),
                )

            transformed_files.append(
                os.path.join(os.path.dirname(file), "data_clean.csv")
            )

    all_smiles = set()
    for file in transformed_files:
        df = pd.read_csv(file)
        all_smiles.update(df["SMILES"].tolist())
        del df  # ensure memory is freed

    all_smiles = list(all_smiles)
    splits = _create_scaffold_split(
        all_smiles, frac=[train_frac, val_frac, test_frac], seed=42
    )

    # select the right indices for each split
    train_smiles = [all_smiles[i] for i in splits["train"]]
    val_smiles = [all_smiles[i] for i in splits["valid"]]
    print(
        "Train smiles:",
        len(train_smiles),
        "Val smiles:",
        len(val_smiles),
        "Test smiles:",
        len(splits["test"]),
    )

    # now for each dataframe, add a split column based on in which list in the `splits` dict the smiles is
    # smiles are in the SMILES column
    # if the override option is true, we write this new file to `data_clean.csv` in the same directory
    # otherwise, we copy the old `data_clean.csv` to `data_clean_old.csv` and write the new file to `data_clean.csv`
    # we print the train/val/test split sizes and fractions for each dataset

    split_for_smiles_curried = partial(
        split_for_smiles, train_smiles=train_smiles, val_smiles=val_smiles
    )
    for file in transformed_files:
        df = pd.read_csv(file)
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


if __name__ == "__main__":
    fire.Fire(main)
