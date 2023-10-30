import ast
from collections import defaultdict
from glob import glob

import pandas as pd
from datasets import Dataset

allowed_single_output_features = [
    "['num_valence_electrons']",
    "['monoisotopic_molecular_mass']",
    "['molecular_formula']",
    "['num_hydrogen_bond_acceptors']",
    "['num_hydrogen_bond_donors']",
    "['num_lipinski_violations']",
    "['inertial_shape_factor']",
    "['eccentricity']",
    "['asphericity']",
    "['num_chiral_centers']",
]

allowed_multi_output_features = [
    "['rotable_proportion', 'non_rotable_proportion']",
    "['num_unspecified_bond', 'num_single_bonds', 'num_double_bonds', 'num_triple_bonds', 'num_quadruple_bonds', 'num_quintuple_bonds', 'num_hextuple_bonds', 'num_oneandahalf_bonds', 'num_twoandahalf_bonds', 'num_threeandahalf_bonds', 'num_fourandahalf_bonds', 'num_fiveandahalf_bonds', 'num_aromatic_bonds', 'num_ionic_bonds', 'num_hydrogen_bonds', 'num_threecenter_bonds', 'num_dativeone_bonds', 'num_dative_bonds', 'num_other_bonds', 'num_zero_bonds', 'num_bonds']",  # noqa
    "['carbon_mass', 'hydrogen_mass', 'nitrogen_mass', 'oxygen_mass']",
    "['num_carbon_atoms', 'num_hydrogen_atoms', 'num_nitrogen_atoms', 'num_oxygen_atoms']",
    "['npr1_value', 'npr2_value']",
    "['pmi1_value', 'pmi2_value', 'pmi3_value']",
]


def get_allowed_features():
    return allowed_single_output_features + allowed_multi_output_features


def extract_output_feature(row):
    completion = row["completion"]
    completion = ast.literal_eval(completion)
    labels = row["completion_labels"]
    labels = ast.literal_eval(labels)

    return dict(zip(completion, labels))


def extract_features_frame(file):
    molecular_features = defaultdict(dict)
    df = pd.read_json(file, lines=True)
    df["completion_labels"] = df["completion_labels"].astype(str)
    df["completion"] = df["completion"].astype(str)
    subset = df[df["completion_labels"].str.contains("|".join(get_allowed_features()))]

    for _index, row in subset.iterrows():
        molecule = row["representation"]
        representation_type = row["representation_type"]
        features = extract_output_feature(row)
        features["representation_type"] = representation_type
        molecular_features[molecule].update(features)

    list_of_dicts = []
    for k, v in molecular_features.items():
        v["representation"] = k
        list_of_dicts.append(v)

    del molecular_features
    del df

    return pd.DataFrame(list_of_dicts)


if __name__ == "__main__":
    all_files = glob("*.jsonl")
    all_dfs = []
    for file in all_files:
        df = extract_features_frame(file)
        all_dfs.append(df)

    df = pd.concat(all_dfs)

    ds = Dataset.from_pandas(df)
    ds.push_to_hub(repo_id="kjappelbaum/chemnlp-chem-caption", config_name="rdkit_feat")
