import concurrent.futures

import numpy as np
import pandas as pd
from chemcaption.featurize.adaptor import ValenceElectronCountAdaptor
from chemcaption.featurize.base import MultipleFeaturizer
from chemcaption.featurize.composition import (
    ElementCountFeaturizer,
    ElementMassFeaturizer,
    ElementMassProportionFeaturizer,
    MolecularFormulaFeaturizer,
    MonoisotopicMolecularMassFeaturizer,
)
from chemcaption.featurize.electronicity import (
    HydrogenAcceptorCountFeaturizer,
    HydrogenDonorCountFeaturizer,
)
from chemcaption.featurize.rules import LipinskiFilterFeaturizer
from chemcaption.featurize.stereochemistry import ChiralCenterCountFeaturizer
from chemcaption.featurize.substructure import FragmentSearchFeaturizer
from chemcaption.molecules import SMILESMolecule
from chemcaption.presets import ORGANIC

ORGANIC = dict(zip(ORGANIC["names"], ORGANIC["smarts"]))


def get_smarts_featurizers():
    featurizers = []
    for name, smarts in ORGANIC.items():
        featurizers.append(FragmentSearchFeaturizer([smarts], names=[name]))
    return featurizers


FEATURIZER = MultipleFeaturizer(
    get_smarts_featurizers()
    + [
        ValenceElectronCountAdaptor(),
        MolecularFormulaFeaturizer(),
        MonoisotopicMolecularMassFeaturizer(),
        ElementMassFeaturizer(),
        ElementCountFeaturizer(),
        ElementMassProportionFeaturizer(),
        HydrogenAcceptorCountFeaturizer(),
        HydrogenDonorCountFeaturizer(),
        LipinskiFilterFeaturizer(),
        ChiralCenterCountFeaturizer(),
    ]
)


def to_smiles_molecule(smiles: str):
    return SMILESMolecule(smiles)


def featurize_smiles(smiles: str):
    molecule = to_smiles_molecule(smiles)
    return FEATURIZER.featurize(molecule)


def transform():
    df_1 = pd.read_csv("../solubility_aqsoldb/data_clean.csv")
    df_2 = pd.read_csv("../sr_atad5_tox21/data_clean.csv")

    # merge on SMILES, keep where we have SMILES in both datasets
    merged = pd.merge(df_1, df_2, on="SMILES", how="inner")
    print(len(merged))

    features = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for feature in executor.map(featurize_smiles, merged["SMILES"]):
            features.append(feature)

    feature_names = FEATURIZER.feature_labels()
    print(feature_names)
    features = np.concatenate(features)
    print(features.shape)
    # add features to dataframe
    for i, name in enumerate(feature_names):
        merged[name] = features[:, i]
    merged.dropna(
        subset=[
            "SMILES",
        ]
        + FEATURIZER.feature_names()
        + ["aqeuous_solubility", "toxicity_SR-ATAD5"],
        inplace=True,
    )
    merged[
        [
            "carboxyl_count",
            "carbonyl_count",
            "ether_count",
            "alkanol_count",
            "thiol_count",
            "halogen_count",
            "amine_count",
            "amide_count",
            "ketone_count",
            "num_valence_electrons",
            "num_carbon_atoms",
            "num_hydrogen_atoms",
            "num_nitrogen_atoms",
            "num_oxygen_atoms",
            "num_hydrogen_bond_acceptors",
            "num_hydrogen_bond_donors",
            "num_lipinski_violations",
            "num_chiral_centers",
        ]
    ] = merged[
        [
            "carboxyl_count",
            "carbonyl_count",
            "ether_count",
            "alkanol_count",
            "thiol_count",
            "halogen_count",
            "amine_count",
            "amide_count",
            "ketone_count",
            "num_valence_electrons",
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
    print(len(merged))
    merged["split"] = merged["split_x"]
    merged.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    transform()
