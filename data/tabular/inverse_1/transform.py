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
from chemcaption.featurize.rules import LipinskiViolationCountFeaturizer
from chemcaption.featurize.stereochemistry import ChiralCenterCountFeaturizer
from chemcaption.featurize.substructure import SMARTSFeaturizer
from chemcaption.molecules import SMILESMolecule
from chemcaption.presets import ORGANIC

ORGANIC = dict(zip(ORGANIC["names"], ORGANIC["smarts"]))


def get_smarts_featurizers():
    featurizers = []
    for name, smarts in ORGANIC.items():
        featurizers.append(SMARTSFeaturizer([smarts], names=[name]))
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
        LipinskiViolationCountFeaturizer(),
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
    df_2 = pd.read_csv("../nr_ar_tox21/data_clean.csv")

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
        + ["aqeuous_solubility", "toxicity_NR-AR"],
        inplace=True,
    )
    print(len(merged))
    merged.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    transform()
