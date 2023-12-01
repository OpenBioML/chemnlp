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
from tqdm import tqdm

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
    try:
        molecule = to_smiles_molecule(smiles)
        return FEATURIZER.featurize(molecule)
    except Exception:
        return np.array([np.nan] * len(FEATURIZER.feature_labels())).reshape(1, -1)


def transform():
    df_1 = pd.read_csv("../choline_transporter_butkiewicz/data_clean.csv")
    df_2 = pd.read_csv("../kcnq2_potassium_channel_butkiewicz/data_clean.csv")

    # merge on SMILES, keep where we have SMILES in both datasets
    merged = pd.merge(df_1, df_2, on="SMILES", how="inner")
    print(len(merged))

    features = []
    feature_names = FEATURIZER.feature_labels()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for feature in tqdm(
            executor.map(featurize_smiles, merged["SMILES"]), total=len(merged)
        ):
            features.append(feature)

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
        + ["activity_choline_transporter", "activity_kcnq2_potassium_channel"],
        inplace=True,
    )
    print(len(merged))
    merged.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    transform()
