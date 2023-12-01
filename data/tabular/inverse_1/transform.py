import concurrent.futures

import pandas as pd
from chemcaption.featurize.adaptor import ValenceElectronCountAdaptor
from chemcaption.featurize.base import MultipleFeaturizer
from chemcaption.featurize.bonds import (
    BondTypeCountFeaturizer,
    BondTypeProportionFeaturizer,
    RotableBondCountFeaturizer,
)
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
from chemcaption.featurize.spatial import (
    AsphericityFeaturizer,
    EccentricityFeaturizer,
    InertialShapeFactorFeaturizer,
    NPRFeaturizer,
    PMIFeaturizer,
)
from chemcaption.featurize.stereochemistry import ChiralCenterCountFeaturizer
from chemcaption.featurize.substructure import SMARTSFeaturizer
from chemcaption.molecules import InChIMolecule, SELFIESMolecule, SMILESMolecule
from chemcaption.presets import ALL_SMARTS


def get_smarts_featurizers():
    featurizers = []
    for name, smarts in ALL_SMARTS.items():
        featurizers.append(SMARTSFeaturizer([smarts], names=[name]))
    return featurizers


FEATURIZER = MultipleFeaturizer(
    get_smarts_featurizers()
    + [
        ValenceElectronCountAdaptor(),
        RotableBondCountFeaturizer(),
        BondTypeCountFeaturizer(),
        BondTypeProportionFeaturizer(),
        MolecularFormulaFeaturizer(),
        MonoisotopicMolecularMassFeaturizer(),
        ElementMassFeaturizer(),
        ElementCountFeaturizer(),
        ElementMassProportionFeaturizer(),
        HydrogenAcceptorCountFeaturizer(),
        HydrogenDonorCountFeaturizer(),
        LipinskiViolationCountFeaturizer(),
        InertialShapeFactorFeaturizer(),
        EccentricityFeaturizer(),
        AsphericityFeaturizer(),
        InertialShapeFactorFeaturizer(),
        NPRFeaturizer(),
        PMIFeaturizer(),
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

    feature_names = FEATURIZER.feature_names()

    # add features to dataframe
    for i, name in enumerate(feature_names):
        merged[name] = features[:, i]

    merged.to_csv("merged.csv", index=False)


if __name__ == "__main__":
    transform()
