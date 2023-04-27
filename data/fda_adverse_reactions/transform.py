import concurrent.futures
import os
from xml.etree import ElementTree

import pandas as pd
import requests
import yaml

DATASET_URL = "ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.02/output/etl/json/fda/significantAdverseDrugReactions"  # noqa
DOWNLOAD_FOLDER = "./fda/significantAdverseDrugReactions"
EBI_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/{}"

META_TEMPLATE = {
    "name": "fda_adverse_reactions",  # unique identifier, we will also use this for directory names
    "description": "A dataset of adverse reaction statistics for drugs and reaction events.",
    "targets": [
        {
            "id": "count",  # name of the column in a tabular dataset
            "description": "A count of how many reaction events occurred for this chembl id.",
            "units": None,  # units of the values in this column (leave empty if unitless)
            "type": "ordinal",  # can be "categorical", "ordinal", "continuous", "string"
            "names": [  # names for the property (to sample from for building the prompts)
                {"noun": "adverse reaction count"},
            ],
            "pubchem_aids": [],
            "uris": [],
        },
        {
            "id": "event",  # name of the column in a tabular dataset
            "description": "The type of event that occurred for this molecule interaction.",
            "units": None,  # units of the values in this column (leave empty if unitless)
            "type": "string",  # can be "categorical", "ordinal", "continuous", "string"
            "names": [  # names for the property (to sample from for building the prompts)
                {"noun": "adverse event reaction"},
            ],
            "pubchem_aids": [],
            "uris": [],
        },
    ],
    "identifiers": [
        {
            "id": "SMILES",
            "type": "SMILES",
            "description": "This is the SMILES identifier for a given molecule.",
        },
    ],
    "license": "CC BY-SA 3.0",  # license under which the original dataset was published
    "links": [  # list of relevant links (original dataset, other uses, etc.)
        {
            "name": "Dataset",
            "url": "https://platform.opentargets.org/downloads",
            "description": "The website which we download the dataset from during the transformation script.",
        }
    ],
    "benchmarks": [],
    "num_points": None,  # number of datapoints in this dataset
    "bibtex": [],
}


def str_presenter(dumper, data: dict):
    """Configures yaml for dumping multiline strings"""
    if data.count("\n") > 0:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def create_meta_yaml(num_points: int):
    """Create meta configuration file for the dataset"""
    # create meta yaml
    META_TEMPLATE["num_points"] = num_points
    with open("meta.yaml", "w+") as f:
        yaml.dump(META_TEMPLATE, f, sort_keys=False)
    print(f"Finished processing chebi-20 {META_TEMPLATE['name']} dataset!")


def get_data():
    os.system(
        f"wget --recursive --no-parent --no-host-directories --cut-dirs 8 {DATASET_URL}"
    )


def read_data() -> pd.DataFrame:
    dfs = []
    for file in os.listdir(DOWNLOAD_FOLDER):
        if file.endswith(".json"):
            data = pd.read_json(f"{DOWNLOAD_FOLDER}/{file}", lines=True)
            dfs.append(data)
    return pd.concat(dfs, ignore_index=True)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["SMILES"])


def parallelise_smiles_retrieval(df: pd.DataFrame) -> dict:
    unique_chemblids = df["chembl_id"].unique()
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        smiles = executor.map(get_smiles_from_chembl_id, unique_chemblids)
    return {c: s for c, s in zip(unique_chemblids, smiles)}


def get_smiles_from_chembl_id(id: str) -> str:
    try:
        xml_content = requests.get(EBI_URL.format(id))
        xml_tree = ElementTree.fromstring(xml_content.content)
        smile = xml_tree[17][0].text
        return smile
    except Exception:
        print(f"Could not find SMILES for CHEMBL ID {id}.")


if __name__ == "__main__":
    get_data()
    df = read_data()

    chemblid_to_smile = parallelise_smiles_retrieval(df)
    df["SMILES"] = df["chembl_id"].apply(lambda x: chemblid_to_smile[x])
    df_clean = validate_data(df)
    print(df_clean.head(1))
    print(df_clean.info())

    yaml.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)

    create_meta_yaml(num_points=len(df_clean))
