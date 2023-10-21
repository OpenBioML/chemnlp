import os
import urllib
import urllib.request

import pandas as pd
import yaml

DATASET_URL = (
    "https://huggingface.co/datasets/adamoyoung/mona/resolve/main/data/mona_df.json"
)
DATASET_RAW = "data_raw.json"
META_YAML_PATH = __file__.replace("transform.py", "meta.yaml")
DF_CSV_PATH = __file__.replace("transform.py", "data_clean.csv")

META_TEMPLATE = {
    "name": "mona",
    "description": "MassBank of North America, public repository of mass spectra for small molecules",
    # Unclear with this type of data what exactly the targets should be
    # Arbitrarily chose spectral entropy since it is a continuous value
    "targets": [
        {
            "id": "spectral_entropy",
            "type": "continuous",
            "units": "nat",
            "names": [{"noun": "spectral entropy"}],
            "description": "The entropy of the spectrum.",
        },
        {
            "id": "normalized_entropy",
            "type": "continuous",
            "units": None,
            "names": [{"noun": "normalized entropy"}],
            "description": "The normalized entropy of the spectrum (ratio of spectral entropy to maximum possible "
            + "entropy for a spectrum with the same number of peaks).",
        },
    ],
    "identifiers": [
        {"id": "SMILES", "type": "SMILES", "description": "SMILES"},
        {"id": "inchi", "type": "InChI", "description": "InChI"},
        {"id": "inchikey", "type": "InChIKey", "description": "InChIKey"},
        {"id": "id", "type": "Other", "description": "MassBank ID", "sample": "False"},
    ],
    "license": "CC BY 4.0",
    "links": [
        {
            "name": "MassBank of North America",
            "url": "https://mona.fiehnlab.ucdavis.edu/",
            "description": "original repository",
        },
        {
            "name": "HuggingFace dataset upload",
            "url": "https://huggingface.co/datasets/adamoyoung/mona",
            "description": "HuggingFace dataset upload",
        },
    ],
    "benchmarks": [],
    "num_points": 0,  # will be updated
    "bibtex": [
        # noqa
    ],
    "templates": [
        "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} has a {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}.",  # noqa: E501
        "Based on the {SMILES__description} {#representation of |!}{SMILES#}, the molecule has a {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}.",  # noqa: E501
        "The {SMILES__description} {SMILES#} {#represents|is representing!} a molecule {#that has a|with a!} {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}.",  # noqa: E501
        "The molecule with the {SMILES__description} {SMILES#} has a {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}.",  # noqa: E501
        # Instruction tuning text templates
        """Task: Please predict a molecule feature based on the description.
Description: Predict the {spectral_entropy__names__noun} in {spectral_entropy__units}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a numeric value in {spectral_entropy__units} without using any {#other|additional!} words.
Result: {spectral_entropy#} {spectral_entropy__units}""",  # noqa: E501
        """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that has a {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}.
Result: {SMILES#}""",  # noqa: E501
        # Conversational text templates
        """User: Can you {#tell me|derive|estimate!} the {spectral_entropy__names__noun} in {spectral_entropy__units} of the molecule with the {SMILES__description} {SMILES#}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, this molecule has a {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}.""",  # noqa: E501
        """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that has a {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
        """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that has a {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}.
Assistant: This is a molecule that has a {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}: {SMILES#}""",  # noqa: E501
        """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should have a {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} represents a molecule that has a {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}: {SMILES#}""",  # noqa: E501
        """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should have a {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} represents a molecule that has a {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}: {SMILES#}""",  # noqa: E501
        # Benchmarking text templates
        "The {spectral_entropy__names__noun} of the molecule with the {SMILES__description} {SMILES#} is:<EOI>{spectral_entropy#} {spectral_entropy__units}",  # noqa: E501
        "The {spectral_entropy__names__noun} of the {SMILES__description} {SMILES#} is:<EOI>{spectral_entropy#} {spectral_entropy__units}",  # noqa: E501
        "The {spectral_entropy__names__noun} of the molecule {SMILES__description} {SMILES#} is:<EOI>{spectral_entropy#} {spectral_entropy__units}",  # noqa: E501
        """Task: Please predict a molecule feature based on the description.
Description: Predict the {spectral_entropy__names__noun} in {spectral_entropy__units} of a molecule.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a numeric value in {spectral_entropy__units} without using any {#other|additional!} words.
Result:<EOI>{spectral_entropy#} {spectral_entropy__units}""",  # noqa: E501
        """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that has {spectral_entropy__names__noun} of {spectral_entropy#} {spectral_entropy__units}.
Result:<EOI>{SMILES#}""",  # noqa: E501
    ],
}


def get_raw_data(raw_dataset: str = DATASET_URL) -> pd.DataFrame:
    """Load the raw dataset into a pandas dataframe"""
    if not (os.path.isfile(DATASET_RAW)):
        # use the repo URL directly since it avoid unnecessary huggingface processing
        urllib.request.urlretrieve(DATASET_URL, DATASET_RAW)
    else:
        print("Using already downloaded raw data.")
    df_raw = pd.read_json(DATASET_RAW)
    df_raw.columns = [
        "spectrum",
        "id",
        "score",
        "library",
        "inchi",
        "inchikey",
        "molecular_formula",
        "SMILES",
        "accession",
        "date",
        "license",
        "instrument",
        "instrument_type",
        "ms_level",
        "ionization_mode",
        "spectral_entropy",
        "normalized_entropy",
        "precursor_type",
        "precursor_m/z",
        "mass_accuracy",
        "mass_error",
        "collision_energy",
        "fragmentation_mode",
        "derivatization_type",
        "ionization_energy",
    ]
    return df_raw


def create_meta_yaml(num_points: int):
    """Create meta configuration file for the dataset"""
    META_TEMPLATE["num_points"] = num_points

    def str_presenter(dumper, data):
        """configures yaml for dumping multiline strings
        Ref:
        https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
        """
        if data.count("\n") > 0:  # check for multiline string
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)
    with open(META_YAML_PATH, "w") as f:
        yaml.dump(META_TEMPLATE, f, sort_keys=False, default_flow_style=False)
    print(f"Finished processing {META_TEMPLATE['name']} dataset!")


if __name__ == "__main__":
    num_samples = 0
    raw_df = get_raw_data()
    num_samples += len(raw_df)
    create_meta_yaml(num_samples)
    raw_df.to_csv(DF_CSV_PATH, index=False)
