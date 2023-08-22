import urllib
import urllib.request

import pandas as pd
import yaml

DATASET_URL = (
    "https://huggingface.co/datasets/adamoyoung/mona/resolve/main/data/mona_df.json"
)
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
            "units": "nats",
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
        {"id": "smiles", "type": "SMILES", "description": "SMILES"},
        {"id": "inchi", "type": "InChI", "description": "InChI"},
        {"id": "inchikey", "type": "InChIKey", "description": "InChIKey"},
        {"id": "id", "type": "Other", "description": "MassBank ID", "sample": "False"},
    ],
    "templates": [
        {
            "prompt": "Please answer the following chemistry question.\n"
            + "What is the spectral entropy of the following mass spectrum?\n<spectrum#text>",
            "completion": "<spectral_entropy#value>",
        },
        {
            "prompt": "Please answer the following chemistry question.\n"
            + "Which of the molecules, <molecule1#text> or <molecule2#text>, is more likely to "
            + "produce the following mass spectrum?\n<spectrum#text>",
            "completion": "<molecule1#text>",
        },
    ],
    "fields": {
        "exp_values": {
            "values": [
                {
                    "name": "spectrum",
                    "column": "spectrum",
                    "text": "Raw mass spectrum represented as a set of (m/z location, intensity) pairs",
                },
                {
                    "name": "spectral_entropy",
                    "column": "spectral_entropy",
                    "text": "The entropy of the spectrum",
                },
                {
                    "name": "normalized_entropy",
                    "column": "normalized_entropy",
                    "text": "The normalized entropy of the spectrum (ratio of spectral entropy to maximum possible "
                    + "entropy for a spectrum with the same number of peaks)",
                },
            ],
        },
        "metadata": {
            "values": [
                {"name": "id", "column": "id", "text": "MassBank ID"},
                {
                    "name": "score",
                    "column": "score",
                    "text": "Quality score of the spectrum (1-5, 1 being low and 5 being high)",
                },
                {
                    "name": "library",
                    "column": "library",
                    "text": "Library the spectrum was obtained from",
                },
                {
                    "name": "molecular_formula",
                    "column": "molecular_formula",
                    "text": "Molecular formula",
                },
                {
                    "name": "accession",
                    "column": "accession",
                    "text": "Accession number",
                },
                {"name": "date", "column": "date", "text": "Date of upload"},
                {"name": "license", "column": "license", "text": "License"},
                {
                    "name": "instrument",
                    "column": "instrument",
                    "text": "Specific model of mass spectrometer",
                },
                {
                    "name": "instrument_type",
                    "column": "instrument_type",
                    "text": "General type of mass spectrometer",
                },
                {
                    "name": "ms_level",
                    "column": "ms_level",
                    "text": "MS level for MSn data (MS1-MS5, can also be composite)",
                },
                {
                    "name": "ionization_mode",
                    "column": "ionization_mode",
                    "text": "Ionization mode (positive or negative)",
                },
                {
                    "name": "precursor_m/z",
                    "column": "precursor_m/z",
                    "text": "The mass to charge ratio (m/z) of the precursor ion",
                },
                {
                    "name": "precursor_type",
                    "column": "precursor_type",
                    "text": "The precursor adduct",
                },
                {
                    "name": "mass_accuracy",
                    "column": "mass_accuracy",
                    "text": "The mass accuracy of the spectrum",
                },
                {
                    "name": "mass_error",
                    "column": "mass_error",
                    "text": "The mass error of the spectrum",
                },
                {
                    "name": "collision_energy",
                    "column": "collision_energy",
                    "text": "The collision energy of the spectrum "
                    + "(depending on the fragmentation_mode, can be normalized)",
                },
                {
                    "name": "fragmentation_mode",
                    "column": "fragmentation_mode",
                    "text": "The fragmentation mode of the spectrum (i.e. HCD, CID)",
                },
                {
                    "name": "derivatization_type",
                    "column": "derivatization_type",
                    "text": "Chemical derivatization used (for gas chromatrography spectra)",
                },
                {
                    "name": "ionization_energy",
                    "column": "ionization_energy",
                    "text": "The ionization energy (for electron ionization spectra)",
                },
            ],
        },
        "molecule": {
            "values": [
                {"name": "inchi", "column": "inchi", "text": "InChI"},
                {"name": "inchikey", "column": "inchikey", "text": "InChIKey"},
                {"name": "smiles", "column": "smiles", "text": "SMILES"},
            ],
        },
    },
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
}


def get_raw_data(raw_dataset: str = DATASET_URL) -> pd.DataFrame:
    """Load the raw dataset into a pandas dataframe"""
    # use the repo URL directly since it avoid unnecessary huggingface processing
    df_file = urllib.request.urlretrieve(DATASET_URL)[0]
    df_raw = pd.read_json(df_file)
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
