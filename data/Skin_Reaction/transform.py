import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    data = Tox(name="Skin Reaction")
    fn_data_original = "data_original.csv"
    data.get_data().to_csv(fn_data_original, index=False)

    # create dataframe
    df = pd.read_csv(
        fn_data_original,
        delimiter=",",
    )  # not necessary but ensure we can load the saved data

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "Drug_ID",
        "Drug",
        "Y",
    ]

    # overwrite column names = fields
    fields_clean = [
        "compound_name",
        "SMILES",
        "Skin_Reaction",
    ]
    df.columns = fields_clean

    # data cleaning
    df.compound_name = (
        df.compound_name.str.strip()
    )  # remove leading and trailing white space characters

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "Skin_Reaction",  # unique identifier, we will also use this for directory names
        "description": """Repetitive exposure to a chemical agent can induce an immune reaction in inherently susceptible individuals that leads to skin sensitization. The dataset used in this study was retrieved from the ICCVAM (Interagency Coordinating Committee on the Validation of Alternative Methods) report on the rLLNA.""",
        "targets": [
            {
                "id": "Skin_Reaction",  # name of the column in a tabular dataset
                "description": "whether it can cause skin reaction (1) or not (0).",  # description of what this column means
                "units": "Skin_Reaction",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "Skin Reaction",
                    "skin sensitization",
                    "agent Induced Skin Reaction",
                    "drug induced skin immune reaction",
                ],
                "uris": [
                    "https://bioportal.bioontology.org/ontologies/SNOMEDCT?p=classes&conceptid=http%3A%2F%2Fpurl.bioontology.org%2Fontology%2FSNOMEDCT%2F264519003",
                    "https://bioportal.bioontology.org/ontologies/MEDDRA?p=classes&conceptid=http%3A%2F%2Fpurl.bioontology.org%2Fontology%2FMEDDRA%2F10040914",
                ],
            },
        ],
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "description": "SMILES",  # description (optional, except for "Other")
            },
            {
                "id": "compound_name",
                "type": "Synonyms",
                "description": "drug name",
                "names": [
                    "compound",
                    "compound name",
                    "chemical agent",
                ],
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1016/j.taap.2014.12.014",
                "description": "corresponding publication",
            },
            {
                "url": "https://ntp.niehs.nih.gov/iccvam/docs/immunotox_docs/llna-ld/tmer.pdf",
                "description": "related publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#skin-reaction",
                "description": "Data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Alves2015,
                  doi = {10.1016/j.taap.2014.12.014},
                  url = {https://doi.org/10.1016/j.taap.2014.12.014},
                  year = {2015},
                  month = apr,
                  publisher = {Elsevier {BV}},
                  volume = {284},
                  number = {2},
                  pages = {262--272},
                  author = {Vinicius M. Alves and Eugene Muratov and Denis Fourches and Judy Strickland and Nicole Kleinstreuer and Carolina H. Andrade and Alexander Tropsha},
                  title = {Predicting chemically-induced skin reactions. Part I: {QSAR} models of skin sensitization and their application to identify potentially hazardous compounds},
                  journal = {Toxicology and Applied Pharmacology}}""",
        ],
    }

    def str_presenter(dumper, data):
        """configures yaml for dumping multiline strings
        Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
        """
        if data.count("\n") > 0:  # check for multiline string
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(
        str, str_presenter
    )  # to use with safe_dum
    fn_meta = "meta.yaml"
    with open(fn_meta, "w") as f:
        yaml.dump(meta, f, sort_keys=False)

    print(f"Finished processing {meta['name']} dataset!")


if __name__ == "__main__":
    get_and_transform_data()
