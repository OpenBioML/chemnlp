import pandas as pd
import yaml
from tdc.generation import MolGen

DATASET_NAME = "zinc"


def get_and_transform_data():

    # get raw data per dataset
    def get_single_dataset(dataset_name):
        splits = MolGen(name=dataset_name).get_split()
        df_train = splits["train"]
        df_valid = splits["valid"]
        df_test = splits["test"]
        df_train["split"] = "train"
        df_valid["split"] = "valid"
        df_test["split"] = "test"
        df = pd.concat([df_train, df_valid, df_test], axis=0)
        df["dataset"] = dataset_name
        return df

    # get raw data
    df = get_single_dataset(DATASET_NAME)

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "smiles",
        "split",
        "dataset",
    ]
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": DATASET_NAME,  # unique identifier, we will also use this for directory names
        "description": """
        ZINC is a free database of commercially-available compounds for virtual screening.
        It contains over 230 million purchasable compounds in ready-to-dock, 3D formats.
        TDC uses a 250,000 sampled version from the original Mol-VAE paper.
        """,
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
                "description": "SMILES",  # description (optional, except for "OTHER")
            },
        ],
        # license under which the original dataset was published
        "license": """ZINC is free to use for everyone.
        Redistribution of significant subsets requires written permission from the authors.
        """,
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00559",
                "description": "Article about original dataset",
            },
            {
                "url": "https://pubs.acs.org/doi/abs/10.1021/acscentsci.7b00572",
                "description": "Exemplary related article shown in tdc's website",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """
        @article{doi:10.1021/acs.jcim.5b00559,
                author = {Sterling, Teague and Irwin, John J.},
                title = {ZINC 15 – Ligand Discovery for Everyone},
                journal = {Journal of Chemical Information and Modeling},
                volume = {55},
                number = {11},
                pages = {2324-2337},
                year = {2015},
                doi = {10.1021/acs.jcim.5b00559},
                note ={PMID: 26479676},
                URL = {
                        https://doi.org/10.1021/acs.jcim.5b00559
                },
                eprint = {
                        https://doi.org/10.1021/acs.jcim.5b00559
                }
        }
            """,
        ],
    }

    def str_presenter(dumper, data):
        """configures yaml for dumping multiline strings
        Ref:
        https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
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
