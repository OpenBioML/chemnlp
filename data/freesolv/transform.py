import pandas as pd
import requests
import yaml


def get_and_transform_data():
    # get raw data
    data_path = (
        "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt"
    )
    fn_data_original = "data_original.txt"
    data = requests.get(data_path)
    with open(fn_data_original, "wb") as f:
        f.write(data.content)

    # create dataframe
    df = pd.read_csv(fn_data_original, delimiter=";", skiprows=2)

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "# compound id (and file prefix)",
        " SMILES",
        " iupac name (or alternative if IUPAC is unavailable or not parseable by OEChem)",
        " experimental value (kcal/mol)",
        " experimental uncertainty (kcal/mol)",
        " Mobley group calculated value (GAFF) (kcal/mol)",
        " calculated uncertainty (kcal/mol)",
        " experimental reference (original or paper this value was taken from)",
        " calculated reference",
        " text notes.",
    ]

    # overwrite column names = fields
    fields_clean = [
        "compound_id",
        "SMILES",
        "iupac_name",
        "exp_value",
        "exp_uncertainty",
        "GAFF",
        "calc_uncertainty",
        "exp_ref",
        "calc_reference",
        "notes",
    ]
    df.columns = fields_clean

    # data cleaning
    df.notes = (
        df.notes.str.strip()
    )  # remove leading and trailing white space characters

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "freesolv",  # unique identifier, we will also use this for directory names
        "description": "Experimental and calculated small molecule hydration free energies",
        "targets": [
            {
                "id": "exp_value",  # name of the column in a tabular dataset
                "description": "experimental hydration free energy value",  # description of what this column means
                "units": "kcal/mol",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "hydration free energy",
                ],
            },
            {
                "id": "exp_uncertainty",
                "description": "experimental hydration free energy uncertainty",
                "units": "kcal/mol",
                "type": "continuous",
                "names": [
                    "hydration free energy uncertainty",
                ],
            },
            {
                "id": "GAFF",  # name of the column in a tabular dataset
                "description": "mobley group calculated value",  # description of what this column means
                "units": "kcal/mol",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "hydration free energy computed using the GAFF force field",
                ],
            },
            {
                "id": "calc_uncertainty",
                "description": "mobley group calculated value calculated uncertainty",
                "units": "kcal/mol",
                "type": "continuous",
                "names": [
                    "uncertainty in hydration free energy computed using the GAFF force field",
                ],
            },
        ],
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
                "description": "SMILES",  # description (optional, except for "OTHER")
            },
            {
                "id": "iupac_name",
                "type": "IUPAC",
                "description": "IUPAC",
            },
        ],
        "license": "CC BY-NC-SA 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://github.com/MobleyLab/FreeSolv",
                "description": "issue tracker and source data",
            },
            {
                "url": "https://escholarship.org/uc/item/6sd403pz",
                "description": "repository with data",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{mobley2013experimental,
title={Experimental and calculated small molecule hydration free energies},
author={Mobley, David L},
year={2013}""",
        ],
    }
    fn_meta = "meta.yaml"
    with open(fn_meta, "w") as f:
        yaml.dump(meta, f, sort_keys=False)

    print(f"Finished processing {meta['name']} dataset!")


if __name__ == "__main__":
    get_and_transform_data()
