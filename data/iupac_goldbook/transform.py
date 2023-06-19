import json

import pandas as pd
import requests
import yaml


def get_and_transform_data():
    # get raw data
    data_path = "https://goldbook.iupac.org/terms/vocab/json/download"
    fn_data_original = "data_original.json"
    data = requests.get(data_path)
    with open(fn_data_original, "wb") as f:
        f.write(data.content)
    del data

    # create dataframe
    with open(fn_data_original) as f:
        data = json.load(f)

    entries = []
    for key in data["entries"].keys():
        d = data["entries"][key].copy()
        d["entry_id"] = int(key)
        entries.append(d)
    df = pd.DataFrame(entries)

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "code",
        "term",
        "index",
        "status",
        "doi",
        "version",
        "lastupdated",
        "newcode",
        "url",
        "definition",
        "identifiers",
        "related",
        "entry_id",
        "replacedby",
    ]

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": data[
            "title"
        ],  # unique identifier, we will also use this for directory names
        "description": """The Compendium is popularly referred to as the Gold
Book, in recognition of the contribution of the late Victor Gold, who
initiated work on the first edition. It is one of the series of IUPAC
Colour Books on chemical nomenclature, terminology, symbols and units
(see the list of source documents), and collects together terminology
definitions from IUPAC recommendations already published in Pure and
Applied Chemistry and in the other Colour Books. Terminology
definitions published by IUPAC are drafted by international committees
of experts in the appropriate chemistry sub-disciplines, and ratified
by IUPAC's Interdivisional Committee on Terminology, Nomenclature and
Symbols (ICTNS). In this edition of the Compendium these IUPAC-approved
definitions are supplemented with some definitions from ISO and from
the International Vocabulary of Basic and General Terms in Metrology,
both these sources are recognised by IUPAC as authoritative. The result
is a collection of nearly 7000 terms, with authoritative definitions,
spanning the whole range of chemistry.""",
        "targets": [
            {
                "id": "definition",  # name of the column in a tabular dataset
                "description": "definition of a chemistry term",  # description of what this column means
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "string",  # can be "categorical", "ordinal", "continuous", "string"
                "names": [  # names for the property (to sample from for building the prompts)
                    "definition",
                    "definition of a chemistry term",
                ],
            },
        ],
        "identifiers": [
            {
                "id": "term",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
                "description": "chemistry term",  # description (optional, except for "OTHER")
                "names": [
                    {"noun": "term"},
                    {"noun": "chemistry term"},
                ],
            },
        ],
        "license": "CC BY-NC-ND 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://goldbook.iupac.org",
                "description": "home page",
            },
            {
                "url": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
                "description": "license description",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            "@article{iupac2023,"
            f"title={{{data['title']}}},\n"
            f"publisher={{{data['publisher']}}},\n"
            f"isbn={{{data['isbn']}}},\n"
            f"doi={{{data['doi']}}},\n"
            f"accessdate={{{data['accessdate']}}},\n"
            "}",
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
    )  # to use with safe_dump

    fn_meta = "meta.yaml"
    with open(fn_meta, "w") as f:
        yaml.dump(meta, f, sort_keys=False)

    print(f"Finished processing {meta['name']} dataset!")


if __name__ == "__main__":
    get_and_transform_data()
