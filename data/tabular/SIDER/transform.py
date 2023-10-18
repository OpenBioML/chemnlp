import inspect
from typing import List, Tuple

import pandas as pd
import yaml


def load_dataset():
    sider = pd.read_csv(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"
    )
    return sider


def transform_data() -> Tuple[pd.DataFrame, pd.Index]:
    sider = load_dataset()
    old_columns = sider.columns.str.lower()
    sider.columns = sider.columns.str.lower().str.replace(" ", "_").str.replace(",", "")
    sider = sider.rename(columns={"smiles": "SMILES"})
    sider.to_csv("data_clean.csv", index=False)

    return sider, old_columns


def write_meta(column_ids: pd.Index, descriptions: List[str], num_points: int) -> None:
    # Write metadata
    targets = [
        {
            "id": f"{col_id}",
            "description": f"{description}",
            "type": "boolean",
            "names": [{"noun": f"{description}".lower()}],
        }
        for col_id, description in zip(column_ids[1:], descriptions[1:])
    ]

    templates = [
        "The {#molecule|compound|chemical|molecular species|chemical compound!} with the {SMILES__description}"
        + "{#representation of |!}{SMILES#} is {"
        + col_id
        + "#not a &a }"
        + "{#potential cause|potential reason!} for {"
        + col_id
        + "__names__noun}."  # noqa: E501
        for col_id in column_ids[1:]
    ]

    meta = {
        "name": "SIDER",  # unique identifier, we will also use this for directory names
        "description": """Database of marketed drugs and adverse drug reactions (ADR), grouped into 27 system organ classes.""",
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",
                "description": "SMILES",  # description (optional, except for "Other")
            }
        ],
        "targets": targets,
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://academic.oup.com/nar/article/44/D1/D1075/2502602?login=false",
                "description": "corresponding publication",
            },
            {
                "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
                "description": "Data source",
            },
        ],
        "num_points": num_points,  # number of datapoints in this dataset
        "bibtex": [
            """@article{10.1093/nar/gkv1075,
author = {Kuhn, Michael and Letunic, Ivica and Jensen, Lars Juhl and Bork, Peer},
title = "{The SIDER database of drugs and side effects}",
journal = {Nucleic Acids Research},
volume = {44},
number = {D1},
pages = {D1075-D1079},
year = {2015},
month = {10},
issn = {0305-1048},
doi = {10.1093/nar/gkv1075},
url = {https://doi.org/10.1093/nar/gkv1075},
}""",
        ],
        "templates": templates,
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

    return


def main():
    sider, old_columns = transform_data()
    write_meta(sider.columns, old_columns, len(sider))


if __name__ == "__main__":
    main()
