import pandas as pd
import yaml
from tdc.single_pred import Develop
from tdc.utils import retrieve_label_name_list


def get_and_transform_data():
    # get raw data
    target_subfolder = "TAP"
    label_list = retrieve_label_name_list(target_subfolder)
    df = pd.DataFrame()
    for i, label in enumerate(label_list):
        print(f"Get data subset {label}:")
        splits = Develop(name=target_subfolder, label_name=label).get_split()
        df_train = splits["train"]
        df_valid = splits["valid"]
        df_test = splits["test"]
        df_train["split"] = "train"
        df_valid["split"] = "valid"
        df_test["split"] = "test"
        df_cat = pd.concat([df_train, df_valid, df_test], axis=0)
        assert df_cat.columns.tolist() == ["Antibody_ID", "Antibody", "Y", "split"]
        df_cat.columns = ["Antibody_ID", "Antibody", label, "split"]
        if i > 0:
            df = pd.merge(df, df_cat, on=["Antibody_ID", "Antibody", "split"])
        else:
            df = df_cat

    fn_data_raw = "data_raw.csv"
    df.to_csv(fn_data_raw, index=False)
    del df

    # proceed raw data
    df = pd.read_csv(fn_data_raw, sep=",")
    fields_orig = df.columns.tolist()

    assert fields_orig == [
        "Antibody_ID",
        "Antibody",
        "CDR_Length",
        "split",
        "PSH",
        "PPC",
        "PNC",
        "SFvCSP",
    ]
    fields_clean = [
        "antibody_name",
        "antibody_sequences",
        "CDR_Length",
        "split",
        "PSH",
        "PPC",
        "PNC",
        "SFvCSP",
    ]
    df.columns = fields_clean
    #  convert list columns to two columns
    antibody_list = df.antibody_sequences.tolist()

    def s2l(list_string):
        return list(map(str.strip, list_string.strip("][").replace("'", "").split(",")))

    def antibody2list(list_string):
        return [x.strip() for x in s2l(list_string)[0].split("\\n")]

    df["heavy_chain"] = [antibody2list(x)[0] for x in antibody_list]
    df["light_chain"] = [antibody2list(x)[1] for x in antibody_list]
    fn_data_original = "data_original.csv"
    df.to_csv(fn_data_original, index=None)

    #  load raw data and assert columns
    df = pd.read_csv(fn_data_original, sep=",")
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "antibody_name",
        "antibody_sequences",
        "CDR_Length",
        "split",
        "PSH",
        "PPC",
        "PNC",
        "SFvCSP",
        "heavy_chain",
        "light_chain",
    ]

    df = df[
        [
            "antibody_name",
            "heavy_chain",
            "light_chain",
            "CDR_Length",
            "split",
            "PSH",
            "PPC",
            "PNC",
            "SFvCSP",
        ]
    ]
    fields_clean = [
        "antibody_name",
        "heavy_chain",
        "light_chain",
        "CDR_Length",
        "split",
        "PSH",
        "PPC",
        "PNC",
        "SFvCSP",
    ]

    df.columns = fields_clean
    assert fields_orig != fields_clean
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    meta = {
        "name": "tap",  # unique identifier, we will also use this for directory names
        "description": """Immunogenicity, instability, self-association,
high viscosity, polyspecificity, or poor expression can all preclude
an antibody from becoming a therapeutic. Early identification of these
negative characteristics is essential. Akin to the Lipinski guidelines,
which measure druglikeness in small molecules,
Therapeutic Antibody Profiler (TAP) highlights antibodies
that possess characteristics that are rare/unseen in
clinical-stage mAb therapeutics.""",
        "targets": [
            {
                "id": "CDR_Length",  # name of the column in a tabular dataset
                "description": "complementarity-determining regions (CDR) length",
                "units": "amino acids",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "antibody complementarity-determining regions length",
                    "antibody complementarity-determining regions (CDR) length",
                    "antibody CDR length",
                    "complementarity-determining regions (CDR) length",
                    "complementarity-determining regions length",
                    "CDR length",
                ],
                "uris": None,
            },
            {
                "id": "PSH",  # name of the column in a tabular dataset
                "description": "patches of surface hydrophobicity (PSH) score",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "antibody patches of surface hydrophobicity (PSH) score",
                    "antibody patches of surface hydrophobicity score",
                    "antibody PSH score",
                    "patches of surface hydrophobicity (PSH) score",
                    "patches of surface hydrophobicity score",
                    "PSH score",
                ],
                "uris": None,
            },
            {
                "id": "PPC",  # name of the column in a tabular dataset
                "description": "patches of positive charge (PPC) score",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "antibody patches of positive charge (PPC) score",
                    "antibody patches of positive charge score",
                    "antibody PPC score",
                    "patches of positive charge (PPC) score",
                    "patches of positive charge score",
                    "PPC score",
                ],
                "uris": None,
            },
            {
                "id": "PNC",  # name of the column in a tabular dataset
                "description": "patches of negative charge (PNC) score",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "antibody patches of negative charge (PNC) score",
                    "antibody patches of negative charge score",
                    "antibody PNC score",
                    "patches of negative charge (PNC) score",
                    "patches of negative charge score",
                    "PNC score",
                ],
                "uris": None,
            },
            {
                "id": "SFvCSP",  # name of the column in a tabular dataset
                "description": "structural Fv charge symmetry parameter (SFvCSP) score",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "antibody structural Fv charge symmetry parameter (SFvCSP) score",
                    "antibody structural Fv charge symmetry parameter score",
                    "antibody SFvCSP score",
                    "structural Fv charge symmetry parameter (SFvCSP) score",
                    "structural Fv charge symmetry parameter score",
                    "SFvCSP score",
                ],
                "uris": None,
            },
        ],
        "identifiers": [
            {
                "id": "antibody_name",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": [
                    "antibody name",
                    "name of the antibody",
                    "name of the antibody drug",
                ],
                "description": "antibody name",  # description (optional, except for "Other")
            },
            {
                "id": "heavy_chain",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": [
                    "amino acid sequence",
                    "heavy chain amino acid sequence",
                    "heavy chain AA sequence",
                ],
                "description": "antibody heavy chain amino acid sequence",  # description (optional, except for "Other")
            },
            {
                "id": "light_chain",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": [
                    "amino acid sequence",
                    "light chain amino acid sequence",
                    "light chain AA sequence",
                ],
                "description": "antibody light chain amino acid sequence",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1073/pnas.1810576116",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/develop/#tap",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Raybould2019,
doi = {10.1073/pnas.1810576116},
url = {https://doi.org/10.1073/pnas.1810576116},
year = {2019},
month = feb,
publisher = {Proceedings of the National Academy of Sciences},
volume = {116},
number = {10},
pages = {4025--4030},
author = {Matthew I. J. Raybould and Claire Marks and Konrad Krawczyk
and Bruck Taddese and Jaroslaw Nowak and Alan P. Lewis and Alexander Bujotzek
and Jiye Shi and Charlotte M. Deane},
title = {Five computational developability guidelines for therapeutic antibody profiling},
journal = {Proceedings of the National Academy of Sciences}""",
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
