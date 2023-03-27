import pandas as pd
import yaml
from tdc.single_pred import Develop
from tdc.utils import retrieve_label_name_list


def get_and_transform_data():
    # get raw data
    target_subfolder = "TAP"
    label_list = retrieve_label_name_list(target_subfolder)
    data = Develop(name=target_subfolder, label_name=label_list[0])
    # proceed raw data
    df = pd.read_csv("data/tap.tab", sep="\t")
    fields_orig = df.columns.tolist()
    assert fields_orig == ["X", "ID", "CDR_Length", "PSH", "PPC", "PNC", "SFvCSP"]
    fields_clean = [
        "antibody_two_sequences",
        "antibody_name",
        "CDR_Length",
        "PSH",
        "PPC",
        "PNC",
        "SFvCSP",
    ]
    df.columns = fields_clean
    #  convert list columns to two columns
    antibody_list = df.antibody_two_sequences.tolist()
    s2l = lambda list_string: list(
        map(str.strip, list_string.strip("][").replace("'", "").split(","))
    )
    antibody2list = lambda list_string: [
        x.strip() for x in s2l(list_string)[0].split("\\n")
    ]
    df["heavy_chain"] = [antibody2list(x)[0] for x in antibody_list]
    df["light_chain"] = [antibody2list(x)[1] for x in antibody_list]
    fn_data_original = "data_original.csv"
    df.to_csv(fn_data_original, index=None)

    #  load raw data and assert columns
    df = pd.read_csv(fn_data_original, sep=",")
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "antibody_two_sequences",
        "antibody_name",
        "CDR_Length",
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
                "description": "CDR Complementarity-determining regions length", 
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "Antibody Complementarity-determining regions length",
                    "Therapeutic Antibody Profiler",
                    "antibody developability",
                    "monoclonal anitbody",
                ],
                "uris": [
                    "https://rb.gy/s9gv88",
                    "https://rb.gy/km77hq",
                    "https://rb.gy/b8cx8i",
                ],
            },
            {
                "id": "PSH",  # name of the column in a tabular dataset
                "description": "patches of surface hydrophobicity",  
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "antibody patches of surface hydrophobicity",
                    "Therapeutic Antibody Profiler",
                    "antibody developability",
                    "monoclonal anitbody",
                ],
                "uris": [
                    "https://rb.gy/bchhaa",
                    "https://rb.gy/2irr4l",
                    "https://rb.gy/b8cx8i",
                ],
            },
            {
                "id": "PPC",  # name of the column in a tabular dataset
                "description": "patches of positive charge", 
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "patches of positive charge",
                    "Therapeutic Antibody Profiler",
                    "antibody developability",
                    "monoclonal anitbody",
                ],
                "uris": [
                    "https://rb.gy/b8cx8i",
                ],
            },
            {
                "id": "PNC",  # name of the column in a tabular dataset
                "description": "patches of negative charge", 
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "anitbody patches of negative charge",
                    "Therapeutic Antibody Profiler",
                    "antibody developability",
                    "monoclonal anitbody",
                ],
                "uris": [
                    "https://rb.gy/b8cx8i",
                ],
            },
            {
                "id": "SFvCSP",  # name of the column in a tabular dataset
                "description": "structural Fv charge symmetry parameter",
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "antibody structural Fv charge symmetry parameter",
                    "Therapeutic Antibody Profiler",
                    "antibody developability",
                    "monoclonal anitbody",
                ],
                "uris": [
                    "https://rb.gy/uxyhc3",
                    "https://rb.gy/b8cx8i",
                ],
            },
        ],
        "benchmarks": [
        {
            "name": "TDC",  # unique benchmark name
            "link": "https://tdcommons.ai/",  # benchmark URL
            "split_column": "split",  # name of the column that contains the split information
        },
        ],
        "identifiers": [
            {
                "id": "antibody_name",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names":[
                "Name of the antibody",
                         "Name of the antibody drug",
                         "Name of drug"
                ],
                "description": "anitbody name",  # description (optional, except for "Other")
            },
            {
                "id": "heavy_chain",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names":[
                "Fastq",
                "gene sequence",
                ],
                "description": "anitbody heavy chain amino acid sequence",  # description (optional, except for "Other")
            },
            {
                "id": "light_chain",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names":[
                "Fastq",
                "gene sequence",
                ],
                "description": "anitbody light chain amino acid sequence",  # description (optional, except for "Other")
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
