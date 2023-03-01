import pandas as pd
import yaml
from tdc.single_pred import ADME


def get_and_transform_data():
    # get raw data
    data = ADME(name="PAMPA_NCATS")
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
        "compound_id",
        "SMILES",
        "permeability",
    ]
    df.columns = fields_clean

    # data cleaning
    df.drop(columns=["compound_id"], inplace=True)
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "pampa_ncats",  # unique identifier, we will also use this for directory names
        "description": """PAMPA (parallel artificial membrane permeability assay) is a commonly
        employed assay to evaluate drug permeability across the cellular membrane. PAMPA is a
        non-cell-based, low-cost and high-throughput alternative to cellular models. Although
        PAMPA does not model active and efflux transporters, it still provides permeability values
        that are useful for absorption prediction because the majority of drugs are absorbed by
        passive diffusion through the membrane.""",
        "targets": [
            {
                "id": "permeability",  # name of the column in a tabular dataset
                "description": "Binary permeability in PAMPA assay.",  # description of what this column means
                "units": "Bool",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "binary permeability in PAMPA assay",
                    "permeability in PAMPA assay",
                    "PAMPA permeability",
                ],
            },
        ],
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "description": "SMILES",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#pampa-permeability-ncats",
                "description": "original dataset link",
            },
            {
                "url": "https://journals.sagepub.com/doi/full/10.1177/24725552211017520",
                "description": "corresponding publication",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{siramshetty2021validating,
    title={Validating ADME QSAR Models Using Marketed Drugs},
    author={Siramshetty, Vishal and Williams, Jordan and Nguyen, DHac-Trung and Neyra, Jorge and Southall,
    Noel and Math'e, Ewy and Xu, Xin and Shah, Pranav},
    journal={SLAS DISCOVERY: Advancing the Science of Drug Discovery},
    volume={26},
    number={10},
    pages={1326--1336},
    year={2021},
    publisher={SAGE Publications Sage CA: Los Angeles, CA}
    }""",
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
