import pandas as pd
import yaml
from tdc.multi_pred import TCREpitopeBinding 

def get_and_transform_data():
    # get raw data
    data = TCREpitopeBinding(name = 'weber', path = './data')
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
        "epitope_aa",
        "epitope_smi",
        "tcr",
        "tcr_aa",
        "label",
    ]

    # overwrite column names = fields
    fields_clean = [
        "epitope_aa",
        "epitope_smiles",
        "tcr",
        "tcr_aa",
        "binding",
    ]
    df.columns = fields_clean

    # data cleaning
    df.epitope_aa = (
        df.epitope_aa.str.strip()
    )  # remove leading and trailing white space characters

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "tcr_epitope_binding",  # unique identifier, we will also use this for directory names
        "description": """T-cells are an integral part of the adaptive immune system, whose survival, proliferation, activation
        and function are all governed by the interaction of their T-cell receptor (TCR) with immunogenic peptides (epitopes).
        A large repertoire of T-cell receptors with different specificity is needed to provide protection against a wide range of pathogens.
        This new task aims to predict the binding affinity given a pair of TCR sequence and epitope sequence.""",
        "targets": [
            {
                "id": "binding",  # name of the column in a tabular dataset
                "description": "TCR epitope binding.",  # description of what this column means
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "binary classification",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "tcr binding affinity",
                    "binding affinity",
                    "binding",

                ],
            },
        ],
        "identifiers": [
            {
                "id": "epitope_smiles",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
                "description": "epitope smiles",  # description (optional, except for "OTHER")
            },
            {
                "id": "epitope_aa",
                "type": "Other",
                "description": "epitope amino acid sequence",
    
            },
            {
                "id": "tcr_full_aa",
                "type": "Other",
                "description": "tcr amino acid sequence",
    
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://tdcommons.ai/multi_pred_tasks/tcrepitope/",
                "description": "original data set link",
            },
            {
                "url": "https://doi.org/10.1093/bioinformatics/btab294",
                "description": "corresponding publication",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{weber2021titan,
            title={TITAN: T-cell receptor specificity prediction with bimodal attention network},
            author={Weber Anna,Born Janis, Martinez Maria Rodriguez},
            journal={Bioinformatics},
            volume={56},
            number={4},
            pages={i237-i234},
            year={2021},
            publisher={Oxford Academic}
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
