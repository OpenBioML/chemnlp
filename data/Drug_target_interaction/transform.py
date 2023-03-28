import pandas as pd
import yaml
from tdc.multi_pred import DTI

def get_and_transform_data():
    # get raw data
    data = DTI(name="BindingDB_Kd")
    splits = data.get_split()
    df_train = splits["train"]
    df_valid = splits["valid"]
    df_test = splits["test"]
    df_train["split"] = "train"
    df_valid["split"] = "valid"
    df_test["split"] = "test"

    df = pd.concat([df_train, df_valid, df_test], axis=0)

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "Drug_ID",
        "Drug",
        "Target_ID",
        "Target",
        "Y",
        "split",
    ]

    # overwrite column names = fields
    fields_clean = [
        "compound_name",
        "SMILES",
        "target_name",
        "Target_aa",
        "binding",
        "split"
    ]
    df.columns = fields_clean

    # data cleaning
    '''
    df.compound_name = (
        df.compound_name.str.strip()
    )  # remove leading and trailing white space characters
    '''
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "Drug-Target Interaction",  # unique identifier, we will also use this for directory names
        "description": """The activity of a small-molecule drug is measured by its binding affinity with the target protein.
        Given a new target protein, the very first step is to screen a set of potential compounds to find their activity.
        Traditional method to gauge the affinities are through high-throughput screening wet-lab experiments.
        However, they are very expensive and are thus restricted by their abilities to search over a large set of candidates
        Drug-target interaction prediction task aims to predict the interaction activity score in silico given only the accessible compound structural information and protein amino acid sequence.""",
        "targets": [
            {
                "id": "binding",  # name of the column in a tabular dataset
                "description": "small-molecule protein interaction.",  # description of what this column means
                "units": "Kd",  # units of the values in this column (leave empty if unitless)
                "type": "regression",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "Drug-Target Interaction"
                    "small-molecule binding affinity",
                    "small-molecule binding",
                    "protein-ligand binding",
                    "protein-ligand"
                    "binding affinity",
                    "binding",

                ],
            },
        ],
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
                "description": "small-molecule",  # description (optional, except for "OTHER")
            },
            {
                "id": "Target",
                "type": "Other",
                "description": "Target amino acid sequence",
    
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://tdcommons.ai/multi_pred_tasks/dti/",
                "description": "original data set link",
            },
            {
                "url": "https://doi.org/10.1093/nar/gkl999",
                "description": "corresponding publication",
            },
        ],
        "benchmarks": [
            {
                "name": "TDC",
                "link": "https://tdcommons.ai/",
                "split_column": "split",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Liu2006bindingdb,
            title={BindingDB: a web-accessible database of experimentally determined protein-ligand binding affinities},
            author={Tiqing Liu, Yuhmei Lin, Xin Wen, Robert N. Jorissen, Micahel, K. Gilson},
            journal={Journal of Chemical Information and Modeling},
            volume={35},
            number={4},
            pages={D198-D201},
            year={2006},
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
