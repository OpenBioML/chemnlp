import pandas as pd
import yaml
from tdc.generation import MolGen

DATASET_NAME = "moses"


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
        "description": """Molecular Sets (MOSES) is a benchmark platform
for distribution learning based molecule generation.
Within this benchmark, MOSES provides a cleaned dataset of molecules that are ideal of optimization.
It is processed from the ZINC Clean Leads dataset.""",
        "benchmarks": [
            {
                "name": "TDC",  # unique benchmark name
                "link": "https://tdcommons.ai/",  # benchmark URL
                "split_column": "split",  # name of the column that contains the split information
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
                "url": "https://arxiv.org/abs/1811.12823",
                "description": "Article about original dataset",
            },
            {
                "url": "https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559",
                "description": "Link to publication of associated dataset - zinc",
            },
            {
                "url": "https://github.com/molecularsets/moses",
                "description": "Github repository concering the datset",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{10.3389/fphar.2020.565644,
title={{M}olecular {S}ets ({MOSES}): {A} {B}enchmarking {P}latform for {M}olecular {G}eneration {M}odels},
author={Polykovskiy, Daniil and Zhebrak, Alexander and Sanchez-Lengeling, Benjamin and Golovanov,
Sergey and Tatanov, Oktai and Belyaev, Stanislav and Kurbanov, Rauf and Artamonov,
Aleksey and Aladinskiy, Vladimir and Veselov, Mark and Kadurin, Artur and Johansson,
Simon and  Chen, Hongming and Nikolenko, Sergey and Aspuru-Guzik, Alan and Zhavoronkov, Alex},
journal={Frontiers in Pharmacology},
year={2020}
}""",
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
