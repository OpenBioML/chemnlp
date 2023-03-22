import pandas as pd
import yaml
from tdc.single_pred import ADME


def get_and_transform_data():
    # get raw data
    splits = ADME(name="Caco2_Wang").get_split()
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
        "Y",
        "split",
    ]

    # overwrite column names = fields
    fields_clean = [
        "compound_name",
        "SMILES",
        "permeability",
        "split",
    ]
    df.columns = fields_clean

    # data cleaning
    df.compound_name = (
        df.compound_name.str.strip()
    )  # remove leading and trailing white space characters

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "caco2_wang",  # unique identifier, we will also use this for directory names
        "description": """The human colon epithelial cancer cell line, Caco-2,
is used as an in vitro model to simulate the human intestinal tissue.
The experimental result on the rate of drug passing through
the Caco-2 cells can approximate the rate at which the drug permeates
through the human intestinal tissue.""",
        "targets": [
            {
                "id": "permeability",  # name of the column in a tabular dataset
                "description": "Caco-2 cell effective permeability.",  # description of what this column means
                "units": "cm/s",
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "Caco-2 cell effective permeability",
                    "Caco-2 cell permeability",
                    "Caco-2 permeability",
                ],
                "pubchem_aids": [678378],
                "uris": [
                    "http://www.bioassayontology.org/bao#BAO_0010008",
                    "http://purl.obolibrary.org/obo/MI_2162",
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
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
                "description": "SMILES",  # description (optional, except for "OTHER")
            },
            {
                "id": "compound_name",
                "type": "Other",
                "description": "compound name",
                "names": [
                    "compound",
                    "compound name",
                ],
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#caco-2-cell-effective-permeability-wang-et-al",
                "description": "original data set link",
            },
            {
                "url": "https://pubs.acs.org/doi/10.1021/acs.jcim.5b00642",
                "description": "corresponding publication",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{wang2016adme,
title={ADME properties evaluation in drug discovery: prediction of Caco-2 cell permeability
using a combination of NSGA-II and boosting},
author={Wang, Ning-Ning and Dong, Jie and Deng, Yin-Hua and Zhu, Min-Feng and Wen, Ming and Yao,
Zhi-Jiang and Lu, Ai-Ping and Wang, Jian-Bing and Cao, Dong-Sheng},
journal={Journal of Chemical Information and Modeling},
volume={56},
number={4},
pages={763--773},
year={2016},
publisher={ACS Publications}
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
