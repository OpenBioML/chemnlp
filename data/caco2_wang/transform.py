from tdc.single_pred import ADME
import pandas as pd
import yaml


def get_and_transform_data():
    # get raw data
    data = ADME(name = 'Caco2_Wang')
    fn_data_original = "data_original.csv"
    data.get_data().to_csv(fn_data_original, index=False)

    # create dataframe
    df = pd.read_csv(fn_data_original, delimiter=",")  # not necessary but ensure we can load the saved data

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
    df.compound_id = (
        df.compound_id.str.strip()
    )  # remove leading and trailing white space characters
    
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "caco2_wang",  # unique identifier, we will also use this for directory names
        "description": "The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.",
        "targets": [
            {
                "id": "permeability",  # name of the column in a tabular dataset
                "description": "Caco-2 cell effective permeability.",  # description of what this column means
                "units": "?",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "Caco-2 cell effective permeability",
                    "Caco-2 cell permeability",
                    "Caco-2 permeability",
                    "permeability",
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
                "id": "compound_id",
                "type": "OTHER",
                "description": "Compound id / name",
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            "https://tdcommons.ai/single_pred_tasks/adme/#caco-2-cell-effective-permeability-wang-et-al",
            "https://pubs.acs.org/doi/10.1021/acs.jcim.5b00642",
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "url": "https://tdcommons.ai/single_pred_tasks/adme/#caco-2-cell-effective-permeability-wang-et-al",
        "bibtex": [
            """@article{wang2016adme,
            title={ADME properties evaluation in drug discovery: prediction of Caco-2 cell permeability using a combination of NSGA-II and boosting},
            author={Wang, Ning-Ning and Dong, Jie and Deng, Yin-Hua and Zhu, Min-Feng and Wen, Ming and Yao, Zhi-Jiang and Lu, Ai-Ping and Wang, Jian-Bing and Cao, Dong-Sheng},
            journal={Journal of Chemical Information and Modeling},
            volume={56},
            number={4},
            pages={763--773},
            year={2016},
            publisher={ACS Publications}
            }""",
        ],
    }
    fn_meta = "meta.yaml"
    with open(fn_meta, "w") as f:
        yaml.dump(meta, f, sort_keys=False)

    print(f"Finished processing {meta['name']} dataset!")


if __name__ == "__main__":
    get_and_transform_data()
