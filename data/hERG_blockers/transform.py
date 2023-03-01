import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    data = Tox(name = 'hERG')
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
        "compound_name",
        "SMILES",
        "hERG_blocker",
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
        "name": "hERG",  # unique identifier, we will also use this for directory names
        "description": """Human ether-à-go-go related gene (hERG) is crucial for the coordination of the heart's beating. Thus, if a drug blocks the hERG, it could lead to severe adverse effects. Therefore, reliable prediction of hERG liability in the early stages of drug design is quite important to reduce the risk of cardiotoxicity-related attritions in the later development stages.""",
        "targets": [
            {
                "id": "hERG_blocker",  # name of the column in a tabular dataset
                "description": "hERG active compound(blocker)",  # description of what this column means
                "units": "ld50",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "hERG activity",
                    "hERG active compound",
                    "hERG blocker",
                    "Human ether-à-go-go related gene (hERG) blocker",
                    "Activity against Human ether-à-go-go related gene (hERG)",
                ],
            },
        ],
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "description": "SMILES",  # description (optional, except for "Other")
            },
            {
                "id": "compound_name",
                "type": "Other",
                "description": "compound name",
                "names": [
                    "compound",
                    "compound name",
                    "drug",
                ],
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1021/acs.molpharmaceut.6b00471",
                "description": "corresponding publication",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "url": "https://tdcommons.ai/single_pred_tasks/tox/#herg-blockers",
        "bibtex": [
            """@article{Wang2016,
      doi = {10.1021/acs.molpharmaceut.6b00471},
      url = {https://doi.org/10.1021/acs.molpharmaceut.6b00471},
      year = {2016},
      month = jul,
      publisher = {American Chemical Society ({ACS})},
      volume = {13},
      number = {8},
      pages = {2855--2866},
      author = {Shuangquan Wang and Huiyong Sun and Hui Liu and Dan Li and Youyong Li and Tingjun Hou},
      title = {{ADMET} Evaluation in Drug Discovery. 16. Predicting {hERG} Blockers by Combining Multiple Pharmacophores and Machine Learning Approaches},
      journal = {Molecular Pharmaceutics}}""",
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
