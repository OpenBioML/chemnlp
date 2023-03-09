import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    data = Tox(name = 'Carcinogens_Lagunin')
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
    fields_clean =[
        "compound_id",
        "SMILES",
        "carcinogen",
    ]
    df.columns = fields_clean

    # data cleaning
#     df.compound_id = (
#         df.compound_id.str.strip()
#     )  # remove leading and trailing white space characters

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta =  {
        "name": "Carcinogens",  # unique identifier, we will also use this for directory names
        "description": """A carcinogen is any substance, radionuclide, or radiation that promotes carcinogenesis, the formation of cancer. This may be due to the ability to damage the genome or to the disruption of cellular metabolic processes.""",
        "targets": [
            {
                "id": "carcinogen",  # name of the column in a tabular dataset
                "description": "whether it can cause carcinogen (1) or not (0).",  # description of what this column means
                "units": "carcinogen",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "carcinogen",
                    "promotes carcinogenesis",
                    "carcinogenesis",
                    "any substance, radionuclide, or radiation that promotes carcinogenesis",
                    "damage the genome",
                    "substance promotes carcinogenesis"
                ],
                "uris":[
                "https://bioportal.bioontology.org/ontologies/NCIT?p=classes&conceptid=http%3A%2F%2Fncicb.nci.nih.gov%2Fxml%2Fowl%2FEVS%2FThesaurus.owl%23C347",
                 "https://bioportal.bioontology.org/ontologies/SNOMEDCT?p=classes&conceptid=http%3A%2F%2Fpurl.bioontology.org%2Fontology%2FSNOMEDCT%2F88376000"
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
                "url": "https://doi.org/10.1002/qsar.200860192",
                "description": "corresponding publication",
            },
            {
                "url": "https://doi.org/10.1021/ci300367a",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#carcinogens",
                "description": "Data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Lagunin2009,
          doi = {10.1002/qsar.200860192},
          url = {https://doi.org/10.1002/qsar.200860192},
          year = {2009},
          month = jun,
          publisher = {Wiley},
          volume = {28},
          number = {8},
          pages = {806--810},
          author = {Alexey Lagunin and Dmitrii Filimonov and Alexey Zakharov and Wei Xie and Ying Huang and Fucheng Zhu and Tianxiang Shen and Jianhua Yao and Vladimir Poroikov},
          title = {Computer-Aided Prediction of Rodent Carcinogenicity by {PASS} and {CISOC}-{PSCT}},
          journal = {{QSAR} {\&}amp$\mathsemicolon$ Combinatorial Science}}"""
            ,
            """@article{Cheng2012,
              doi = {10.1021/ci300367a},
              url = {https://doi.org/10.1021/ci300367a},
              year = {2012},
              month = nov,
              publisher = {American Chemical Society ({ACS})},
              volume = {52},
              number = {11},
              pages = {3099--3105},
              author = {Feixiong Cheng and Weihua Li and Yadi Zhou and Jie Shen and Zengrui Wu and Guixia Liu and Philip W. Lee and Yun Tang},
              title = {{admetSAR}: A Comprehensive Source and Free Tool for Assessment of Chemical {ADMET} Properties},
              journal = {Journal of Chemical Information and Modeling}}""",
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
