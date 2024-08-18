import pandas as pd
import yaml
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list

def get_and_transform_data():
    label = 'ToxCast'

    df_all = []
    toxcast_list = retrieve_label_name_list('Toxcast')
    for x in toxcast_list:
        df = Tox(name = label, label_name = x).get_data(format = 'df')
        df['Assay'] = x
        df_all.append(df)

    df_data = pd.concat(df_all)
    fn_data_original = 'data_original.csv'
    df_data.to_csv(fn_data_original, index=None)
    df = pd.read_csv(fn_data_original, delimiter=",")
    df = df.dropna()
    fields_orig = df.columns.tolist()
    assert fields_orig == ['Drug_ID', 'Drug', 'Y', 'Assay']
    fields_clean = ['Drug_ID', 'SMILES', 'Toxicity', 'Assay']
    df.columns = fields_clean
    assert fields_orig != fields_clean
    assert not df.duplicated().sum()
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    meta = {
        "name": "toxcast",  # unique identifier, we will also use this for directory names
        "description": """ToxCast includes qualitative results of over 600 experiments on 8k compounds.""",
        "targets": [
            {
                "id": "Toxicity",  # name of the column in a tabular dataset
                "description": "Presence of a compound toxicity in a specific assay.",  # description of what this column means
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "Toxicity"},
                    {"noun": "Toxic compound"},
                    {"noun": "Toxicity in specific assay"},
                ],
            },
            {
                "id": "Assay",  # name of the column in a tabular dataset
                "description": "Assay where the compound tested",  # description of what this column means
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "Toxicity"},
                    {"noun": "Toxicity assay"},
                    {"noun": "Experimental assay"},

                ],
                      "uris":[
            "https://ncit.nci.nih.gov/ncitbrowser/ConceptReport.jsp?dictionary=NCI_Thesaurus&ns=ncit&code=C60819",
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
                "url": "https://doi.org/10.1021/acs.chemrestox.6b00135",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#toxcast",
                "description": "data source",
            }
        ],
        "num_points": len(df),  # number of datapoints in this dataset   
        "bibtex": [
            """@article{Richard2016,
              doi = {10.1021/acs.chemrestox.6b00135},
              url = {https://doi.org/10.1021/acs.chemrestox.6b00135},
              year = {2016},
              month = jul,
              publisher = {American Chemical Society (ACS)},
              volume = {29},
              number = {8},
              pages = {1225--1251},
              author = {Ann M. Richard and Richard S. Judson and Keith A. Houck and Christopher M. Grulke and Patra Volarath and Inthirany Thillainadarajah and Chihae Yang and James Rathman and Matthew T. Martin and John F. Wambaugh and Thomas B. Knudsen and Jayaram Kancherla and Kamel Mansouri and Grace Patlewicz and Antony J. Williams and Stephen B. Little and Kevin M. Crofton and Russell S. Thomas},
              title = {ToxCast Chemical Landscape: Paving the Road to 21st Century Toxicology},
              journal = {Chemical Research in Toxicology}""",
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
