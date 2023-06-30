import pandas as pd
import yaml
import requests

def get_and_transform_data():
    target_folder = "toxcast_with_description"
    url = "https://huggingface.co/datasets/phalem/awesome_chem_clean_data/resolve/main/Toxcast_with_description.csv.gz"
    df = pd.read_csv(url, compression='gzip')
    fn_data_original = 'data_original.csv'
    df.to_csv(fn_data_original, index=None)
    df = pd.read_csv(fn_data_original, delimiter=",")
    df = df.dropna()
    fields_orig = df.columns.tolist()
    assert fields_orig == ['Drug_ID',
     'Drug',
     'Y',
     'Target',
     'assay_desc',
     'organism',
     'tissue',
     'cell_format']

    fields_clean = ['Drug_ID',
     'SMILES',
     'Toxicity',
     'Assay',
     'assay_desc',
     'organism',
     'tissue',
     'cell_format']

    df.columns = fields_clean
    assert fields_orig != fields_clean
    df = df.drop_duplicates()
    assert not df.duplicated().sum()
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    meta = {
        "name": f"{target_folder}",  # unique identifier, we will also use this for directory names
        "description": """ToxCast includes qualitative results of over 600 experiments on 8k compounds.""",
        "targets": [
            {
                "id": "Toxicity",  # name of the column in a tabular dataset
                "description": "Presence (1) or absence(0) of a compound toxicity in a specific assay.",  # description of what this column means
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
            {
                "id": "assay_desc",  # name of the column in a tabular dataset
                "description": "Description of the Assay where the compound tested",  # description of what this column means
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "text",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "Toxicity description"},
                    {"noun": "Toxicity assay description"},
                    {"noun": "Experimental assay description"},
                    {"noun": "Description of a given assay"},
                ],
                      "uris":[
            "https://ncit.nci.nih.gov/ncitbrowser/ConceptReport.jsp?dictionary=NCI_Thesaurus&ns=ncit&code=C60819",
                    ],
            },
                        {
                    "id": "Organism",  # name of the column in a tabular dataset
                    "description": "cell origin for which assay is tested",  # description of what this column means
                    "units": "",  # units of the values in this column (leave empty if unitless)
                    "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                    "names": [  # names for the property (to sample from for building the prompts).
                    {"noun": "The organism that the cell isolate from"},
                    {"noun": "For which organism cell related to"},
                    {"noun": "living that the cell isolate from"},
                    ],
                      "uris":[
                    "http://purl.bioontology.org/ontology/CCON", #organism
            ],
                },
            {
                "id": "tissue",  # name of the column in a tabular dataset
                "description": "Tissue where cell isolated from",  # description of what this column means
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "Tissue"},
                    {"noun": "body"},
                    {"noun": "Experimental assay description"},
                ],
                      "uris":[
                          "http://purl.obolibrary.org/obo/NCIT_C12801"
                    ],
            },
            {
                "id": "cell_format",  # name of the column in a tabular dataset
                "description": "cell format",  # description of what this column means
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "cell format"},

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
                "url": "https://doi.org/10.23645/epacomptox.6062623.v10",
                "description": "data source",
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
    #        """@misc{https://doi.org/10.23645/epacomptox.6062623.v8,
    #   doi = {10.23645/EPACOMPTOX.6062623.V8},
    #   url = {https://epa.figshare.com/articles/dataset/ToxCast_Database_invitroDB_/6062623/8},
    #   author = {Toxicology,  EPA's National Center for Computational},
    #   keywords = {Toxicology (incl. clinical toxicology)},
    #   title = {ToxCast Database (invitroDB)},
    #   publisher = {The United States Environmental Protection Agency’s Center for Computational Toxicology and Exposure},
    #   year = {2022},
    #   copyright = {Creative Commons Zero v1.0 Universal}""" 

        ]
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
