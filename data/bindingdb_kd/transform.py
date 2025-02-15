import pandas as pd
import yaml
from tdc.multi_pred import DTI

def get_and_transform_data():
    target_subfolder ="BindingDB_Kd"
    target_folder = str(target_subfolder).lower()
    splits = DTI(name = target_subfolder).get_split()
    df_train = splits["train"]
    df_valid = splits["valid"]
    df_test = splits["test"]
    df_train["split"] = "train"
    df_valid["split"] = "valid"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_valid, df_test], axis=0)

    fn_data_original = "data_original.csv"
    df.to_csv(fn_data_original, index=False)
    del df
    # create dataframe
    df = pd.read_csv(
        fn_data_original,
        delimiter=",",
    )  # not necessary but ensure we can load the saved data
    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == ['Drug_ID', 'Drug', 'Target_ID', 'Target', 'Y', 'split']
    # overwrite column names = fields
    fields_clean = ['Drug_ID', 'SMILES', 'Target_ID', 'Target', 'BindingDB_Kd', 'split']
    df.columns = fields_clean
    # data cleaning
    # remove leading and trailing white space characters
    df = df.dropna()
    assert not df.duplicated().sum()
    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    meta = {
            "name": "bindingdb_kd",  # unique identifier, we will also use this for directory names
            "description": """BindingDB is a public, web-accessible database of
            measured binding affinities, focusing chiefly on the interactions of
            protein considered to be drug-targets with small, drug-like molecules.""",
            "targets": [
                {
                    "id": "BindingDB_Kd",  # name of the column in a tabular dataset
                    "description": "binding affinity of the given compound for a given target or protein",
                    "units": "KD",  # units of the values in this column (leave empty if unitless)
                    "type": "continuous",
                    "names": [  # names for the property (to sample from for building the prompts)
                        {"noun": "The strength of binding of a single molecule to its ligand"},
                        {"noun": "Drug potency for certain protein target"},
                        {"verb": "Inhibit certain protein"},
                        {"verb": "Change the functionality and protein conformation"},
                        {"adjective": "Inhibition of certain protein target"},
                        {
                            "adjective": "Inhibition of certain protein target to change its function"
                        },
                    ],
                    "uris": [
                        "http://purl.obolibrary.org/obo/MI_0646"

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
                    "type": "SMILES",
                    "description": "SMILES",  # description (optional, except for "Other")
                },
                {
                    "id": "Target_ID",  # column name
                    "type": "Other",
                    "names": [
                        {"noun": "protein target id"},
                        {"noun": "protein id"},
                    ],
                    "description": "protein target id",  # description (optional, except for "Other")
                },
                {
                    "id": "Target",  # column name
                    "type": "Other",
                    "names": [
                        {"noun": "protein sequence"},
                        {"noun": "protein fastq"},
                    ],
                    "description": "protein sequence in fastq",  # description (optional, except for "Other")
                },            
            ],
            "license": "CC BY 3.0 US.",  # license under which the original dataset was published
            "links": [  # list of relevant links (original dataset, other uses, etc.)
                {
                    "url": "https://doi.org/10.1093/nar/gkl999",
                    "description": "corresponding publication",
                },
                {
                    "url": "https://arxiv.org/abs/2004.08919",
                    "description": "corresponding publication",
                },
                {
                    "url": "https://tdcommons.ai/single_pred_tasks/adme/#bbb-blood-brain-barrier-martins-et-al",
                    "description": "data source",
                },
            ],
            "num_points": len(df),  # number of datapoints in this dataset
            "bibtex": [
     """@article{https://doi.org/10.48550/arxiv.2004.08919,
    doi = {10.48550/ARXIV.2004.08919},
    url = {https://arxiv.org/abs/2004.08919},
    author = {Huang,  Kexin and Fu,  Tianfan and Glass,  Lucas and Zitnik,  Marinka and Xiao,  Cao and Sun,  Jimeng},
    keywords = {Machine Learning (cs.LG),  Quantitative Methods (q-bio.QM),  Machine Learning (stat.ML),  FOS: Computer and information sciences,  FOS: Computer and information sciences,  FOS: Biological sciences,  FOS: Biological sciences},
    title = {DeepPurpose: a Deep Learning Library for Drug-Target Interaction Prediction},
    publisher = {arXiv},
    year = {2020},
    copyright = {Creative Commons Attribution 4.0 International}""",
                """@article{Liu2007,
    doi = {10.1093/nar/gkl999},
    url = {https://doi.org/10.1093/nar/gkl999},
    year = {2007},
    month = jan,
    publisher = {Oxford University Press ({OUP})},
    volume = {35},
    number = {Database},
    pages = {D198--D201},
    author = {T. Liu and Y. Lin and X. Wen and R. N. Jorissen and M. K. Gilson},
    title = {BindingDB: a web-accessible database of experimentally determined protein-ligand binding affinities},
    journal = {Nucleic Acids Research}""",
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
