import pandas as pd
import yaml
from tdc.single_pred import Epitope


def get_and_transform_data():
    # get raw data
    target_folder = "IEDB_Jespersen_et_al"
    target_subfolder = "IEDB_Jespersen"
    data = Epitope(name=target_subfolder)

    def get_active_position(seq, active_poisition, sequence_only=False):
        """
        Input: given a sequence and list of active index
        Output: return active sequence and other sequence convert to _
        MASQKRPS ,[1,2,3,4,6] -> _ASQK_P_
        """
        if sequence_only:
            _seq = "".join([seq[x] for x in active_poisition])
            return _seq
        _seq = ["_" for a in range(len(seq))]
        for x in active_poisition:
            _seq[x] = seq[x]
        _seq = "".join(_seq)
        return _seq

    df = pd.read_pickle("data/iedb_jespersen.pkl")
    fields_orig = df.columns.tolist()
    assert fields_orig == ["ID", "X", "Y"]

    # Rename columns of raw data
    fields_clean = ["Antigen_ID", "Antigen_sequence", "active_positions_indices"]
    df.columns = fields_clean

    # get active position
    antigen_seq = df.Antigen_sequence.tolist()
    a_pos_ind_list = df.active_positions_indices.tolist()
    df["active_position"] = [
        get_active_position(x, o) for x, o in zip(antigen_seq, a_pos_ind_list)
    ]

    # save data to original
    fn_data_original = "data_original.csv"
    df.to_csv(fn_data_original, index=None)
    df = pd.read_csv(fn_data_original, sep=",")
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "Antigen_ID",
        "Antigen_sequence",
        "active_positions_indices",
        "active_position",
    ]

    # get right columns

    df = df[["Antigen_sequence", "active_position"]]
    fields_clean = ["Antigen_sequence", "active_position"]
    df.columns = fields_clean
    assert fields_orig != fields_clean
    assert not df.duplicated().sum()
    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    meta = {
        "name": "iedb_jespersen_et_al",  # unique identifier, we will also use this for directory names
        "description": """Epitope prediction is to predict the active region in the antigen.
This dataset is from Bepipred, which curates a dataset from IEDB. It collects B-cell
epitopes and non-epitope amino acids determined from crystal structures.""",
        "targets": [
            {
                "id": "active_position",  # name of the column in a tabular dataset
                "description": "amino acids sequence position that is active in binding",  # description of what this column means
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "amino acids sequence active in binding",
                    "Epitope",
                ],
                "uris": [
                    "https://bioportal.bioontology.org/ontologies/NCIT?p=classes&conceptid=http%3A%2F%2Fncicb.nci.nih.gov%2Fxml%2Fowl%2FEVS%2FThesaurus.owl%23C13189",
                ],
            }
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
                "id": "Antigen_sequence",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": [
                "amino acid sequence",
                "FASTQ",
                "fastq sequence",
                "Protien sequence"
                ],
                "description": "amino acid sequence",  # description (optional, except for "Other")
            }
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1093/nar/gky1006",
                "description": "corresponding publication",
            },
            {
                "url": "https://doi.org/10.1093/nar/gkx346",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/epitope/#iedb-jespersen-et-al",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Vita2018,
doi = {10.1093/nar/gky1006},
url = {https://doi.org/10.1093/nar/gky1006},
year = {2018},
month = oct,
publisher = {Oxford University Press (OUP)},
volume = {47},
number = {D1},
pages = {D339--D343}},
author = {Randi Vita and Swapnil Mahajan and James A Overton and
Sandeep Kumar Dhanda and Sheridan Martini and Jason R Cantrell and
Daniel K Wheeler and Alessandro Sette and Bjoern Peters},
title = {The Immune Epitope Database (IEDB): 2018 update},
journal = {Nucleic Acids Research}""",
            """@article{Jespersen2017,
doi = {10.1093/nar/gkx346},
url = {https://doi.org/10.1093/nar/gkx346},
year = {2017},
month = may,
publisher = {Oxford University Press (OUP)},
volume = {45},
number = {W1},
pages = {W24--W29},
author = {Martin Closter Jespersen and Bjoern Peters and Morten Nielsen and Paolo Marcatili},
title = {BepiPred 2.0: improving sequence-based B-cell epitope prediction using
conformational epitopes},
journal = {Nucleic Acids Research}""",
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
