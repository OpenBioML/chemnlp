import pandas as pd
import yaml
from tdc.single_pred import Epitope


def get_and_transform_data():
    # get raw data
    target_subfolder = "PDB_Jespersen"
    splits = Epitope(name=target_subfolder).get_split()
    df_train = splits["train"]
    df_valid = splits["valid"]
    df_test = splits["test"]
    df_train["split"] = "train"
    df_valid["split"] = "valid"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_valid, df_test], axis=0)

    fn_data_raw = "data_raw.csv"
    df.to_csv(fn_data_raw, index=False)
    del df

    def get_active_position(seq, active_position, sequence_only=False):
        """
        Input: given a sequence and list of active index
        Output: return active sequence and other sequence convert to _
        MASQKRPS ,[1,2,3,4,6] -> _ASQK_P_
        """
        if isinstance(
            active_position, str
        ):  # if list is casted to string after loading from raw csv data file.
            active_position = [int(x) for x in active_position[1:-1].split(", ")]

        if sequence_only:
            _seq = "".join([seq[x] for x in active_position])
            return _seq

        _seq = ["_" for a in range(len(seq))]
        for x in active_position:
            _seq[x] = seq[x]
        _seq = "".join(_seq)
        return _seq

    # proceed raw data
    df = pd.read_csv(fn_data_raw, sep=",")
    fields_orig = df.columns.tolist()
    assert fields_orig == ["Antigen_ID", "Antigen", "Y", "split"]

    # Rename columns of raw data
    fields_clean = [
        "Antigen_ID",
        "Antigen_sequence",
        "active_positions_indices",
        "split",
    ]
    df.columns = fields_clean

    # get active position
    antigen_seq = df.Antigen_sequence.tolist()
    a_pos_ind_list = df.active_positions_indices.tolist()
    df["active_position"] = [
        get_active_position(x, o, sequence_only=True)
        for x, o in zip(antigen_seq, a_pos_ind_list)
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
        "split",
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
        "name": "pdb_jespersen_et_al",  # unique identifier, we will also use this for directory names
        "description": """Epitope prediction is to predict the active region in the antigen.
This dataset is from Bepipred, which curates a dataset from PDB.
It collects B-cell epitopes and non-epitope amino acids determined from crystal structures.""",
        "targets": [
            {
                "id": "active_position",  # name of the column in a tabular dataset
                "description": "amino acids sequence position that is active in binding",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "epitope"},
                    {"noun": "amino acids sequence active in antigen binding"},
                    {"noun": "epitope sequence active in antigen binding"},
                    {"noun": "epitope sequence active in binding"},
                ],
                "uris": [
                    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C13189",
                ],
            }
        ],
        "identifiers": [
            {
                "id": "Antigen_sequence",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": [
                    "amino acid sequence",
                    "AA sequence",
                    "epitope amino acid sequence",
                    "epitope AA sequence",
                ],
                "description": "amino acid sequence",  # description (optional, except for "Other")
            }
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1093/nar/gkx346",
                "description": "corresponding publication",
            },
            {
                "url": "https://doi.org/10.1093/nar/28.1.235",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/epitope/#pdb-jespersen-et-al",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
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
            """@article{Berman2000,
doi = {10.1093/nar/28.1.235},
url = {https://doi.org/10.1093/nar/28.1.235},
year = {2000},
month = jan,
publisher = {Oxford University Press (OUP)},
volume = {28},
number = {1},
pages = {235--242},
author = {H. M. Berman},
title = {The Protein Data Bank},
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
