import pandas as pd
import yaml
from tdc.single_pred import Develop


def get_and_transform_data():
    # get raw data
    target_subfolder = "SAbDab_Chen"
    splits = Develop(name=target_subfolder).get_split()
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

    # proceed raw data
    df = pd.read_csv(fn_data_raw, sep=",")

    fields_orig = df.columns.tolist()
    assert fields_orig == ["Antibody_ID", "Antibody", "Y", "split"]

    fn_data_original = "data_original.csv"

    antibody_list = df.Antibody.tolist()

    def s2l(list_string):
        return list(map(str.strip, list_string.strip("][").replace("'", "").split(",")))

    df["heavy_chain"] = [s2l(x)[0] for x in antibody_list]
    df["light_chain"] = [s2l(x)[1] for x in antibody_list]
    df = df[["Antibody_ID", "heavy_chain", "light_chain", "Y", "split"]]
    df.to_csv(fn_data_original, index=False)

    #  load raw data and assert columns
    df = pd.read_csv(fn_data_original, sep=",")
    fields_orig = df.columns.tolist()
    assert fields_orig == ["Antibody_ID", "heavy_chain", "light_chain", "Y", "split"]
    fields_clean = [
        "antibody_pdb_ID",
        "heavy_chain",
        "light_chain",
        "developability",
        "split",
    ]
    df.columns = fields_clean
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    meta = {
        "name": "sabdab_chen",  # unique identifier, we will also use this for directory names
        "description": """Antibody data from Chen et al, where they process from the SAbDab.
From an initial dataset of 3816 antibodies, they retained 2426 antibodies that
satisfy the following criteria: 1.have both sequence (FASTA) and Protein Data
Bank (PDB) structure files, 2. contain both a heavy chain and a light chain,
and 3. have crystal structures with resolution < 0.3 nm. The DI label is derived
from BIOVIA's pipelines.""",
        "targets": [
            {
                "id": "developability",  # name of the column in a tabular dataset
                "description": "functional antibody candidate to be developed into a manufacturable one (1) or not (0)",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    {
                        "noun": "functional antibody candidate to be developed into a manufacturable one"
                    },
                    {"noun": "manufacturable and functional antibody candidate"},
                ],
                "uris": None,
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
                "id": "antibody_pdb_ID",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": [
                    "pdb id",
                    "Protein Data Bank id",
                ],
                "description": "anitbody pdb id",  # description (optional, except for "Other")
            },
            {
                "id": "heavy_chain",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": [
                    "amino acid sequence",
                    "heavy chain amino acid sequence",
                    "heavy chain AA sequence",
                ],
                "description": "anitbody heavy chain amino acid sequence in FASTA",
            },
            {
                "id": "light_chain",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": [
                    "amino acid sequence",
                    "light chain amino acid sequence",
                    "light chain AA sequence",
                ],
                "description": "anitbody light chain amino acid sequence in FASTA",
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1101/2020.06.18.159798",
                "description": "corresponding publication",
            },
            {
                "url": "https://doi.org/10.1093/nar/gkt1043",
                "description": "corresponding publication",
            },
            {
                "url": "https://www.3ds.com/products-services/biovia/products/data-science/pipeline-pilot/",
                "description": "corresponding tools used",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/develop/#sabdab-chen-et-al",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Chen2020,
doi = {10.1101/2020.06.18.159798},
url = {https://doi.org/10.1101/2020.06.18.159798},
year = {2020},
month = jun,
publisher = {Cold Spring Harbor Laboratory},
author = {Xingyao Chen and Thomas Dougherty and
Chan Hong and Rachel Schibler and Yi Cong Zhao and
Reza Sadeghi and Naim Matasci and Yi-Chieh Wu and Ian Kerman},
title = {Predicting Antibody Developability from Sequence
using Machine Learning}""",
            """@article{Dunbar2013,
doi = {10.1093/nar/gkt1043},
url = {https://doi.org/10.1093/nar/gkt1043},
year = {2013},
month = nov,
publisher = {Oxford University Press ({OUP})},
volume = {42},
number = {D1},
pages = {D1140--D1146},
author = {James Dunbar and Konrad Krawczyk and Jinwoo Leem
and Terry Baker and Angelika Fuchs and Guy Georges and Jiye Shi and
Charlotte M. Deane},
title = {SAbDab: the structural antibody database},
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
