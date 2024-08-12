import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    splits = Tox(name="hERG_Karim").get_split()
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
    assert fields_orig == [
        "Drug_ID",
        "Drug",
        "Y",
        "split",
    ]

    # overwrite column names = fields
    fields_clean = [
        "compound_id",
        "SMILES",
        "herg_blocker",
        "split",
    ]
    df.columns = fields_clean

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "herg_karim_et_al",  # unique identifier, we will also use this for directory names
        "description": """A integrated Ether-à-go-go-related gene (hERG) dataset consisting
of molecular structures labelled as hERG (<10uM) and non-hERG (>=10uM) blockers in
the form of SMILES strings was obtained from the DeepHIT, the BindingDB database,
ChEMBL bioactivity database, and other literature.""",
        "targets": [
            {
                "id": "herg_blocker",
                "description": "whether it blocks hERG (1, <10uM) or not (0, >=10uM)",
                "units": None,
                "type": "boolean",
                "names": [
                    {"noun": "hERG blocker (<10uM)"},
                    {"noun": "hERG blocking compound (<10uM)"},
                    {"noun": "human ether-à-go-go related gene (hERG) blocker (<10uM)"},
                    {
                        "noun": "human ether-à-go-go related gene (hERG) blocking compound (<10uM)",
                    },
                    {"verb": "block hERG (<10uM)"},
                    {
                        "verb": "block the human ether-à-go-go related gene (hERG) (<10uM)"
                    },
                ],
                "uris": [
                    "http://purl.obolibrary.org/obo/MI_2136",
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
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "description": "SMILES",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1186/s13321-021-00541-z",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#herg-karim-et-al",
                "description": "Data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Karim2021,
doi = {10.1186/s13321-021-00541-z},
url = {https://doi.org/10.1186/s13321-021-00541-z},
year = {2021},
month = aug,
publisher = {Springer Science and Business Media LLC},
volume = {13},
number = {1},
author = {Abdul Karim and Matthew Lee and Thomas Balle and Abdul Sattar},
title = {CardioTox net: a robust predictor for hERG channel blockade
based on deep learning meta-feature ensembles},
journal = {Journal of Cheminformatics}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} is {herg_blocker#not a hERG blocker (>= 10uM)&a hERG blocker (<10uM)}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule is {herg_blocker#not a human ether-à-go-go related gene (hERG) blocker (>= 10uM)&a human ether-à-go-go related gene (hERG) blocker (<10uM)}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} represents a molecule that is {herg_blocker#not a hERG blocker (>= 10uM)&a hERG blocker (<10uM)}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {herg_blocker#not a human ether-à-go-go related gene (hERG) blocker (>= 10uM)&a human ether-à-go-go related gene (hERG) blocker (<10uM)}.",  # noqa: E501
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that does {herg_blocker__names__verb}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result: {herg_blocker#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is a {herg_blocker__names__noun}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {herg_blocker#no &NULL}{herg_blocker__names__noun}.""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is a {herg_blocker__names__noun}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the molecule with the {SMILES__description} {SMILES#} is a {herg_blocker__names__noun}?
Assistant: {herg_blocker#No&Yes}, this molecule is {herg_blocker#not &NULL}a {herg_blocker__names__noun}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} a {herg_blocker__names__noun}?
Assistant: {herg_blocker#No&Yes}, it is {herg_blocker#not &NULL}a {herg_blocker__names__noun}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {herg_blocker#not &NULL}a {herg_blocker__names__noun}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that does {herg_blocker#not &NULL}{herg_blocker__names__verb}?
Assistant: This is a molecule that is {herg_blocker#not &NULL}a {herg_blocker__names__noun}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {herg_blocker#not &NULL}be a {herg_blocker__names__noun}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {herg_blocker#not &NULL}a {herg_blocker__names__noun}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {herg_blocker#not &NULL}be a {herg_blocker__names__noun}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {herg_blocker#not &NULL}a {herg_blocker__names__noun}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} a {herg_blocker__names__noun}:<EOI>{herg_blocker#no&yes}",  # noqa: E501 for the benchmarking setup <EOI>separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is a {herg_blocker__names__noun}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI>{herg_blocker#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is a {herg_blocker__names__noun}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result:<EOI>This molecule is {herg_blocker#not &NULL}a {herg_blocker__names__noun}.""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} a {herg_blocker__names__noun}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{herg_blocker%}
Answer:<EOI>{%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {herg_blocker#not &NULL}a {herg_blocker__names__noun}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%herg_blocker%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {herg_blocker#not &NULL}a {herg_blocker__names__noun}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%herg_blocker%}
Answer:<EOI>{%multiple_choice_result}""",  # noqa: E501
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
