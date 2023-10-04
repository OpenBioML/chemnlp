import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    splits = Tox(name="hERG").get_split()
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
        "compound_name",
        "SMILES",
        "herg_blocker",
        "split",
    ]
    df.columns = fields_clean

    # data cleaning
    df.compound_name = (
        df.compound_name.str.strip()
    )  # remove leading and trailing white space characters

    assert not df.duplicated().sum()

    # save to csv
    df.herg_blocker = df.herg_blocker.astype(int)
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "herg_blockers",  # unique identifier, we will also use this for directory names
        "description": """Human ether-à-go-go related gene (hERG) is crucial for the coordination
of the heart's beating. Thus, if a drug blocks the hERG, it could lead to severe
adverse effects. Therefore, reliable prediction of hERG liability in the early
stages of drug design is quite important to reduce the risk of cardiotoxicity
related attritions in the later development stages.""",
        "targets": [
            {
                "id": "herg_blocker",  # name of the column in a tabular dataset
                "description": "whether it blocks hERG (1) or not (0)",  # description of what this column means
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "hERG blocker"},
                    # {"noun": "hERG active compound"},
                    # {"noun": "hERG active compound (<10uM)"},
                    {"noun": "hERG blocking compound"},
                    {"noun": "hERG blocking compound (<10uM)"},
                    {"noun": "human ether-à-go-go related gene (hERG) blocker"},
                    {
                        "noun": "human ether-à-go-go related gene (hERG) blocking compound"
                    },
                    {"verb": "blocks hERG"},
                    {"verb": "blocks the human ether-à-go-go related gene (hERG)"},
                    {"verb": "is active against hERG (<10uM)"},
                    {
                        "verb": "is active against the human ether-à-go-go related gene (hERG)"
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
            {
                "id": "compound_name",
                "type": "Other",
                "description": "compound name",
                "names": [
                    {"noun": "compound"},
                    {"noun": "compound name"},
                    {"noun": "drug"},
                ],
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1021/acs.molpharmaceut.6b00471",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#herg-blockers",
                "description": "Data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Wang2016,
doi = {10.1021/acs.molpharmaceut.6b00471},
url = {https://doi.org/10.1021/acs.molpharmaceut.6b00471},
year = {2016},
month = jul,
publisher = {American Chemical Society (ACS)},
volume = {13},
number = {8},
pages = {2855--2866},
author = {Shuangquan Wang and Huiyong Sun and Hui Liu and Dan Li and
Youyong Li and Tingjun Hou},
title = {ADMET Evaluation in Drug Discovery. 16. Predicting hERG Blockers
by Combining Multiple Pharmacophores and Machine Learning Approaches},
journal = {Molecular Pharmaceutics}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} is {herg_blocker#not &NULL}a {herg_blocker__names__noun}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule is {herg_blocker#not &NULL}a {herg_blocker__names__noun}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} represents a molecule that is {herg_blocker#not &NULL}a {herg_blocker__names__noun}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {herg_blocker#not &NULL}a {herg_blocker__names__noun}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that {herg_blocker__names__verb}.
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
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that {herg_blocker#not &NULL}{herg_blocker__names__verb}?
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
            "Is the {SMILES__description} {SMILES#} a {herg_blocker__names__noun}:<EOI> {herg_blocker#no&yes}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is a {herg_blocker__names__noun}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {herg_blocker#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is a {herg_blocker__names__noun}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result:<EOI> This molecule is {herg_blocker#not &NULL}a {herg_blocker__names__noun}.""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} a {herg_blocker__names__noun}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{herg_blocker%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
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
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
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
