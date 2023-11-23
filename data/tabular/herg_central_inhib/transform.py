import pandas as pd
import yaml
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list


def get_and_transform_data():
    # get raw data
    name = "herg_central"
    label_names = retrieve_label_name_list("herg_central")

    # select datasubset
    ln = label_names[2]  # herg_inhib
    print(ln)

    # get raw data
    splits = Tox(name=name, label_name=ln).get_split()
    df_train = splits["train"]
    df_valid = splits["valid"]
    df_test = splits["test"]
    df_train["split"] = "train"
    df_valid["split"] = "valid"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_valid, df_test], axis=0)

    fn_data_original = "data_original.csv"
    df.to_csv(fn_data_original, index=False)
    del df, df_train, df_valid, df_test

    # create dataframe
    df = pd.read_csv(
        fn_data_original,
        delimiter=",",
    )  # not necessary but ensure we can load the saved data

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == ["Drug_ID", "Drug", "Y", "split"]

    # overwrite column names = fields
    fields_clean = ["compound_id", "SMILES", "herg_inhib", "split"]
    df.columns = fields_clean

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "herg_central_inhib",
        "description": """Human ether-à-go-go related gene (hERG) is crucial for the coordination
of the heart's beating. Thus, if a drug blocks the hERG, it could lead to severe
adverse effects. Therefore, reliable prediction of hERG liability in the early
stages of drug design is quite important to reduce the risk of cardiotoxicity-related
attritions in the later development stages. There are three targets: hERG_at_1microM,
hERG_at_10microM, and herg_inhib.""",
        "targets": [
            {
                "id": "herg_inhib",
                "description": """whether it blocks (1) or not blocks (0) hERG
(This is equivalent to whether hERG_at_10microM < -50, i.e.,
whether the compound has an IC50 of less than 10microM.)""",
                "units": None,
                "type": "boolean",
                "names": [
                    {"noun": "hERG blocker"},
                    {"noun": "hERG blocking compound"},
                    {"noun": "hERG blocking compound (IC50 < 10uM)"},
                    {"noun": "hERG blocking compound (IC50 less than 10uM)"},
                    {"noun": "human ether-à-go-go related gene (hERG) blocker"},
                    {
                        "noun": "human ether-à-go-go related gene (hERG) blocking compound"
                    },
                    {"verb": "block hERG"},
                    {"verb": "block hERG (IC50 < 10uM)"},
                    {"verb": "block hERG (IC50 less than 10uM)"},
                    {"verb": "block the human ether-à-go-go related gene (hERG)"},
                ],
                "uris": [
                    "http://purl.obolibrary.org/obo/MI_2136",
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
                "url": "https://doi.org/10.1089/adt.2011.0425",
                "description": "corresponding publication",
            },
            {
                "url": "https://bbirnbaum.com/",
                "description": "TDC Contributer",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#herg-central",
                "description": "Data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Du2011,
doi = {10.1089/adt.2011.0425},
url = {https://doi.org/10.1089/adt.2011.0425},
year = {2011},
month = dec,
publisher = {Mary Ann Liebert Inc},
volume = {9},
number = {6},
pages = {580--588},
author = {Fang Du and Haibo Yu and Beiyan Zou and Joseph Babcock
and Shunyou Long and Min Li},
title = {hERGCentral: A Large Database to Store,  Retrieve,  and Analyze Compound Human
Ether-à-go-go Related Gene Channel Interactions to Facilitate Cardiotoxicity Assessment in Drug Development},
journal = {ASSAY and Drug Development Technologies}""",
        ],
        "templates": [
            # herg_inhib__names__noun
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} is {herg_inhib#not &NULL}a {herg_inhib__names__noun}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule is {herg_inhib#not &NULL}a {herg_inhib__names__noun}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} represents a molecule that is {herg_inhib#not &NULL}a {herg_inhib__names__noun}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {herg_inhib#not &NULL}a {herg_inhib__names__noun}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that {herg_inhib__names__verb}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result: {herg_inhib#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is a {herg_inhib__names__noun}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {herg_inhib#no &NULL}{herg_inhib__names__noun}.""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is a {herg_inhib__names__noun}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the molecule with the {SMILES__description} {SMILES#} is a {herg_inhib__names__noun}?
Assistant: {herg_inhib#No&Yes}, this molecule is {herg_inhib#not &NULL}a {herg_inhib__names__noun}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} a {herg_inhib__names__noun}?
Assistant: {herg_inhib#No&Yes}, it is {herg_inhib#not &NULL}a {herg_inhib__names__noun}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {herg_inhib#not &NULL}a {herg_inhib__names__noun}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that does {herg_inhib#not &NULL}{herg_inhib__names__verb}?
#Assistant: This is a molecule that is {herg_inhib#not &NULL}a {herg_inhib__names__noun}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {herg_inhib#not &NULL}be a {herg_inhib__names__noun}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {herg_inhib#not &NULL}a {herg_inhib__names__noun}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {herg_inhib#not &NULL}be a {herg_inhib__names__noun}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {herg_inhib#not &NULL}a {herg_inhib__names__noun}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} a {herg_inhib__names__noun}:<EOI> {herg_inhib#no&yes}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is a {herg_inhib__names__noun}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {herg_inhib#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is a {herg_inhib__names__noun}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result:<EOI> This molecule is {herg_inhib#not &NULL}a {herg_inhib__names__noun}.""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} a {herg_inhib__names__noun}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{herg_inhib%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {herg_inhib#not &NULL}a {herg_inhib__names__noun}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%herg_inhib%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {herg_inhib#not &NULL}a {herg_inhib__names__noun}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%herg_inhib%}
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
