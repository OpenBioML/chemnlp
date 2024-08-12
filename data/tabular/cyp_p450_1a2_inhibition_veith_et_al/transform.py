import pandas as pd
import yaml
from tdc.single_pred import ADME


def get_and_transform_data():
    # get raw data
    target_subfolder = "CYP1A2_Veith"
    splits = ADME(name=target_subfolder).get_split()
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
    assert fields_orig == ["Drug_ID", "Drug", "Y", "split"]

    # overwrite column names = fields
    fields_clean = [
        "chembl_id",
        "SMILES",
        f"{target_subfolder.split('_')[0]}_inhibition",
        "split",
    ]

    df.columns = fields_clean

    # data cleaning
    df = df.dropna()
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)
    meta = {
        "name": "cyp_p450_1a2_inhibition_veith_et_al",  # unique identifier, we will also use this for directory names
        "description": """The CYP P450 genes are involved in the formation and breakdown (metabolism)
of various molecules and chemicals within cells. Specifically, CYP1A2 localizes
to the endoplasmic reticulum and its expression is induced by some polycyclic
aromatic hydrocarbons (PAHs), some of which are found in cigarette smoke. It
is able to metabolize some PAHs to carcinogenic intermediates. Other xenobiotic
substrates for this enzyme include caffeine, aflatoxin B1, and acetaminophen.""",
        "targets": [
            {
                "id": f"{target_subfolder.split('_')[0]}_inhibition",  # name of the column in a tabular dataset
                "description": "ability of the drug to inhibit CYP P450 1A2 (1) or not (0)",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "inhibition of CYP1A2"},
                    {"noun": "inhibition of CYP P450 1A2"},
                    {"adjective": "CYP1A2 inhibition"},
                    {"adjective": "CYP P450 1A2 inhibition"},
                    {"verb": "inhibits CYP P450 1A2"},
                    {"verb": "inhibits CYP1A2"},
                    {"gerund": "inhibiting CYP P450 1A2"},
                    {"gerund": "inhibiting CYP1A2"},
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
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "description": "SMILES",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1038/nbt.1581",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#cyp-p450-1a2-inhibition-veith-et-al",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Veith2009,
doi = {10.1038/nbt.1581},
url = {https://doi.org/10.1038/nbt.1581},
year = {2009},
month = oct,
publisher = {Springer Science and Business Media LLC},
volume = {27},
number = {11},
pages = {1050--1055},
author = {Henrike Veith and Noel Southall and Ruili Huang and Tim James
and Darren Fayne and Natalia Artemenko and Min Shen and James Inglese
and Christopher P Austin and David G Lloyd and Douglas S Auld},
title = {Comprehensive characterization of cytochrome P450 isozyme selectivity
across chemical libraries},
journal = {Nature Biotechnology}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} {#shows|exhibits|displays!} {CYP1A2_inhibition#no &NULL}{CYP1A2_inhibition__names__noun}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule {#shows|exhibits|displays!} {CYP1A2_inhibition#no &NULL}{CYP1A2_inhibition__names__noun}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} represents a molecule that {#shows|exhibits|displays!} {CYP1A2_inhibition#no &NULL}{CYP1A2_inhibition__names__noun}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {CYP1A2_inhibition#not &NULL}{CYP1A2_inhibition__names__gerund}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {CYP1A2_inhibition__names__gerund}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result: {CYP1A2_inhibition#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {CYP1A2_inhibition__names__gerund}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {CYP1A2_inhibition#not &NULL}{CYP1A2_inhibition__names__gerund}.""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {CYP1A2_inhibition__names__gerund}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the molecule with the {SMILES__description} {SMILES#} is {CYP1A2_inhibition__names__gerund}?
Assistant: {CYP1A2_inhibition#No&Yes}, this molecule is {CYP1A2_inhibition#not &NULL}{CYP1A2_inhibition__names__gerund}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {CYP1A2_inhibition__names__gerund}?
Assistant: {CYP1A2_inhibition#No&Yes}, it is {CYP1A2_inhibition#not &NULL}{CYP1A2_inhibition__names__gerund}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {CYP1A2_inhibition#not &NULL}{CYP1A2_inhibition__names__gerund}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {CYP1A2_inhibition#not &NULL}{CYP1A2_inhibition__names__gerund}?
Assistant: This is a molecule that is {CYP1A2_inhibition#not &NULL}{CYP1A2_inhibition__names__gerund}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {CYP1A2_inhibition#not &NULL}be {CYP1A2_inhibition__names__gerund}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {CYP1A2_inhibition#not &NULL}{CYP1A2_inhibition__names__gerund}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {CYP1A2_inhibition#not &NULL}be {CYP1A2_inhibition__names__gerund}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {CYP1A2_inhibition#not &NULL}{CYP1A2_inhibition__names__gerund}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {CYP1A2_inhibition__names__gerund}:<EOI>{CYP1A2_inhibition#no&yes}",  # noqa: E501 for the benchmarking setup <EOI>separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {CYP1A2_inhibition__names__gerund}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI>{CYP1A2_inhibition#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {CYP1A2_inhibition__names__gerund}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result:<EOI>This molecule is {CYP1A2_inhibition#not &NULL}{CYP1A2_inhibition__names__gerund}.""",  # noqa: E501
            # noqa: E501"""Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
            # Description: A molecule that is {CYP1A2_inhibition__names__gerund}.
            # Result:<EOI>{SMILES#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {CYP1A2_inhibition__names__gerund}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{CYP1A2_inhibition%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {CYP1A2_inhibition__names__gerund}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{CYP1A2_inhibition%}
Answer:<EOI>{%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {CYP1A2_inhibition#not &NULL}{CYP1A2_inhibition__names__gerund}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%CYP1A2_inhibition%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {CYP1A2_inhibition#not &NULL}{CYP1A2_inhibition__names__gerund}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%CYP1A2_inhibition%}
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
