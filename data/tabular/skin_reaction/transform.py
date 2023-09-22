import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    splits = Tox(name="Skin Reaction").get_split()
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
        "skin_reaction",
        "split",
    ]
    df.columns = fields_clean

    # data cleaning
    df.compound_name = (
        df.compound_name.str.strip()
    )  # remove leading and trailing white space characters

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "skin_reaction",  # unique identifier, we will also use this for directory names
        "description": """Repetitive exposure to a chemical agent can induce an immune reaction
in inherently susceptible individuals that leads to skin sensitization. The
dataset used in this study was retrieved from the ICCVAM (Interagency Coordinating
Committee on the Validation of Alternative Methods) report on the rLLNA.""",
        "targets": [
            {
                "id": "skin_reaction",  # name of the column in a tabular dataset
                "description": "whether it can cause skin reaction (1) or not (0).",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "skin reaction"},
                    {"noun": "skin sensitization"},
                    {"noun": "agent induced skin reaction"},
                    {"noun": "drug induced skin immune reaction"},
                    {"verb": "causes skin reaction"},
                    {"verb": "causes skin sensitization"},
                    {"verb": "causes drug induced skin immune reaction"},
                    {"adjective": "reactive in contact with the skin"},
                    {"gerund": "causing skin reaction"},
                ],
                "uris": [
                    "http://purl.bioontology.org/ontology/MEDDRA/10040914",
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
                "description": "drug name",
                "names": [
                    {"noun": "compound"},
                    {"noun": "compound name"},
                ],
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1016/j.taap.2014.12.014",
                "description": "corresponding publication",
            },
            {
                "url": "https://ntp.niehs.nih.gov/iccvam/docs/immunotox_docs/llna-ld/tmer.pdf",
                "description": "related publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#skin-reaction",
                "description": "Data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Alves2015,
doi = {10.1016/j.taap.2014.12.014},
url = {https://doi.org/10.1016/j.taap.2014.12.014},
year = {2015},
month = apr,
publisher = {Elsevier BV},
volume = {284},
number = {2},
pages = {262--272},
author = {Vinicius M. Alves and Eugene Muratov and Denis Fourches and Judy Strickland
and Nicole Kleinstreuer and Carolina H. Andrade and Alexander Tropsha},
title = {Predicting chemically-induced skin reactions. Part I: QSAR models of skin sensitization
and their application to identify potentially hazardous compounds},
journal = {Toxicology and Applied Pharmacology}""",
        ],
        "templates": [
            "The {#molecule |!}{SMILES__description} {SMILES#} causes {skin_reaction#no &NULL}reaction with the skin.",  # noqa: E501
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} is not {skin_reaction#no &NULL}{skin_reaction__names__adjective}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, there is {skin_reaction#no &NULL}reaction in contact with the skin.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is from!} a molecule that is {skin_reaction#not &NULL}identified as {skin_reaction__names__adjective}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {skin_reaction#not &NULL}{skin_reaction__names__adjective}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {skin_reaction__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result: {skin_reaction#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {skin_reaction__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete|definite!} sentence.
Result: This molecule is {skin_reaction#not &NULL}{skin_reaction__names__adjective}.""",
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {skin_reaction__names__adjective}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the molecule with the {SMILES__description} {SMILES#} is {skin_reaction__names__adjective}?
Assistant: {skin_reaction#No&Yes}, this molecule is {skin_reaction#not &NULL}{skin_reaction__names__adjective}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {skin_reaction__names__adjective}?
Assistant: {skin_reaction#No&Yes}, it is {skin_reaction#not &NULL}{skin_reaction__names__adjective}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {skin_reaction#not &NULL}{skin_reaction__names__adjective}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {skin_reaction#not &NULL}{skin_reaction__names__adjective}?
Assistant: This is a molecule that is {skin_reaction#not &NULL}{skin_reaction__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {skin_reaction#not &NULL}be {skin_reaction__names__adjective}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {skin_reaction#not &NULL}{skin_reaction__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {skin_reaction#not &NULL}be {skin_reaction__names__adjective}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {skin_reaction#not &NULL}{skin_reaction__names__adjective}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {skin_reaction__names__adjective}:<EOI> {skin_reaction#yes&no}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {skin_reaction__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {skin_reaction#False&True}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {skin_reaction__names__adjective}.
Result:<EOI> {SMILES#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {skin_reaction__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{skin_reaction%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {skin_reaction__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{skin_reaction%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {skin_reaction#not &NULL}{skin_reaction__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%skin_reaction%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {skin_reaction#not &NULL}{skin_reaction__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%skin_reaction%}
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
