import pandas as pd
import yaml
from tdc.single_pred import HTS


def get_and_transform_data():
    # get raw data
    splits = HTS(name="HIV").get_split()
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
        "activity_HIV",
        "split",
    ]
    df.columns = fields_clean

    # data cleaning
    df.compound_id = (
        df.compound_id.str.strip()
    )  # remove leading and trailing white space characters

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "hiv",  # unique identifier, we will also use this for directory names
        "description": """The HIV dataset was introduced by the Drug Therapeutics Program (DTP)
AIDS Antiviral Screen, which tested the ability to inhibit HIV replication for
over 40,000 compounds.""",
        "targets": [
            {
                "id": "activity_HIV",  # name of the column in a tabular dataset
                "description": "whether it is active against HIV virus (1) or not (0).",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "activity against the human immunodeficiency virus"},
                    {"noun": "activity against HIV"},
                    {"adjective": "active against the human immunodeficiency virus"},
                    {"adjective": "active against HIV"},
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
                "url": "https://rb.gy/wphpqg",
                "description": "data source",
            },
            {
                "url": "https://rb.gy/0xx91v",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/hts/#hiv",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Wu2018,
doi = {10.1039/c7sc02664a},
url = {https://doi.org/10.1039/c7sc02664a},
year = {2018},
publisher = {Royal Society of Chemistry (RSC)},
volume = {9},
number = {2},
pages = {513--530},
author = {Zhenqin Wu and Bharath Ramsundar and Evan~N. Feinberg and Joseph Gomes
and Caleb Geniesse and Aneesh S. Pappu and Karl Leswing and Vijay Pande},
title = {MoleculeNet: a benchmark for molecular machine learning},
journal = {Chemical Science}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} {#shows|exhibits|displays!} {activity_HIV#no &NULL}{activity_HIV__names__noun}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule {#shows|exhibits|displays!} {activity_HIV#no &NULL}{activity_HIV__names__noun}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is from!} a molecule that {#shows|exhibits|displays!} {activity_HIV#no &NULL}{activity_HIV__names__noun}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {activity_HIV#not &NULL}{activity_HIV__names__adjective}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_HIV__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result: {activity_HIV#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_HIV__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {activity_HIV#not &NULL}{activity_HIV__names__adjective}.""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {activity_HIV__names__adjective}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the molecule with the {SMILES__description} {SMILES#} is {activity_HIV__names__adjective}?
Assistant: {activity_HIV#No&Yes}, this molecule is {activity_HIV#not &NULL}{activity_HIV__names__adjective}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {activity_HIV__names__adjective}?
Assistant: {activity_HIV#No&Yes}, it is {activity_HIV#not &NULL}{activity_HIV__names__adjective}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {activity_HIV#not &NULL}{activity_HIV__names__adjective}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {activity_HIV#not &NULL}{activity_HIV__names__adjective}?
Assistant: This is a molecule that is {activity_HIV#not &NULL}{activity_HIV__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {activity_HIV#not &NULL}be {activity_HIV__names__adjective}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {activity_HIV#not &NULL}{activity_HIV__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {activity_HIV#not &NULL}be {activity_HIV__names__adjective}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {activity_HIV#not &NULL}{activity_HIV__names__adjective}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {activity_HIV__names__adjective}:<EOI> {activity_HIV#no&yes}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_HIV__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {activity_HIV#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_HIV__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result:<EOI> This molecule is {activity_HIV#not &NULL}{activity_HIV__names__adjective}.""",  # noqa: E501
            # noqa: E501 """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
            # Description: A molecule that is {activity_HIV__names__adjective}.
            # Result:<EOI> {SMILES#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {activity_HIV__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{activity_HIV%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {activity_HIV#not &NULL}{activity_HIV__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%activity_HIV%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {activity_HIV#not &NULL}{activity_HIV__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%activity_HIV%}
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
