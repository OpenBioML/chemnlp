import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    splits = Tox(name="AMES").get_split()
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
        "mutagenic",
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
        "name": "ames_mutagenicity",  # unique identifier, we will also use this for directory names
        "description": """Mutagenicity means the ability of a drug to induce genetic alterations.
Drugs that can cause damage to the DNA can result in cell death or other severe
adverse effects. Nowadays, the most widely used assay for testing the mutagenicity
of compounds is the Ames experiment which was invented by a professor named
Ames. The Ames test is a short term bacterial reverse mutation assay detecting
a large number of compounds which can induce genetic damage and frameshift mutations.
The dataset is aggregated from four papers.""",
        "targets": [
            {
                "id": "mutagenic",  # name of the column in a tabular dataset
                "description": "whether it is mutagenic (1) or not mutagenic (0)",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "mutagenicity"},
                    {"noun": "Ames mutagenicity"},
                    {"adjective": "mutagenic"},
                    {"adjective": "Ames mutagenic"},
                    {"verb": "has the ability to induce genetic alterations"},
                    {"gerund": "having the potential to cause mutations"},
                    {"gerund": "having the potential to induce genetic alterations"},
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
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1021/ci300400a",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#ames-mutagenicity",
                "description": "Data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Xu2012,
doi = {10.1021/ci300400a},
url = {https://doi.org/10.1021/ci300400a},
year = {2012},
month = oct,
publisher = {American Chemical Society (ACS)},
volume = {52},
number = {11},
pages = {2840--2847},
author = {Congying Xu and Feixiong Cheng and Lei Chen and
Zheng Du and Weihua Li and Guixia Liu and Philip W. Lee and Yun Tang},
title = {In silico Prediction of Chemical Ames Mutagenicity},
journal = {Journal of Chemical Information and Modeling}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} {#shows|exhibits|displays!} {mutagenic#no &NULL}{mutagenic__names__adjective} properties.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule has {mutagenic#no &NULL}{mutagenic__names__adjective} {#properties|characteristics|features!}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is from!} a molecule that is {mutagenic#not &NULL}identified as {mutagenic__names__adjective}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} is {mutagenic#no &NULL}{mutagenic__names__adjective}.",
            "The {#molecule |!}{SMILES__description} {SMILES#} is {mutagenic__names__adjective}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {mutagenic__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result: {mutagenic#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {mutagenic__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {mutagenic#not &NULL}{mutagenic__names__adjective}.""",
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {mutagenic__names__adjective}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the molecule with the {SMILES__description} {SMILES#} is {mutagenic__names__adjective}?
Assistant: {mutagenic#No&Yes}, this molecule is {mutagenic#not &NULL}{mutagenic__names__adjective}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {mutagenic__names__adjective}?
Assistant: {mutagenic#No&Yes}, it is {mutagenic#not &NULL}{mutagenic__names__adjective}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {mutagenic#not &NULL}{mutagenic__names__adjective}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {mutagenic#not &NULL}{mutagenic__names__adjective}?
Assistant: This is a molecule that is {mutagenic#not &NULL}{mutagenic__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {mutagenic#not &NULL}be {mutagenic__names__adjective}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {mutagenic#not &NULL}{mutagenic__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {mutagenic#not &NULL}be {mutagenic__names__adjective}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {mutagenic#not &NULL}{mutagenic__names__adjective}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {mutagenic__names__adjective}:<EOI> {mutagenic#yes&no}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {mutagenic__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {mutagenic#False&True}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {mutagenic__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{mutagenic%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {mutagenic__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{mutagenic%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {mutagenic#not &NULL}{mutagenic__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%mutagenic%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {mutagenic#not &NULL}{mutagenic__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%mutagenic%}
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
