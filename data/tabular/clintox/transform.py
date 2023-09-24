import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    splits = Tox(name="ClinTox").get_split()
    df_train = splits["train"]
    df_valid = splits["valid"]
    df_test = splits["test"]
    df_train["split"] = "train"
    df_valid["split"] = "valid"
    df_test["split"] = "test"

    df = pd.concat([df_train, df_valid, df_test], axis=0)

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == ["Drug_ID", "Drug", "Y", "split"]

    # overwrite column names = fields
    fields_clean = ["compound_id", "SMILES", "clinical_toxicity", "split"]
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
        "name": "clintox",  # unique identifier, we will also use this for directory names
        "description": """The ClinTox dataset includes drugs that have failed
clinical trials for toxicity reasons and also drugs that are associated
with successful trials.""",
        "targets": [
            {
                "id": "clinical_toxicity",  # name of the column in a tabular dataset
                "description": "whether it can cause clinical toxicity (1) or not (0).",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "toxicity"},
                    {"noun": "clinical toxicity"},
                    {"adjective": "toxic"},
                    {"adjective": "clinically toxic"},
                    {"gerund": "displaying clinical toxicity"},
                ],
                "uris": [
                    "http://purl.bioontology.org/ontology/MESH/Q000633",
                    "https://ncit.nci.nih.gov/ncitbrowser/ConceptReport.jsp?dictionary=NCI_Thesaurus&ns=ncit&code=C27990",  # noqa: E501
                    "https://ncit.nci.nih.gov/ncitbrowser/ConceptReport.jsp?dictionary=NCI_Thesaurus&ns=ncit&code=C27955",  # noqa: E501
                ],
            },
        ],
        "benchmarks": [
            {
                "name": "TDC",
                "link": "https://tdcommons.ai/",
                "split_column": "split",
            }
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
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#clintox",
                "description": "original dataset",
            },
            {
                "url": "https://doi.org/10.1016/j.chembiol.2016.07.023",
                "description": "corresponding publication",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Gayvert2016,
doi = {10.1016/j.chembiol.2016.07.023},
url = {https://doi.org/10.1016/j.chembiol.2016.07.023},
year = {2016},
month = oct,
publisher = {Elsevier {BV}},
volume = {23},
number = {10},
pages = {1294--1301},
author = {Kaitlyn~M. Gayvert and Neel~S. Madhukar and Olivier Elemento},
title = {A Data-Driven Approach to Predicting Successes and Failures of Clinical Trials},
journal = {Cell Chemical Biology}}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} {#shows|exhibits|displays!} {clinical_toxicity#no &NULL}{clinical_toxicity__names__noun}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule has {clinical_toxicity#no &NULL}{clinical_toxicity__names__adjective} {#properties|characteristics|features|traits!}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is from!} a molecule that is {clinical_toxicity#not &NULL}identified as {clinical_toxicity__names__adjective}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {clinical_toxicity#not &NULL}{clinical_toxicity__names__adjective}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {clinical_toxicity__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result: {clinical_toxicity#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {clinical_toxicity__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {clinical_toxicity#not &NULL}{clinical_toxicity__names__adjective}.""",
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {clinical_toxicity__names__adjective}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the molecule with the {SMILES__description} {SMILES#} is {clinical_toxicity__names__adjective}?
Assistant: {clinical_toxicity#No&Yes}, this molecule is {clinical_toxicity#not &NULL}{clinical_toxicity__names__adjective}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {clinical_toxicity__names__adjective}?
Assistant: {clinical_toxicity#No&Yes}, it is {clinical_toxicity#not &NULL}{clinical_toxicity__names__adjective}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {clinical_toxicity#not &NULL}{clinical_toxicity__names__adjective}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {clinical_toxicity#not &NULL}{clinical_toxicity__names__adjective}?
Assistant: This is a molecule that is {clinical_toxicity#not &NULL}{clinical_toxicity__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {clinical_toxicity#not &NULL}be {clinical_toxicity__names__adjective}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {clinical_toxicity#not &NULL}{clinical_toxicity__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {clinical_toxicity#not &NULL}be {clinical_toxicity__names__adjective}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {clinical_toxicity#not &NULL}{clinical_toxicity__names__adjective}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {clinical_toxicity__names__adjective}:<EOI> {clinical_toxicity#yes&no}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {clinical_toxicity__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {clinical_toxicity#False&True}""",  # noqa: E501
            # noqa: E501 """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
            # Description: A molecule that is {clinical_toxicity__names__adjective}.
            # Result:<EOI> {SMILES#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {clinical_toxicity__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{clinical_toxicity%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {clinical_toxicity__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{clinical_toxicity%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {clinical_toxicity#not &NULL}{clinical_toxicity__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%clinical_toxicity%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {clinical_toxicity#not &NULL}{clinical_toxicity__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%clinical_toxicity%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
        ],
    }

    def str_presenter(dumper, data):
        """configures yaml for dumping multiline strings

        Ref:
        https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
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
