import pandas as pd
import yaml
from tdc.single_pred import ADME


def get_and_transform_data():
    # get raw data
    target_subfolder = "HIA_Hou"
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

    df["Drug_ID"] = [x.split(".")[0] for x in df.Drug_ID.tolist()]

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == ["Drug_ID", "Drug", "Y", "split"]

    # overwrite column names = fields
    fields_clean = [
        "compound_name",
        "SMILES",
        f"absorption_{target_subfolder}",
        "split",
    ]
    df.columns = fields_clean

    # data cleaning
    # remove leading and trailing white space characters
    df.compound_name = df.compound_name.str.strip()

    df = df.dropna()
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "human_intestinal_absorption",  # unique identifier, we will also use this for directory names
        "description": """When a drug is orally administered, it needs to be absorbed from the
human gastrointestinal system into the bloodstream of the human body. This ability
of absorption is called human intestinal absorption (HIA) and it is crucial
for a drug to be delivered to the target.""",
        "targets": [
            {
                "id": f"absorption_{target_subfolder}",  # name of the column in a tabular dataset
                "description": "whether it is absorbed from the human gastrointestinal system (1) or not (0)",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "human intestinal absorption"},
                    {"noun": "human intestinal absorption (HIA)"},
                    {"noun": "HIA"},
                    {"adjective": "absorbed from the human gastrointestinal system"},
                ],
                "uris": [
                    "http://purl.bioontology.org/ontology/MESH/D007408",
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
                "id": "compound_name",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": [
                    {"noun": "compound name"},
                    {"noun": "drug name"},
                    {"noun": "generic drug name"},
                ],
                "description": "drug name",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1021/ci600343x",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#hia-human-intestinal-absorption-hou-et-al",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Hou2006,
doi = {10.1021/ci600343x},
url = {https://doi.org/10.1021/ci600343x},
year = {2006},
month = nov,
publisher = {American Chemical Society (ACS)},
volume = {47},
number = {1},
pages = {208--218},
author = {Tingjun Hou and Junmei Wang and Wei Zhang and Xiaojie Xu},
title = {ADME Evaluation in Drug Discovery. 7. Prediction of Oral Absorption
by Correlation and Classification},
journal = {Journal of Chemical Information and Modeling}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} {#shows|exhibits|displays!} {absorption_HIA_Hou#no &NULL}{absorption_HIA_Hou__names__noun} properties.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule has {absorption_HIA_Hou#no &NULL}{absorption_HIA_Hou__names__noun} {#properties|characteristics|features!}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is from!} a molecule that {#shows|exhibits|displays!} {absorption_HIA_Hou#not &NULL}{absorption_HIA_Hou__names__noun}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {absorption_HIA_Hou#not &NULL}{absorption_HIA_Hou__names__adjective}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {absorption_HIA_Hou__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result: {absorption_HIA_Hou#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {absorption_HIA_Hou__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {absorption_HIA_Hou#not &NULL}{absorption_HIA_Hou__names__adjective}.""",
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {absorption_HIA_Hou__names__adjective}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the molecule with the {SMILES__description} {SMILES#} is {absorption_HIA_Hou__names__adjective}?
Assistant: {absorption_HIA_Hou#No&Yes}, this molecule is {absorption_HIA_Hou#not &NULL}{absorption_HIA_Hou__names__adjective}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {absorption_HIA_Hou__names__adjective}?
Assistant: {absorption_HIA_Hou#No&Yes}, it is {absorption_HIA_Hou#not &NULL}{absorption_HIA_Hou__names__adjective}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {absorption_HIA_Hou#not &NULL}{absorption_HIA_Hou__names__adjective}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {absorption_HIA_Hou#not &NULL}{absorption_HIA_Hou__names__adjective}?
Assistant: This is a molecule that is {absorption_HIA_Hou#not &NULL}{absorption_HIA_Hou__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {absorption_HIA_Hou#not &NULL}be {absorption_HIA_Hou__names__adjective}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {absorption_HIA_Hou#not &NULL}{absorption_HIA_Hou__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {absorption_HIA_Hou#not &NULL}be {absorption_HIA_Hou__names__adjective}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {absorption_HIA_Hou#not &NULL}{absorption_HIA_Hou__names__adjective}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {absorption_HIA_Hou__names__adjective}:<EOI> {absorption_HIA_Hou#no&yes}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {absorption_HIA_Hou__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {absorption_HIA_Hou#False&True}""",  # noqa: E501
            # noqa: E501 """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
            # Description: A molecule that is {absorption_HIA_Hou__names__adjective}.
            # Result:<EOI> {SMILES#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {absorption_HIA_Hou__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{absorption_HIA_Hou%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {absorption_HIA_Hou__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{absorption_HIA_Hou%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {absorption_HIA_Hou#not &NULL}{absorption_HIA_Hou__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%absorption_HIA_Hou%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {absorption_HIA_Hou#not &NULL}{absorption_HIA_Hou__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%absorption_HIA_Hou%}
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
