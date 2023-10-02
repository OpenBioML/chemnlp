import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    splits = Tox(name="Carcinogens_Lagunin").get_split()
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
        "carcinogen",
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
        "name": "carcinogens",  # unique identifier, we will also use this for directory names
        "description": """A carcinogen is any substance, radionuclide, or radiation that promotes
carcinogenesis, the formation of cancer. This may be due to the ability to damage
the genome or to the disruption of cellular metabolic processes.""",
        "targets": [
            {
                "id": "carcinogen",  # name of the column in a tabular dataset
                "description": "whether it is carcinogenic (1) or not (0).",
                "units": None,
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "carcinogen"},
                    # {"noun": "substance that promotes carcinogenesis"},
                    {"adjective": "carcinogenic"},
                    {"gerund": "having the potential to cause cancer"},
                ],
                "uris": [
                    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C347",
                    "http://purl.bioontology.org/ontology/SNOMEDCT/88376000",
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
                "url": "https://doi.org/10.1002/qsar.200860192",
                "description": "corresponding publication",
            },
            {
                "url": "https://doi.org/10.1021/ci300367a",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#carcinogens",
                "description": "Data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Lagunin2009,
doi = {10.1002/qsar.200860192},
url = {https://doi.org/10.1002/qsar.200860192},
year = {2009},
month = jun,
publisher = {Wiley},
volume = {28},
number = {8},
pages = {806--810},
author = {Alexey Lagunin and Dmitrii Filimonov and Alexey Zakharov and Wei Xie
and Ying Huang and Fucheng Zhu and Tianxiang Shen and Jianhua Yao and Vladimir Poroikov},
title = {Computer-Aided Prediction of Rodent Carcinogenicity by PASS and CISOC PSCT},
journal = {QSAR & Combinatorial Science}""",
            """@article{Cheng2012,
doi = {10.1021/ci300367a},
url = {https://doi.org/10.1021/ci300367a},
year = {2012},
month = nov,
publisher = {American Chemical Society (ACS)},
volume = {52},
number = {11},
pages = {3099--3105},
author = {Feixiong Cheng and Weihua Li and Yadi Zhou and Jie Shen and Zengrui Wu
and Guixia Liu and Philip W. Lee and Yun Tang},
title = {admetSAR: A Comprehensive Source and Free Tool for Assessment of Chemical ADMET Properties},
journal = {Journal of Chemical Information and Modeling}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} {#shows|exhibits|displays!} {carcinogen#no &NULL}{carcinogen__names__adjective} {#properties|effects!}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule has {carcinogen#no &NULL}{carcinogen__names__adjective} {#effects|properties|characteristics|features!}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} represents a molecule that is {carcinogen#not &NULL}identified as {carcinogen__names__adjective}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} is {carcinogen#not &NULL}{carcinogen__names__adjective}.",
            "The {#molecule |!}{SMILES__description} {SMILES#} is {carcinogen#not &NULL}{carcinogen__names__adjective}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {carcinogen__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result: {carcinogen#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {carcinogen__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {carcinogen#not &NULL}{carcinogen__names__adjective}.""",
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {carcinogen__names__adjective}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the molecule with the {SMILES__description} {SMILES#} is {carcinogen__names__adjective}?
Assistant: {carcinogen#No&Yes}, this molecule is {carcinogen#not &NULL}{carcinogen__names__adjective}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {carcinogen__names__adjective}?
Assistant: {carcinogen#No&Yes}, it is {carcinogen#not &NULL}{carcinogen__names__adjective}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {carcinogen#not &NULL}{carcinogen__names__adjective}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {carcinogen#not &NULL}{carcinogen__names__adjective}?
Assistant: This is a molecule that is {carcinogen#not &NULL}{carcinogen__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {carcinogen#not &NULL}be {carcinogen__names__adjective}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {carcinogen#not &NULL}{carcinogen__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {carcinogen#not &NULL}be {carcinogen__names__adjective}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {carcinogen#not &NULL}{carcinogen__names__adjective}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {carcinogen__names__adjective}:<EOI> {carcinogen#no&yes}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {carcinogen__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {carcinogen#False&True}""",  # noqa: E501
            # noqa: E501 """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
            # Description: A molecule that is {carcinogen__names__adjective}.
            # Result:<EOI> {SMILES#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {carcinogen__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{carcinogen%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {carcinogen__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{carcinogen%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {carcinogen#not &NULL}{carcinogen__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%carcinogen%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {carcinogen#not &NULL}{carcinogen__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%carcinogen%}
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
