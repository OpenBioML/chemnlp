import pandas as pd
import yaml
from tdc.single_pred import ADME


def get_and_transform_data():
    # get raw data
    target_subfolder = "Bioavailability_Ma"
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
    fields_clean = ["compound_name", "SMILES", "bioavailable", "split"]
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
        "name": "bioavailability_ma_et_al",  # unique identifier, we will also use this for directory names
        "description": """Oral bioavailability is defined as the rate and extent to which the
active ingredient or active moiety is absorbed from a drug product and becomes
available at the site of action.""",
        "targets": [
            {
                "id": "bioavailable",  # name of the column in a tabular dataset
                "description": "whether it is bioavailable (1) or not (0)",  # description of what this column means
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "oral bioavailability"},
                    {"adjective": "orally bioavailable"},
                ],
                "uris": [
                    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C70913",
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
            {
                "id": "compound_name",  # column name
                "type": "Other",
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
                "url": "https://doi.org/10.1016/j.jpba.2008.03.023",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#bioavailability-ma-et-al",
                "description": "data source",
                # note: this is not the original data, it is their modified version
                # original larger dataset: http://modem.ucsd.edu/adme/databases/databases_bioavailability.htm
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Ma2008,
doi = {10.1016/j.jpba.2008.03.023},
url = {https://doi.org/10.1016/j.jpba.2008.03.023},
year = {2008},
month = aug,
publisher = {Elsevier BV},
volume = {47},
number = {4-5},
author = {Chang-Ying Ma and Sheng-Yong Yang and Hui Zhang
and Ming-Li Xiang and Qi Huang and Yu-Quan Wei},
title = {Prediction models of human plasma protein binding rate and
oral bioavailability derived by using GA-CG-SVM method},
journal = {Journal of Pharmaceutical and Biomedical Analysis}""",
        ],
        "templates": [
            "The {#molecule with the |!}{SMILES__description} {#representation of |!}{SMILES#} has a {bioavailable#low&high} {bioavailable__names__noun}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation of |!}{SMILES#}, the molecule has a {bioavailable#low&high} {bioavailable__names__noun}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is from!} a molecule that has a {bioavailable#low&high} {bioavailable__names__noun}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} has a {bioavailable#low&high} {bioavailable__names__noun}.",
            "The {#molecule with the |!}{SMILES__description} {SMILES#} has a {bioavailable#low&high} {bioavailable__names__noun}.",  # noqa: E501
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: Predict if the molecule has a low or high {bioavailable__names__noun}?
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "low" or "high" without using any {#other|additional!} words.
Result: {bioavailable#low&high}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: Predict if the molecule has a low or high {bioavailable__names__noun}?
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule has a {bioavailable#low&high} {bioavailable__names__noun}.""",
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that has a {bioavailable#low&high} {bioavailable__names__noun}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the {#molecule with the |!}{SMILES__description} {SMILES#} has a low or high {bioavailable__names__noun}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, this molecule has a {bioavailable#low&high} {bioavailable__names__noun}.""",  # noqa: E501
            """User: Has this {#molecule with the |!}{SMILES__description} {SMILES#} a low or high {bioavailable__names__noun}?
Assistant: It has a {bioavailable#low&high} {bioavailable__names__noun}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that has a {bioavailable#low&high} {bioavailable__names__noun}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that has a {bioavailable#low&high} {bioavailable__names__noun}?
Assistant: {#Ok, this|This!} is a molecule that has a {bioavailable#low&high} {bioavailable__names__noun}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should have a {bioavailable#low&high} {bioavailable__names__noun}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} has a {bioavailable#low&high} {bioavailable__names__noun}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should have a {bioavailable#low&high} {bioavailable__names__noun}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} has a {bioavailable#low&high} {bioavailable__names__noun}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {bioavailable__names__adjective}?<EOI> {bioavailable#yes&no}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please answer the multiple choice question.
Question: Has the {#molecule with the |!}{SMILES__description} {#representation of |!}{SMILES#} a high {bioavailable__names__noun}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{bioavailable%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Has the {#molecule with the |!}{SMILES__description} {#representation of |!}{SMILES#} a high {bioavailable__names__noun}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{bioavailable%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules have a high {bioavailable__names__noun}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%bioavailable%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules have a high {bioavailable__names__noun}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%bioavailable%}
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
