import pandas as pd
import yaml
from tdc.single_pred import ADME


def get_and_transform_data():
    # get raw data
    target_subfolder = "CYP2C9_Substrate_CarbonMangels"
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
        "compound_name",
        "SMILES",
        f"{'_'.join(target_subfolder.split('_')[:2])}",
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
    meta = {
        "name": "cyp2c9_substrate_carbonmangels",  # unique identifier, we will also use this for directory names
        "description": """CYP P450 2C9 plays a major role in the oxidation of both xenobiotic
and endogenous compounds. Substrates are drugs that are metabolized by the enzyme.
TDC used a dataset from Carbon Mangels et al, which merged information on substrates
and nonsubstrates from six publications.""",
        "targets": [
            {
                "id": f"{'_'.join(target_subfolder.split('_')[:2])}",  # name of the column in a tabular dataset
                "description": "drugs that are metabolized by CYP P450 2C9 (1) or not (0)",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "CYP P450 2C9 substrate"},
                    {"noun": "CYP2C9 substrate"},
                    {"noun": "substrate for CYP2C9"},
                    {"noun": "substrate for CYP P450 2C9"},
                    {"verb": "metabolized by CYP2C9"},
                    {"verb": "metabolized by CYP P450 2C9"},
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
                "url": "https://doi.org/10.1002/minf.201100069",
                "description": "corresponding publication",
            },
            {
                "url": "https://doi.org/10.1021/ci300367a",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#cyp2c9-substrate-carbon-mangels-et-al",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{CarbonMangels2011,
doi = {10.1002/minf.201100069},
url = {https://doi.org/10.1002/minf.201100069},
year = {2011},
month = sep,
publisher = {Wiley},
volume = {30},
number = {10},
pages = {885--895},
author = {Miriam Carbon-Mangels and Michael C. Hutter},
title = {Selecting Relevant Descriptors for Classification by Bayesian Estimates:
A Comparison with Decision Trees and Support Vector Machines Approaches for Disparate Data Sets},
journal = {Molecular Informatics}""",
            """@article{Cheng2012,
doi = {10.1021/ci300367a},
url = {https://doi.org/10.1021/ci300367a},
year = {2012},
month = nov,
publisher = {American Chemical Society (ACS)},
volume = {52},
number = {11},
number = {3099--3105},
author = {Feixiong Cheng and Weihua Li and Yadi Zhou and Jie Shen and Zengrui Wu
and Guixia Liu and Philip W. Lee and Yun Tang},
title = {admetSAR: A Comprehensive Source and Free Tool for Assessment of Chemical ADMET Properties},
journal = {Journal of Chemical Information and Modeling}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} is {CYP2C9_Substrate#not &NULL}a {CYP2C9_Substrate__names__noun}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule is {CYP2C9_Substrate#not &NULL}{CYP2C9_Substrate__names__verb}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is from!} a molecule that is {CYP2C9_Substrate#not &NULL}identified as a {CYP2C9_Substrate__names__noun}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {CYP2C9_Substrate#not &NULL}{CYP2C9_Substrate__names__verb}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is a {CYP2C9_Substrate__names__noun}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
esult: {CYP2C9_Substrate#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {CYP2C9_Substrate__names__verb}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {CYP2C9_Substrate#not &NULL}a {CYP2C9_Substrate__names__noun}.""",
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is a {CYP2C9_Substrate__names__noun}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the molecule with the {SMILES__description} {SMILES#} is a {CYP2C9_Substrate__names__noun}?
Assistant: {CYP2C9_Substrate#No&Yes}, this molecule is {CYP2C9_Substrate#not &NULL}{CYP2C9_Substrate__names__verb}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {CYP2C9_Substrate__names__verb}?
Assistant: {CYP2C9_Substrate#No&Yes}, it is {CYP2C9_Substrate#not &NULL}a {CYP2C9_Substrate__names__noun}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is a {CYP2C9_Substrate#not &NULL}{CYP2C9_Substrate__names__noun}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {CYP2C9_Substrate#not &NULL}{CYP2C9_Substrate__names__verb}?
Assistant: This is a molecule that is {CYP2C9_Substrate#not &NULL}a {CYP2C9_Substrate__names__noun}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {CYP2C9_Substrate#not &NULL}be {CYP2C9_Substrate__names__verb}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {CYP2C9_Substrate#not &NULL}{CYP2C9_Substrate__names__verb}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {CYP2C9_Substrate#not &NULL}be a {CYP2C9_Substrate__names__noun}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {CYP2C9_Substrate#not &NULL}a {CYP2C9_Substrate__names__noun}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} a {CYP2C9_Substrate__names__noun}:<EOI> {CYP2C9_Substrate#yes&no}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is a {CYP2C9_Substrate__names__noun}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {CYP2C9_Substrate#False&True}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {CYP2C9_Substrate__names__verb}.
Result:<EOI> {SMILES#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {CYP2C9_Substrate__names__verb}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{CYP2C9_Substrate%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} a {CYP2C9_Substrate__names__noun}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{CYP2C9_Substrate%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {CYP2C9_Substrate#not &NULL}a {CYP2C9_Substrate__names__noun}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%CYP2C9_Substrate%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {CYP2C9_Substrate#not &NULL}{CYP2C9_Substrate__names__verb}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%CYP2C9_Substrate%}
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
