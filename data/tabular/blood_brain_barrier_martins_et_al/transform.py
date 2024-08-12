import pandas as pd
import yaml
from tdc.single_pred import ADME


def get_and_transform_data():
    # get raw data
    target_subfolder = "BBB_Martins"
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
    fields_clean = ["compound_name", "SMILES", "penetrate_BBB", "split"]
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
        "name": "blood_brain_barrier_martins_et_al",  # unique identifier, we will also use this for directory names
        "description": """As a membrane separating circulating blood and brain extracellular
fluid, the blood-brain barrier (BBB) is the protection layer that blocks most
foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver
to the site of action forms a crucial challenge in development of drugs for the
central nervous system.""",
        "targets": [
            {
                "id": "penetrate_BBB",  # name of the column in a tabular dataset
                "description": "The ability of a drug to penetrate the blood brain barrier (1) or not (0)",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "blood brain barrier penetration"},
                    {"noun": "ADME blood-brain barrier penetration"},
                    {"verb": "penetrates the blood brain barrier to reach the brain"},
                    {"verb": "penetrates the blood brain barrier"},
                    {"adjective": "penetrating the blood brain barrier"},
                    {
                        "adjective": "penetrating the blood brain barrier to reach the brain"
                    },
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
                "description": "compound name",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1021/ci300124c",
                "description": "corresponding publication",
            },
            {
                "url": "https://rb.gy/0xx91v",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#bbb-blood-brain-barrier-martins-et-al",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Martins2012,
doi = {10.1021/ci300124c},
url = {https://doi.org/10.1021/ci300124c},
year = {2012},
month = jun,
publisher = {American Chemical Society (ACS)},
volume = {52},
number = {6},
pages = {1686--1697},
author = {Ines Filipa Martins and Ana L. Teixeira and Luis Pinheiro
and Andre O. Falcao},
title = {A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling},
journal = {Journal of Chemical Information and Modeling}""",
            """@article{Wu2018,
doi = {10.1039/c7sc02664a},
url = {https://doi.org/10.1039/c7sc02664a},
year = {2018},
publisher = {Royal Society of Chemistry (RSC)},
volume = {9},
number = {2},
pages = {513--530},
author = {Zhenqin Wu and Bharath Ramsundar and Evan~N. Feinberg and Joseph
Gomes and Caleb Geniesse and Aneesh S. Pappu and Karl Leswing and Vijay Pande},
title = {MoleculeNet: a benchmark for molecular machine learning},
journal = {Chemical Science}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {SMILES#} is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} represents a molecule that is {penetrate_BBB#not &NULL}identified as {penetrate_BBB__names__adjective}.",  # noqa: E501
            "The molecule represented with the {SMILES__description} {SMILES#} is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}.",  # noqa: E501
            "{SMILES#} represents a molecule that is {penetrate_BBB#not &NULL}identified as {penetrate_BBB__names__adjective}.",  # noqa: E501
            "{SMILES#} represents a molecule that is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}.",  # noqa: E501
            "{SMILES#} is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}.",
            "The {#molecule |!}{SMILES__description} {SMILES#} is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}.",  # noqa: E501
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {penetrate_BBB__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result: {penetrate_BBB#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {penetrate_BBB__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}.""",  # noqa: E501
            """Task: Please {#give me|create|generate!} the {SMILES__description} of {#molecule|chemical|chemical structure!} based on the {#text |!}description{# below|!}.
Description: A molecule that is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the molecule with the {SMILES__description} {SMILES#} is {penetrate_BBB__names__adjective}?
Assistant: {penetrate_BBB#No&Yes}, this molecule is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {penetrate_BBB__names__adjective}?
Assistant: {penetrate_BBB#No&Yes}, it is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}?
Assistant: This is a molecule that is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} the {SMILES__description} of a molecule.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {penetrate_BBB#not &NULL}be {penetrate_BBB__names__adjective}.
Assistant: Got it, this {SMILES__description} is {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {penetrate_BBB__names__adjective}:<EOI>{penetrate_BBB#no&yes}",  # noqa: E501 for the benchmarking setup <EOI>separates input and output
            # todo: check if we go for multiple choice only and remove the benchmarking template above and below
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {penetrate_BBB__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI>{penetrate_BBB#False&True}""",  # noqa: E501
            # noqa: E501 """Task: Please {#give me|create|generate!} a molecule {SMILES__description} based on the {#text |!}description{# below|!}.
            # Description: A molecule that is {penetrate_BBB__names__adjective}.
            # Result:<EOI>{SMILES#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} of {SMILES#} {penetrate_BBB__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{penetrate_BBB%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%penetrate_BBB%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} of {SMILES#} {penetrate_BBB__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{penetrate_BBB%}
Answer:<EOI>{%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%penetrate_BBB%}
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
