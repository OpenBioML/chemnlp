import pandas as pd
import yaml
from tdc.single_pred import ADME


def get_and_transform_data():
    # get raw data
    splits = ADME(name="PAMPA_NCATS").get_split()
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
    fields_clean = ["compound_id", "SMILES", "permeability", "split"]
    df.columns = fields_clean

    # data cleaning
    df.drop(columns=["compound_id"], inplace=True)
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "pampa_ncats",  # unique identifier, we will also use this for directory names
        "description": """PAMPA (parallel artificial membrane permeability assay) is a commonly
employed assay to evaluate drug permeability across the cellular membrane.
PAMPA is a non-cell-based, low-cost and high-throughput alternative to cellular models.
Although PAMPA does not model active and efflux transporters, it still provides permeability values
that are useful for absorption prediction because the majority of drugs are absorbed
by passive diffusion through the membrane.""",
        "targets": [
            {
                "id": "permeability",  # name of the column in a tabular dataset
                "description": "Binary permeability in PAMPA assay.",  # description of what this column means
                "units": None,
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "permeability"},
                    {"verb": "is permeable in the PAMPA assay"},
                    {
                        "verb": "shows permeability in parallel artificial membrane permeability assay (PAMPA) assay"
                    },
                    {"adjective": "permeable in the PAMPA assay"},
                    {"gerund": "permeating in the PAMPA assay"},
                ],
                "pubchem_aids": [1508612],
                "uris": ["http://purl.bioontology.org/ontology/MESH/D002463"],
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
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#pampa-permeability-ncats",
                "description": "original dataset link",
            },
            {
                "url": "https://journals.sagepub.com/doi/full/10.1177/24725552211017520",
                "description": "corresponding publication",
            },
        ],
        "benchmarks": [
            {
                "name": "TDC",
                "link": "https://tdcommons.ai/",
                "split_column": "split",
            }
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{siramshetty2021validating,
title={Validating ADME QSAR Models Using Marketed Drugs},
author={Siramshetty, Vishal and Williams, Jordan and Nguyen, DHac-Trung and Neyra, Jorge and Southall,
Noel and Math'e, Ewy and Xu, Xin and Shah, Pranav},
journal={SLAS DISCOVERY: Advancing the Science of Drug Discovery},
volume={26},
number={10},
pages={1326--1336},
year={2021},
publisher={SAGE Publications Sage CA: Los Angeles, CA}
}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} is {permeability#not &NULL}{permeability__names__adjective}.",  # noqa: E501
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} is {permeability#not &NULL}{permeability__names__gerund}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule has {permeability#no &NULL}{permeability__names__noun} {#properties|characteristics|features!}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is from!} a molecule that is {permeability#not &NULL}identified as {permeability__names__adjective}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {permeability#not &NULL}{permeability__names__adjective}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {permeability__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional|extra!} words.
Result: {permeability#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {permeability__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {permeability#not &NULL}{permeability__names__adjective}.""",
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {permeability__names__adjective}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|figure out|estimate!} if the molecule with the {SMILES__description} {SMILES#} is {permeability__names__adjective}?
Assistant: {permeability#No&Yes}, this molecule is {permeability#not &NULL}{permeability__names__adjective}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {permeability__names__adjective}?
Assistant: {permeability#No&Yes}, it is {permeability#not &NULL}{permeability__names__adjective}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {permeability#not &NULL}{permeability__names__adjective}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {permeability#not &NULL}{permeability__names__adjective}?
Assistant: This is a molecule that is {permeability#not &NULL}{permeability__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: This sounds {#very exciting. |very interesting. | very curious. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {permeability#not &NULL}be {permeability__names__adjective}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {permeability#not &NULL}{permeability__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {permeability#not &NULL}be {permeability__names__adjective}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {permeability#not &NULL}{permeability__names__adjective}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {permeability__names__adjective}:<EOI> {permeability#yes&no}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {permeability__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {permeability#False&True}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {permeability__names__adjective}.
Result:<EOI> {SMILES#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {permeability__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{permeability%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {permeability__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{permeability%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {permeability#not &NULL}{permeability__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%permeability%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {permeability#not &NULL}{permeability__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%permeability%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501,
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
