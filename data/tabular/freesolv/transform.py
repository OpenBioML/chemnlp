import pandas as pd
import requests
import yaml
from pandas.api.types import is_string_dtype


def get_and_transform_data():
    # get raw data
    data_path = (
        "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt"
    )
    fn_data_original = "data_original.txt"
    data = requests.get(data_path)
    with open(fn_data_original, "wb") as f:
        f.write(data.content)

    # create dataframe
    df = pd.read_csv(fn_data_original, delimiter=";", skiprows=2)

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "# compound id (and file prefix)",
        " SMILES",
        " iupac name (or alternative if IUPAC is unavailable or not parseable by OEChem)",
        " experimental value (kcal/mol)",
        " experimental uncertainty (kcal/mol)",
        " Mobley group calculated value (GAFF) (kcal/mol)",
        " calculated uncertainty (kcal/mol)",
        " experimental reference (original or paper this value was taken from)",
        " calculated reference",
        " text notes.",
    ]

    # overwrite column names = fields
    fields_clean = [
        "compound_id",
        "SMILES",
        "iupac_name",
        "exp_value",
        "exp_uncertainty",
        "GAFF",
        "calc_uncertainty",
        "exp_ref",
        "calc_reference",
        "notes",
    ]
    df.columns = fields_clean

    # data cleaning
    for col in df.columns:
        if is_string_dtype(df[col]):
            df[col] = df[
                col
            ].str.strip()  # remove leading and trailing white space characters

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "freesolv",  # unique identifier, we will also use this for directory names
        "description": "Experimental and calculated small molecule hydration free energies",
        "targets": [
            {
                "id": "exp_value",  # name of the column in a tabular dataset
                "description": "experimental hydration free energy value",  # description of what this column means
                "units": "kcal/mol",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "hydration free energy"},
                ],
            },
            {
                "id": "exp_uncertainty",
                "description": "experimental hydration free energy uncertainty",
                "units": "kcal/mol",
                "type": "continuous",
                "names": [
                    {"noun": "hydration free energy uncertainty"},
                ],
            },
            {
                "id": "GAFF",  # name of the column in a tabular dataset
                "description": "mobley group calculated value",  # description of what this column means
                "units": "kcal/mol",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",
                "names": [  # names for the property (to sample from for building the prompts)
                    {
                        "noun": "hydration free energy computed using the GAFF force field"
                    },
                ],
            },
            {
                "id": "calc_uncertainty",
                "description": "mobley group calculated value calculated uncertainty",
                "units": "kcal/mol",
                "type": "continuous",
                "names": [
                    {
                        "noun": "uncertainty in hydration free energy computed using the GAFF force field"
                    },
                ],
            },
        ],
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "description": "SMILES",  # description (optional, except for "Other")
            },
            {
                "id": "iupac_name",
                "type": "IUPAC",
                "description": "IUPAC",
            },
        ],
        "license": "CC BY-NC-SA 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://github.com/MobleyLab/FreeSolv",
                "description": "issue tracker and source data",
            },
            {
                "url": "https://escholarship.org/uc/item/6sd403pz",
                "description": "repository with data",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{mobley2013experimental,
title={Experimental and calculated small molecule hydration free energies},
author={Mobley, David L},
year={2013}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} has a {exp_value__names__noun} of {exp_value#} {exp_value__units}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation of |!}{SMILES#}, the molecule has a {exp_value__names__noun} of {exp_value#} {exp_value__units}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is representing!} a molecule {#that has a|with a!} {exp_value__names__noun} of {exp_value#} {exp_value__units}.",  # noqa: E501
            "The molecule with the {SMILES__description} {SMILES#} has a {exp_value__names__noun} of {exp_value#} {exp_value__units}.",  # noqa: E501
            # GAFF
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} has a {GAFF__names__noun} of {GAFF#} {GAFF__units}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation of |!}{SMILES#}, the molecule has a {GAFF__names__noun} of {GAFF#} {GAFF__units}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is representing!} a molecule {#that has a|with a!} {GAFF__names__noun} of {GAFF#} {GAFF__units}.",  # noqa: E501
            "The molecule with the {SMILES__description} {SMILES#} has a {GAFF__names__noun} of {GAFF#} {GAFF__units}.",
            # Instruction tuning text templates
            """Task: Please predict a molecule feature based on the description.
Description: Predict the {exp_value__names__noun} in {exp_value__units}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a numeric value in {exp_value__units} without using any {#other|additional!} words.
Result: {exp_value#} {exp_value__units}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that has {exp_value__names__noun} of {exp_value#} {exp_value__units}.
Result: {SMILES#}""",  # noqa: E501
            # GAFF
            """Task: Please predict a molecule feature based on the description.
Description: Predict the {GAFF__names__noun} in {GAFF__units}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a numeric value in {GAFF__units} without using any {#other|additional!} words.
Result: {GAFF#} {GAFF__units}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that has {GAFF__names__noun} of {GAFF#} {GAFF__units}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} the {exp_value__names__noun} in {exp_value__units} of the molecule with the {SMILES__description} {SMILES#}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, this molecule has a {exp_value__names__noun} of {exp_value#} {exp_value__units}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that has a {exp_value__names__noun} of {exp_value#} {exp_value__units}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that has a {exp_value__names__noun} of {exp_value#} {exp_value__units}.
Assistant: This is a molecule that has a {exp_value__names__noun} of {exp_value#} {exp_value__units}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should have a {exp_value__names__noun} of {exp_value#} {exp_value__units}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} represents a molecule that has a {exp_value__names__noun} of {exp_value#} {exp_value__units}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should have a {exp_value__names__noun} of {exp_value#} {exp_value__units}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} represents a molecule that has a {exp_value__names__noun} of {exp_value#} {exp_value__units}: {SMILES#}""",  # noqa: E501
            # GAFF
            """User: Can you {#tell me|derive|estimate!} the {GAFF__names__noun} in {GAFF__units} of the molecule with the {SMILES__description} {SMILES#}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, this molecule has a {GAFF__names__noun} of {GAFF#} {GAFF__units}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that has a {GAFF__names__noun} of {GAFF#} {GAFF__units}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that has a {GAFF__names__noun} of {GAFF#} {GAFF__units}.
Assistant: This is a molecule that has a {GAFF__names__noun} of {GAFF#} {GAFF__units}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should have a {GAFF__names__noun} of {GAFF#} {GAFF__units}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} represents a molecule that has a {GAFF__names__noun} of {GAFF#} {GAFF__units}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should have a {GAFF__names__noun} of {GAFF#} {GAFF__units}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} represents a molecule that has a {GAFF__names__noun} of {GAFF#} {GAFF__units}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "The {exp_value__names__noun} of the molecule with the {SMILES__description} {SMILES#} is:<EOI> {exp_value#} {exp_value__units}",  # noqa: E501
            "The {exp_value__names__noun} of the {SMILES__description} {SMILES#} is:<EOI> {exp_value#} {exp_value__units}",  # noqa: E501
            "The {exp_value__names__noun} of the molecule {SMILES__description} {SMILES#} is:<EOI> {exp_value#} {exp_value__units}",  # noqa: E501
            """Task: Please predict a molecule feature based on the description.
Description: Predict the {exp_value__names__noun} in {exp_value__units} of a molecule.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a numeric value in {exp_value__units} without using any {#other|additional!} words.
Result:<EOI> {exp_value#} {exp_value__units}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that has {exp_value__names__noun} of {exp_value#} {exp_value__units}.
Result:<EOI> {SMILES#}""",  # noqa: E501
            # GAFF
            "The {GAFF__names__noun} of the molecule with the {SMILES__description} {SMILES#} is:<EOI> {GAFF#} {GAFF__units}",  # noqa: E501
            "The {GAFF__names__noun} of the {SMILES__description} {SMILES#} is:<EOI> {GAFF#} {GAFF__units}",  # noqa: E501
            "The {GAFF__names__noun} of the molecule {SMILES__description} {SMILES#} is:<EOI> {GAFF#} {GAFF__units}",  # noqa: E501
            """Task: Please predict a molecule feature based on the description.
Description: Predict the {GAFF__names__noun} in {GAFF__units} of a molecule.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a numeric value in {GAFF__units} without using any {#other|additional!} words.
Result:<EOI> {GAFF#} {GAFF__units}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that has {GAFF__names__noun} of {GAFF#} {GAFF__units}.
Result:<EOI> {SMILES#}""",  # noqa: E501
        ],
    }
    fn_meta = "meta.yaml"
    with open(fn_meta, "w") as f:
        yaml.dump(meta, f, sort_keys=False)

    print(f"Finished processing {meta['name']} dataset!")


if __name__ == "__main__":
    get_and_transform_data()
