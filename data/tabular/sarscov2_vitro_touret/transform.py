import pandas as pd
import yaml
from tdc.single_pred import HTS


def get_and_transform_data():
    # get raw data
    splits = HTS(name="SARSCoV2_Vitro_Touret").get_split()
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
    fields_clean = ["compound_id", "SMILES", "activity_SARSCoV2", "split"]
    df.columns = fields_clean

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "sarscov2_vitro_touret",  # unique identifier, we will also use this for directory names
        "description": """An in-vitro screen of the Prestwick chemical library composed of 1,480
approved drugs in an infected cell-based assay.""",
        "targets": [
            {
                "id": "activity_SARSCoV2",  # name of the column in a tabular dataset
                "description": "whether it is active against SARSCoV2 (1) or not (0).",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "activity against the Corona virus"},
                    {"noun": "activity against SARSCoV2"},
                    {"noun": "activity against COVID19"},
                    {"adjective": "active against the Corona virus"},
                    {"adjective": "active against SARSCoV2"},
                    {"adjective": "active against COVID19"},
                    {"gerund": "fighting the Corona virus"},
                    {"gerund": "fighting against SARSCoV2"},
                    {"gerund": "fighting against COVID19"},
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
                "url": "https://doi.org/10.1038/s41598-020-70143-6",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/hts/#sars-cov-2-in-vitro-touret-et-al",
                "description": "Data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Touret2020,
doi = {10.1038/s41598-020-70143-6},
url = {https://doi.org/10.1038/s41598-020-70143-6},
year = {2020},
month = aug,
publisher = {Springer Science and Business Media LLC},
volume = {10},
number = {1},
author = {Franck Touret and Magali Gilles and Karine Barral and  Antoine Nougairede
and Jacques van Helden and Etienne Decroly and Xavier de Lamballerie and Bruno Coutard},
title = {In vitro screening of a FDA approved chemical library reveals potential inhibitors of
SARS-CoV-2 replication},
journal = {Scientific Reports}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} {#shows|exhibits|displays!} {activity_SARSCoV2#no &NULL}{activity_SARSCoV2__names__noun}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule is {activity_SARSCoV2#effectively &ineffectevely}{activity_SARSCoV2__names__gerund}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is from!} a molecule that {#shows|exhibits|displays!} {activity_SARSCoV2#no &NULL}{activity_SARSCoV2__names__noun}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {activity_SARSCoV2#not &NULL}{activity_SARSCoV2__names__gerund}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_SARSCoV2__names__gerund}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional|extra|!} words.
Result: {activity_SARSCoV2#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_SARSCoV2__names__gerund}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {activity_SARSCoV2#not &NULL}{activity_SARSCoV2__names__gerund}.""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {activity_SARSCoV2__names__gerund}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the molecule with the {SMILES__description} {SMILES#} is {activity_SARSCoV2__names__gerund}?
Assistant: {activity_SARSCoV2#No&Yes}, this molecule is {activity_SARSCoV2#not &NULL}{activity_SARSCoV2__names__gerund}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {activity_SARSCoV2__names__gerund}?
Assistant: {activity_SARSCoV2#No&Yes}, it is {activity_SARSCoV2#not &NULL}{activity_SARSCoV2__names__gerund}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {activity_SARSCoV2#not &NULL}{activity_SARSCoV2__names__gerund}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {activity_SARSCoV2#not &NULL}{activity_SARSCoV2__names__gerund}?
Assistant: This is a molecule that is {activity_SARSCoV2#not &NULL}{activity_SARSCoV2__names__gerund}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {activity_SARSCoV2#not &NULL}be {activity_SARSCoV2__names__gerund}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {activity_SARSCoV2#not &NULL}{activity_SARSCoV2__names__gerund}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {activity_SARSCoV2#not &NULL}be {activity_SARSCoV2__names__gerund}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {activity_SARSCoV2#not &NULL}{activity_SARSCoV2__names__gerund}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {activity_SARSCoV2__names__gerund}:<EOI> {activity_SARSCoV2#yes&no}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_SARSCoV2__names__gerund}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {activity_SARSCoV2#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_SARSCoV2__names__gerund}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result:<EOI> This molecule is {activity_SARSCoV2#not &NULL}{activity_SARSCoV2__names__gerund}.""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {activity_SARSCoV2__names__gerund}.
Result:<EOI> {SMILES#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {activity_SARSCoV2__names__gerund}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{activity_SARSCoV2%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {activity_SARSCoV2#not &NULL}{activity_SARSCoV2__names__gerund}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%activity_SARSCoV2%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {activity_SARSCoV2#not &NULL}{activity_SARSCoV2__names__gerund}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%activity_SARSCoV2%}
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
