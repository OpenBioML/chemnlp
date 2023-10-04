import pandas as pd
import yaml
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list


def get_and_transform_data():
    # get raw data
    target_folder = "Tox21"
    label_list = retrieve_label_name_list(f"{target_folder}")
    target_subfolder = f"{label_list[1]}"
    splits = Tox(name=f"{target_folder}", label_name=target_subfolder).get_split()
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
    fields_clean = ["compound_id", "SMILES", f"toxicity_{target_subfolder}", "split"]
    df.columns = fields_clean

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df["toxicity_NR-AR-LBD"] = df["toxicity_NR-AR-LBD"].astype(int)
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "nr_ar_lbd_tox21",  # unique identifier, we will also use this for directory names
        "description": """Tox21 is a data challenge which contains qualitative toxicity measurements
for 7,831 compounds on 12 different targets, such as nuclear receptors and stress
response pathways.""",
        "targets": [
            {
                "id": f"toxicity_{target_subfolder}",  # name of the column in a tabular dataset
                "description": "whether it shows activity in the NR-AR-LBD toxicity assay (1) or not (0)",
                "units": None,
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "NR-AR-LBD toxicity"},
                    {"noun": "androgen receptor ligand-binding domain toxicity"},
                    {"verb": "is active in the NR-AR-LBD toxicity assay"},
                    {"adjective": "toxic in the NR-AR-LBD assay"},
                    {
                        "adjective": "toxic in the androgen receptor ligand-binding domain assay"
                    },
                    {
                        "gerund": "displaying toxicity in the NR-AR ligand binding domain assay"
                    },
                    {
                        "gerund": "exhibiting toxicity in the NR-androgen-LBD receptor alpha assay"
                    },
                    {
                        "gerund": "demonstrating toxicity in the NR-androgen-LBD receptor alpha assay"
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
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "description": "SMILES",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "http://dx.doi.org/10.3389/fenvs.2017.00003",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#tox21",
                "description": "data source",
            },
            {
                "url": "https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2523-5/tables/3",
                "description": "assay name",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Huang2017,
doi = {10.3389/fenvs.2017.00003},
url = {https://doi.org/10.3389/fenvs.2017.00003},
year = {2017},
month = jan,
publisher = {Frontiers Media SA},
volume = {5},
author = {Ruili Huang and Menghang Xia},
title = {Editorial: Tox21 Challenge to Build Predictive Models of Nuclear Receptor
and Stress Response Pathways As Mediated by Exposure to Environmental Toxicants and Drugs},
journal = {Frontiers in Environmental Science}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} is {toxicity_NR-AR-LBD#not &NULL}{toxicity_NR-AR-LBD__names__adjective}.",  # noqa: E501
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} is {toxicity_NR-AR-LBD#not &NULL}{toxicity_NR-AR-LBD__names__gerund}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule has {toxicity_NR-AR-LBD#no &NULL}{toxicity_NR-AR-LBD__names__noun} {#properties|characteristics|features!}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is from!} a molecule that is {toxicity_NR-AR-LBD#not &NULL}identified as {toxicity_NR-AR-LBD__names__adjective}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {toxicity_NR-AR-LBD#not &NULL}{toxicity_NR-AR-LBD__names__adjective}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {toxicity_NR-AR-LBD__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional|extra!} words.
Result: {toxicity_NR-AR-LBD#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {toxicity_NR-AR-LBD__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {toxicity_NR-AR-LBD#not &NULL}{toxicity_NR-AR-LBD__names__adjective}.""",
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {toxicity_NR-AR-LBD__names__adjective}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|figure out|estimate!} if the molecule with the {SMILES__description} {SMILES#} is {toxicity_NR-AR-LBD__names__adjective}?
Assistant: {toxicity_NR-AR-LBD#No&Yes}, this molecule is {toxicity_NR-AR-LBD#not &NULL}{toxicity_NR-AR-LBD__names__adjective}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {toxicity_NR-AR-LBD__names__adjective}?
Assistant: {toxicity_NR-AR-LBD#No&Yes}, it is {toxicity_NR-AR-LBD#not &NULL}{toxicity_NR-AR-LBD__names__adjective}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {toxicity_NR-AR-LBD#not &NULL}{toxicity_NR-AR-LBD__names__adjective}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {toxicity_NR-AR-LBD#not &NULL}{toxicity_NR-AR-LBD__names__adjective}?
Assistant: This is a molecule that is {toxicity_NR-AR-LBD#not &NULL}{toxicity_NR-AR-LBD__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: This sounds {#very exciting. |very interesting. | very curious. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {toxicity_NR-AR-LBD#not &NULL}be {toxicity_NR-AR-LBD__names__adjective}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {toxicity_NR-AR-LBD#not &NULL}{toxicity_NR-AR-LBD__names__adjective}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {toxicity_NR-AR-LBD#not &NULL}be {toxicity_NR-AR-LBD__names__adjective}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {toxicity_NR-AR-LBD#not &NULL}{toxicity_NR-AR-LBD__names__adjective}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {toxicity_NR-AR-LBD__names__adjective}:<EOI> {toxicity_NR-AR-LBD#no&yes}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {toxicity_NR-AR-LBD__names__adjective}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {toxicity_NR-AR-LBD#False&True}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {toxicity_NR-AR-LBD__names__adjective}.
Result:<EOI> {SMILES#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {toxicity_NR-AR-LBD__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{toxicity_NR-AR-LBD%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {toxicity_NR-AR-LBD__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{toxicity_NR-AR-LBD%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {toxicity_NR-AR-LBD#not &NULL}{toxicity_NR-AR-LBD__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%toxicity_NR-AR-LBD%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {toxicity_NR-AR-LBD#not &NULL}{toxicity_NR-AR-LBD__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%toxicity_NR-AR-LBD%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501,
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
