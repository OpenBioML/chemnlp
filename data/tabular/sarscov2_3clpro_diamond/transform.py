import pandas as pd
import yaml
from tdc.single_pred import HTS


def get_and_transform_data():
    # get raw data
    splits = HTS(name="SARSCoV2_3CLPro_Diamond").get_split()
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
    fields_clean = ["compound_id", "SMILES", "activity_SARSCoV2_3CLPro", "split"]
    df.columns = fields_clean

    # data cleaning
    df = df.dropna()
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "sarscov2_3clpro_diamond",  # unique identifier, we will also use this for directory names
        "description": """A large XChem crystallographic fragment screen against SARS-CoV-2
main protease at high resolution. From MIT AiCures.""",
        "targets": [
            {
                "id": "activity_SARSCoV2_3CLPro",  # name of the column in a tabular dataset
                "description": "activity against the SARSCoV2 3CL protease (1) or not (0)",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "activity against the SARSCoV2 3CL protease"},
                    {"noun": "activity against the SARS-CoV-2 3CL protease"},
                    {"adjective": "active against the SARSCoV2 3CL protease"},
                    {"adjective": "active against the SARS-CoV-2 3CL protease"},
                    {"gerund": "targeting the SARSCoV2 3CL protease"},
                    {"gerund": "acting against the SARSCoV2 3CL protease"},
                    {"gerund": "successfully targeting the SARSCoV2 3CL protease"},
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
                "url": "https://www.diamond.ac.uk/covid-19/for-scientists/Main-protease-structure-and-XChem.html",
                "description": "data source",
            },
            {
                "url": "https://www.diamond.ac.uk/dam/jcr:9fdc4297-15b6-47e2-8d53-befb0970bf7c/COVID19-summary-20200324.xlsx",  # noqa E501
                "description": "data source",
            },
            {
                "url": "http://doi.org/10.1021/jacs.9b02822",
                "description": "corresponding publication",
            },
            {
                "url": "https://doi.org/10.1016/j.jmb.2006.11.073",
                "description": "corresponding publication",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Resnick_2019,
doi = {10.1021/jacs.9b02822},
url = {https://doi.org/10.1021%2Fjacs.9b02822},
year = {2019},
month = may,
publisher = {American Chemical Society (ACS)},
volume = {141},
number = {22},
pages = {8951--8968},
author = {Efrat Resnick and Anthony Bradley and Jinrui Gan and Alice Douangamath
and Tobias Krojer and Ritika Sethi and Paul P. Geurink and Anthony Aimon and Gabriel Amitai
and Dom Bellini and James Bennett and Michael Fairhead and Oleg Fedorov and Ronen Gabizon and Jin Gan
and Jingxu Guo and Alexander Plotnikov and Nava Reznik and Gian Filippo Ruda and Laura Diaz-Saez and
Verena M. Straub and Tamas Szommer and Srikannathasan Velupillai and Daniel Zaidman and Yanling Zhang
and Alun R. Coker and Christopher G. Dowson and Haim M. Barr and Chu Wang and Kilian V.M. Huber
and Paul E. Brennan and Huib Ovaa and Frank von Delft and Nir London},
title = {Rapid Covalent-Probe Discovery by Electrophile-Fragment Screening},
journal = {Journal of the American Chemical Society}""",
            """@article{Xue_2007,
doi = {10.1016/j.jmb.2006.11.073},
url = {https://doi.org/10.1016%2Fj.jmb.2006.11.073},
year = {2007},
month = feb,
publisher = {Elsevier BV},
volume = {366},
number = {3},
pages = {965--975},
author = {Xiaoyu Xue and Haitao Yang and Wei Shen and Qi Zhao and Jun Li and Kailin Yang and
Cheng Chen and Yinghua Jin and Mark Bartlam and Zihe Rao},
title = {Production of Authentic {SARS}-{CoV} Mpro with Enhanced Activity: Application as
a Novel Tag-cleavage Endopeptidase for Protein Overproduction},
journal = {Journal of Molecular Biology}""",
        ],
        "templates": [
            "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} {#shows|exhibits|displays!} {activity_SARSCoV2_3CLPro#no &NULL}{activity_SARSCoV2_3CLPro__names__noun}.",  # noqa: E501
            "Based on the {SMILES__description} {#representation |!}{SMILES#}, the molecule is {activity_SARSCoV2_3CLPro#not &NULL}{activity_SARSCoV2_3CLPro__names__gerund}.",  # noqa: E501
            "The {SMILES__description} {SMILES#} {#represents|is from!} a molecule that {#shows|exhibits|displays!} {activity_SARSCoV2_3CLPro#no &NULL}{activity_SARSCoV2_3CLPro__names__noun}.",  # noqa: E501
            "The {#molecule |!}{SMILES__description} {SMILES#} is {activity_SARSCoV2_3CLPro#not &NULL}{activity_SARSCoV2_3CLPro__names__gerund}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_SARSCoV2_3CLPro__names__gerund}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional|extra|!} words.
Result: {activity_SARSCoV2_3CLPro#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_SARSCoV2_3CLPro__names__gerund}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This molecule is {activity_SARSCoV2_3CLPro#not &NULL}{activity_SARSCoV2_3CLPro__names__gerund}.""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {activity_SARSCoV2_3CLPro__names__gerund}.
Result: {SMILES#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the molecule with the {SMILES__description} {SMILES#} is {activity_SARSCoV2_3CLPro__names__gerund}?
Assistant: {activity_SARSCoV2_3CLPro#No&Yes}, this molecule is {activity_SARSCoV2_3CLPro#not &NULL}{activity_SARSCoV2_3CLPro__names__gerund}.""",  # noqa: E501
            """User: Is the molecule with the {SMILES__description} {SMILES#} {activity_SARSCoV2_3CLPro__names__gerund}?
Assistant: {activity_SARSCoV2_3CLPro#No&Yes}, it is {activity_SARSCoV2_3CLPro#not &NULL}{activity_SARSCoV2_3CLPro__names__gerund}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that is {activity_SARSCoV2_3CLPro#not &NULL}{activity_SARSCoV2_3CLPro__names__gerund}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that is {activity_SARSCoV2_3CLPro#not &NULL}{activity_SARSCoV2_3CLPro__names__gerund}?
Assistant: This is a molecule that is {activity_SARSCoV2_3CLPro#not &NULL}{activity_SARSCoV2_3CLPro__names__gerund}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should {activity_SARSCoV2_3CLPro#not &NULL}be {activity_SARSCoV2_3CLPro__names__gerund}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} is {activity_SARSCoV2_3CLPro#not &NULL}{activity_SARSCoV2_3CLPro__names__gerund}: {SMILES#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#molecule |!}{SMILES__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should {activity_SARSCoV2_3CLPro#not &NULL}be {activity_SARSCoV2_3CLPro__names__gerund}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} is {activity_SARSCoV2_3CLPro#not &NULL}{activity_SARSCoV2_3CLPro__names__gerund}: {SMILES#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {SMILES__description} {SMILES#} {activity_SARSCoV2_3CLPro__names__gerund}:<EOI> {activity_SARSCoV2_3CLPro#no&yes}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_SARSCoV2_3CLPro__names__gerund}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {activity_SARSCoV2_3CLPro#False&True}""",  # noqa: E501
            """Task: Please classify a molecule based on the description.
Description: A molecule that is {activity_SARSCoV2_3CLPro__names__gerund}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result:<EOI> This molecule is {activity_SARSCoV2_3CLPro#not &NULL}{activity_SARSCoV2_3CLPro__names__gerund}.""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that is {activity_SARSCoV2_3CLPro__names__gerund}.
Result:<EOI> {SMILES#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {activity_SARSCoV2_3CLPro__names__gerund}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{activity_SARSCoV2_3CLPro%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {activity_SARSCoV2_3CLPro#not &NULL}{activity_SARSCoV2_3CLPro__names__gerund}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%activity_SARSCoV2_3CLPro%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which molecules are {activity_SARSCoV2_3CLPro#not &NULL}{activity_SARSCoV2_3CLPro__names__gerund}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{SMILES%activity_SARSCoV2_3CLPro%}
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
