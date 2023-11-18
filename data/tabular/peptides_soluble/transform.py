import urllib.request

import numpy as np
import pandas as pd
import yaml


def decoder(seq_vector):
    alphabet = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    seq = []
    for _, index in enumerate(seq_vector.astype("int")):
        if index == 0:
            break
        seq.append(alphabet[index - 1])
    seq = "".join(seq)
    return seq


def get_and_transform_data():
    urllib.request.urlretrieve(
        "https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/insoluble.npz",
        "negative.npz",
    )
    urllib.request.urlretrieve(
        "https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/soluble.npz",
        "positive.npz",
    )
    with np.load("positive.npz") as r:
        pos_data = r["arr_0"]
    with np.load("negative.npz") as r:
        neg_data = r["arr_0"]

    pos_seq = [decoder(s) for s in pos_data]
    neg_seq = [decoder(s) for s in neg_data]
    unique_pos = list(set(pos_seq))
    unique_neg = list(set(neg_seq))

    seq_dict = {
        "sequence": unique_pos + unique_neg,
        "soluble": [True for _ in range(len(unique_pos))]
        + [False for _ in range(len(unique_neg))],
    }
    df = pd.DataFrame.from_dict(seq_dict, orient="index").transpose()
    df["split"] = pd.NA
    split = ["train", "valid", "test"]
    df["split"] = df["split"].apply(
        lambda x: np.random.choice(split, p=[0.70, 0.15, 0.15])
    )
    print(df["split"].value_counts())
    df.to_csv("data_clean.csv", index=False)

    # create meta yaml
    meta = {
        "name": "peptides_soluble",  # unique identifier, we will also use this for directory names
        "description": """Solubility was estimated by retrospective analysis of electronic laboratory notebooks.
        The notebooks were part of a large effort called the Protein Structure Initiative and consider sequences
        linearly through the following stages: Selected,  Cloned,  Expressed,  Soluble,  Purified, Crystallized,
        HSQC (heteronuclear single quantum coherence), Structure, and deposited in PDB. The peptides were identified
        as soluble or insoluble by \"Comparing the experimental status at two time points, September 2009 and May 2010,
        we were able to derive a set of insoluble proteins defined as those which were not
        soluble in September 2009 and still remained in that state 8 months later.\"""",
        "targets": [
            {
                "id": "soluble",  # name of the column in a tabular dataset
                "description": "The solubility of a peptide sequence (1) or not (0).",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "solubility"},
                    {"adjective": "soluble"},
                ],
                "uris": None,
            },
        ],
        "benchmarks": [],
        "identifiers": [
            {
                "id": "sequence",  # column name
                "type": "Other",
                "description": "amino acid sequence",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1021/acs.jcim.2c01317",
                "description": "corresponding publication",
            },
            {
                "url": "https://doi.org/10.1111/j.1742-4658.2012.08603.x",
                "description": "data source",
            },
        ],
        "num_points": 6541,  # number of datapoints in this dataset
        "bibtex": [
            """@article{berman2009protein,
        title={The protein structure initiative structural genomics knowledgebase},
        author={Berman, Helen M and Westbrook, John D and Gabanyi, Margaret J and Tao,
          Wendy and Shah, Raship and Kouranov, Andrei and Schwede, Torsten and Arnold,
            Konstantin and Kiefer, Florian and Bordoli, Lorenza and others},
        journal={Nucleic acids research},
        volume={37},
        number={suppl1},
        pages={D365--D368},
        year={2009},
        publisher={Oxford University Press}""",
            """@article{smialowski2012proso,
        title={PROSO II--a new method for protein solubility prediction},
        author={Smialowski, Pawel and Doose, Gero and Torkler, Phillipp and Kaufmann,
          Stefanie and Frishman, Dmitrij},
        journal={The FEBS journal},
        volume={279},
        number={12},
        pages={2192--2200},
        year={2012},
        publisher={Wiley Online Library}""",
        ],
        "templates": [
            "The sequence of {#aminoacids|AAs!} {sequence#} {#shows|exhibits|demonstrates!} {soluble#no &NULL}{soluble__names__adjective} properties.",  # noqa: E501
            "The aminoacid sequence {sequence#} {#shows|exhibits|displays!} {soluble#no &NULL}{soluble__names__adjective} properties.",  # noqa: E501
            "Based on the {sequence__description} {#representation |!}{sequence#}, the aminoacid sequence has {soluble#no &NULL}{soluble__names__adjective} {#properties|characteristics|features!}.",  # noqa: E501
            "The {sequence__description} {sequence#} {#represents|is from!} an aminoacid sequence that is {soluble#not &NULL}identified as {soluble__names__adjective}.",  # noqa: E501
            "The {#aminoacid sequence |!}{sequence__description} {sequence#} is {soluble#not &NULL}{soluble__names__adjective}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a aminoacid sequence based on the description.
Description: A aminoacid sequence that is {soluble__names__adjective}.
{#aminoacid sequence |!}{sequence__description}: {sequence#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result: {soluble#False&True}""",  # noqa: E501
            """Task: Please classify a aminoacid sequence based on the description.
Description: A aminoacid sequence that is {soluble__names__adjective}.
{#aminoacid sequence |!}{sequence__description}: {sequence#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This aminoacid sequence is {soluble#not &NULL}{soluble__names__adjective}.""",
            """Task: Please {#give me|create|generate!} a {#aminoacid sequence |!}{sequence__description} based on the {#text |!}description{# below|!}.
Description: A aminoacid sequence that is {soluble__names__adjective}.
Result: {sequence#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the aminoacid sequence with the {sequence__description} {sequence#} is {soluble__names__adjective}?
Assistant: {soluble#No&Yes}, this aminoacid sequence is {soluble#not &NULL}{soluble__names__adjective}.""",  # noqa: E501
            """User: Is the aminoacid sequence with the {sequence__description} {sequence#} {soluble__names__adjective}?
Assistant: {soluble#No&Yes}, it is {soluble#not &NULL}{soluble__names__adjective}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {sequence__description} of a aminoacid sequence that is {soluble#not &NULL}{soluble__names__adjective}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {sequence#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {sequence__description} of a aminoacid sequence that is {soluble#not &NULL}{soluble__names__adjective}?
Assistant: This is a aminoacid sequence that is {soluble#not &NULL}{soluble__names__adjective}: {sequence#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#aminoacid sequence |!}{sequence__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The aminoacid sequence should {soluble#not &NULL}be {soluble__names__adjective}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {sequence__description} is {soluble#not &NULL}{soluble__names__adjective}: {sequence#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#aminoacid sequence |!}{sequence__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#aminoacid sequence|one!}?
User: Yes, the aminoacid sequence should {soluble#not &NULL}be {soluble__names__adjective}.
Assistant: {#Understood|Got it|Ok!}, this {sequence__description} is {soluble#not &NULL}{soluble__names__adjective}: {sequence#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {sequence__description} {sequence#} {soluble__names__adjective}:<EOI> {soluble#no&yes}",  # noqa: E501 for the benchmarking setup <EOI> separates input and output
            """Task: Please classify a aminoacid sequence based on the description.
Description: A aminoacid sequence that is {soluble__names__adjective}.
{#aminoacid sequence |!}{sequence__description}: {sequence#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI> {soluble#False&True}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#aminoacid sequence |!}{sequence__description} based on the {#text |!}description{# below|!}.
Description: A aminoacid sequence that is {soluble__names__adjective}.
Result:<EOI> {sequence#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the aminoacid sequence with the {sequence__description} {#representation of |!}{sequence#} {soluble__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{soluble%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the aminoacid sequence with the {sequence__description} {#representation of |!}{sequence#} {soluble__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{soluble%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which aminoacid sequences are {soluble#not &NULL}{soluble__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{sequence%soluble%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which aminoacid sequences are {soluble#not &NULL}{soluble__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{sequence%soluble%}
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
