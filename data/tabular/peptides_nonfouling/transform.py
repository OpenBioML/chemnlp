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
        "https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/human-positive.npz",
        "positive.npz",
    )
    urllib.request.urlretrieve(
        "https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/human-negative.npz",
        "negative.npz",
    )
    with np.load("positive.npz") as r:
        pos_data = r[list(r.keys())[0]]
    with np.load("negative.npz") as r:
        neg_data = r["seqs"]

    pos_seq = [decoder(s) for s in pos_data]
    neg_seq = [decoder(s) for s in neg_data]
    unique_pos = list(set(pos_seq))
    unique_neg = list(set(neg_seq))
    print(len(unique_pos) + len(unique_neg))

    seq_dict = {
        "sequence": unique_pos + unique_neg,
        "nonfouling": [True for _ in range(len(unique_pos))]
        + [False for _ in range(len(unique_neg))],
    }
    df = pd.DataFrame.from_dict(seq_dict, orient="index").transpose()
    df.to_csv("data_clean.csv", index=False)

    # create meta yaml
    meta = {
        "name": "peptides_nonfouling",  # unique identifier, we will also use this for directory names
        "description": """Non-fouling is defined as resistance to non-specific interactions.
        A non-fouling peptide (positive example) is defined using the mechanism proposed in
        ref white2012decoding. Briefly, ref white2012decoding, showed that the exterior surfaces
        of proteins have a significantly different frequency of amino acids, and this increases
        in aggregation prone environments, like the cytoplasm. Synthesizing self-assembling peptides
        that follow this amino acid distribution and coating surfaces with the peptides creates
        non-fouling surfaces. This pattern was also found inside chaperone proteins,
        another area where resistance to non-specific interactions is important (ref white2012role).""",
        "targets": [
            {
                "id": "nonfouling",  # name of the column in a tabular dataset
                "description": "The nonfouling activity of a peptide sequence (1) or not (0).",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "nonfouling activity"},
                    {"adjective": "nonfouling"},
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
                "url": "https://doi.org/10.18653/v1/K18-1030",
                "description": "data source",
            },
        ],
        "num_points": 6541,  # number of datapoints in this dataset
        "bibtex": [
            """@article{white2012decoding,
        title={Decoding nonspecific interactions from nature},
        author={White, Andrew D and Nowinski, Ann K and Huang, Wenjun and Keefe,
          Andrew J and Sun, Fang and Jiang, Shaoyi},
        journal={Chemical Science},
        volume={3},
        number={12},
        pages={3488--3494},
        year={2012},
        publisher={Royal Society of Chemistry}""",
            """@article{barrett2018classifying,
        title={Classifying antimicrobial and multifunctional peptides with Bayesian network models},
        author={Barrett, Rainier and Jiang, Shaoyi and White, Andrew D},
        journal={Peptide Science},
        volume={110},
        number={4},
        pages={e24079},
        year={2018},
        publisher={Wiley Online Library}""",
        ],
        "templates": [
            "The sequence of {#aminoacids|AAs!} {sequence#} {#shows|exhibits|demonstrates!} {nonfouling#no &NULL}{nonfouling__names__adjective} properties.",  # noqa: E501
            "The aminoacid sequence {sequence#} {#shows|exhibits|displays!} {nonfouling#no &NULL}{nonfouling__names__adjective} properties.",  # noqa: E501
            "Based on the {sequence__description} {#representation |!}{sequence#}, the aminoacid sequence has {nonfouling#no &NULL}{nonfouling__names__adjective} {#properties|characteristics|features!}.",  # noqa: E501
            "The {sequence__description} {sequence#} {#represents|is from!} an aminoacid sequence that is {nonfouling#not &NULL}identified as {nonfouling__names__adjective}.",  # noqa: E501
            "The {#aminoacid sequence |!}{sequence__description} {sequence#} is {nonfouling#not &NULL}{nonfouling__names__adjective}.",  # noqa: E501 not all variables need to be used
            # Instruction tuning text templates
            """Task: Please classify a aminoacid sequence based on the description.
Description: A aminoacid sequence that is {nonfouling__names__adjective}.
{#aminoacid sequence |!}{sequence__description}: {sequence#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result: {nonfouling#False&True}""",  # noqa: E501
            """Task: Please classify a aminoacid sequence based on the description.
Description: A aminoacid sequence that is {nonfouling__names__adjective}.
{#aminoacid sequence |!}{sequence__description}: {sequence#}
Constraint: Answer the question in a {#full|complete!} sentence.
Result: This aminoacid sequence is {nonfouling#not &NULL}{nonfouling__names__adjective}.""",
            """Task: Please {#give me|create|generate!} a {#aminoacid sequence |!}{sequence__description} based on the {#text |!}description{# below|!}.
Description: A aminoacid sequence that is {nonfouling__names__adjective}.
Result: {sequence#}""",  # noqa: E501
            # Conversational text templates
            """User: Can you {#tell me|derive|estimate!} if the aminoacid sequence with the {sequence__description} {sequence#} is {nonfouling__names__adjective}?
Assistant: {nonfouling#No&Yes}, this aminoacid sequence is {nonfouling#not &NULL}{nonfouling__names__adjective}.""",  # noqa: E501
            """User: Is the aminoacid sequence with the {sequence__description} {sequence#} {nonfouling__names__adjective}?
Assistant: {nonfouling#No&Yes}, it is {nonfouling#not &NULL}{nonfouling__names__adjective}.""",  # noqa: E501
            """User: Can you {#give me|create|generate!} the {sequence__description} of a aminoacid sequence that is {nonfouling#not &NULL}{nonfouling__names__adjective}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {sequence#}""",  # noqa: E501
            """User: I'm {#searching|looking!} for the {sequence__description} of a aminoacid sequence that is {nonfouling#not &NULL}{nonfouling__names__adjective}?
Assistant: This is a aminoacid sequence that is {nonfouling#not &NULL}{nonfouling__names__adjective}: {sequence#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#aminoacid sequence |!}{sequence__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The aminoacid sequence should {nonfouling#not &NULL}be {nonfouling__names__adjective}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {sequence__description} is {nonfouling#not &NULL}{nonfouling__names__adjective}: {sequence#}""",  # noqa: E501
            """User: I want to {#come up with|create|generate!} a {#aminoacid sequence |!}{sequence__description}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#aminoacid sequence|one!}?
User: Yes, the aminoacid sequence should {nonfouling#not &NULL}be {nonfouling__names__adjective}.
Assistant: {#Understood|Got it|Ok!}, this {sequence__description} is {nonfouling#not &NULL}{nonfouling__names__adjective}: {sequence#}""",  # noqa: E501
            # Benchmarking text templates
            "Is the {sequence__description} {sequence#} {nonfouling__names__adjective}:<EOI>{nonfouling#no&yes}",  # noqa: E501 for the benchmarking setup <EOI>separates input and output
            """Task: Please classify a aminoacid sequence based on the description.
Description: A aminoacid sequence that is {nonfouling__names__adjective}.
{#aminoacid sequence |!}{sequence__description}: {sequence#}
Constraint: Even if you are {#uncertain|not sure!}, you must pick either "True" or "False" without using any {#other|additional!} words.
Result:<EOI>{nonfouling#False&True}""",  # noqa: E501
            """Task: Please {#give me|create|generate!} a {#aminoacid sequence |!}{sequence__description} based on the {#text |!}description{# below|!}.
Description: A aminoacid sequence that is {nonfouling__names__adjective}.
Result:<EOI>{sequence#}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the aminoacid sequence with the {sequence__description} {#representation of |!}{sequence#} {nonfouling__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{nonfouling%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Is the aminoacid sequence with the {sequence__description} {#representation of |!}{sequence#} {nonfouling__names__adjective}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{nonfouling%}
Answer:<EOI>{%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which aminoacid sequences are {nonfouling#not &NULL}{nonfouling__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{sequence%nonfouling%}
Answer: {%multiple_choice_result}""",  # noqa: E501
            """Task: Please answer the multiple choice question.
Question: Which aminoacid sequences are {nonfouling#not &NULL}{nonfouling__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any {#other|additional!} words.
Options:
{sequence%nonfouling%}
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
