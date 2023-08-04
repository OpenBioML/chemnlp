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
        "https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/hemo-positive.npz",
        "positive.npz",
    )
    urllib.request.urlretrieve(
        "https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/hemo-negative.npz",
        "negative.npz",
    )
    with np.load("positive.npz") as r:
        pos_data = r[list(r.keys())[0]]
    with np.load("negative.npz") as r:
        neg_data = r[list(r.keys())[0]]

    pos_seq = [decoder(s) for s in pos_data]
    neg_seq = [decoder(s) for s in neg_data]
    unique_pos = list(set(pos_seq))
    unique_neg = list(set(neg_seq))

    seq_dict = {
        "sequence": unique_pos + unique_neg,
        "hemolytic": [True for _ in range(len(unique_pos))]
        + [False for _ in range(len(unique_neg))],
    }
    df = pd.DataFrame.from_dict(seq_dict, orient="index").transpose()
    df.to_csv("clean_data.csv", index=False)

    # create meta yaml
    meta = {
        "name": "peptides_hemolytic",  # unique identifier, we will also use this for directory names
        "description": """Hemolysis is referred to the disruption of erythrocyte
membranes that decrease the life span of red blood cells and causes
the release of Hemoglobin. It is critical to identify non-hemolytic
antimicrobial peptides as a non-toxic and safe measure against bacterial
infections. However, distinguishing between hemolytic and non-hemolytic
peptides is a challenge, since they primarily exert their activity at the
charged surface of the bacterial plasma membrane.
The data here comes from the Database of Antimicrobial Activity and Structure of
Peptides (DBAASP v3). Hemolytic activity is defined by extrapolating a measurement
assuming dose response curves to the point
at which 50% of red blood cells are lysed. Activities below 100 mu g/ml, are
considered hemolytic.
The data contains sequences of only L- and canonical amino acids. Each measurement
is treated independently, so sequences can appear multiple times. This experimental
dataset contains noise, and in some observations (40%), an identical sequence appears
in both negative and positive class. As an example, sequence "RVKRVWPLVIRTVIAGYNLYRAIKKK"
is found to be both hemolytic and
non-hemolytic in two different lab experiments (i.e. two different training examples). """,
        "targets": [
            {
                "id": "hemolytic",  # name of the column in a tabular dataset
                "description": "The ability of a peptide sequence to lyse red blood cells (1) or not (0).",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "hemolytic activity"},
                    {"noun": "hemolysis"},
                    {"verb": "kill red blood cells"},
                    {"verb": "lyse red blood cells"},
                    {"adjective": "hemolytic"},
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
                "url": "https://doi.org/10.1093/nar/gkaa991",
                "description": "data source",
            },
        ],
        "num_points": 6541,  # number of datapoints in this dataset
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
