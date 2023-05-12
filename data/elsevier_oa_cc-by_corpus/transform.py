import glob
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import requests
import yaml
from tqdm import tqdm


def get_text_from_json_data(data):
    # sort sentences
    data["body_text"].sort(key=lambda e: e["startOffset"])

    # get text data from json dict
    data_text = {}

    data_text["title"] = data["metadata"]["title"]

    if "subjareas" in data["metadata"]:
        data_text["subject_classification"] = [
            e for i, e in enumerate(data["metadata"]["subjareas"])
        ]
        data_text["subject_classification"] = [
            e + "," for e in data_text["subject_classification"][:-1]
        ] + [data_text["subject_classification"][-1]]

    if "asjc" in data["metadata"]:
        data_text["all_science_journal_classification"] = [
            e for i, e in enumerate(data["metadata"]["asjc"])
        ]
        data_text["all_science_journal_classification"] = [
            e + "," for e in data_text["all_science_journal_classification"][:-1]
        ] + [data_text["all_science_journal_classification"][-1]]

    if "keywords" in data["metadata"]:
        data_text["keywords"] = [
            f"{e}," for i, e in enumerate(data["metadata"]["keywords"])
        ]

    data_text["abstract"] = data["abstract"]

    if "author_highlights" in data:
        # sort author highlights
        data["author_highlights"].sort(key=lambda e: e["startOffset"])
        data_text["author_highlights"] = [
            f"{i+1}. {e['sentence']}\n"
            for i, e in enumerate(data["author_highlights"])
        ]
        data_text["author_highlights"] = "".join(data_text["author_highlights"])

    for e in data["body_text"]:
        if e["title"] in data_text:
            data_text[e["title"]].append(e["sentence"])
        else:
            data_text[e["title"]] = []
            data_text[e["title"]].append(e["sentence"])

    # create full_text
    full_text = ""

    for key in data_text:
        full_text += f"\n\n# {key.replace('_',' ').title()}\n"
        if isinstance(data_text[key], str):
            full_text += data_text[key]
        else:
            full_text += " ".join(data_text[key])

    return data_text, full_text


def get_and_transform_data():
    # get raw data
    fn_zip_original = Path("elsevier_oa_cc-by_corpus_v3.zip")
    if not (fn_zip_original.is_file()):
        zip_path = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zm33cdndxs-3.zip"
        data = requests.get(zip_path)

        with open(fn_zip_original, "wb") as f:
            f.write(data.content)
        del data

        hash_func = hashlib.new("sha256")
        with open(fn_zip_original, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        sha256_hash = hash_func.hexdigest()
        sha256_hash_orig = (
            "380ee2737cc341ab15d4de1481d2c2f809de8dfefdf639c158c479249a085e6a"
        )
        assert (
            sha256_hash == sha256_hash_orig
        ), "Sha256 checksum doesn't match, delete downloaded file and restart this script."
        del sha256_hash

    # unzip data
    fn_dir_original = Path("elsevier_oa_cc-by_corpus_v3")
    if not (fn_dir_original.is_dir()):
        ZipFile(fn_zip_original).extractall(fn_dir_original)
        ZipFile(fn_dir_original / "json.zip").extractall(fn_dir_original)

    # load json data
    data_json_clean    = []
    data_json_separate = []
    for filename in tqdm(glob.glob(str(fn_dir_original / "json/*.json"))):
        dict_json_clean    = {}
        dict_json_separate = {}
        dict_json_clean["filename"]    = filename.split("elsevier_oa_cc-by_corpus_v3/json/")[-1]
        dict_json_separate["filename"] = filename.split("elsevier_oa_cc-by_corpus_v3/json/")[-1]
        with open(filename) as f:
            data_json_raw = json.load(f)
        if "abstract" in data_json_raw:
            data_text, dict_json_clean["text"] = get_text_from_json_data(data_json_raw)
            dict_json_separate = {**dict_json_separate, **data_text}
        else:
            continue
        data_json_clean.append(dict_json_clean)
        data_json_separate.append(dict_json_separate)

    assert len(data_json_clean) == len(data_json_separate)

    # save data as jsonl
    with open("data_clean.jsonl", "w") as file_out:
        for element in data_json_clean:
            json.dump(element, file_out)
            file_out.write("\n")

    with open("data_separate.jsonl", "w") as file_out:
        for element in data_json_separate:
            json.dump(element, file_out)
            file_out.write("\n")

    # create meta yaml
    meta = {
        "name": "elsevier_oa_cc-by_corpus",  # unique identifier, we will also use this for directory names
        "description": """This is a corpus of 40k (40,001) open access (OA)
CC-BY articles from across Elsevier’s journals represent the first
cross-discipline research of data at this scale to support NLP and ML
research.
This dataset was released to support the development of ML and NLP models
targeting science articles from across all research domains. While the release
builds on other datasets designed for specific domains and tasks, it will allow
for similar datasets to be derived or for the development of models which can
be applied and tested across domains.""",
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://elsevier.digitalcommonsdata.com/datasets/zm33cdndxs/3",
                "description": "original dataset link",
            },
        ],
        "num_points": len(data_json_clean),  # number of datapoints in this dataset
        "bibtex": [
            """@article{DBLP:journals/corr/abs-2008-00774,
author       = {Daniel Kershaw and Rob Koeling},
title        = {Elsevier {OA} CC-By Corpus},
journal      = {CoRR},
volume       = {abs/2008.00774},
year         = {2020},
url          = {https://arxiv.org/abs/2008.00774},
eprinttype   = {arXiv},
eprint       = {2008.00774},
timestamp    = {Fri, 07 Aug 2020 15:07:21 +0200},
biburl       = {https://dblp.org/rec/journals/corr/abs-2008-00774.bib},
bibsource    = {dblp computer science bibliography, https://dblp.org}
}""",
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
