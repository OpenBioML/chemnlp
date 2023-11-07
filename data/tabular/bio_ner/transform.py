import glob
import os
import subprocess

import pandas as pd
import yaml

from chemnlp.data.ner import cleaner

# create meta yaml
meta_template = {
    "name": None,
    "description": "NER data.",
    "identifiers": None,
    "targets": None,
    "benchmarks": [
        {
            "name": "???",
            "link": "???",
            "split_column": "split",  # name of the column that contains the split information
        },
    ],
    "license": "NEEDS TO BE DEFINED",
    "links": [
        {
            "url": "https://github.com/ML4LitS/bio-datasets",
            "description": "???",
        },
    ],
    "num_points": None,
    "bibtex": ["???"],
    "templates": [
        """Task: Please carry out the {#named entity recognition (NER)|named entity recognition|NER!} task for the the text below.
Text: {#Sentence}.
Constrain: Please, {#only |!}list the entities in the form NER entity, span start, span end, and type {#in separate lines |!}with a high probability of being in the text.
Result: {#entity_1}""",  # noqa: E501
    ],
}


def get_entity_count(x):
    return int(x.split("/")[-1].split("_")[1])


def read_and_concat(fns):
    dfs = []
    for fn in fns:
        df = pd.read_csv(fn, sep="\t")
        dfs.append(df)
    return pd.concat(dfs)


def str_presenter(dumper, data: str):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
    """
    if data.count("\n") > 0:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def get_and_transform_data():
    path_base = os.getcwd()
    print(path_base)

    # clone repo if it is not there
    if not (os.path.isdir(path_base + "/bio-datasets")):
        subprocess.check_call(
            ["git", "clone", "https://github.com/ML4LitS/bio-datasets.git"]
        )
    else:
        print("Reusing the cloned repo.")

    # get tsv data paths
    paths = sorted(glob.glob(path_base + "/bio-datasets/processed_NER/**/*.tsv"))

    # create dict with entity count as key
    data = {}
    for path in paths:
        entity_count = get_entity_count(path)
        if entity_count in data:
            data[entity_count].append(path)
        else:
            data[entity_count] = [path]

    # merge data with the same entity count to one file in separate dirs then also save yaml
    for entity_count in data:
        # subselect entity count
        # if entity_count > 2:
        #    continue

        df = read_and_concat(data[entity_count])

        if entity_count == 1:
            path_export = path_base
        else:
            path_export = (
                "/".join(path_base.split("/")[:-1]) + f"/bio_ner_{entity_count}"
            )
            os.makedirs(path_export, exist_ok=True)

        df["Sentence"] = df["Sentence"].apply(cleaner)
        fn_data_clean = path_export + "/data_clean.csv"
        df.to_csv(fn_data_clean, index=False)

        dataset_name = path_export.split("/")[-1]
        meta_copy = meta_template.copy()
        meta_copy["name"] = dataset_name
        meta_copy["num_points"] = len(df)
        meta_copy["identifiers"] = [
            {"id": c, "description": c, "type": "Other"}
            for c in df.columns
            if (("1" not in c) and (c != "json") and (c != "split"))
        ]
        meta_copy["targets"] = [
            {
                "id": c,
                "description": c,
                "type": "Other",
                "units": c,
                "names": [{"noun": c}],
            }
            for c in df.columns
            if c[-1].isnumeric()
        ]
        meta_copy["targets"] += [
            {
                "id": "json",
                "description": "json",
                "type": "Other",
                "units": None,
                "names": [{"noun": "JSON output"}],
            }
        ]
        # adapt templates for more entities than 1
        if entity_count > 1:
            entity_str = "\n".join(
                ["{#entity_" + str(i + 1) + "}" for i in range(entity_count)]
            )
            meta_copy["templates"] = [
                t.replace("{#entity_1}", entity_str) for t in meta_copy["templates"]
            ]

        yaml.add_representer(str, str_presenter)
        yaml.representer.SafeRepresenter.add_representer(str, str_presenter)
        fn_meta = path_export + "/meta.yaml"
        with open(fn_meta, "w") as f:
            yaml.dump(meta_copy, f, sort_keys=False)

        print(dataset_name)


if __name__ == "__main__":
    get_and_transform_data()
