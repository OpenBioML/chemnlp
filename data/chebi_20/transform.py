import datasets
import yaml

SPLITS = ["train", "test", "validation"]
ORIGINAL_COLUMNS = ["CID", "SMILES", "description"]
NEW_COLUMNS = ["compound_id", "SMILES", "description"]

META_YAML_PATH = "./data/chebi_20/meta.yaml"
META_TEMPLATE = {
    "name": "chebi_20",  # unique identifier, we will also use this for directory names
    "description": "A dataset of pairs of natural language descriptions and SMILEs.",
    "targets": [
        {
            "id": "description",  # name of the column in a tabular dataset
            "description": "A natural language description of a SMILE",  # description of what this column means
            "units": None,  # units of the values in this column (leave empty if unitless)
            "type": "string",  # can be "categorical", "ordinal", "continuous"
            "names": [  # names for the property (to sample from for building the prompts)
                "natural language description",
            ],
        },
    ],
    "identifiers": [
        {
            "id": "SMILES",  # column name
            "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
            "description": "SMILES",  # description (optional, except for "OTHER")
        }
    ],
    "license": "CC BY 4.0",  # license under which the original dataset was published
    "links": [  # list of relevant links (original dataset, other uses, etc.)
        {
            "url": "https://aclanthology.org/2021.emnlp-main.47/",
            "description": "Original Text2Mol paper which introduced the dataset.",
        },
        {
            "url": "https://github.com/cnedwards/text2mol",
            "description": "Text2Mol data repository.",
        },
        {
            "url": "https://huggingface.co/datasets/OpenBioML/chebi_20",
            "description": "Hugging Face dataset for OpenBioML.",
        },
    ],
    "num_points": None,  # number of datapoints in this dataset
    "bibtex": [
        """@inproceedings{edwards2021text2mol,
            title={Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries},
            author={Edwards, Carl and Zhai, ChengXiang and Ji, Heng},
            booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
            pages={595--607},
            year={2021},
            url = {https://aclanthology.org/2021.emnlp-main.47/}
            }""",
    ],
}


def str_presenter(dumper, data: dict):
    """Configures yaml for dumping multiline strings"""
    if data.count("\n") > 0:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def get_dataset(split: str) -> datasets.Dataset:
    """
    Retrieve the dataset from Hugging Face
    Details on how to upload to Hugging Face
    https://huggingface.co/docs/datasets/upload_dataset
    """
    # 3 splits of train, val, test
    return datasets.load_dataset("OpenBioML/chebi_20", split=split, delimiter="\t")


def remove_whitespace(sample: dict) -> dict:
    sample["description"] = sample["description"].strip()
    return sample


def clean_dataset(hf_data: datasets.Dataset) -> datasets.Dataset:
    """Clean the dataset"""
    assert list(hf_data.features.keys()) == ORIGINAL_COLUMNS
    for old, new in zip(ORIGINAL_COLUMNS, NEW_COLUMNS):
        if old != new:
            hf_data.rename_column(old, new)
    return hf_data.map(remove_whitespace, num_proc=4)


def create_meta_yaml(num_points: int):
    """Create meta configuration file for the dataset"""
    # create meta yaml
    META_TEMPLATE["num_points"] = num_points
    with open(META_YAML_PATH, "w+") as f:
        yaml.dump(META_TEMPLATE, f, sort_keys=False)
    print(f"Finished processing chebi-20 {META_TEMPLATE['name']} dataset!")


if __name__ == "__main__":
    num_samples = 0
    for split in SPLITS:
        hf_data = get_dataset(split)
        hf_data_clean = clean_dataset(hf_data)
        num_samples += hf_data_clean.num_rows

    yaml.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)
    create_meta_yaml(num_samples)
