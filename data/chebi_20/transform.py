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
            "description": "A natural language description of the molecule SMILE",
            "units": None,  # units of the values in this column (leave empty if unitless)
            "type": "string",  # can be "categorical", "ordinal", "continuous", "string"
            "names": [  # names for the property (to sample from for building the prompts)
                "natural language description",
            ],
            "pubchem_aids": [],
            "uris": [],
        },
    ],
    "identifiers": [
        {
            "id": "SMILES",  # column name
            "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
            "description": "SMILES",  # description (optional, except for "OTHER")
        },
        {
            "id": "compound_id",
            "type": "OTHER",
            "description": "This is the PubChem CID to identify a given molecule.",
        },
    ],
    "license": "CC BY 4.0",  # license under which the original dataset was published
    "links": [  # list of relevant links (original dataset, other uses, etc.)
        {
            "name": "Research Paper",
            "url": "https://aclanthology.org/2021.emnlp-main.47/",
            "description": "Original Text2Mol paper which introduced the chebi_20 dataset.",
        },
        {
            "name": "Dataset",
            "url": "https://github.com/cnedwards/text2mol",
            "description": "Text2Mol original data repository on GitHub.",
        },
        {
            "name": "Hugging Face dataset upload",
            "url": "https://huggingface.co/datasets/OpenBioML/chebi_20",
            "description": "Hugging Face dataset uploaded to the OpenBioML organisation.",
        },
    ],
    "benchmarks": [],
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
        """@inproceedings{edwards-etal-2022-translation,
            title = "Translation between Molecules and Natural Language",
            author = "Edwards, Carl  and
                Lai, Tuan  and
                Ros, Kevin  and
                Honke, Garrett  and
                Cho, Kyunghyun  and
                Ji, Heng",
            booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
            month = dec,
            year = "2022",
            address = "Abu Dhabi, United Arab Emirates",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2022.emnlp-main.26",
            pages = "375--413",
            abstract = "We present MolT5 - a self-supervised learning framework for pretraining models on a vast amount of unlabeled natural language text and molecule strings. MolT5 allows for new, useful, and challenging analogs of traditional vision-language tasks, such as molecule captioning and text-based de novo molecule generation (altogether: translation between molecules and language), which we explore for the first time. Since MolT5 pretrains models on single-modal data, it helps overcome the chemistry domain shortcoming of data scarcity. Furthermore, we consider several metrics, including a new cross-modal embedding-based metric, to evaluate the tasks of molecule captioning and text-based molecule generation. Our results show that MolT5-based models are able to generate outputs, both molecules and captions, which in many cases are high quality.",
            }""",  # noqa
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
