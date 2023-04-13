import pandas as pd
import yaml
from datasets import load_dataset

LINES_TO_REMOVE = "/workspaces/chemnlp/data/libre_textbooks/lines_to_remove.jsonl"
RAW_DATASET = "Hack90/libre_chem_textbooks"


META_YAML_PATH = "./data/libre_textbooks/meta.yaml"
META_TEMPLATE = {
    "name": "libre_textbooks",  # unique identifier, we will also use this for directory names
    "description": "A dataset of scraped articles from libre textbooks",
    "targets": [
        {
            "id": "html",  # name of the column in a tabular dataset
            "description": "A scraped page from libre textbooks",
            "units": None,  # units of the values in this column (leave empty if unitless)
            "type": "string",  # can be "categorical", "ordinal", "continuous", "string"
            "names": [  # names for the property (to sample from for building the prompts)
                "natural language article",
            ],
            "pubchem_aids": [],
            "uris": [],
        },
    ],
    "identifiers": [
        {
            "id": "url ",  # column name
            "type": "string",  # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
            "description": "url of the page the content is scraped from",
        },
        {
            "id": "text_length",  # text character count
            "type": "int",  # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
            "description": "text character count",
        },
    ],
    "license": "CC BY 4.0",  # license under which the original dataset was published
    "links": [  # list of relevant links (original dataset, other uses, etc.)
        {
            "name": "Libre Textbooks",
            "url": "https://chem.libretexts.org/Bookshelves",
            "description": "",
        },
        {
            "name": "Hugging Face dataset upload",
            "url": "https://huggingface.co/datasets/Hack90/libre_chem_textbooks",
            "description": "Hugging Face dataset uploaded to HF account",  # Hopefully will move this to the openbioml space
        },
    ],
    "benchmarks": [],
    "num_points": 3740,  # number of datapoints in this dataset
    "bibtex": [
        # noqa
    ],
}


def get_raw_data(raw_dataset: str = RAW_DATASET) -> pd.DataFrame:
    """Load the raw dataset into a pandas dataframe"""
    dataset = load_dataset(raw_dataset)
    df_raw = pd.DataFrame(dataset["train"].to_pandas())
    return df_raw


def create_meta_yaml(num_points: int):
    """Create meta configuration file for the dataset"""
    # create meta yaml
    META_TEMPLATE["num_points"] = num_points
    with open(META_YAML_PATH, "w+") as f:
        yaml.dump(META_TEMPLATE, f, sort_keys=False)
    print(f"Finished processing libre_textbooks {META_TEMPLATE['name']} dataset!")


if __name__ == "__main__":
    num_samples = 0
    raw_df = get_raw_data()
    num_samples += len(raw_df)
    create_meta_yaml(num_samples)
