import re
from datasets import  load_dataset
import pandas as pd
import yaml 


LINES_TO_REMOVE = "/workspaces/chemnlp/data/libre_textbooks/lines_to_remove.jsonl"
RAW_DATASET = "Hack90/libre_chem_textbooks"
SPLITS = ["train"]


META_YAML_PATH = "./data/libre_textbooks/meta.yaml"
META_TEMPLATE = {
    "name": "libre_textbooks",  # unique identifier, we will also use this for directory names
    "description": "A dataset of pairs of natural language descriptions and SMILEs.",
    "targets": [
        {
            "id": "text",  # name of the column in a text dataset
            "description": "A line of string from the textbook",
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
            "id": "book",  # column name
            "type": "string",  # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
            "description": "name of the textbook the string line is from",  
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
            "name": "Google Drive host",
            "url": "https://drive.google.com/drive/folders/1iZFXnIw_wnXoR04-hiRmW2ax2b0Nx4qm?usp=sharing",
            "description": "Website hosting textbook pdfs.", # I'm hoping to move this to github soon
        },
        {
            "name": "Hugging Face dataset upload",
            "url": "https://huggingface.co/datasets/Hack90/libre_chem_textbooks",
            "description": "Hugging Face dataset uploaded to HF account", # Hopefully will move this the openbioml space
        },
    ],
    "benchmarks": [],
    "num_points": None,  # number of datapoints in this dataset
    "bibtex": [
      # noqa
    ],
}


def remove_hyperlinks_licences_and_chapter_heads(text: str):
    """Remove hyperlinks, licences, and chapter heads from text."""
    if bool(re.search(r'\d+\.\d+:', text)):
        text = ''
    if bool(re.search(r'\d+\.\d+\.\d+', text)):
        text = ''
    if bool(re.search(r'\d+:', text)):
        text = ''
    if 'CC BY' in text:
        text = ''
    if 'yout' in text:
        text = ''
    if 'Libre' in text:
        text = ''
    if 'curated by' in text:
        text = ''
    if 'licens' in text.lower():
        text = ''
    if 'licens' in text.lower():
        text = ''
    text = re.sub(r'http\S+', '', text)
    return text


def get_raw_data(raw_dataset: str = RAW_DATASET) -> pd.DataFrame:
    """Load the raw dataset into a pandas dataframe"""
    dataset = load_dataset(raw_dataset)
    df_raw = pd.DataFrame(dataset['train'].to_pandas())
    df_raw['text'] = df_raw['text'].apply(remove_hyperlinks_licences_and_chapter_heads)
    return df_raw


def clean_dataset(df_raw: pd.DataFrame, lines_to_remove: str = LINES_TO_REMOVE) -> pd.DataFrame:
    """Clean the raw dataset and return a clean pandas dataframe""" 
    df_to_remove  = pd.read_json(lines_to_remove, lines=True)
    df_to_remove = df_to_remove[df_to_remove['answer'] == 'reject']
    df_clean = pd.DataFrame()
    df_clean = df_raw
    df_clean['common_values'] = df_clean["text"].isin(df_to_remove["text"])
    df_clean = df_clean[df_clean['common_values'] == False]
    df_clean = df_clean[df_clean['text'].str.strip() != '']
    return df_clean


def create_meta_yaml(num_points: int):
    """Create meta configuration file for the dataset"""
    # create meta yaml
    META_TEMPLATE["num_points"] = num_points
    with open(META_YAML_PATH, "w+") as f:
        yaml.dump(META_TEMPLATE, f, sort_keys=False)
    print(f"Finished processing libre_textbooks {META_TEMPLATE['name']} dataset!")


if __name__ == "__main__":
    num_samples = 0
    for split in SPLITS:
        raw_df = get_raw_data()
        clean_df = clean_dataset(raw_df)
        num_samples += len(clean_df)

    create_meta_yaml(num_samples)
