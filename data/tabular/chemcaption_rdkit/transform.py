import pandas as pd 
from huggingface_hub import list_repo_files
from tqdm import tqdm
import fire
from datasets import load_dataset


def process():
    df = pd.read_parquet('https://huggingface.co/datasets/kjappelbaum/chemnlp-chem-caption/resolve/main/rdkit_feat/train-00000-of-00001-27355e7935aa33a9.parquet')
    print(df.columns)
    df.to_csv('data_clean.csv', index=False)

if __name__ == '__main__':
    fire.Fire(process)