import pandas as pd
from huggingface_hub import hf_hub_download


def process():
    file_train = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-uspto",
        filename="US_patents_1976-Sep2016_1product_reactions_test_prompts.json",
        repo_type="dataset",
    )
    df_train = pd.read_json(file_train)
    df_train['split'] = 'train'

    file_test = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-uspto",
        filename="US_patents_1976-Sep2016_1product_reactions_test_prompts.json",
        repo_type="dataset",
    )
    df_test = pd.read_json(file_test)
    df_test['split'] = 'test'

    file_valid = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-uspto",
        filename="US_patents_1976-Sep2016_1product_reactions_test_prompts.json",
        repo_type="dataset",
    )

    df_valid = pd.read_json(file_valid)
    df_valid['split'] = 'valid'

    df = pd.concat([df_train, df_test, df_valid])

    print(df.head())

if __name__ == "__main__":
    process()

