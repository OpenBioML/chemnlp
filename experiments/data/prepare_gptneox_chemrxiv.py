"""
Preparing chemrxiv dataset as per GPT-NeoX guidelines
NOTE this needs to be run from the root of this repository directory

Example usage:
    python experiments/chem_data_prep.py /fsx/proj-chemnlp/data/ chemnlp/gpt-neox/
"""
import argparse
import os

import datasets
import jsonlines

DATASET = "marianna13/chemrxiv"
GPT_NEOX_KEY = "text"

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_dir", help="Where you want to store the prepared dataset."
    )
    parser.add_argument(
        "gptneox_dir", help="Where you can find the GPT-NeoX repository."
    )
    args = parser.parse_args()

    # save initial strings from chemrxiv articles as jsonlines
    chem_data = datasets.load_dataset(DATASET)
    all_full_text_samples = [
        {GPT_NEOX_KEY: paper["TEXT"]} for paper in chem_data["train"]
    ]
    save_path = f"{args.save_dir}/{DATASET}"
    data_path = f"{save_path}/data.jsonl"
    os.makedirs(save_path, exist_ok=True)
    with jsonlines.open(data_path, "w") as writer:
        writer.write_all(all_full_text_samples)

    # execute gpt-neox processing
    gpt_tool_path = f"{args.gptneox_dir}/tools/preprocess_data.py"
    os.system(
        f"""
        python {gpt_tool_path}
        --input {data_path}
        --output-prefix {save_path}/data
        --vocab /fsx/pile/20B_tokenizer.json
        --dataset-impl mmap
        --tokenizer-type HFTokenizer --append-eod
        """
    )
