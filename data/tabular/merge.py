import os
from glob import glob
from pathlib import Path
from typing import Union

import fire
import pandas as pd
from tqdm import tqdm


def merge_files(dir):
    fns = sorted(glob(os.path.join(dir, "data_clean-*.csv")))
    fn_merged = os.path.join(dir, "data_clean.csv")
    if os.path.exists(fn_merged):
        os.remove(fn_merged)
    for fn in fns:
        df = pd.read_csv(fn, index_col=False, low_memory=False)
        df.to_csv(
            fn_merged, mode="a", index=False, header=not os.path.exists(fn_merged)
        )
        os.remove(fn)
        del df


def process_file(file: Union[str, Path]):
    dir = Path(file).parent
    # check if there is any csv file
    if not glob(os.path.join(dir, "*.csv")):
        return
    if len(glob(os.path.join(dir, "data_clean-0*.csv"))) >= 1:
        merge_files(dir)


def process_all_files(data_dir):
    all_yaml_files = sorted(glob(os.path.join(data_dir, "**", "**", "meta.yaml")))
    all_yaml_files = [f for f in all_yaml_files if "fda" in f]
    print(all_yaml_files)
    for yaml_file in tqdm(all_yaml_files):
        print(f"Processing {yaml_file}")
        try:
            process_file(yaml_file)
        except Exception as e:
            print(f"Could not process {yaml_file}: {e}")


if __name__ == "__main__":
    fire.Fire(process_all_files)
