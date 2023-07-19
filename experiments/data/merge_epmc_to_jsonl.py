"""
This file is specifically for taking a nested folder of jsonlines files 
and merging them into one complete jsonlines file.

e.g. 
<dir>/2022_05_27/file1.jsonl
<dir>/2022_05_25/file2.jsonl
...
"""
from typing import List
import os
import jsonlines
import multiprocessing
from tqdm import tqdm

# NOTE hardcoded paths
ROOT = "/fsx/proj-chemnlp/jeb/europmc_deduped"
OUT_DIR = "/fsx/proj-chemnlp/jeb/europmc_deduped/merged_paper_and_abstracts"


def write_to_merged_file(all_files: List[str], name: str):
    """Loops over all_files, reads them and writes them to the out_file"""
    out_file = f"{OUT_DIR}/merged_all_{name}.jsonl"
    print(f"Writing {len(all_files)} {name} files to {out_file}")
    # start context manager for writing
    with jsonlines.open(out_file, mode="w") as writer:
        for file in tqdm(all_files):
            # as writing is serial, this cannot be easily parallelised
            with jsonlines.open(file) as reader:
                all_entries = [*reader]
                writer.write_all(all_entries)


if __name__ == "__main__":
    # collect all files
    result = os.popen(f"find {ROOT} -type f -name '*.jsonl'")
    parsed_result = result.read().split("\n")
    all_files = [x for x in parsed_result if x]
    paper_files = [x for x in all_files if "ft" in x]
    abstract_files = [x for x in all_files if "abs" in x]
    print(
        f"{len(all_files)} files, {len(paper_files)} paper and {len(abstract_files)} abstract files."
    )

    # merge and process
    with multiprocessing.Pool(os.cpu_count()) as p:
        p.starmap(
            write_to_merged_file,
            [(paper_files, "papers"), (abstract_files, "abstracts")],
        )
