"""This script parses MSDS data parsed from Sigma Aldrich
(https://huggingface.co/datasets/chemNLP/MSDS/tree/main) and flattens it.

You need to change filepaths before running this script
"""

import json
import os


def get_text(d, text="", level=1, linebreaks=2):
    for k in d:
        if k in [
            "SECTION 6: Acidental release measures",  # always empty
            "SECTION 1: Toxicological information",  # always empty
            "SECTION 16: Other information",  # always the same information
        ]:
            continue

        text += "#" * level + " " + k + "\n" * linebreaks

        if isinstance(d[k], str):
            if d[k] != "":
                text += d[k].rstrip() + "\n" * linebreaks
        elif isinstance(d[k], dict):
            text = get_text(d[k], text=text, level=level + 1)
    return text


if __name__ == "__main__":
    path_jsonl_in = "/fsx/proj-chemnlp/micpie/chemnlp/data/natural/msds/msds.jsonl"

    # load
    with open(path_jsonl_in) as f:
        data = [json.loads(line) for line in f]

    # process
    data = list(map(get_text, data))
    data = [{"text": x} for x in data]

    # save
    path_jsonl_out = path_jsonl_in.replace(".jsonl", "_clean.jsonl")
    if os.path.isfile(path_jsonl_out):
        print(f"Output file already exists, please check: {path_jsonl_out}")
    else:
        with open(path_jsonl_out, "a") as fout:
            for sample in data:
                fout.write(json.dumps(sample) + "\n")
        print(f"JSONL saved to: {path_jsonl_out}")
