import json
import os
import re

rm_ref_brackets = re.compile(r"\s?\(\d+([\,\-\;] ?\d+)*\)"), ""
rm_ref_square_brackets = re.compile(r"\s?\[\d+([\,\-\;] ?\d+)*\]"), ""
rm_figure_caption_start = re.compile(r"[Ff]igure \d+\w?\.?[:\|]?\s"), ""
rm_schema_caption_start = re.compile(r"[Ss]chema \d+\w?\.?[:\|]?\s"), ""
rm_fig_caption_start = re.compile(r"[Ff]ig.? \d+\w?\.?[:\|]?\s"), ""
rm_figure_in_brackets = re.compile(r" \([Ff]igure \d+\w?\.?\)"), ""
rm_fig_in_brackets = re.compile(r" \([Ff]ig.? \d+\w?\.?\)"), ""
rm_email_with_text = re.compile(r"[Ee]mail[:\s] \S*@\S*\s?"), ""
rm_email = re.compile(r"\S*@\S*\s?"), ""


def clean_text(text):
    # low level cleaning
    reg_replace = [
        rm_ref_brackets,
        rm_ref_square_brackets,
        rm_figure_caption_start,
        rm_schema_caption_start,
        rm_fig_caption_start,
        rm_figure_in_brackets,
        rm_fig_in_brackets,
        rm_email_with_text,
        rm_email,
    ]
    for reg, replace in reg_replace:
        text = reg.sub(replace, text)

    return text


def clean_jsonl(path_jsonl_in):
    print(f"{path_jsonl_in=}")
    path_jsonl_out = path_jsonl_in.replace(".jsonl", "_clean.jsonl")
    if os.path.isfile(path_jsonl_out):
        print(f"Output file already exists, please check: {path_jsonl_out}")
        return

    # basic setup for large files with line-by-line processing
    with open(path_jsonl_in, "r") as fin:
        for line in fin:
            data = json.loads(line)
            # clean only full text papers
            if data["source"].startswith("s2orc"):
                data["text"] = clean_text(data["text"])
            with open(path_jsonl_out, "a") as fout:
                fout.write(json.dumps(data) + "\n")

    print(f"{path_jsonl_out=}")


if __name__ == "__main__":
    path_base = ""
    clean_jsonl(path_base)
