"""This script performs cleaning of the EuroPMC natural text dataset. 

For this it reads `.jsonl` files, which have `text` keys with the content and then uses heuristics
encoded in regular expression to find references, authors, captions, etc. 

Before running this scripts, the filepaths need to be changed.
import json
import os
import re

import numpy as np

# import pandas as pd

STR_CUTOFF = 5000
GAP_CUTOFF = 1000
SENTENCE_END_IDX = -1


rm_ref_brackets = re.compile(r"\s?\(\d+([\,\-\;] ?\d+)*\)"), ""
rm_ref_square_brackets = re.compile(r"\s?\[\d+([\,\-\;] ?\d+)*\]"), ""
rm_figure_caption_start = re.compile(r"[Ff]igure \d+\w?\.?[:\|]?\s"), ""
rm_schema_caption_start = re.compile(r"[Ss]chema \d+\w?\.?[:\|]?\s"), ""
rm_fig_caption_start = re.compile(r"[Ff]ig.? \d+\w?\.?[:\|]?\s"), ""
rm_figure_in_brackets = re.compile(r" \([Ff]igure \d+\w?\.?\)"), ""
rm_fig_in_brackets = re.compile(r" \([Ff]ig.? \d+\w?\.?\)"), ""
rm_email_with_text = re.compile(r"[Ee]mail[:\s] \S*@\S*\s?"), ""
rm_email = re.compile(r"\S*@\S*\s?"), ""

year_numbers = re.compile(r"(19|20)\d{2}")
sentence_end = re.compile(r"[a-z]\.\s[A-Z]")


def clean_text_from_citation_section(text, gap_cutoff=1000, sentence_end_idx=-1):
    # get spans of year numbers
    year_number_spans = [x.span() for x in year_numbers.finditer(text)]
    if len(year_number_spans) == 0:
        return text
    relative_span_diffs = []
    for s in year_number_spans:
        if len(s) != 0:
            relative_span_diffs.append((np.array(s) / np.max(s)).tolist())
    year_number_span_starts = [s[0] for s in year_number_spans]

    # absolute span gaps
    abs_span_gaps = [
        x - y for x, y in zip(year_number_span_starts[1:], year_number_span_starts[:-1])
    ]

    # get last absolute gap based on gap cutoff
    last_gap = None
    for s, d in zip(reversed(year_number_spans), reversed(abs_span_gaps)):
        if d > gap_cutoff:
            last_gap = s[0]  # get start
            break
    if last_gap is None:
        if year_number_span_starts[0] > len(text) // 2:
            last_gap = year_number_span_starts[0]

    # get sentence ends
    spans_sentence_end = [x.span() for x in sentence_end.finditer(text[:last_gap])]
    if len(spans_sentence_end) == 0:
        return text
    return text[: spans_sentence_end[sentence_end_idx][0] + 2]


def clean_text(text, gap_cutoff=GAP_CUTOFF, sentence_end_idx=SENTENCE_END_IDX):
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

    # citation section removal
    text = clean_text_from_citation_section(text, gap_cutoff, sentence_end_idx)
    return text


def clean_jsonl(path_jsonl_in):
    print(f"{path_jsonl_in=}")
    path_jsonl_out = path_jsonl_in.replace(".jsonl", "_clean.jsonl")
    if os.path.isfile(path_jsonl_out):
        print(f"Output file already exists, please check: {path_jsonl_out}")
        return

    # pandas setup for smaller files
    # df = pd.read_json(path_jsonl_in, lines=True)
    # df.text = df.text.apply(clean_text)
    # df.to_json(path_jsonl_out, orient='records', lines=True)

    # basic setup for large files with line-by-line processing
    with open(path_jsonl_in, "r") as fin:
        for line in fin:
            data = json.loads(line)
            data["text"] = clean_text(data["text"])
            if len(data["text"]) <= STR_CUTOFF:
                # print(f"Too short text in: {fn}")
                continue
            else:
                with open(path_jsonl_out, "a") as fout:
                    fout.write(json.dumps(data) + "\n")

    print(f"{path_jsonl_out=}")


if __name__ == "__main__":
    paths = [
        "/scratch/micpie/ft_results_test.jsonl",
        "/scratch/micpie/ft_results_valid.jsonl",
        "/scratch/micpie/ft_results_train.jsonl",
    ]
    for path_base in paths:
        print(path_base)
        clean_jsonl(path_base)
