import glob
import json
import os
import re

from tqdm import tqdm

STR_CUTOFF = 5000
KEEP_FIRST_HEADERS = [
    "main",
    "abstract",
    "introduction",
    "summary",
]


def load_mmd_from_path(path):
    with open(path) as f:
        data = f.read()
    return data


rm_ref_brackets = re.compile(r"\s?\(\d+([\,\-\;] ?\d+)*\)"), ""
rm_ref_square_brackets = re.compile(r"\s?\[\d+([\,\-\;] ?\d+)*\]"), ""
change_asterisk_headers = re.compile(r"\n\*\*(.*)\*\*.?\n"), r"\n## \1\n\n"
change_asterisk_headers_inline = re.compile(r"\n\*\*(.*)\*\*.?\s"), r"\n## \1\n\n"
change_underline_headers = re.compile(r"\n\_(.*)\_.?\n"), r"\n## \1\n"
# rm_double_asterisk = re.compile(r"\*\*"), ""
rm_line_number = re.compile(r"\n\* \d+\s"), "\n"
rm_missing_page_fail_a = re.compile(r"\n\n\[MISSING_PAGE_FAIL:\d+\]"), ""
rm_missing_page_fail_b = re.compile(r"\[MISSING_PAGE_FAIL:\d+\]"), ""
rm_missing_page_empty_a = re.compile(r"\n\n\[MISSING_PAGE_EMPTY:\d+\]"), ""
rm_missing_page_empty_b = re.compile(r"\[MISSING_PAGE_EMPTY:\d+\]"), ""
rm_missing_page_post_a = re.compile(r"\n\n\[MISSING_PAGE_POST\]"), ""
rm_missing_page_post_b = re.compile(r"\[MISSING_PAGE_POST\]"), ""
rm_figure_caption_start = re.compile(r"[Ff]igure \d+\w?\.?[:\|]?\s"), ""
rm_schema_caption_start = re.compile(r"[Ss]chema \d+\w?\.?[:\|]?\s"), ""
rm_fig_caption_start = re.compile(r"[Ff]ig.? \d+\w?\.?[:\|]?\s"), ""
rm_figure_in_brackets = re.compile(r" \([Ff]igure \d+\w?\.?\)"), ""
rm_fig_in_brackets = re.compile(r" \([Ff]ig.? \d+\w?\.?\)"), ""
rm_fig_in_brackets_asterisk = re.compile(r" \(\*\*[Ff]ig. \d+.*\*\*\)"), ""
# rm_figure_reference = re.compile(r", see [Ff]igure \d+\w?"), ""
# rm_fig_reference = re.compile(r", see [Ff]ig. \d+\w?"), ""
rm_email_with_text = re.compile(r"[Ee]mail[:\s] \S*@\S*\s?"), ""
rm_email = re.compile(r"\S*@\S*\s?"), ""
rm_empty_table = re.compile(r"\n\n\\begin{table}\n\n\\end{table}\nTable.+?\."), "\n"
rm_incomplete_sentence_start_para = re.compile(r"\n\n[a-z].+?\.\s"), "\n\n"
rm_incomplete_sentence_end_para = (
    re.compile(r"\.\s[A-Z,a-z][^\.]+?[a-z][,]?[\s]?\n"),
    ".\n",
)

find_headers = re.compile("(#{1,6}.*)\\n")

year_numbers = re.compile(r"(19|20)\d{2}")


def get_headers(mmd, show=False):
    headers = []
    for match in find_headers.finditer(mmd):
        span = match.span()
        headers.append((mmd[span[0] : span[1]], span))
    if show:
        for h in headers:
            print(h)
    return headers


def get_next_header_to_remove(mmd, exclude_headers):
    headers = get_headers(mmd)
    for header, span in headers:
        for eh in exclude_headers:
            if header.lower().find(eh) != -1:
                return (header, span)
    return False


def remove_nested_headers(mmd, header_span, verbose):
    header, span = header_span
    count_hashtag = header.count("#")
    headers = get_headers(mmd)
    header_idx = headers.index(header_span)
    for i, (next_header, next_span) in enumerate(headers):
        if i + 1 == len(headers):
            next_header_pos = len(mmd) - 1
        if i <= header_idx:
            continue
        if count_hashtag == next_header.count("#"):
            next_header_pos = next_span[0]
    if verbose:
        print(f"Removed span: {span[0]}:{next_header_pos}")
    mmd = mmd[: span[0]] + mmd[next_header_pos + 1 :]
    return mmd


def remove_first_header(mmd):
    headers = get_headers(mmd)
    if len(headers) <= 1:
        return mmd
    header, span = headers[0]
    if span[0] > 0:
        _, next_span = headers[1]
        mmd = mmd[next_span[0] :]

    headers = get_headers(mmd)
    if len(headers) <= 1:
        return mmd
    header, span = headers[0]
    if all([header.lower().find(kfh) == -1 for kfh in KEEP_FIRST_HEADERS]):
        _, next_span = headers[1]
        mmd = mmd[next_span[0] :]
    return mmd


def clean_mmd(mmd, rm_first_header=False, verbose=False):
    # low level cleaning
    reg_replace = [
        rm_ref_brackets,
        rm_ref_square_brackets,
        change_asterisk_headers,
        change_asterisk_headers_inline,
        change_underline_headers,
        # rm_double_asterisk,
        rm_missing_page_fail_a,
        rm_missing_page_fail_b,
        rm_missing_page_empty_a,
        rm_missing_page_empty_b,
        rm_missing_page_post_a,
        rm_missing_page_post_b,
        rm_figure_caption_start,
        rm_schema_caption_start,
        rm_fig_caption_start,
        rm_figure_in_brackets,
        rm_fig_in_brackets,
        rm_fig_in_brackets_asterisk,
        # rm_figure_reference,
        # rm_fig_reference,
        rm_email_with_text,
        rm_email,
        rm_empty_table,
        rm_incomplete_sentence_start_para,
        rm_incomplete_sentence_end_para,
    ]
    for reg, replace in reg_replace:
        mmd = reg.sub(replace, mmd)

    # section cleaning
    if verbose:
        _ = get_headers(mmd, show=True)

    if rm_first_header:
        # we try to remove the first header two times
        for _ in range(2):
            mmd = remove_first_header(mmd)
    header_span = True  # start value
    while header_span is not False:
        header_span = get_next_header_to_remove(mmd, exclude_headers)
        if verbose:
            print(f"{header_span=}")
        if isinstance(header_span, tuple):
            mmd = remove_nested_headers(mmd, header_span, verbose)

    return mmd


exclude_headers = [
    "accession codes",
    "acknowledgement",
    "acknowledgment",
    "additional files",
    "additional information",
    "associated content",
    "author",  # incl: "author information", "corresponding author",
    "availability",
    "bibliography",
    "competing interest",
    "contributions",
    "conflict of interest",
    "conflicts of interest",
    "consent",
    "data and software availability",
    "data availability",
    "declaration",
    "dedication",
    "disclaimer",
    "disclosure",
    "figure legends",
    "financial support",
    "funding",
    "graphical toc",
    "graphical abstract",
    "keywords",
    "note",
    "orcid",
    "present address",
    "reference",
    "supplementary material",
    "supporting formation available",
    "supporting information",
    "table of contents",
    # "abbreviations",
    # "toc",  # creates false positives
]


def create_jsonl_from_dir(path):
    print(f"{path=}")
    paths = sorted(glob.glob(path + "/*.mmd"))
    #path_jsonl = path + "/out.jsonl"
    path_jsonl = path.replace("rxiv/", "rxiv_clean.jsonl")
    if os.path.isfile(path_jsonl):
        print(f"Output file already exists, please check: {path_jsonl}")
        return

    print(f"{path_jsonl=}")
    for path in (pbar := tqdm(paths)):
        fn = path.split("/")[-1].split(".mmd")[0]
        pbar.set_postfix_str(fn)
        mmd = load_mmd_from_path(path)
        text = clean_mmd(mmd, rm_first_header=True, verbose=False)
        if len(text) <= STR_CUTOFF:
            # print(f"Too short text in: {fn}")
            continue
        elif text.count("Journal of") > 10:
            # print(f'Too many "Journal of" in text: {fn}')
            continue
        elif text.count(" doi:") > 10:
            # print(f'Too many " doi:" in text: {fn}')
            continue
        elif len(year_numbers.findall(text)) > 10:
            # print(f"Too many year numbers in text: {fn}")
            continue
        else:
            out = {"fn": fn, "text": text}
            with open(path_jsonl, "a") as f:
                f.write(json.dumps(out) + "\n")
            # uncomment to diff individual files for debugging
            # with open(path.replace(".mmd", "_.mmd"), "w") as f:
            #     f.write(text)


if __name__ == "__main__":
    for path_base in [
            "/fsx/proj-chemnlp/data/nougat_processed_chemrxiv/",
            "/fsx/proj-chemnlp/data/nougat_processed_biorxiv/",
            "/fsx/proj-chemnlp/data/nougat_processed_medrxiv/",
            ]:
        create_jsonl_from_dir(path_base)

    #path_jsonl = "/fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_chemrxiv.jsonl"
    #path_base = "/fsx/proj-chemnlp/data/nougat_processed_chemrxiv/"

    #path_base = "/fsx/proj-chemnlp/data/nougat_processed_biorxiv/"
    #path_jsonl = "/fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_biorxiv.jsonl"
