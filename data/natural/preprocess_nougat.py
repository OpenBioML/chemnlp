import glob
import json
import re

from tqdm import tqdm

TEXT_CUTOFF = 0


def load_mmd_from_path(path):
    with open(path) as f:
        data = f.read()
    return data


rm_double_asterisk_start = re.compile(r"\n\*\*"), "\n## "
rm_double_asterisk_end = re.compile(r"\*\*\n"), ""
rm_double_asterisk = re.compile(r"\*\*"), ""
rm_missing_page_fail = re.compile(r"\n\n\[MISSING_PAGE_FAIL:\d+\]"), ""
rm_missing_page_empty = re.compile(r"\n\n\[MISSING_PAGE_EMPTY:\d+\]"), ""
rm_missing_page_post = re.compile(r"\n\n\[MISSING_PAGE_POST\]"), ""
rm_figure_caption_start = re.compile(r"[Ff]igure \d+\w?\.?[:\|]?\s"), ""
rm_schema_caption_start = re.compile(r"[Ss]chema \d+\w?\.?[:\|]?\s"), ""
rm_schema_caption_start = re.compile(r"[Ss]cheme \d+\w?\.?[:\|]?\s"), ""
rm_fig_caption_start = re.compile(r"[Ff]ig. \d+\w?\.?[:\|]?\s"), ""
rm_figure_in_brackets = re.compile(r" \([Ff]igure \d+\w?\.?\)"), ""
rm_fig_in_brackets = re.compile(r" \([Ff]ig. \d+\w?\.?\)"), ""
rm_figure_reference = re.compile(r", see [Ff]igure \d+\w?"), ""
rm_fig_reference = re.compile(r", see [Ff]ig. \d+\w?"), ""
rm_ref_single = re.compile(r"\s?\[\d+]"), ""
rm_ref_multi = re.compile(r"\s?\[\d+.+\d\]"), ""
rm_email_with_text = re.compile(r"[Ee]mail[:\s] \S*@\S*\s?"), ""
rm_email = re.compile(r"\S*@\S*\s?"), ""
rm_empty_table = re.compile(r"\n\n\\begin{table}\n\n\\end{table}\nTable.+?\."), "\n"
rm_incomplete_sentence_start_para = re.compile(r"\n\n[a-z].+?\.\s"), "\n\n"
rm_incomplete_sentence_end_para = re.compile(r"\.\s[A-Z,a-z][^\.]+?[a-z][,]?\n"), ".\n"

year_numbers = re.compile(r"[19,20]\d\d\,")


def clean_mmd(mmd, verbose=False):
    # section cleaning
    find_headers = re.compile("(#{1,6}.*)\\n")

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

    def remove_header(mmd, header_span):
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

    if verbose:
        _ = get_headers(mmd, show=True)
    header_span = True  # dummy start value
    while header_span is not False:
        header_span = get_next_header_to_remove(mmd, exclude_headers)
        if verbose:
            print(f"{header_span=}")
        if isinstance(header_span, tuple):
            mmd = remove_header(mmd, header_span)

    # low level cleaning
    reg_replace = [
        rm_double_asterisk_start,
        rm_double_asterisk_end,
        # rm_double_asterisk,
        rm_missing_page_fail,
        rm_missing_page_empty,
        rm_missing_page_post,
        rm_figure_caption_start,
        rm_schema_caption_start,
        rm_fig_caption_start,
        rm_figure_in_brackets,
        rm_fig_in_brackets,
        rm_figure_reference,
        rm_fig_reference,
        rm_ref_single,
        rm_ref_multi,
        rm_email_with_text,
        rm_email,
        rm_empty_table,
        rm_incomplete_sentence_start_para,
        rm_incomplete_sentence_end_para,
    ]
    for reg, replace in reg_replace:
        mmd = reg.sub(replace, mmd)

    return mmd


exclude_headers = [
    "accession codes",
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "additional files",
    "additional information",
    "associated content",
    "author contributions",
    "author information",
    "author",
    "authors",
    "bibliography",
    "code availability",
    "competing interest",
    "competing interests statement",
    "competing interests",
    "conflict of interest",
    "conflicts of interest",
    "corresponding author",
    "data and software availability",
    "data availability",
    "declaration of competing interest",
    "dedication",
    "disclaimer",
    "financial support",
    "funding acs",
    "funding sources",
    "graphical toc entry",
    "graphical abstract",
    "keywords",
    "note",
    "notes",
    "orcid",
    "present address",
    "reference",
    "references",
    "supplementary material",
    "supporting formation available",
    "supporting information available",
    "supporting information",
    "table of contents",
    # "toc",  # creates false positives
    "corresponding authors:",
    # "abbreviations",
]


def create_jsonl_from_dir(path):
    print(f"{path=}")
    paths = sorted(glob.glob(path + "/*.mmd"))
    path_jsonl = path + "/out.jsonl"
    print(f"{path_jsonl=}")
    for path in (pbar := tqdm(paths)):
        fn = path.split("/")[-1].split(".mmd")[0]
        pbar.set_postfix_str(fn)
        mmd = load_mmd_from_path(path)
        text = clean_mmd(mmd)
        if len(text) <= TEXT_CUTOFF:
            print(f"Too short text in: {fn}")
        elif text.count("Journal of") > 10:
            print(f'Too many "Journal of" in text: {fn}')
        elif len(year_numbers.findall(text)) > 10:
            print(f"Too many year numbers in text: {fn}")
        else:
            out = {"fn": fn, "text": text}
            with open(path_jsonl, "a") as f:
                f.write(json.dumps(out) + "\n")
            # uncomment to diff individual files for debugging
            # with open(path.replace(".mmd", "_.mmd"), "w") as f:
            #     f.write(text)


if __name__ == "__main__":
    path_base = ""
    create_jsonl_from_dir(path_base)
