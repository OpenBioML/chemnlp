import glob
import json
import re

import markdown
from bs4 import BeautifulSoup
from tqdm import tqdm

TEXT_CUTOFF = 0


def load_mmd_from_path(path):
    with open(path) as f:
        data = f.read()
    return data


rm_missing_page_fail = re.compile(r"\n\n\[MISSING_PAGE_FAIL:\d+\]"), ""
rm_missing_page_empty = re.compile(r"\n\n\[MISSING_PAGE_EMPTY:\d+\]"), ""
rm_missing_page_post = re.compile(r"\n\n\[MISSING_PAGE_POST\]"), ""
rm_figure_caption_start = re.compile(r"[Ff]igure \d+\w?\.?[:\|]?\s"), ""
rm_schema_caption_start = re.compile(r"[Ss]chema \d+\w?\.?[:\|]?\s"), ""
rm_fig_caption_start = re.compile(r"[Ff]ig. \d+\w?\.?[:\|]?\s"), ""
rm_figure_in_brackets = re.compile(r" \([Ff]igure \d+\w?\.?\)"), ""
rm_fig_in_brackets = re.compile(r" \([Ff]ig. \d+\w?\.?\)"), ""
rm_figure_reference = re.compile(r", see [Ff]igure \d+\w?"), ""
rm_fig_reference = re.compile(r", see [Ff]ig. \d+\w?"), ""
rm_ref_single = re.compile(r"\s?\[\d+]"), ""
rm_ref_multi = re.compile(r"\s?\[\d+.+\d\]"), ""
rm_email_with_text = re.compile(r"[Ee]mail[:\s] \S*@\S*\s?"), ""
rm_email = re.compile(r"\S*@\S*\s?"), ""
rm_incomplete_sentence_start_para = re.compile(r"\n\n[a-z].+?\.\s"), "\n\n"
rm_incomplete_sentence_end_para = re.compile(r"\.\s[A-Z,a-z][^\.]+?[a-z][,]?\n"), ".\n"
rm_empty_table = re.compile(r"\n\n\\begin{table}\n\n\\end{table}\nTable.+?\."), "\n"
rm_double_asterisk = re.compile(r"\*\*"), ""


def clean_mmd(mmd):
    reg_replace = [
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
        rm_incomplete_sentence_start_para,
        rm_incomplete_sentence_end_para,
        rm_empty_table,
        rm_double_asterisk,
    ]
    for reg, replace in reg_replace:
        mmd = reg.sub(replace, mmd)
    return mmd


def mmd_to_html(mmd):
    return markdown.markdown(mmd)


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
    "corresponding authors:",
    # "abbreviations",
]


def html_to_clean_text(html, verbose=False):
    soup = BeautifulSoup(html, "html.parser")
    text = ""
    headers = soup.find_all(re.compile("^h\d$"))  # noqa: W605
    if verbose:
        for i, h in enumerate(headers):
            print(i, h.text)
        print()

    for i, h in enumerate(headers):
        header_text = h.text

        cont = True  # continue after exclude header check
        for eh in exclude_headers:
            if header_text.lower().find(eh) != -1:
                cont = False

        first_headers = ["abstract", "introduction"]
        if i == 0:
            # check if we don't find a first_headers
            if not (any([header_text.lower().find(fh) > -1 for fh in first_headers])):
                cont = False

        if cont:
            if header_text.isupper():
                text += header_text.capitalize()  # or .title() ?
            else:
                text += header_text
            text += "\n"

            for sibling in h.next_siblings:
                if sibling.name is None:
                    continue
                elif sibling.name.startswith("h"):
                    break
                else:
                    text += sibling.text
                    text += "\n\n"
            text += "\n"

    text = text.replace("\n\n\n", "\n\n")  # clean up line breaks
    return text


def create_jsonl_from_dir(path):
    print(f"{path=}")
    paths = sorted(glob.glob(path + "/*.mmd"))
    path_jsonl = path + "/out.jsonl"
    print(f"{path_jsonl=}")
    for path in (pbar := tqdm(paths)):
        fn = path.split("/")[-1].split(".mmd")[0]
        pbar.set_postfix_str(fn)
        mmd = load_mmd_from_path(path)
        mmd = clean_mmd(mmd)
        html = mmd_to_html(mmd)
        text = html_to_clean_text(html)
        if len(text) <= TEXT_CUTOFF:
            print(f"Too short text in: {fn}")
        elif text.count("Journal of") > 10:
            print(f'Too many "Journal of" in text: {fn}')
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
