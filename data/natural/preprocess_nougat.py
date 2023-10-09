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


rm_ref_single = re.compile(r"\[\d+\]")
rm_ref_multi = re.compile(r"\[\d+.+\d\]")
rm_missing_page_fail = re.compile(r"\n\n\[MISSING_PAGE_FAIL:\d+\]")
rm_missing_page_empty = re.compile(r"\n\n\[MISSING_PAGE_EMPTY:\d+\]")
rm_d = re.compile(r"\d")


def clean_mmd(mmd):
    res = [rm_ref_single, rm_ref_multi, rm_missing_page_fail, rm_missing_page_empty]
    for r in res:
        mmd = r.sub("", mmd)
    return mmd


def mmd_to_html(mmd):
    return markdown.markdown(mmd)


exclude_headers = [
    "accession codes",
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
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

        if cont:
            if header_text.isupper():
                text += header_text.capitalize()  # or .title() ?
            else:
                text += header_text
            text += "\n"

            if (
                i != 0
            ):  # the first header comes usually with unwanted infos, e.g., author information
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


if __name__ == "__main__":
    path_base = ""
    create_jsonl_from_dir(path_base)
