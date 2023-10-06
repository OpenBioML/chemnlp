import glob
import json
import re

import markdown
from bs4 import BeautifulSoup
from tqdm import tqdm


def load_mmd_from_path(path):
    with open(path) as f:
        data = f.read()
    return data


rm_ref_single = re.compile(r"\[\d+\]")
rm_ref_multi = re.compile(r"\[\d+.+\d\]")
rm_missing_page = re.compile(r"\n\n\[MISSING_PAGE_FAIL:\d+\]")
rm_d = re.compile(r"\d")


def clean_mmd(mmd):
    res = [rm_ref_single, rm_ref_multi, rm_missing_page]
    for r in res:
        mmd = r.sub("", mmd)
    return mmd


def mmd_to_html(mmd):
    return markdown.markdown(mmd)


exclude_headers = [
    "associated content",
    "accession codes",
    "author information",
    "corresponding author",
    "authors",
    "author contributions",
    "notes",
    "acknowledgments",
    "references",
    "notes",
    "supporting information",
    "code availability",
    "acknowledgements",
    "author contributions",
    "competing interests",
    "funding sources",
    "acknowledgment",
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
        out = {"fn": fn, "text": text}
        with open(path_jsonl, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    path_base = ""
    create_jsonl_from_dir(path_base)
