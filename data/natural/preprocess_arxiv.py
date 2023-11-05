import re

import pandas as pd
import pypandoc
from datasets import Dataset, load_dataset


def latex_to_markdown_with_pandoc(latex_text):
    # Remove \cite statements, also if there are options passed. Note that there might be non, or
    # multiple square brackets and they might or might not be filled
    # remove also \citet, \citep, \citeyear, \citeauthor

    latex_text = re.sub(r"\\cite(\[.*?\])?{.*?}", "", latex_text)
    latex_text = re.sub(r"\\citet(\[.*?\])?{.*?}", "", latex_text)
    latex_text = re.sub(r"\\citep(\[.*?\])?{.*?}", "", latex_text)
    latex_text = re.sub(r"\\citeyear(\[.*?\])?{.*?}", "", latex_text)
    latex_text = re.sub(r"\\citeauthor(\[.*?\])?{.*?}", "", latex_text)

    # Remove figure environments
    latex_text = re.sub(
        r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", "", latex_text, flags=re.DOTALL
    )

    # Remove everything starting with \x
    latex_text = re.sub(r"\\x.*", "", latex_text)

    # Remove linebreaks in the python string
    latex_text = latex_text.replace("\n", " ")

    # Replace  ,. with .
    latex_text = latex_text.replace(",.", ".")

    # Replace ' .' with '.'
    latex_text = latex_text.replace(" .", ".")

    # Replace ' .\n' with '.\n'
    latex_text = latex_text.replace(" .\n", ".\n")

    # Replace ' , ' with ', '
    latex_text = latex_text.replace(" , ", ", ")

    # Replace ' : ' with ': '
    latex_text = latex_text.replace(" : ", ": ")

    # Replace ' ; ' with '; '
    latex_text = latex_text.replace(" ; ", "; ")

    # Replace ' - ' with '-'
    latex_text = latex_text.replace(" - ", "-")

    # Replace '- ' with '-'
    latex_text = latex_text.replace("- ", "-")

    # Replace " -" with "-"
    latex_text = latex_text.replace(" -", "-")

    exclude = [
        "Acknowledgment",
        "accession codes",
        "acknowledgement",
        "acknowledgment",
        "additional files",
        "additional information",
        "associated content",
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
        "references",
        "supplementary information",
        "table of contents",
    ]

    # Convert LaTeX to Markdown using Pandoc
    try:
        markdown_text = pypandoc.convert_text(
            latex_text,
            "markdown",
            format="latex",
            extra_args=["--wrap=none", "--fail-if-warnings"],
        )

        # convert equation syntax into the one used by Nougat
        markdown_text = re.sub(
            r"\$\$(.*?)\$\$", r"\\[\1\\]", markdown_text, flags=re.DOTALL
        )

        markdown_text = re.sub(r"\$(.*?)\$", r"\\(\1\\)", markdown_text)

        for excl in exclude:
            # Define a regular expression pattern to match headings containing the target word
            pattern = rf"(^#+\s+{excl}.*?\n)(.*?)(?=(^#)|\Z)"
            # Use regular expressions to find and remove sections with matching headings
            markdown_text = re.sub(
                pattern,
                "",
                markdown_text,
                flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
            )

            target_marker = ":::"

            # Define a regular expression pattern to match sections delimited by the target marker
            pattern_2 = rf"(^{re.escape(target_marker)}.*?{re.escape(target_marker)}\n)"

            markdown_text = re.sub(
                pattern_2,
                "",
                markdown_text,
                flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
            )

            # remove empty floats. I.e. cases with [](something), where the square brackets are empty
            # for example, []{#tab:samples label="tab:samples"}
            markdown_text = re.sub(r"\[\]\{.*?\}", "", markdown_text, flags=re.DOTALL)

            # remove more than one empty line
            markdown_text = re.sub(r"\n\n\n+", "\n\n", markdown_text, flags=re.DOTALL)

            # remove empty lines at the beginning and end of the string
            markdown_text = markdown_text.strip()

            # remove double square brackets

            # remove the headings matched with the first pattern if it is the final section, too,
            # i.e., if there is no other section after it
            markdown_text = re.sub(
                rf"(^#+\s+{excl}.*?\n)(.*?)(?=\Z)",
                "",
                markdown_text,
                flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
            )

            # remove it also if it only is a heading, without any content and there is no line break
            markdown_text = re.sub(
                rf"(^#+\s+{excl}.*?)(?=\Z)",
                "",
                markdown_text,
                flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
            )
    except Exception as e:
        print(f"An error occurred during Pandoc conversion: {e}")
        return None

    return markdown_text


def clean_ds():
    ds_arxiv = load_dataset("EleutherAI/proof-pile-2", "arxiv")
    clean_entries = []

    for entry in ds_arxiv["train"]:
        entry["text"] = latex_to_markdown_with_pandoc(entry["text"])
        if entry["text"] is not None:
            clean_entries.append(entry)

    df = pd.DataFrame(clean_entries)
    df.to_json("arxiv.jsonl", orient="records", lines=True)

    ds = Dataset.from_pandas(df)
    ds.push_to_hub("chemnlp/arxiv")


if __name__ == "__main__":
    clean_ds()
