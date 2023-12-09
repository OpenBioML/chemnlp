import os
import urllib.request
import zipfile

import pandas as pd
import yaml


def get_and_transform_data():
    url = "https://ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem/NLM-Chem-corpus.zip"
    local_zip_path = "NLM-Chem-corpus.zip"

    # Download the ZIP file
    urllib.request.urlretrieve(url, local_zip_path)

    # Open the ZIP file and extract the TSV file
    with zipfile.ZipFile(local_zip_path, "r") as z:
        with z.open("FINAL_v1/abbreviations.tsv") as f:
            df = pd.read_csv(f, sep="\t", header=None)

    # Set column names
    df.columns = ["MeSH_Identifier", "Abbreviation", "Full_Form"]

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Check duplicates
    assert not df.duplicated().sum(), "Found duplicate rows in the dataframe"

    # Save to CSV
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # Create meta.yaml
    meta = {
        "name": "NLM-Chem",
        "description": (
            "NLM-Chem is a new resource for chemical entity recognition in PubMed full text literature."
        ),
        "identifiers": [
            {
                "id": "Abbreviation",
                "description": "abbreviation of a term",
                "type": "Other",
                "names": [{"noun": "abbreviation"}],
            },
            {
                "id": "MeSH_Identifier",
                "description": "unique codes for Medical Subject Headings",
                "type": "categorical",
                "names": [{"noun": "MeSH identifier"}],
                "sample": False,
            },
        ],
        "targets": [
            {
                "id": "Full_Form",
                "description": "full form or meaning of the abbreviation",
                "type": "categorical",
                "names": [{"noun": "full form or meaning"}],
            },
        ],
        "license": "CC BY 4.0",
        "links": [
            {
                "url": "https://ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem/",
                "description": "data source",
            },
            {
                "url": "https://www.nature.com/articles/s41597-021-00875-1",
                "description": "publication",
            },
        ],
        "num_points": len(df),
        "bibtex": [
            """@article{Islamaj2021,
author = {Islamaj, R. and Leaman, R. and Kim, S. and Lu, Z.},
title = {NLM-Chem, a new resource for chemical entity recognition in PubMed full text literature},
journal = {Nature Scientific Data},
volume = {8},
number = {91},
year = {2021},
doi = {10.1038/s41597-021-00875-1},
url = {https://doi.org/10.1038/s41597-021-00875-1}
}""",
        ],
        "templates": [
            'The {Abbreviation__names__noun} "{Abbreviation#}" stands for "{#Full_Form}".',  # noqa
            """Task: Please give me the {Full_Form__names__noun} of the {Abbreviation__names__noun}.
Abbreviation: {Abbreviation#}
Constraint: Answer the question with {#full|complete!} words.
Result: {Full_Form#}""",  # noqa
            """Task: Please give me the {Abbreviation__names__noun} of the following {Full_Form__names__noun}.
Full form or meaning of the abbreviation: {Full_Form#}
Constraint: Answer the question with an {Abbreviation__names__noun}.
Result: {Abbreviation#}""",  # noqa
            """User: Can you give me the {Abbreviation__names__noun} of the following {Full_Form__names__noun}: {#Full_Form}
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {Abbreviation#}""",  # noqa
            """User: Can you give me the {Full_Form__names__noun} of the following {Abbreviation__names__noun}: {#Abbreviation}
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {Full_Form#}""",  # noqa
            """User: I'm {#searching|looking!} for the {Abbreviation__names__noun} for: {#Full_Form}
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {Abbreviation#}""",  # noqa
            """Task: Please give me the {Full_Form__names__noun} of the {Abbreviation__names__noun}.
Abbreviation: {Abbreviation#}
Constraint: Answer the question with {#full|complete!} words.
Result:<EOI>{Full_Form#}""",  # noqa
            """Task: Please give me the {Abbreviation__names__noun} of the following {Full_Form__names__noun}.
Full form or meaning of the abbreviation: {Full_Form#}
Constraint: Answer the question with an {Abbreviation__names__noun}.
Result:<EOI>{Abbreviation#}""",  # noqa
        ],
    }

    def str_presenter(dumper, data):
        """Configure yaml for dumping multiline strings
        Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
        """
        if data.count("\n") > 0:  # Check for multiline string
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)
    fn_meta = "meta.yaml"
    with open(fn_meta, "w") as f:
        yaml.dump(meta, f, sort_keys=False)

    # Add the file_path code here
    file_path = os.path.abspath(fn_meta)
    print(f"Meta.yaml is being saved at: {file_path}")
    print(f"Finished processing {meta['name']} dataset!")


if __name__ == "__main__":
    get_and_transform_data()
