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
        "identifier": [
            {
                "id": "Abbreviation",
                "description": "abbreviation of a term",
                "type": "categorical",
                "names": ["abbreviation"],
            },
        ],
        "targets": [
            {
                "id": "MeSH_Identifier",
                "description": "unique codes for Medical Subject Headings",
                "type": "categorical",
                "names": ["MeSH identifier"],
                "sample": False,
            },
            {
                "id": "Full_Form",
                "description": "full form or meaning of the abbreviation",
                "type": "categorical",
                "names": ["full form or meaning of the abbreviation"],
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
