There are many different ways to contribute to ChemNLP! 
You can get in touch via the GitHub issues and our [Discord](https://t.co/YMzpevmkiN).

# Implementing a dataset 


## Pre-Requisites
Please make a GitHub account prior to implementing a dataset; you can follow instructions to install git here.


## Contributing a dataset 
One of the most important way to contribute to the ChemNLP efforts is to implement a dataset. 
With "implementing" we mean the following: 

- Take a dataset from our [awesome list]() (if it is not there, please add it there first, so we keep track)
- Make an issue in this repository that you want to add this dataset 
- Make a PR that adds 
  - `meta.yaml` describing the dataset 
  - `transform.py` python code that transforms the original dataset (linked in `meta.yaml`) into a form that can be consumed by the loader 



The `meta.yaml` has the following structure:

```yaml
name: aquasoldb # unique identifier, we will also use this for directory names
description: | # short description what this dataset is about
  Curation of nine open source datasets on aqueous solubility.
  The authors also assigned reliability groups.
targets:
  - name: Solubility # name of the column in a tabular dataset
    description: Experimental aqueous solubility value (LogS) # description of what this column means
    units: log(mol/L) # units of the values in this column (leave empty if unitless)
    type: continuos # can be "categorical", "ordinal", "continuos"
  - name: SD
    description: Standard deviation of the experimental aqueous solubility value for multiple occurences
    units: log(mol/L)
    type: continuos
identifiers:
  - name: InChI # column name 
    type: InChI # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
    description: International Chemical Identifier # description (optional, except for "OTHER")
license: CC0 1.0 # license under which the original dataset was published
links: # list of relevant links (original dataset, other uses, etc.)
  - name: dataset
    url: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8
    description: Original dataset
bibtex: # citation(s) for this dataset in BibTeX format
  - |
    "@article{Sorkun_2019,
    doi = {10.1038/s41597-019-0151-1},
    url = {https://doi.org/10.1038%2Fs41597-019-0151-1},
    year = 2019,
    month = {aug},
    publisher = {Springer Science and Business Media {LLC}},
    volume = {6},
    number = {1},
    author = {Murat Cihan Sorkun and Abhishek Khetan and Süleyman Er},
    title = {{AqSolDB}, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds},
    journal = {Sci Data}
    }"
```

For the typical material-property datasets, we will later use the `identifier` and `property` columns to create and fill prompt templates. 
In case your dataset isn't a simple tabular dataset with chemical compounds and properties, please also add the following additional fields for the templates:
