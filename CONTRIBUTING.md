There are many different ways to contribute to ChemNLP! 
<<<<<<< HEAD
You can get in touch via the GitHub [task board](https://github.com/orgs/OpenBioML/projects/5?query=is:open+sort:updated-desc) and [issues](https://github.com/OpenBioML/chemnlp/issues?q=is:issue+is:open+sort:updated-desc&query=is:open+sort:updated-desc) and our [Discord](https://t.co/YMzpevmkiN).
=======
You can get in touch via the GitHub [task board]() and [issues]() and our [Discord](https://t.co/YMzpevmkiN).
>>>>>>> 14b24af (feat: explain templating)

## Pre-Requisites
Please make a [GitHub account](https://github.com/) prior to implementing a dataset; you can follow instructions to install git [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the [ChemNLP repository](https://github.com/OpenBioML/chemnlp)
2. [Clone the you fork](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
3. [Make a new branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)
4. Please try using [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) for formatting your commit messages

## Create a development environment (For code/dataset contributions)

For code and data contributions, we recommend you creata a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). If you do not have conda already installed on your system, we recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html):

```bash
conda env create -f conda.yml  # Creates a conda env
conda activate chemnlp  # Activate your conda environment
```

## Contributing a dataset 
One of the most important ways to contribute to the ChemNLP efforts is to implement a dataset. 
With "implementing" we mean the following: 

- Take a dataset from our [awesome list](https://github.com/kjappelbaum/awesome-chemistry-datasets) (if it is not there, please add it there first, so we keep track)
- Make an issue in this repository that you want to add this dataset (we will label this issue and assign it to you)
- Make a PR that adds in a new folder in `data`
  - `meta.yaml` describing the dataset 
  - `transform.py` Python code that transforms the original dataset (linked in `meta.yaml`) into a form that can be consumed by the loader
  - If you need additional dependencies, 



The `meta.yaml` has the following structure:

```yaml
name: aquasoldb # unique identifier, we will also use this for directory names
description: | # short description what this dataset is about
  Curation of nine open source datasets on aqueous solubility.
  The authors also assigned reliability groups.
targets:
  - id: Solubility # name of the column in a tabular dataset
    description: Experimental aqueous solubility value (LogS) # description of what this column means
    units: log(mol/L) # units of the values in this column (leave empty if unitless)
    type: continuous # can be "categorical", "ordinal", "continuous"
    names: # names for the property (to sample from for building the prompts)
      - solubility
      - water solubility
  - id: SD
    description: Standard deviation of the experimental aqueous solubility value for multiple occurences
    units: log(mol/L)
    type: continuous
    names:
      - solubility
      - water solubility
      - solubility in water
identifiers:
  - id: InChI # column name
    type: InChI # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
    description: International Chemical Identifier # description (optional, except for "OTHER")
license: CC0 1.0 # license under which the original dataset was published
num_points: 10000 # number of datapoints in this dataset
links: # list of relevant links (original dataset, other uses, etc.)
num_points: 10000 # number of datapoints in this dataset
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


```yaml
<<<<<<< HEAD
templates:
  - prompt: "Please answer the following chemistry question.\nDerive for the molecule with the <molecule_text> <molecule> the <expt_value_text>."
    completion: "<exp_value>"
  - prompt: "Please answer the following question.\nPredict the <expt_value_text> for <molecule>."
    completion: "<exp_value>"
fields:
  exp_value:
    values:
      - name: exp_value
        column: exp_value
        text: adsorption energy
      - name: calc_value
        column: calc_value
        text: adsorption free energy
  molecule:
    values:
      - name: smiles
        column: smiles
        text:
=======
templates: 
    - prompt: "Please answer the following chemistry question.\nDerive for the molecule with the <molecule_text> <molecule> the <expt_value_text>."
      completion: "<exp_value>"
    - prompt: "Please answer the following question.\nPredict the <expt_value_text> for <molecule>."
      completion: "<exp_value>"
fields: 
  exp_value:
    values:
      - name: exp_value
        column: exp_value 
        text: adsorption energy
      - name: calc_value
        column: calc_value 
        text: adsorption free energy 
  molecule: 
    values: 
      - name: smiles
        column: smiles
        text: 
>>>>>>> 14b24af (feat: explain templating)
      - name: smiles
        column: smiles
        text: SMILES
```

This templating syntax should allow for quite some flexibility: For every template field we will look for the key, e.g., `exp_value` as well as `exp_value_text` (which can be used to describe to the field type/value).
If this (`text`) is a column name, we will use the values from the column (therefore, effectively, jointly sample the `column` and `text` columns).
If there are multiple values for one field, we will sample combinations. 
If you want to suggest sampling from different prompt prefixes, you can do so by specifying a template fields and different `text` (but no `column`).

## Implementing a dataloader

TBD. 


## Implementing tokenizers 

TBD.

