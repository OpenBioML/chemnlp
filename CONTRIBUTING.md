There are many different ways to contribute to ChemNLP!
You can get in touch via the GitHub [task board](https://github.com/orgs/OpenBioML/projects/5?query=is:open+sort:updated-desc) and [issues](https://github.com/OpenBioML/chemnlp/issues?q=is:issue+is:open+sort:updated-desc&query=is:open+sort:updated-desc) and our [Discord](https://t.co/YMzpevmkiN).

## Pre-Requisites
Please make a [GitHub account](https://github.com/) prior to implementing a dataset; you can follow instructions to install git [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the [ChemNLP repository](https://github.com/OpenBioML/chemnlp)
2. [Clone your fork](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
3. [Make a new branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)
4. Please try using [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) for formatting your commit messages

## Create a development environment (For code/dataset contributions)

For code and data contributions, we recommend you creata a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). If you do not have conda already installed on your system, we recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html):

```bash
conda env create -f conda.yaml  # Creates a conda env
conda activate chemnlp  # Activate your conda environment
```

Then, please run

```bash
pre-commit install
```

to install the [pre-commit hooks](https://pre-commit.com/). These will automatically format and lint your code upon every commit.
There might be some warnings, e.g., by `flake8`. If you struggle with them, do not hestiate to contact us.

# Implementing a dataset

## Contributing a dataset
One of the most important ways to contribute to the ChemNLP efforts is to implement a dataset.
With "implementing" we mean the following:

- Take a dataset from our [awesome list](https://github.com/kjappelbaum/awesome-chemistry-datasets) (if it is not there, please add it there first, so we keep track)
- Make an issue in this repository that you want to add this dataset (we will label this issue and assign it to you)
- Make a PR that adds in a new folder in `data`
  - `meta.yaml` describing the dataset in the form that `transform.py` produces. We will use this later to construct the prompts.
  - `transform.py` Python code that transforms the original dataset (linked in `meta.yaml`) into a form that can be consumed by the loader.
    For tabular datasets that will mostly involve: Removing/merging duplicated entries, renaming columns, dropping unused columns.
    Try to keep the output your `transform.py` uses as lean as possible (i.e. no columns that will not be used).
    In some cases, you might envision that extra columns might be useful. If this is the case, please add them (e.g., indicating some grouping, etc.)
    Even though some examples create the `meta.yaml` in `transform.py` there is no need to do so. You can also do it by hand.


    In the `transform.py` please try to download the data from an official resource.
    We encourage you to upload the raw data to HuggingFace, Foundry or some other repository and then retrieve the data from there with your script.

  - If you need additional dependencies, add them to `dev-requirements.txt` (those are needed for linting/testing/validation) or `requirements.txt` (those are the ones for running `transform.py`)



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
Please do not simply copy/paste generic descriptions but try to give a concise and specific description for the dataset you are adding.

For the typical material-property datasets, we will later use the `identifier` and `property` columns to create and fill prompt templates.
In case your dataset isn't a simple tabular dataset with chemical compounds and properties, please also add the following additional fields for the templates:


```yaml
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
      - name: smiles
        column: smiles
        text: SMILES
```

This templating syntax should allow for quite some flexibility: For every template field we will look for the key, e.g., `exp_value` as well as `exp_value_text` (which can be used to describe to the field type/value).
If this (`text`) is a column name, we will use the values from the column (therefore, effectively, jointly sample the `column` and `text` columns).
If there are multiple values for one field, we will sample combinations.
If you want to suggest sampling from different prompt prefixes, you can do so by specifying a template fields and different `text` (but no `column`).

In case you run into issues (or think you don't have enough compute or storage, please let us know). Also, in some cases `csv` might not be the best format. If you think that `csv` is not suitable for your dataset, let us know.

For now, you do not need to upload the transformed datasets anywhere.
We will collect the URLs of the raw data in `meta.yaml` and the code to produce curated data in `transform.py` and then run in this on dedicated infrastructure.

### How will the datasets be used?

If your dataset is in tabular form, we will construct prompts using, for example, the LIFT framework.
In this case, we will sample from the identifier and targets columns. If you specify prompt templates, we will also sample from those.
Therefore, it is very important that the column names in the `meta.yaml` match the ones in the file that `transform.py` produces.
One example of a prompt we might construct is `"What is the <target_name> of <identifier>"`, where we sample `target_name` from the names of the targets listed in `meta.yaml` and `identifier` from the identifiers provided in `meta.yaml`.

#### Splits 

If your dataset is part of a benchmark, please indicate what fold your data is part of using an additional `split_col` in which you use `train`, `valid`, `test` to indicate the split type. 
#### Identifiers

We ask you to add `uris` and `pubchem_aids` in case you find suitable references.


##### Uniform Resource Identifiers (URIs)

If you have a uniform resource identifier (URI) that links to a suitable name of a property, please list it in the `uris` list for a given `target`.
Please ensure that the link is specific. If you have a boolean target that measures inhibition of a protein, link to `inhbitor of XY` and _not_ to the protein.
If such a link does not exist, leave the field empty.

You might find suitable links using the following resources:

- https://bioportal.bioontology.org/search
- https://goldbook.iupac.org/


#### PubChem Assay IDs

For some targets, the activity was measured using assays. In this case, please list the assays using with their _numeric_ PubChem assay id in the field `pubchem_aids`.
Please ensure that the _first_ entry in this list is a primary scan for which corresponds to the target property (and not to its inverse or a control).
Keep in mind that we plan to look up the name and the description of the assay to build prompt. That is, the name of the assay of the _first entry_ in this list should also work in a prompt such as `Is <identifier> active in `<pubchem assay name>?`

#### Prompt examples

##### Boolean variables

- `Is <name> <identifier>?`
- ```
  What molecules in the list are <name>?

  - <identifier_1>
  - <identifier_2>
  - <identifier_3>
  ```


#### Continuous variables

- `What is <name> of <identifier>?`
- ```
  What is the molecule with largest <name> in the following list?

  - <identifier_1>
  - <identifier_2>
  - <identifier_3>
  ```



For datasets that are not in tabular form, we are still discussing the best process, but we also envision that we might perform some named-entity-recognition to also use some of the text datasets in a framework such as LIFT. Otherwise, we will simple use them in the typical GPT pretraining task.


## Implementing structured data sampler

TBD.


## Implementing tokenizers

TBD.


## Implementing model adaptations

Our first experiments will be based on [Pythia model](https://github.com/EleutherAI/pythia) suite from [EleuetherAI](https://www.eleuther.ai) that is based on [GPT-NeoX](https://github.com/EleutherAI/gpt-neox).

If you are not familiar LLM training have a look at this very good guide: [Large-scale language modeling tutorials with PyTorch from TUNiB](https://nbviewer-org.translate.goog/github/tunib-ai/large-scale-lm-tutorials/blob/main/notebooks/01_introduction.ipynb?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=de&_x_tr_pto=wapp)

Please have a look for the details in the [corresponding section in our proposal](https://docs.google.com/document/d/1C44EKSJRojm39P2CaxnEq-0FGwDRaknKxJ8lZI6xr5M/edit#heading=h.aww08l8o9tti).
