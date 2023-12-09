There are many different ways to contribute to ChemNLP!
You can get in touch via the GitHub [task board](https://github.com/orgs/OpenBioML/projects/5?query=is:open+sort:updated-desc) and [issues](https://github.com/OpenBioML/chemnlp/issues?q=is:issue+is:open+sort:updated-desc&query=is:open+sort:updated-desc) and our [Discord](https://t.co/YMzpevmkiN).

## Prerequisites
Please make a [GitHub account](https://github.com/) prior to implementing a dataset; you can follow instructions to install git [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the [ChemNLP repository](https://github.com/OpenBioML/chemnlp)
2. [Clone your fork](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
3. [Make a new branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)
4. Please try using [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) for formatting your commit messages

If you wish to work on one of the submodules for the project, please see the [git workflow](SUBMODULES.md) docs.

## Create a development environment (For code/dataset contributions)

For code and data contributions, we recommend you creata a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). If you do not have conda already installed on your system, we recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html):

To create your developer environment please follow the guidelines in the `Installation and set-up` of [README.md](README.md)

## Work package leads

If you are contributing to an existing task which contains a `work package: <name>` label, please refer to the list below to find a main point of contact for that piece of work. If you've any questions or wish to contribute additional issues feel free to reach out to these work package leads from the core team on the [OpenBioML Discord](https://discord.gg/GgDBFP8ZEt) or message directly on GitHub issues.

| Name (discord & github)                                | Main Work Packages                                            |
| ------------------------------------------------------ | ------------------------------------------------------------- |
| Michael Pieler (MicPie#9427 & MicPie)                  | 💾 Structured Data, Knowledge Graph, Tokenisers, Data Sampling |
| Kevin Jablonka (Kevin Jablonka#1694 & kjappelbaum)     | 💾 Structured Data, Knowledge Graph, Tokenisers, Data Sampling |
| Bethany Connolly (bethconnolly#3951 & bethanyconnolly) | 📊 Model Evaluation                                            |
| Jack Butler (Jack Butler#8114 & jackapbutler)          | ⚙️ Model Training                                              |
| Mark Worrall (Mark Worrall#3307 & maw501)              | 🦑 Model Adaptations                                           |

# Implementing a dataset

## Contributing a dataset
One of the most important ways to contribute to the ChemNLP efforts is to implement a dataset.
With "implementing" we mean the following:

- Take a dataset from our [awesome list](https://github.com/kjappelbaum/awesome-chemistry-datasets) (if it is not there, please add it there first, so we keep track)
- Make an issue in this repository that you want to add this dataset (we will label this issue and assign it to you)
- Make a PR that adds in a new folder in `data`
  - `meta.yaml` describing the dataset in the form that `transform.py` produces. We will use this later to construct the prompts.
    > If your dataset has multiple natural splits (i.e. train, test, validation) you can create a <split>\_meta.yaml for each.
  - `transform.py` Python code that transforms the original dataset (linked in `meta.yaml`) into a form that can be consumed by the loader.
    For tabular datasets that will mostly involve: Removing/merging duplicated entries, renaming columns, dropping unused columns.
    Try to keep the output your `transform.py` uses as lean as possible (i.e. no columns that will not be used).
    In some cases, you might envision that extra columns might be useful. If this is the case, please add them (e.g., indicating some grouping, etc.)
    Even though some examples create the `meta.yaml` in `transform.py` there is no need to do so. You can also do it by hand.
    In most cases the data will be stored in a tabular format and should be named `data_clean.csv`.


    In the `transform.py` please try to download the data from an official resource.
    We encourage you to upload the raw data to HuggingFace Hub, Foundry or some other repository and then retrieve the data from there with your script, if the raw data license permits it.

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
    type: continuous , "boolean"
    names: # names for the property (to sample from for building the prompts)
      - noun: aqueous solubility
      - noun: solubility in water
  - id: SD
    description: Standard deviation of the experimental aqueous solubility value for multiple occurences
    units: log(mol/L)
    type: continuous
    names:
      - noun: standard deviation of the aqueous solubility
      - noun: tandard deviation of the solubility in water
benchmarks: # lists all benchmarks this dataset has been part of. split_column is a column in this dataframe with the value "train", "valid", "test" - indicating to which fold a specific entry belongs to
    - name: TDC
      link: https://tdcommons.ai/
      split_column: split
identifiers:
  - id: InChI # column name
    type: InChI # can be "SMILES", "SELFIES", "IUPAC", "Other", "InChI", "InChiKey", "RXNSMILES", "RXNSMILESWAdd" see IdentifierEnum
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

### Text templates

With our text template setup for the sampling you can:
* use all the data from the `meta.yaml` file,
* recode categorical data, and
* chain together multiple data fields from the tabular and meta data.

#### Example text template 1 (mainly used for tabular data)
```
The molecule with the {SMILES__description} representation of {SMILES#} exhibits {mutagenic#no &NULL}{mutagenic__names__adjective} properties.
```
* `SMILES__description` gets you the text from the description field of the SMILES identifier. The `__` dunder (double underscore) is used to indicate the levels in the `meta.yaml` file.
* `SMILES#` gets you the data of the sampled SMILES entry (= row from the tabular data). The `#` is used to get the corresponding data.
* `mutagenic#no &NULL` gets you the data with `#` and recodes it. The recoding options are separated with a `&`. In this example the binary variable `mutagenic` that can be `0` or `1` gets recoded to `no ` and `NULL`. `NULL` is a "reserved word" an indicates [no value](https://en.wikipedia.org/wiki/Null_(SQL)). Thus, the `no ` gets added in front of the `mutagenic__names__adjective` if `mutagenic# == 0`.
* `mutagenic__names__adjective` gets from the `mutagenic` target the adjective names.

#### Example text template 2 (mainly used for KG data)`
```
The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_type#} {node3_name#}.
```
* `node1_name#|node1_smiles#` chains together two data fields from the tabular data with `|` so they are jointly sampled for this position. This means that we sample in this case from the name and the SMILES representation.
* A similar setup can be used in a single data entry (= row from the tabular data) of the tabular data: For `node2_protein_names` the field can include several protein names separated by a `|`, e.g., `Pyruvate dehydrogenase E1 component subunit beta, mitochondrial|PDHE1-B` which then samples from `Pyruvate dehydrogenase E1 component subunit beta, mitochondrial` or `PDHE1-B`.

#### Example text templates 3 for multiple choice setups
Multiple choice setups are also supported. For this we need three components:
* `%multiple_choice_enum%2%aA1` can be used to list the multiple choice enumerations, i.e., `1, 2, or 3`, `A or B`, etc., The second `%` starts the multiple choice number sequence. Single integers and a range consisting of two integers separated by a `-` are supported to set the lower and higher number, e.g., `2-5` will sample a value between 2 and 5, including the boundaries, for the answer options. The third `%` is used to subselect multiple choice enumerations, i.e., `a` for lower case alphabetical enumerations, `A` for upper case alphabetical, and `1` for numerical enumerations.
* `mutagenic%` is used to list the multiple choice enumerations with the corresponding possible answer options after the multiple choice enumerations, and
* `%multiple_choice_result` is used to get the multiple choice enumeration of the answer, i.e., `1`, `c`.
Please pay attention to the `%` symbol and its position as this is used to parse the different control elements from the text template.
The sampling procedure incorporates a range of different multiple choice enumerations that are sampled randomly:
* numerical (`1, 2, 3, ...`) and alphabetical (`a, b, c, ...` or `A, B, C, ...`) enumerations combined with
* different suffixes, i.e., ` ` (no suffix), `.`, `.)`, `)`, and `:`, to create a range of different multiple choice enumerations.
If only the choices `0` or `1` are available they will be recoded with `False` and `True`.

##### Standard template
```
Task: Please answer the multiple choice question below with {%multiple_choice_enum%2%aA1}.
Question: Is the molecule with the {SMILES__description} representation of {SMILES#} {mutagenic__names__adjective}?
Options:
{mutagenic%}
Answer: {%multiple_choice_result}
```
Example output:
```
Task: Please answer the multiple choice question below with A or B.
Question: Is the molecule with the SMILES representation of CC(C)NCC(O)c1ccc2ccccc2c1 Ames mutagenic?
Options:
A) False
B) True
Answer: A"
```

##### Template for benchmarking
```
Task: Please answer the multiple choice question below with {%multiple_choice_enum%2%aA1}.
Question: Is the molecule with the {SMILES__description} representation of {SMILES#} {mutagenic__names__adjective}?
Options:
{mutagenic%}
Answer:<EOI>{%multiple_choice_result}
```
The benchmarking setup exports additional fields for the benchmarking setup, see the example below:
`{"input":"Task: Please answer the multiple choice question below with 1 or 2.\nQuestion: Is the molecule with the SMILES representation of BrCBr Ames mutagenic?\nOptions:\n1.) False\n2.) True\nAnswer:","output":" 2","output_choices":["1","2"],"correct_output_index":"1"}`
Please have a look at the following section below about the general benchmarking template setup.

#### Example text templates 4 for flexible multiple choice setups
More flexible multiple choice setups are also supported. The standard multiple choice setup from "Example text templates 3 for multiple choice setups" is intended for features of molecules as those are deduplicated during the sampling process. In contrast, this flexible multiple choice setup also lets you use the molecule identifiers, e.g., SMILES, in the multiple choice options.

For this we only need to add one component to the previously outlined multiple choice format:
* In order to let the model predict which `SMILES` has or has not the boolean variable `penetrate_BBB` we simply add `SMILES%penetrate_BBB%` as an enumeration placeholder for the possible options. With that the list of the multiple choice enumerations shows the SMILES data. Note that the `penetrate_BBB#not &NULL` is needed because the sampling is based on the individual sample (= row from the tabular data) and depending on if `penetrate_BBB` is `True` or `False` we look for a different result label because in the code we compare the sampled options to the `penetrate_BBB` value of the specific sample (= entry from the specific row from the tabular data).

```
Task: Please answer the multiple choice question.
Question: Which molecules are {penetrate_BBB#not &NULL}{penetrate_BBB__names__adjective}?
Constraint: You must select none, one or more options from {%multiple_choice_enum%2-5%aA1} without using any other words.
Options:
{SMILES%penetrate_BBB%}
Answer: {%multiple_choice_result}
```

```
Task: Please answer the multiple choice question.
Question: Which molecules are not penetrating the blood brain barrier?
Constraint: You must select none, one or more options from A, B, or C without using any other words.
Options:
A. Cc1ccsc1C(=CCCN1CCC[C@@H](C(=O)O)C1)c1sccc1C
B. CC(=O)N1CCN(c2ccc(OC[C@H]3CO[C@](Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1
C. CCCC(C)C1(CC)C(=O)NC(=S)NC1=O
Answer: B, C
```

#### Benchmarking text templates
There are two versions of text templates, i.e., text templates with and without the end-of-input token `<EOI>`:
```
The {SMILES__description} {SMILES#} is {mutagenic#no &NULL}{mutagenic__names__adjective}.
Is the {SMILES__description} {SMILES#} {mutagenic__names__adjective}:<EOI>{mutagenic# yes& no}
```
The `<EOI>` token indicates the splitting position for the benchmarking export, i.e., everything before it will be written to the `input` field and everything afterwards to the `output` field. Without `<EOI>` everything will be in the `text` field.
In the current setup, you can switch with the `benchmarking_templates` flag of the [`TemplateSampler` class](https://github.com/OpenBioML/chemnlp/blob/text_sampling/text_sampling/text_sampling.py#L104) between text templates with and without `<EOI>`.

The filename scheme uses the split information for the export, i.e., `train.jsonl`, `test.jsonl`, etc., and if no split information is available this will be set to `full` and exported to `full.jsonl`. With `<EOI>` the filename ends with `_benchmark.jsonl` instead of `.jsonl`.

Have a look at the [`meta.yaml` file](https://github.com/OpenBioML/chemnlp/blob/text_sampling/data/ames_mutagenicity/meta.yaml) to see the corresponding structure there.

In case you run into issues (or think you don't have enough compute or storage), please let us know. Also, in some cases `csv` might not be the best format. If you think that `csv` is not suitable for your dataset, let us know.

For now, you do not need to upload the transformed datasets anywhere.
We will collect the URLs of the raw data in the `meta.yaml` files and the code to produce the curated data in `transform.py` and then run in this on dedicated infrastructure.

### How will the datasets be used?

If your dataset is in tabular form, we will construct prompts using, for example, the LIFT framework.
In this case, we will sample from the identifier and targets columns. If you specify prompt templates, we will also sample from those.
Therefore, it is very important that the column names in the `meta.yaml` match the ones in the file that `transform.py` produces.
One example of a prompt we might construct is `"What is the <target_name> of <identifier>"`, where we sample `target_name` from the names of the targets listed in `meta.yaml` and `identifier` from the identifiers provided in `meta.yaml`.

#### Splits

If your dataset is part of a benchmark, please indicate what fold your data is part of using an additional `split_col` in which you use `train`, `valid`, `test` to indicate the split type.
Please indicate this in the `meta.yaml` under the field `split_col`.

#### Identifiers

We ask you to add `uris` and `pubchem_aids` in case you find suitable references. We distinguish certain types of identifiers, for which you have to specify the correct strings. The currently allowed types are in the `IdentifierEnum` in `src/chemnlp/data_val/model.py`:

- `SMILES`: Use the canonical form ([RdKit](https://www.rdkit.org/docs/GettingStartedInPython.html))
- `SELFIES`: [Self-referencing embedded strings](https://github.com/aspuru-guzik-group/selfies)
- `IUPAC`: IUPAC-Name, not use it for non-standard, common names
- `InChI`
- `InChIKey`: The key derived from the `InChI`
- `RXNSMILES`: The [reaction SMILES](https://www.daylight.com/meetings/summerschool98/course/dave/smiles-react.html) containing only educt and product
- `RXNSMILESWAdd`: The reaction SMILES also containing solvent and additives
- `Other`: For all other identifiers

##### Uniform Resource Identifiers (URIs)

If you have a uniform resource identifier (URI) that links to a suitable name of a property, please list it in the `uris` list for a given `target`.
Please ensure that the link is specific. If you have a boolean target that measures inhibition of a protein, link to `inhbitor of XY` and _not_ to the protein.
If such a link does not exist, leave the field empty.

You might find suitable links using the following resources:

- https://bioportal.bioontology.org/search
- https://goldbook.iupac.org/


#### PubChem Assay IDs

For some targets, the activity was measured using assays. In this case, please list the assays using with their _numeric_ PubChem assay id in the field `pubchem_aids`.
Please ensure that the _first_ entry in this list is a primary scan which corresponds to the target property (and not to its inverse or a control).
Keep in mind that we plan to look up the name and the description of the assay to build prompt. That is, the name of the assay of the _first entry_ in this list should also work in a prompt such as `Is <identifier> active in `<pubchem assay name>?`

#### Prompt examples


For datasets that are not in tabular form, we are still discussing the best process, but we also envision that we might perform some named-entity-recognition to also use some of the text datasets in a framework such as LIFT. Otherwise, we will simple use them in the typical GPT pretraining task.


## Implementing structured data sampler

TBD.


## Implementing tokenizers

TBD.


## Implementing model adaptations

Our first experiments will be based on [Pythia model](https://github.com/EleutherAI/pythia) suite from [EleuetherAI](https://www.eleuther.ai) that is based on [GPT-NeoX](https://github.com/EleutherAI/gpt-neox).

If you are not familiar LLM training have a look at this very good guide: [Large-scale language modeling tutorials with PyTorch from TUNiB](https://nbviewer-org.translate.goog/github/tunib-ai/large-scale-lm-tutorials/blob/main/notebooks/01_introduction.ipynb?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=de&_x_tr_pto=wapp)

Please have a look for the details in the [corresponding section in our proposal](https://docs.google.com/document/d/1C44EKSJRojm39P2CaxnEq-0FGwDRaknKxJ8lZI6xr5M/edit#heading=h.aww08l8o9tti).

## Hugging Face Hub

We have a preference for using the Hugging Face Hub and processing datasets through the [`datasets`](https://github.com/huggingface/datasets) package when storing larger datasets on the [OpenBioML](https://huggingface.co/OpenBioML) hub as it can offer us a lot of nice features such as

- Easy multiprocessing parallelism for data cleaning
- Version controlling of the datasets as well as our code
- Easy interface into tokenisation & other aspects for model training
- Reuse of utility functions once we have a consistent data structure.

However, don't feel pressured to use this if you're more comfortable contributing an external dataset in another format. We are primarily thinking of using this functionality for processed, combined datasets which are ready for training.

Feel free to reach out to one of the team and read [this guide](https://huggingface.co/docs/datasets/upload_dataset#share-a-dataset-to-the-hub) for more information.
