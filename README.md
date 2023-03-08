[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

# ChemNLP project 🧪🚀
The ChemNLP project aims to
1) create an extensive chemistry dataset and
1) use it to train large language models (LLMs) that can leverage the data for a wide range of chemistry applications.

For more details see our [information material section below](#information-material).


# Information material
* [Introduction presentation](https://docs.google.com/presentation/d/1JkAKJveYsNGtAWoaksU8ykTdrC0aX3FshiFJ13SU6o8/edit?usp=sharing)
* [Project proposal](https://docs.google.com/document/d/1C44EKSJRojm39P2CaxnEq-0FGwDRaknKxJ8lZI6xr5M/edit?usp=sharing)
* [Task board](https://github.com/orgs/OpenBioML/projects/5/views/1)
* [awesome-chemistry-datasets repository](https://github.com/kjappelbaum/awesome-chemistry-datasets) to collect interesting chemistry datasets
* Weekly meetings are set up soon! Please join our [Discord community](#community) for more information.


# Community
Feel free to join our `#chemnlp` channel on our [OpenBioML discord server](https://discord.com/invite/GgDBFP8ZEt) to start the discussion in more detail.


# Contributing
ChemNLP is an open-source project - your involvement is warmly welcome! If you're excited to join us, we recommend the following steps:
* Join our [Discord server](#community).
* Have a look at our [contributing guide](https://github.com/OpenBioML/chemnlp/blob/main/CONTRIBUTING.md).
* Looking for ideas? See our [task board](https://github.com/orgs/OpenBioML/projects/5/views/1) to see what we may need help with.
* Have an idea? Create an [issue](https://github.com/OpenBioML/chemnlp/issues)!


# Note on the "ChemNLP" name
Our OpenBioML ChemNLP project is not afiliated to the [ChemNLP library from NIST](https://arxiv.org/abs/2209.08203) and we use "ChemNLP" as a general term to highlight our project focus. The datasets and models we create through our project will have a unique and recognizable name when we release them.


# About OpenBioML.org
See https://openbioml.org, especially [our approach and partners](https://openbioml.org/approach-and-partners.html).

# Installation and set-up
Create a new conda environment for Python 3.8:
```
conda create -n chemnlp python=3.8
conda activate chemnlp
```
To install the `chemnlp` package (and required dependencies):

```
pip install chemnlp
```

If working on developing the python package:
```
pip install -e "chemnlp[dev]"  # to install development dependencies
```

If working on model training, request access to the `wandb` project `chemnlp`
and log-in to `wandb` with your API key per [here](https://docs.wandb.ai/quickstart).

### Adding a new dataset (to the model training pipline)

We specify datasets by creating a new function [here](src/chemnlp/data/hf_datasets.py) which is named per the dataset on Hugging Face. At present the function must accept a tokenizer and return back the tokenized train and validation datasets.
