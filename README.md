[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

<img src="/assets/logo/chemnlp.png" width="25%" height="25%">

# ChemNLP project 🧪🚀

The ChemNLP project aims to

1. create an extensive chemistry dataset and
1. use it to train large language models (LLMs) that can leverage the data for a wide range of chemistry applications.

For more details see our [information material section below](#information-material).

# Information material

- [Introduction presentation](https://docs.google.com/presentation/d/1JkAKJveYsNGtAWoaksU8ykTdrC0aX3FshiFJ13SU6o8/edit?usp=sharing)
- [Project proposal](https://docs.google.com/document/d/1C44EKSJRojm39P2CaxnEq-0FGwDRaknKxJ8lZI6xr5M/edit?usp=sharing)
- [Task board](https://github.com/orgs/OpenBioML/projects/5/views/1)
- [awesome-chemistry-datasets repository](https://github.com/kjappelbaum/awesome-chemistry-datasets) to collect interesting chemistry datasets
- Weekly meetings are set up soon! Please join our [Discord community](#community) for more information.

# Community

Feel free to join our `#chemnlp` channel on our [OpenBioML discord server](https://discord.com/invite/GgDBFP8ZEt) to start the discussion in more detail.

# Contributing

ChemNLP is an open-source project - your involvement is warmly welcome! If you're excited to join us, we recommend the following steps:

- Join our [Discord server](#community).
- Have a look at our [contributing guide](docs/CONTRIBUTING.md).
- Looking for ideas? See our [task board](https://github.com/orgs/OpenBioML/projects/5/views/1) to see what we may need help with.
- Have an idea? Create an [issue](https://github.com/OpenBioML/chemnlp/issues)!

# Note on the "ChemNLP" name

Our OpenBioML ChemNLP project is not affiliated to the [ChemNLP library from NIST](https://arxiv.org/abs/2209.08203) and we use "ChemNLP" as a general term to highlight our project focus. The datasets and models we create through our project will have a unique and recognizable name when we release them.

# About OpenBioML.org

See https://openbioml.org, especially [our approach and partners](https://openbioml.org/approach-and-partners.html).

# Installation and set-up

Create a new conda environment with Python 3.8:

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

If extra dependencies are required (e.g. for dataset creation) but are not needed for the main package please add to the `pyproject.toml` in the `dataset_creation` variable and ensure this is reflected in the `conda.yml` file.

Then, please run

```bash
pre-commit install
```

to install the [pre-commit hooks](https://pre-commit.com/). These will automatically format and lint your code upon every commit.
There might be some warnings, e.g., by `flake8`. If you struggle with them, do not hestiate to contact us.

**Note**

If working on model training, request access to the `wandb` project `chemnlp`
and log-in to `wandb` with your API key per [here](https://docs.wandb.ai/quickstart).

### Adding a new dataset (to the model training pipline)

We specify datasets by creating a new function [here](src/chemnlp/data/hf_datasets.py) which is named per the dataset on Hugging Face. At present the function must accept a tokenizer and return back the tokenized train and validation datasets.

### Cloning submodules

In order to work on the git submodules (i.e. `gpt-neox`) you will need to ensure you have cloned them.

To do this at the same time as cloning ChemNLP:

  ```
   # using ssh (if you have your ssh key on GitHub)
  git clone --recurse-submodules git@github.com:OpenBioML/chemnlp.git

   # using https (if you use personal access token)
  git clone --recurse-submodules [git@github.com:OpenBioML/chemnlp.git ](https://github.com/OpenBioML/chemnlp.git)
  ```

  > This will automatically initialize and update each submodule in the repository, including nested submodules if any of the submodules in the repository have submodules themselves.

 If you've already cloned ChemNLP and don't have the submodules you can run:
```
git submodule update --init --recursive
```

See [here](docs/SUBMODULES.md) for more information about contributing to submodules.

### Experiments

Follow the guidelines [here](docs/EXPERIMENT.md) for more information about running experiments on the Stability AI cluster.
