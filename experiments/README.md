# Working with the Stability cluster

We currently run our large scale experiments on the Stability AI HPC cluster.
This subdirectory features a few helpful scripts that can help you get up and
running on the cluster.

If you believe you need access to the cluster for your work please reach out
to the core team on Discord.

- [Install Miniconda](scripts/miniconda_install.sh) -
  installs miniconda for your cluster environment.

## GPT-Neox

1. [Create Environment](scripts/env_creation_neox.sh) -
   creates a basic conda environment for experiments.

   - Creates a conda environment at the prefix `CONDA_ENV_PATH` path.
     > Using the positional argument passed into the script
   - Clones `chemnlp` into your personal cluster `USER` directory.
   - Installs the current revision of the `chemnlp` repository and
     dependencies that are in your personal directory into the conda environment.

   ```bash
   # general case
   source experiments/scripts/env_creation_neox.sh where/to/store/conda where/to/build/conda/from

   # for creating a personal environment
   source experiments/scripts/env_creation_neox.sh jack jack
   ```

2. [Training Models](scripts/sbatch_train_neox.sh) -
   runs a GPT-NeoX training pipeline

   - creates a conda environment using the `env_creation_neox.sh` script.
   - runs the GPT-NeoX `train.py` script using the user configuration
     > as GPT-NeoX configurations can be combined, the PEFT configurations are held
     > separately to the full model training and cluster configurations

   ```bash
   # general case
   sbatch experiments/scripts/sbatch_train_neox.sh where/to/store/conda where/to/build/conda/from <cluster-config-name.yml> <training-config-names.yml>

   # for typical small model soft-prompt experiments
   sbatch experiments/scripts/sbatch_train_neox.sh experiments/my-experiment jack cluster_setup.yml 160M.yml soft_prompt.yml
   ```

   > To interact with WandB services you need to authenticate yourself as per the [Stability HPC guidelines](https://www.notion.so/stabilityai/Stability-HPC-Cluster-User-Guide-226c46436df94d24b682239472e36843) to append a username + password to your .netrc file.

## Hugging Face

1. [Create Environment](scripts/env_creation_hf.sh) -
   creates a basic conda environment for experiments.

   - Creates a conda environment at the prefix `CONDA_ENV_PATH` path.
     > Using the positional argument passed into the script
   - Clones `chemnlp` into your personal cluster `USER` directory.
   - Installs the current revision of the `chemnlp` repository and
     dependencies that are in your personal directory into the conda environment.

   ```bash
   # general case
   source experiments/scripts/env_creation_hf.sh where/to/store/conda where/to/build/conda/from

   # for creating a personal environment
   source experiments/scripts/env_creation_hf.sh jack jack
   ```

2. [Training Models](scripts/sbatch_train_hf.sh) -
   runs a Hugging Face training pipeline

   - creates a conda environment using the `env_creation_hf.sh` script.
   - runs the Hugging Face `run_tune.py` script with the user configuration

   ```bash
   # general case
   sbatch experiments/scripts/sbatch_train_hf.sh where/to/store/conda where/to/build/conda/from <training-config-name.yml>

   # for typical finetuning experiments
   sbatch experiments/scripts/sbatch_train_hf.sh experiments/my-experiment jack 160M_peft.yml
   ```

   > To interact with WandB services you need to authenticate yourself as per the [Stability HPC guidelines](https://www.notion.so/stabilityai/Stability-HPC-Cluster-User-Guide-226c46436df94d24b682239472e36843) to append a username + password to your .netrc file.
