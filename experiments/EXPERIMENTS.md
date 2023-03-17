# Working with the Stability cluster

We currently run our large scale experiments on the Stability AI HPC cluster.
This subdirectory features a few helpful scripts that can help you get up and
running on the cluster.

1. [Install Miniconda](stability-cluster/miniconda_install.sh) -
   installs miniconda for your cluster environment.

2. [Create Environment](stability-cluster/env_creation.sh) -
   creates a basic conda environment for experiments.

   - Creates a conda environment at the prefix `CONDA_ENV_PATH` path.
     > Using the positional argument passed into the script
   - Clones `chemnlp` into your personal cluster `USER` directory.
   - Installs the current revision of the `chemnlp` repository and
     dependencies that are in your personal directory into the conda environment.

   ```bash
   # general case
   source experiments/scripts/stability-cluster/env_creation.sh where/to/store/conda where/to/build/conda/from/

   # for creating a personal environment
   source experiments/scripts/stability-cluster/env_creation.sh jack/ jack/
   ```

3. [Running Experiment](stability-cluster/sbatch_run.sh) -
   runs a GPT-NeoX training pipeline

   - creates a conda environment using the `env_creation.sh` script.
   - runs the GPT-NeoX `train.py` script using the user configuration
     > as GPT-NeoX configurations can be combined, the PEFT configurations are held
     > separately to the full model training and cluster configurations

   ```bash
   # general case
   sbatch experiments/scripts/stability-cluster/sbatch_run.sh where/to/store/conda where/to/build/conda/from/ <cluster-config-name.yml> <training-config-names.yml>

   # for typical small model finetuning experiments
   sbatch experiments/scripts/stability-cluster/sbatch_run.sh experiments/my-experiment jack cluster_setup.yml 160M.yml

   # for typical small model soft-prompt experiments
   sbatch experiments/scripts/stability-cluster/sbatch_run.sh experiments/my-experiment jack cluster_setup.yml 160M.yml soft_prompt.yml
   ```
