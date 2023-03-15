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
   # for creating a personal environment
   source experiments/scripts/stability-cluster/env_creation.sh jack/ jack/

   # for creating an experiment environment (usually called by another script)
   source experiments/scripts/stability-cluster/env_creation.sh experiments/my-experiment jack/
   ```
