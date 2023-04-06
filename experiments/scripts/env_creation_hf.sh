#! /bin/bash
### This script creates a conda environment for chemnlp
### The first arg ($1) is the prefix directory where the environment is saved
### The second arg ($2) is the directory to use when building the environment

## Must already have miniconda installed!
export CONDA_ENV_PATH=/fsx/proj-chemnlp/$1/conda/env/chemnlp-hf
export PYTHON_VER=3.8
CUDA_VERSION=11.7
CONDA_BASE=$(conda info --base)

## ensure we can use activate syntax in slurm scripts
source $CONDA_BASE/etc/profile.d/conda.sh

# Create Python environment through conda
if [ -d "${CONDA_ENV_PATH}" ]; then rm -Rf ${CONDA_ENV_PATH}; fi
conda create --force --prefix ${CONDA_ENV_PATH} python=${PYTHON_VER} -y
conda activate ${CONDA_ENV_PATH}

# Python requirements
## cd into your directory inside of proj-chemnlp
cd /fsx/proj-chemnlp/$2

## clone + submodules (ok if exists)
[ ! -d 'chemnlp' ] && git clone --recurse-submodules git@github.com:OpenBioML/chemnlp.git

## install core requirements
conda install -y pytorch torchvision torchaudio pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia --verbose
