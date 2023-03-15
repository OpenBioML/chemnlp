#! /bin/bash
## Must already have miniconda installed!
export CONDA_ENV_PATH=/fsx/proj-chemnlp/$1/conda/env/chemnlp-standard
export PYTHON_VER=3.8

## ensure we can use activate syntax in slurm scripts
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Create Python environment through conda
conda create --force --prefix ${CONDA_ENV_PATH} python=${PYTHON_VER} -y
conda activate ${CONDA_ENV_PATH}

# Python requirements
## cd into your directory inside of proj-chemnlp
cd /fsx/proj-chemnlp/${USER}

## clone + submodules (ok if exists)
[ ! -d 'chemnlp' ] && git clone --recurse-submodules --remote-submodules git@github.com:OpenBioML/chemnlp.git

## install
pip install chemnlp # our repo
pip install -r chemnlp/gpt-neox/requirements/requirements.txt # base gpt-neox requirements
