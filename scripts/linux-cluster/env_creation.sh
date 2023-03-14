#! /bin/bash
# Must already have miniconda installed!

cd ~

# Create Python env
export CONDA_ENV_PATH=/fsx/conda/env/chemnlp-standard
export PYTHON_VER=3.8
conda create --prefix ${CONDA_ENV_PATH} python=${PYTHON_VER} -y
conda activate ${CONDA_ENV_PATH}

# Python requirements
## clone + submodules
git clone --recurse-submodules --remote-submodules git@github.com:OpenBioML/chemnlp.git

## install
pip install chemnlp # our repo
pip install -r chemnlp/gpt-neox/requirements/requirements.txt # basic gpt-neox requirements
