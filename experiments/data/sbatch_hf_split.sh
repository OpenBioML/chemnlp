#! /bin/bash
#SBATCH --job-name="llched-split"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/fsx/proj-chemnlp/experiments/logs/split_%j.out
#SBATCH --error=/fsx/proj-chemnlp/experiments/logs/split_%j.err
#SBATCH --open-mode=append
#SBATCH --account=topchem
#SBATCH --partition=g40x
#SBATCH --exclusive

### This script runs tokenisation of Hugging Face datasets
### The first arg ($1) is the prefix directory where the environment is saved
### The second arg ($2) is the directory to use when building the environment
### The third arg ($3) is the path to the raw jsonlines file
### The fourth arg ($4) is the fractional size of the test / validation sets

set -ex # allow for exiting based on non-0 codes

# set workdir
cd /fsx/proj-chemnlp/$2/chemnlp

# create environment
source experiments/scripts/env_creation_hf.sh $1 $2

# trigger run
python experiments/data/split_data.py $3 $4
