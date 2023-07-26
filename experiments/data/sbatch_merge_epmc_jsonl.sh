#! /bin/bash
#SBATCH --job-name="merge-epmc"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/fsx/proj-chemnlp/experiments/logs/data_merge_%j.out
#SBATCH --error=/fsx/proj-chemnlp/experiments/logs/data_merge_%j.err
#SBATCH --open-mode=append
#SBATCH --account=topchem
#SBATCH --partition=g40x
#SBATCH --exclusive

### The first arg ($1) is the prefix directory where the environment is saved
### The second arg ($2) is the directory to use when building the environment

set -ex # allow for exiting based on non-0 codes

# set workdir
CHEMNLP_PATH=/fsx/proj-chemnlp/$2/chemnlp

# create environment
source $CHEMNLP_PATH/experiments/scripts/env_creation_hf.sh $1 $2

# trigger run
cd $CHEMNLP_PATH
python experiments/data/merge_epmc_to_jsonl.py
