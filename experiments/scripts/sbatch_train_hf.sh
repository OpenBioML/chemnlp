#! /bin/bash
#SBATCH --job-name="llchem-singlenode"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/fsx/proj-chemnlp/experiments/logs/job_%j.out
#SBATCH --error=/fsx/proj-chemnlp/experiments/logs/job_%j.err
#SBATCH --open-mode=append
#SBATCH --account=topchem
#SBATCH --partition=g40
#SBATCH --exclusive

### This script runs a GPT-NeoX experiments
### The first arg ($1) is the prefix directory where the environment is saved
### The second arg ($2) is the directory to use when building the environment
### The third arg ($3) is the name of the training config

set -ex # allow for exiting based on non-0 codes
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://stability.wandb.io"

# set workdir
CHEMNLP_PATH=/fsx/proj-chemnlp/$2/chemnlp

# create environment
source $CHEMNLP_PATH/experiments/scripts/env_creation_hf.sh $1 $2

# trigger run
cd $CHEMNLP_PATH
python -m torch.distributed.launch --use-env --nnodes 1 --nproc-per-node 8 \
    experiments/scripts/run_tune.py experiments/configs/hugging-face/$3
