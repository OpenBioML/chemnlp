#! /bin/bash
#SBATCH --job-name="chemtest"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/fsx/proj-chemnlp/experiments/logs/eval_%j.out
#SBATCH --error=/fsx/proj-chemnlp/experiments/logs/eval_%j.err
#SBATCH --open-mode=append
#SBATCH --account=topchem
#SBATCH --partition=g40x
#SBATCH --exclusive

### This script runs lm_eval2 experiments
### The first arg ($1) is the prefix directory where the environment is saved
### The second arg ($2) is the directory to use when building the environment
### The third arg ($3) is the name of the eval config.yaml file

set -ex # allow for exiting based on non-0 codes
overrides=${4:-'{}'}
# set workdir
CHEMNLP_PATH=/fsx/proj-chemnlp/$1/chemnlp

# create environment
source $CHEMNLP_PATH/experiments/scripts/env_creation_hf.sh $1 $2

export TOKENIZERS_PARALLELISM=false

# trigger run
cd $CHEMNLP_PATH/lm-evaluation-harness
python main_eval.py $3 --config_overrides $overrides
