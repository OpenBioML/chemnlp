#! /bin/bash
#SBATCH --job-name="chemtest"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/fsx/proj-chemnlp/experiments/logs/batch_eval_%j.out
#SBATCH --error=/fsx/proj-chemnlp/experiments/logs/batch_eval_%j.err
#SBATCH --open-mode=append
#SBATCH --account=topchem
#SBATCH --partition=g40
#SBATCH --exclusive

### This script runs lm_eval2 experiments
### The first arg ($1) is the prefix directory where the environment is saved
### The second arg ($2) is the directory to use when building the environment
### The third arg ($3) is the name of the default eval config.yaml file
### The fourth arg ($4) is the path to the parent file containing the models to evaluate

set -ex # allow for exiting based on non-0 codes

# set workdir
CHEMNLP_PATH=/fsx/proj-chemnlp/$1/chemnlp

# create environment
source $CHEMNLP_PATH/experiments/scripts/env_creation_hf.sh $1 $2

# create experiment config for each model
python $CHEMNLP_PATH/experiments/scripts/eval_create_batch_configs.py $3 $4

# evaluate each model
for entry in $4/*/
do
  python $CHEMNLP_PATH/lm-evaluation-harness/main_eval.py "$entry"eval_config.yml
done
