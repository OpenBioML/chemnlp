#! /bin/bash
#SBATCH --job-name="llchem-singlenode"
#SBATCH --nodes=4
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
### The third arg ($3) is the name of the base training config
### The fourth arg ($4) is an optional json of any overriding configuration values

set -ex # allow for exiting based on non-0 codes
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://stability.wandb.io"
export NCCL_DEBUG=INFO
overrides=${4:-'{}'}

# set workdir
CHEMNLP_PATH=/fsx/proj-chemnlp/$2/chemnlp

# create environment
source $CHEMNLP_PATH/experiments/scripts/env_creation_hf.sh $1 $2

# install extras
cd $CHEMNLP_PATH
pip install ".[training]"

# Get multinode information
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

# Run script
srun python -m torch.distributed.launch --use-env --nnodes 4 --nproc_per_node 8 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
experiments/scripts/run_tune.py  experiments/configs/hugging-face/$3 --config_overrides $overrides
