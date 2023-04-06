#! /bin/bash
#SBATCH --job-name="chemtest"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
# #SBATCH --gres=gpu:2
#SBATCH --output=/fsx/proj-chemnlp/experiments/logs/job_%j.out
#SBATCH --error=/fsx/proj-chemnlp/experiments/logs/job_%j.err
#SBATCH --open-mode=append
#SBATCH --account=chemnlp
#SBATCH --partition=g40
#SBATCH --exclusive
# #SBATCH --nodelist=ip-26-0-128-[46,48,85,93-94,101,106,111,123,136,142-143,168-169,175,183,189,211,215,223,231,244],ip-26-0-129-[0-1,4,6,11,45,48,60,81-82,84-85,94,105],ip-26-0-130-[183,193],ip-26-0-131-[4-5,38,51,77,85,89,107-108,111-112,130,143,150-152,168,182-183,188],ip-26-0-132-[130,139,141-142,149,154,184],ip-26-0-133-[159-160,226,242],ip-26-0-134-[0,26-27,43,52,61],ip-26-0-137-[92,94,97,102,115-116,121,124,139,168,175],ip-26-0-139-[191,200,214,216,218,226,229,235,237,241,246],ip-26-0-142-[106,125,144,146,166,184,186,198,204,217,235,237,246,251,254],ip-26-0-143-[30,39,46,53,61,66,145,164,171,175,180,206,225,230,235,250],ip-26-0-129-122,ip-26-0-130-[12-13,19,116,127,132,134,147-148,150,163-164],ip-26-0-131-[239-240,244,247],ip-26-0-132-[7,10,21,37,93,98,107,118],ip-26-0-133-[67,76,81,89,111,115,126,131-133,140,145,148,151],ip-26-0-134-[66,76,83,90-91,105,120,134,141,157,201,219,226-227,248,254],ip-26-0-135-[1,4,22,49,55,64,67,110,118,163,173,184,186,190,192-193,204,208,219,242,255],ip-26-0-136-13,ip-26-0-137-[176,184,196,212,214,240],ip-26-0-138-[3,13,51,62,66,69,71,79,93,101,159,166,171,178,186,188,208,213],ip-26-0-141-[140,146,157,161,166,178,217,228,247],ip-26-0-142-[3,13,21,24,29,33,36,38,41,45,49,67,71,103],ip-26-0-143-[111,121],ip-26-0-128-146,ip-26-0-137-76

### This script runs a GPT-NeoX experiments
### The first arg ($1) is the prefix directory where the environment is saved
### The second arg ($2) is the directory to use when building the environment
### The third arg ($3) is the name of the cluster config
### The fourth arg ($4) is the name of the training config
### The fifth arg ($5) is the name of any supplementary config (prompt tuning)

set -ex # allow for exiting based on non-0 codes

# set workdir
CHEMNLP_PATH=/fsx/proj-chemnlp/$2/chemnlp

# create environment
source $CHEMNLP_PATH/experiments/scripts/env_creation_neox.sh $1 $2

# trigger run
cd $CHEMNLP_PATH/gpt-neox
python3 deepy.py train.py  --conf_dir $CHEMNLP_PATH/experiments/configs/gpt-neox $3 $4 $5
