#! /bin/bash
#SBATCH --job-name="llchem-transfer"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/fsx/proj-chemnlp/experiments/logs/transfer_%j.out
#SBATCH --error=/fsx/proj-chemnlp/experiments/logs/transfer_%j.err
#SBATCH --open-mode=append
#SBATCH --account=topchem
#SBATCH --partition=g40x
#SBATCH --exclusive

## This script recursively copies a directory to S3 storage
### The first arg ($1) is a full path to a checkpoint folder (i.e. <....>/checkpoint-1000)
### The second arg ($2) is the S3 bucket to copy to (i.e. llchem-models)

cutat=checkpoints/
TARGET_DIR=$(echo $1 | awk -F $cutat '{print $2}') # turns /a/b/checkpoints/c/d/ -> c/d/
PARENT_DIR="$(dirname "$1")"
CHILD_FILE="$(basename "$1")"

if [ ! -f "$1.tar" ]; then
    cd $PARENT_DIR && tar -cvf $CHILD_FILE.tar $CHILD_FILE
fi

aws s3 cp $1.tar s3://$2/$TARGET_DIR.tar
