#! /bin/bash
#SBATCH --job-name="llchem-transfer-batch"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/fsx/proj-chemnlp/experiments/logs/transfer_batch_%j.out
#SBATCH --error=/fsx/proj-chemnlp/experiments/logs/transfer_batch_%j.err
#SBATCH --open-mode=append
#SBATCH --account=topchem
#SBATCH --partition=g40
#SBATCH --exclusive

## This script recursively copies a directory to S3 storage
### The first arg ($1) is a full path to a checkpoint folder (i.e. <....>/checkpoint-1000)
### The second arg ($2) is the S3 bucket to copy to (i.e. llchem-models)

CHECKPOINT_DIR=/fsx/proj-chemnlp/experiments/checkpoints
SUBDIR=${$1#"$CHECKPOINT_DIR"}  # get diff

PARENT_DIR="$(dirname "$1")"
CHILD_FILE="$(basename "$1")"
TARGET_DIR=s3://$2/$SUBDIR

echo "Copying from $1 to ${TARGET_DIR}"

# sync only transfers new files from the source directory
cd $PARENT_DIR && tar -cvf $CHILD_FILE.tar $CHILD_FILE
aws s3 cp $1.tar $TARGET_DIR.tar
