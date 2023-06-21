#! /bin/bash
#SBATCH --job-name="llchem-transfer-batch"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/fsx/proj-chemnlp/experiments/logs/transfer_batch_%j.out
#SBATCH --error=/fsx/proj-chemnlp/experiments/logs/transfer_batch_%j.err
#SBATCH --open-mode=append
#SBATCH --account=topchem
#SBATCH --partition=g40x
#SBATCH --exclusive

## This script recursively copies a directory to S3 storage
### The first arg ($1) is the full path to a folder (i.e. <....>/1B_experiments/)
### The second arg ($2) is the S3 bucket to copy to (i.e. llchem-models)
### The third argument is the directory inside proj-chemnlp to find chemnlp

EC2_AVAIL_ZONE=`curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone`
EC2_REGION="`echo \"$EC2_AVAIL_ZONE\" | sed 's/[a-z]$//'`"
CHEMNLP_PATH=/fsx/proj-chemnlp/$3/chemnlp
CHECKPOINT_DIR=/fsx/proj-chemnlp/experiments/checkpoints

echo "Finding checkpoints in $1"
all_checkpoints=( $(find $1 -name "checkpoint-*" -type d) )

echo "Saving checkpoints to region: $EC2_REGION"
for chkpt in ${all_checkpoints[@]}
do
    sbatch $CHEMNLP_PATH/experiments/scripts/transfer_checkpoint_to_s3.sh $chkpt $2
    sleep 1
done
