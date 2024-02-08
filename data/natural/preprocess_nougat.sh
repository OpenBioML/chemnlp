#!/bin/bash
#SBATCH --job-name=preprocess_nougat
#SBATCH --output=/fsx/proj-chemnlp/micpie/chemnlp/data/natural/%x_%j.out
#SBATCH --account chemnlp
#SBATCH --comment chemnlp
#SBATCH --partition=cpu16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

cd /fsx/proj-chemnlp/micpie/chemnlp/data/natural/

## ensure we can use activate syntax in slurm scripts
export CONDA_ENV_PATH=/admin/home-micpie/miniconda3/envs/chemnlp
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_PATH}

python --version

python preprocess_nougat.py

#DATE=$(date -d "today" +"%Y%m%d%H%M")
#echo $DATE

#mv /fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_chemrxiv.jsonl /fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_chemrxiv_$DATE.jsonl
#tar -cvzf /fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_chemrxiv_$DATE.jsonl.tar.gz /fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_chemrxiv_$DATE.jsonl

#mv /fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_biorxiv.jsonl /fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_biorxiv_$DATE.jsonl
#tar -cvzf /fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_biorxiv_$DATE.jsonl.tar.gz /fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_biorxiv_$DATE.jsonl

#mv /fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_medrxiv.jsonl /fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_medrxiv_$DATE.jsonl
#tar -cvzf /fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_medrxiv_$DATE.jsonl.tar.gz /fsx/proj-chemnlp/micpie/chemnlp/data/natural/nougat_processed_medrxiv_$DATE.jsonl
