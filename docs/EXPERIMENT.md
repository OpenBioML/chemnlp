# Running an experiment
Here we describe the set-up for training a model (including on the Stability cluster).

## General set-up
- Configs are in: `experiments/configs`.
-  If you wish to use a new model from Hugging Face as the starting point you will need to tokenise your data. We have an example script for `chemrxiv` which does this here: `experiments/data/prepare_hf_chemrxiv.py`.
-  You will also need to create a configuration file for the model if one does not exist e.g. `experiments/configs/hugging-face/full_160M.yml`.

If the data is already tokenised for the model you wish to use you can proceed to the next step.

## Interactive run
-  Create a conda environment as shown in [the documentation](https://github.com/OpenBioML/chemnlp/tree/main/experiments/scripts) and install `chemnlp`.
- If using Weights and Biases for logging: `export WANDB_BASE_URL="https://stability.wandb.io"`.
- Run using `torchrun`, for example:
    ```
    torchrun --nnodes 1 --nproc-per-node 4 experiments/scripts/run_tune.py experiments/configs/hugging-face/full_160M.yml
    ```
- You can use `nvidia-smi` or `wandb` logging to monitor efficiency during this step.

## Launching an experiment run through SLURM
- Take the `sbatch_<suffix>` script associated with the training run and execute this through an `sbatch` command as shown in [the documentation](https://github.com/OpenBioML/chemnlp/tree/main/experiments). This will build the conda environment and install `chemnlp` before the job begins. Note that building the environment can be a little slow so if you aren't confident your code will run it's best to test it interactively first.
- Example command:

```bash
sbatch experiments/scripts/sbatch_train_hf.sh $1 $2 $3  # see script for description of arguments
sbatch experiments/scripts/sbatch_train_hf.sh experiments/maw501 maw501 160M_full.yml  # explicit example
```
- From within the stability cluster, you can monitor your job at `/fsx/proj-chemnlp/experiments/logs` or as set in the `sbatch` script.

## Using Weights and Biases
If you don't have the required permission to log to W&B, please request this. In the interim you can disable this or log to a project under your name by changing the configuration options e.g. in `experiments/configs/hugging-face/full_160M.yml`.

## Restarting from a checkpoint
This is for Hugging Face fine-tuning only at the moment.

**WARNING:** Hugging Face **does not** know you are restarting from a checkpoint and so you may wish to change `output_dir` in the config file to avoid overwriting old checkpoints. You may wish to use a lower learning rate / different scheduler if continuing training.

You can restart training from a checkpoint by passing `checkpoint_path`, a directory containing the output from a model saved by HF's `Trainer` class.

Example config block:

```yaml
model:
  base: GPTNeoXForCausalLM
  name: EleutherAI/pythia-160m
  revision: main
  checkpoint_path: /fsx/proj-chemnlp/experiments/checkpoints/finetuned/full_160M/checkpoint-1600  # directory to restart training from
```
