"""
A Python script for finetuning language models.

    Usage: python run_tune.py <path-to-config-yml>
"""

import argparse
import json
import os
import pathlib
from typing import Dict, Optional, Union

import datasets
import torch
import transformers
import wandb
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from chemnlp.data_val.config import TrainPipelineConfig
from chemnlp.trainer import LLcheMTrainer
from chemnlp.utils import (
    collect_cpu_memory,
    collect_gpu_memory,
    get_local_ip_address,
    load_and_split,
    load_config,
)

FILE_PATH = pathlib.Path(__file__).parent.resolve()
CONFIG_DIR = FILE_PATH.parent / "configs"
ZERO_RANK = [0, -1]


def should_restart(
    output_dir: str, restart_checkpoint: Union[str, bool]
) -> Union[str, bool]:
    """
    The default behaviour causes an automatic restart from the most recent checkpoint.
    However, you can also specify
        - specific checkpoint folders
        - a value of False to not load a checkpoint
    """
    if isinstance(restart_checkpoint, str):
        # if specific checkpoint provided
        if os.path.isdir(restart_checkpoint):
            return restart_checkpoint
        else:
            raise ValueError(f"checkpoint cannot be found at {restart_checkpoint}")

    elif restart_checkpoint:
        # if we want to restart
        if os.path.isdir(output_dir):
            # if exists
            checkpoint_folders = [
                f for f in os.listdir(output_dir) if "checkpoint" in f
            ]
            if checkpoint_folders:
                # if has any checkpoint-X folder
                return True

    # default to False if no checkpoints exist or none chosen
    return False


def print_zero_rank(rank, x):
    """Print a statement only if the zero rank process"""
    if rank in ZERO_RANK:
        print(x)


def run(config_path: str, config_overrides: Optional[Dict] = None) -> None:
    """Perform a training run for a given YAML defined configuration"""
    raw_config = load_config(config_path)
    config = TrainPipelineConfig(**raw_config)
    if config_overrides:
        config = config.update(config_overrides)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    global_rank = int(os.environ.get("RANK", -1))
    ip_add = get_local_ip_address()
    num_devices = torch.cuda.device_count()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    print_zero_rank(local_rank, config)
    print(
        f"IP: {ip_add} Local: {local_rank} Global: {global_rank}, # GPUs: {num_devices} CUDA_VISIBLE: {visible_devices}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model.name,
        revision=config.model.revision,
    )
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    model_ref = getattr(transformers, config.model.base)
    model = model_ref.from_pretrained(
        pretrained_model_name_or_path=config.model.checkpoint_path or config.model.name,
        revision=(
            config.model.revision if config.model.checkpoint_path is None else None
        ),
    )

    if config.prompt_tuning.enabled:
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            **config.prompt_tuning.dict(exclude={"enabled"}),
            tokenizer_name_or_path=config.model.name,
        )
        model = get_peft_model(model, peft_config)
    total_trainables = sum(
        [param.numel() for param in model.parameters() if param.requires_grad]
    )
    print_zero_rank(
        local_rank,
        f"Total Parameters: {model.num_parameters()} Trainable Parameters: {total_trainables}",
    )

    if isinstance(config.data.path, list):
        # set validation size per dataset or constant
        validation_sizes = (
            [config.data.validation_size] * len(config.data.path)
            if isinstance(config.data.validation_size, float)
            else config.data.validation_size
        )
        assert len(config.data.path) == len(
            validation_sizes
        ), "you must specify an equal number of datasets and validation sizes"

        # collect all datasets
        data_sources = {
            source: load_and_split(source, val_size)
            for source, val_size in zip(config.data.path, validation_sizes)
        }
        train_ds = [d["train"] for d in data_sources.values()]
        if config.data.interleave_probs:
            # interleaving with specific probabilities and stopping criterion
            assert (
                config.data.interleave_probs and config.data.sampling_criterion
            ), "both interleaving probabilities and strategy must be set"
            assert len(config.data.interleave_probs) == len(
                config.data.path
            ), "you must specify an equal number of datasets and probabilities"
            train_dataset = datasets.interleave_datasets(
                train_ds,
                probabilities=config.data.interleave_probs,
                stopping_strategy=config.data.sampling_criterion,
            )
        else:
            # do full random concatenation of training data
            train_dataset = datasets.concatenate_datasets(train_ds)
        # NOTE validation set is independent of the mixing strategy
        val_dataset = datasets.concatenate_datasets(
            [d["test"] for d in data_sources.values()]
        )

    else:
        # load single dataset
        loaded_dataset = load_and_split(config.data.path, config.data.validation_size)
        train_dataset, val_dataset = loaded_dataset["train"], loaded_dataset["test"]
    print_zero_rank(
        local_rank,
        f"{train_dataset.num_rows} / {val_dataset.num_rows} samples for training / validation",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_args = TrainingArguments(
        **config.trainer.dict(exclude={"deepspeed_config", "restart_checkpoint"}),
        report_to="wandb" if config.wandb.enabled else "none",
        local_rank=local_rank,
        deepspeed=(
            CONFIG_DIR / f"deepspeed/{config.trainer.deepspeed_config}"
            if config.trainer.deepspeed_config
            else None
        ),
    )
    print_zero_rank(local_rank, training_args)

    if config.wandb.enabled:
        config.wandb.name += f"_global_{global_rank}_local_{local_rank}_rank"
        wandb.init(**config.wandb.dict(exclude={"enabled"}), config=config.dict())

        # custom logging at start of training
        wandb.log({"Node IP Address": ip_add})
        wandb.log(
            {"CPU_start": collect_cpu_memory(), "GPU_start": collect_gpu_memory()}
        )

    # start train or auto-restart
    trainer = LLcheMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train(
        resume_from_checkpoint=should_restart(
            training_args.output_dir, config.trainer.restart_checkpoint
        )
    )
    if config.trainer.save_strategy == "steps":
        trainer.save_model(config.trainer.output_dir + "/checkpoint-final")

    if config.wandb.enabled:
        # custom logging at end of training
        wandb.log({"CPU_end": collect_cpu_memory(), "GPU_end": collect_gpu_memory()})

    if config_overrides and local_rank in ZERO_RANK and global_rank in ZERO_RANK:
        # only save down successful grid search runs
        config_dir = pathlib.Path(config.trainer.output_dir).parent.absolute()
        with open(f"{config_dir}/{config.wandb.name}_overrides.json", "a+") as fp:
            # record as chkpt: config
            recorded_config = {config.trainer.output_dir: config_overrides}
            json.dump(recorded_config, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="The full path to the YAML config file.")
    parser.add_argument(
        "--config_overrides",
        required=False,
        default="{}",
        help="Any overriding parameters as a JSON.",
    )
    args = parser.parse_args()
    parsed_json_overrides = json.loads(args.config_overrides)
    run(args.config_path, parsed_json_overrides)
