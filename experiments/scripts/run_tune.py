"""
A Python script for finetuning language models.

    Usage: python run_tune.py <path-to-config-yml>
"""
import argparse
import json
import os
import pathlib
from typing import Dict, Optional

import datasets
import transformers
import wandb
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from chemnlp.data_val.config import TrainPipelineConfig
from chemnlp.utils import load_config

FILE_PATH = pathlib.Path(__file__).parent.resolve()
CONFIG_DIR = FILE_PATH.parent / "configs"


def print_zero_rank(rank, x):
    """Print a statement only if the zero rank process"""
    if rank in [0, -1]:
        print(x)


def run(config_path: str, config_overrides: Optional[Dict] = None) -> None:
    """Perform a training run for a given YAML defined configuration"""
    raw_config = load_config(config_path)
    config = TrainPipelineConfig(**raw_config)
    if config_overrides:
        config = config.update(config_overrides)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    global_rank = int(os.environ.get("RANK", -1))
    print_zero_rank(local_rank, config)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model.name,
        revision=config.model.revision,
    )
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    model_ref = getattr(transformers, config.model.base)
    model = model_ref.from_pretrained(
        pretrained_model_name_or_path=config.model.checkpoint_path or config.model.name,
        revision=config.model.revision
        if config.model.checkpoint_path is None
        else None,
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

    dataset = datasets.load_from_disk(config.data.path)
    split_dataset = dataset.train_test_split(test_size=0.025)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        **config.trainer.dict(exclude={"enabled", "deepspeed_config"}),
        report_to="wandb" if config.wandb.enabled else "none",
        local_rank=local_rank,
        deepspeed=CONFIG_DIR / f"deepspeed/{config.trainer.deepspeed_config}"
        if config.trainer.deepspeed_config
        else None,
    )
    print_zero_rank(local_rank, training_args)

    if config.wandb.enabled:
        config.wandb.name += f"_global_{global_rank}_local_{local_rank}_rank"
        wandb.init(**config.wandb.dict(exclude={"enabled"}), config=config.dict())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(config.trainer.output_dir + "/checkpoint-final")

    if config_overrides and local_rank in [0, -1]:
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
        default={},
        help="Any overriding parameters as a JSON.",
    )
    args = parser.parse_args()
    parsed_json_overrides = json.loads(args.config_overrides)
    run(args.config_path, parsed_json_overrides)
