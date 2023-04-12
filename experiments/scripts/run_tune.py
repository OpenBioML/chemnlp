"""
A Python script for finetuning language models.

    Usage: python run_tune.py <path-to-config-yml>
"""
import argparse
import os

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


def run(config_path: str) -> None:
    """Perform a training run for a given YAML defined configuration"""
    raw_config = load_config(config_path)
    config = TrainPipelineConfig(**raw_config)
    gpu_rank = os.environ.get("LOCAL_RANK", -1)
    print(config)

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
        model.print_trainable_parameters()
    else:
        print(f"Total Parameters: {model.num_parameters()}")

    dataset = datasets.load_from_disk(config.data.path)
    split_dataset = dataset.train_test_split(test_size=0.025)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        **config.trainer.dict(exclude={"enabled"}),
        report_to="wandb" if config.wandb.enabled else "none",
        local_rank=gpu_rank,
    )

    if config.wandb.enabled:
        config.wandb.name = f"{config.wandb.name}_rank_{gpu_rank}"
        wandb.init(**config.wandb.dict(exclude={"enabled"}), config=config.dict())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    assert trainer.model.device.type != "cpu", "Stopping as model is on CPU"
    trainer.train()
    trainer.save_model(config.trainer.output_dir + "/checkpoint-final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="The full path to the YAML config file.")
    args = parser.parse_args()
    run(args.config_path)
