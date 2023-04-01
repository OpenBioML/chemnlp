import argparse

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


def run(config_path: str):
    raw_config = load_config(config_path)
    config = TrainPipelineConfig(**raw_config)
    print(config)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model.name,
        revision=config.model.revision,
    )
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    model_ref = getattr(transformers, config.model.base)
    model = model_ref.from_pretrained(
        pretrained_model_name_or_path=config.model.name,
        revision=config.model.revision,
    )

    if config.prompt.enabled:
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=config.prompt.num_virtual_tokens,
            prompt_tuning_init_text=config.prompt.prompt_tuning_init_text,
            tokenizer_name_or_path=config.model.name,
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = datasets.load_from_disk(config.data.path)
    split_dataset = dataset.train_test_split(test_size=0.2)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config.train.output_dir,
        evaluation_strategy="epoch",
        learning_rate=config.train.learning_rate,
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=config.train.per_device_train_batch_size,
        per_device_eval_batch_size=config.train.per_device_eval_batch_size,
        report_to="wandb" if config.train.wandb_enabled else None,
    )

    if config.train.is_wandb:
        wandb.init(
            project=config.train.wandb_project,
            name=f"{config.model.name}-{config.train.run_name}-finetuned",
            config=config.dict(),
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    print(trainer.state.log_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="The full path to the YAML config file.")
    args = parser.parse_args()
    run(args.config_path)
