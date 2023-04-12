from typing import Optional

from pydantic import BaseModel, validator


class Data(BaseModel):
    path: str


class Model(BaseModel):
    base: str
    name: str
    revision: str
    checkpoint_path: Optional[str] = None


class PromptTune(BaseModel):
    enabled: bool = False
    num_virtual_tokens: Optional[int] = None
    prompt_tuning_init_text: str = " "


class TrainerConfig(BaseModel):
    output_dir: str
    num_train_epochs: float = 1.0
    learning_rate: float = 3e-4
    bf16: bool = False
    fp16: bool = False
    evaluation_strategy: str = "steps"
    logging_steps: int = 50
    eval_steps: int = 100
    save_steps: int = 100
    dataloader_num_workers: int = 0
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32

    @validator("learning_rate")
    def small_positive_learning_rate(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Specify a positive learning rate <= 1")
        return v


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "LLCheM"
    group: str
    name: str
    entity: str = "chemnlp"


class TrainPipelineConfig(BaseModel):
    data: Data
    model: Model
    prompt_tuning: PromptTune
    trainer: TrainerConfig
    wandb: WandbConfig
