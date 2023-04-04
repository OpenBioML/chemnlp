from typing import Optional
from pydantic import BaseModel, validator


class Data(BaseModel):
    path: str


class Model(BaseModel):
    base: str
    name: str
    revision: str


class PromptTune(BaseModel):
    enabled: bool = False
    num_virtual_tokens: Optional[int] = None
    prompt_tuning_init_text: str = " "


class TrainerConfig(BaseModel):
    output_dir: str
    num_train_epochs: int = 1
    learning_rate: float = 3e-4
    fp16: bool = False
    logging_steps: int = 50
    save_steps: int = 100
    dataloader_num_workers: int = 0
    torch_compile: bool = False
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32

    @validator("learning_rate")
    def small_positive_learning_rate(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Specify a positive learning rate <= 1")
        return v


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "chemnlp"
    group: str
    name: str


class TrainPipelineConfig(BaseModel):
    data: Data
    model: Model
    prompt_tuning: PromptTune
    trainer: TrainerConfig
    wandb: WandbConfig
