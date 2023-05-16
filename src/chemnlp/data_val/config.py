from typing import List, Optional, Union

from pydantic import BaseModel, validator
from transformers.trainer_utils import SchedulerType


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
    lr_scheduler_type: Union[str, SchedulerType] = "constant"
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    bf16: bool = False
    fp16: bool = False
    evaluation_strategy: str = "steps"
    logging_steps: int = 50
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    dataloader_num_workers: int = 0
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    gradient_checkpointing: bool = False
    deepspeed_config: Optional[str] = None

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


class DataMixingConfig(BaseModel):
    data_paths: List[str]
    num_tokens: List[int]
    context_length: int
    save_path: str
