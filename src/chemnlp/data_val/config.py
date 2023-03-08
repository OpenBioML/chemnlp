from typing import List

from pydantic import BaseModel, Field


class Data(BaseModel):
    datasets: List[str] = Field(default_factory=list)
    subsample: bool
    num_train_samples: int
    num_val_samples: int
    pad_to_multiple_of: int


class Model(BaseModel):
    base: str
    name: str
    revision: str


class PromptTune(BaseModel):
    num_virtual_tokens: int = None
    prompt_tuning_init_text: str = " "


class TrainerConfig(BaseModel):
    output_dir: str
    epochs: int = 1
    learning_rate: float = 3e-4
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32


class TrainPipelineConfig(BaseModel):
    data: Data
    model: Model
    prompt: PromptTune
    train: TrainerConfig
