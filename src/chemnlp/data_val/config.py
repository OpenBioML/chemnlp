from pydantic import BaseModel, validator


class Data(BaseModel):
    path: str
    

class Model(BaseModel):
    base: str
    name: str
    revision: str


class PromptTune(BaseModel):
    enabled: bool = False
    num_virtual_tokens: int = None
    prompt_tuning_init_text: str = " "


class TrainerConfig(BaseModel):
    output_dir: str
    epochs: int = 1
    learning_rate: float = 3e-4
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    wandb_enabled: bool = False
    wandb_project: str = "chemnlp"
    run_name: str

    @validator("learning_rate")
    def small_positive_learning_rate(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Specify a positive learning rate <= 1")
        return v


class TrainPipelineConfig(BaseModel):
    data: Data
    model: Model
    prompt: PromptTune
    train: TrainerConfig
