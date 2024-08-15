from unsloth import FastLanguageModel
import torch
from unsloth import add_new_tokens
from typing import Optional, List
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
import wandb
from datasets import load_dataset
import fire

def load_model(
    rank: int = 128,
    train_embeddings: bool = True,
    add_special_tokens: Optional[List[str]] = None,
):
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    add_new_tokens(model, tokenizer, new_tokens=add_special_tokens)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    if train_embeddings:
        target_modules += ["embed_tokens", "lm_head"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=target_modules,
        lora_alpha=rank / 4,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=True,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    return model, tokenizer


def train(
    model, tokenizer, dataset, run_name: str, batch_size: int = 64, max_seq_length=2048
):
    wandb.init(project="chemnlp-ablations", name=run_name)
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_ratio=0.1,
            num_train_epochs=1,
            learning_rate=5e-5,
            embedding_learning_rate=1e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=f"outputs_{run_name}",
        ),
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    model.save_pretrained(f"lora_model_{run_name}")  # Local saving
    tokenizer.save_pretrained(f"lora_model_{run_name}")


def create_dataset(tokenizer, datasets):
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        outputs = []
        for t in examples["text"]:
            outputs.append(t + EOS_TOKEN)
        return {
            "text": outputs,
        }

    dataset = load_dataset("json", data_files=datasets)
    dataset = dataset["train"]

    dataset = dataset.map(formatting_prompts_func, batched=True)

    return dataset


def run(data_files: List[str], train_embeddings: bool, run_name: str, batch_size: int, add_special_tokens: Optional[List[str]]=None)
    model, tokenizer = load_model(train_embeddings=train_embeddings, add_special_tokens=add_special_tokens )

    dataset = create_dataset(
        tokenizer, data_files
    )

    train(model, tokenizer, dataset, run_name, batch_size=batch_size)


if __name__ == "__main__":
    fire.Fire(run)
