import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# See: https://huggingface.co/docs/transformers/perplexity

MAX_LENGTH = 2048
STRIDE = 512
BASE_STRING = "EleutherAI/pythia-"
PYTHIA_MODELS = ["70m", "160m", "410m", "1b", "1.4b", "2.8b"]
MODELS = [BASE_STRING + x for x in PYTHIA_MODELS]
print(f"Running models: {MODELS}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("aeslc")["test"]
print(f"Loaded dataset, size: {len(dataset)}")

results = {k: None for k in MODELS}

for model_name in MODELS:
    print(f"Starting model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Join dataset into one long document:
    tokenized_dataset = tokenizer(
        "\n\n".join(dataset["email_body"]), return_tensors="pt"
    )
    seq_len = tokenized_dataset.input_ids.size(1)
    prev_end_loc = 0
    nlls = []

    for begin_loc in tqdm(range(0, seq_len, STRIDE)):
        end_loc = min(begin_loc + MAX_LENGTH, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop

        # Get data:
        input_ids = tokenized_dataset.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # set all but last trg_len tokens to -100

        # Forward pass;
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break

    results[model_name] = torch.stack(nlls).sum() / end_loc
    print(results)
