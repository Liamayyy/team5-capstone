# generate_activations.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Directories
activations_dir = "activations"
os.makedirs(activations_dir, exist_ok=True)

# Model setup
base_model_name = "google/gemma-2-2b"
finetuned_model_path = "Liamayyy/gemma-2-2b-medical-v2"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto")
finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path, torch_dtype=torch.bfloat16, device_map="auto")
base_model.eval()
finetuned_model.eval()

# Load only a small shard (1/1000th) of the dataset to avoid full download or processing
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:1000]")

def tokenize_function(examples):
    return tokenizer(examples["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
train_dataset = tokenized_dataset

# DataLoader
dataloader = DataLoader(train_dataset, batch_size=8)

# Hook
def make_hook(activation_list):
    def hook_fn(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        activation_list.append(act.detach().cpu())
    return hook_fn

def process_and_save_activations(model, layer_module, dataloader, save_prefix, chunk_size, layer_idx):
    activation_list = []
    chunk_idx = 0
    hook_handle = layer_module.register_forward_hook(make_hook(activation_list))
    
    for i, batch in enumerate(tqdm(dataloader, desc=f"Processing {save_prefix} Model")):
        inputs = {k: v.to(model.device) for k, v in batch.items() if k in tokenizer.model_input_names}
        _ = model(**inputs)

        if (i + 1) % chunk_size == 0:
            torch.save(torch.cat(activation_list), os.path.join(activations_dir, f"{save_prefix}_activations_layer_{layer_idx}_chunk_{chunk_idx}.pt"))
            activation_list.clear()
            chunk_idx += 1
    if activation_list:
        torch.save(torch.cat(activation_list), os.path.join(activations_dir, f"{save_prefix}_activations_layer_{layer_idx}_chunk_{chunk_idx}.pt"))
    hook_handle.remove()

# Layer and execution
layer_idx = 8
chunk_size = 500
base_layer = base_model.model.layers[layer_idx]
finetuned_layer = finetuned_model.model.layers[layer_idx]
process_and_save_activations(base_model, base_layer, dataloader, "base", chunk_size, layer_idx)
process_and_save_activations(finetuned_model, finetuned_layer, dataloader, "finetuned", chunk_size, layer_idx)
