import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from sae_lens import SAE
from tqdm import tqdm

#############################################
# Create directories for saving results
#############################################
activations_dir = "activations"
sparse_codes_dir = "sparse_codes"
os.makedirs(activations_dir, exist_ok=True)
os.makedirs(sparse_codes_dir, exist_ok=True)

#############################################
# 1. Model and Tokenizer Initialization
#############################################
# Base model name (original Gemma-2-2B)
base_model_name = "google/gemma-2-2b"

# For the fine-tuned model, set the appropriate identifier or path
finetuned_model_path = "Liamayyy/gemma-2-2b-medical-v2"  # update with your fine-tuned checkpoint

# Load the tokenizer (assumed to be shared between models)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load both the base and fine-tuned models.
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
finetuned_model = AutoModelForCausalLM.from_pretrained(
    finetuned_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Set both models in evaluation mode
base_model.eval()
finetuned_model.eval()

#############################################
# 2. Data Loading & Tokenization (Using 1/10 of Dataset)
#############################################
dataset = load_dataset("wikitext", "wikitext-103-v1")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Tokenize the dataset in batched mode.
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
train_dataset = tokenized_dataset["train"]

# Use only one-tenth of the dataset to ensure everything is working.
sample_size = len(train_dataset) // 1000
train_dataset = train_dataset.select(range(sample_size))

# Create a DataLoader for batching
batch_size = 8
dataloader = DataLoader(train_dataset, batch_size=batch_size)

#############################################
# 3. Helper Function: Process & Save Activations
#############################################
def make_hook(activation_list):
    """A hook function that appends module output to activation_list."""
    def hook_fn(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        activation_list.append(act.detach().cpu())
    return hook_fn

def process_and_save_activations(model, layer_module, dataloader, save_prefix, chunk_size, layer_idx):
    """
    Process the dataloader with the given model, collect activations from the specified layer,
    and save them in chunks (to reduce memory usage).
    
    Returns a list of filenames where activations were saved.
    """
    activation_list = []
    chunk_idx = 0
    file_list = []
    hook_handle = layer_module.register_forward_hook(make_hook(activation_list))
    
    for i, batch in enumerate(tqdm(dataloader, desc=f"Processing {save_prefix} Model")):
        inputs = {k: v.to(model.device) for k, v in batch.items() if k in tokenizer.model_input_names}
        _ = model(**inputs)
        
        # Save the activations every chunk_size batches.
        if (i + 1) % chunk_size == 0:
            chunk_tensor = torch.cat(activation_list, dim=0)
            filename = os.path.join(activations_dir, f"{save_prefix}_activations_layer_{layer_idx}_chunk_{chunk_idx}.pt")
            torch.save(chunk_tensor, filename)
            file_list.append(filename)
            print(f"Saved chunk {chunk_idx} to {filename}")
            chunk_idx += 1
            activation_list.clear()  # clear for next chunk
    
    # Save any remaining activations from the final partial chunk.
    if activation_list:
        chunk_tensor = torch.cat(activation_list, dim=0)
        filename = os.path.join(activations_dir, f"{save_prefix}_activations_layer_{layer_idx}_chunk_{chunk_idx}.pt")
        torch.save(chunk_tensor, filename)
        file_list.append(filename)
        print(f"Saved final chunk {chunk_idx} to {filename}")
    
    hook_handle.remove()
    return file_list

# Choose the layer whose activations you want to capture
layer_idx = 8  # Adjust if needed
chunk_size = 500  # Adjust based on available memory (flush every 100 batches)

#############################################
# 4. Process Data & Periodically Save Activations for Both Models
#############################################
# For the base model:
base_layer_module = base_model.model.layers[layer_idx]
base_file_list = process_and_save_activations(base_model, base_layer_module, dataloader, "base", chunk_size, layer_idx)

# For the fine-tuned model:
finetuned_layer_module = finetuned_model.model.layers[layer_idx]
finetuned_file_list = process_and_save_activations(finetuned_model, finetuned_layer_module, dataloader, "finetuned", chunk_size, layer_idx)

#############################################
# 5. Apply SAE to Saved Activation Chunks and Compare Sparse Codes
#############################################
# Specify the SAE identifier related to the chosen layer.
sae_id = f"layer_{layer_idx}/width_65k/canonical"
sae_release = "gemma-scope-2b-pt-att-canonical"  # Update the release identifier as necessary.

device = "cuda" if torch.cuda.is_available() else "cpu"
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=sae_release,
    sae_id=sae_id,
    device=device
)

# Variables to accumulate metrics for overall comparison.
total_l1_sum = 0.0
total_elements = 0
total_cos_sum = 0.0
total_vectors = 0

# Process and compare chunk by chunk (assumes corresponding chunk files for both models)
for base_file, finetuned_file in zip(base_file_list, finetuned_file_list):
    # Load the activations from disk.
    base_chunk = torch.load(base_file)
    finetuned_chunk = torch.load(finetuned_file)
    
    # Apply the SAE encoder on each chunk.
    base_sparse = sae.encode(base_chunk)
    finetuned_sparse = sae.encode(finetuned_chunk)
    
    # Save the sparse codes for each chunk in the sparse_codes folder.
    base_sc_filename = os.path.join(sparse_codes_dir, os.path.basename(base_file).replace("activations", "sparse_codes"))
    finetuned_sc_filename = os.path.join(sparse_codes_dir, os.path.basename(finetuned_file).replace("activations", "sparse_codes"))
    torch.save(base_sparse, base_sc_filename)
    torch.save(finetuned_sparse, finetuned_sc_filename)
    print(f"Saved sparse codes for chunk from {base_file} and {finetuned_file} to {base_sc_filename} and {finetuned_sc_filename}.")
    
    # Compute metrics for this chunk.
    l1_diff = torch.abs(base_sparse - finetuned_sparse)
    chunk_l1_mean = l1_diff.mean().item()
    print(f"Chunk L1 difference mean: {chunk_l1_mean}")
    
    # Compute cosine similarity for each example.
    base_norm = torch.nn.functional.normalize(base_sparse, dim=-1)
    finetuned_norm = torch.nn.functional.normalize(finetuned_sparse, dim=-1)
    cos_sim = (base_norm * finetuned_norm).sum(dim=-1)
    chunk_cos_mean = cos_sim.mean().item()
    print(f"Chunk cosine similarity mean: {chunk_cos_mean}")
    
    # Accumulate sums for global metrics.
    total_l1_sum += l1_diff.sum().item()
    total_elements += base_sparse.numel()
    total_cos_sum += cos_sim.sum().item()
    total_vectors += base_sparse.shape[0]

# Compute and display the global metrics.
global_l1 = total_l1_sum / total_elements
global_cos = total_cos_sum / total_vectors
print("Global average L1 difference between base and fine-tuned sparse codes:", global_l1)
print("Global average cosine similarity between sparse codes:", global_cos)
