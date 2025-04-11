import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SparseAutoencoder, load_pretrained_sae

# Load your fine-tuned Gemma-2-2B model
model_name = "your-username/your-gemma-2-2b-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load a pre-trained SAE (ensure compatibility with your model's hidden size)
sae = load_pretrained_sae("gemma-2b")  # Replace with the appropriate SAE identifier

# Tokenize input text
input_text = "The implications of quantum computing are"
inputs = tokenizer(input_text, return_tensors="pt")

# Get model outputs
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

# Apply SAE to hidden states
sparse_representation = sae.encode(hidden_states)

# Inspect the sparse representation
print("Sparse representation shape:", sparse_representation.shape)
