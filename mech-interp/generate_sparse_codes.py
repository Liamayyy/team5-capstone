# generate_sparse_codes.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
from sae_lens import SAE
from tqdm import tqdm


def encode_in_batches_to_file(sae, activations, out_path_prefix, batch_size=512):
    with torch.no_grad():
        num_batches = (activations.shape[0] + batch_size - 1) // batch_size
        for i in tqdm(range(0, activations.shape[0], batch_size), desc=f"Encoding {out_path_prefix}"):
            batch = activations[i:i+batch_size]
            codes = sae.encode(batch).to_sparse_coo().cpu()

            out_path = f"{out_path_prefix}_part_{i//batch_size}.pt"
            torch.save(codes, out_path)

            del batch, codes
            torch.cuda.empty_cache()


# Directories
activations_dir = "activations"
sparse_codes_dir = "sparse_codes"
os.makedirs(sparse_codes_dir, exist_ok=True)

# SAE config
layer_idx = 8
sae_id = f"layer_{layer_idx}/width_65k/canonical"
sae_release = "gemma-scope-2b-pt-res-canonical"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Sparse Autoencoder
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=sae_release,
    sae_id=sae_id,
    device=device
)

# Process files
for prefix in ["base", "finetuned"]:
    for filename in os.listdir(activations_dir):
        if filename.startswith(prefix) and f"layer_{layer_idx}" in filename:
            full_path = os.path.join(activations_dir, filename)
            size_gb = os.path.getsize(full_path) / 1e9
            print(f"\nProcessing {filename} ({size_gb:.2f} GB)")

            if size_gb > 1.5:
                print("Skipping — file too large for memory-safe processing")
                continue

            try:
                activations = torch.load(full_path, map_location=device)

                out_prefix = os.path.join(
                    sparse_codes_dir,
                    filename.replace("activations", "sparse_codes").replace(".pt", "")
                )
                encode_in_batches_to_file(sae, activations, out_prefix, batch_size=16)

                print(f"Saved sparse codes to prefix: {out_prefix}_part_*.pt")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

            finally:
                del activations
                torch.cuda.empty_cache()
