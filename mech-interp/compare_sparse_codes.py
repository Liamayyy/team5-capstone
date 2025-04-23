# compare_sparse_codes.py (streamed version)
import os
import glob
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

def streamed_cosine_similarity(base_prefix, finetuned_prefix):
    base_parts = sorted(glob.glob(f"sparse_codes/{base_prefix}_sparse_codes_layer_8_chunk_0_part_*.pt"))
    finetuned_parts = sorted(glob.glob(f"sparse_codes/{finetuned_prefix}_sparse_codes_layer_8_chunk_0_part_*.pt"))

    if len(base_parts) != len(finetuned_parts):
        raise ValueError("Mismatch in number of part files between base and finetuned.")

    all_sims = []
    for b_path, f_path in tqdm(list(zip(base_parts, finetuned_parts)), desc="Comparing parts"):
        base = torch.load(b_path).to_dense()
        fine = torch.load(f_path).to_dense()

        min_len = min(base.shape[0], fine.shape[0])
        base, fine = base[:min_len], fine[:min_len]

        sims = cosine_similarity(base, fine, dim=1)
        all_sims.append(sims.cpu())

        del base, fine, sims
        torch.cuda.empty_cache()

    all_sims_tensor = torch.cat(all_sims)
    print(f"Average Cosine Similarity: {all_sims_tensor.mean().item():.4f}")
    print(f"Std Dev of Cosine Similarity: {all_sims_tensor.std().item():.4f}")

if __name__ == "__main__":
    print("Comparing sparse codes using streamed cosine similarity...")
    streamed_cosine_similarity("base", "finetuned")
