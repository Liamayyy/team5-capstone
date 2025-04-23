import os
import glob
import argparse
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import numpy as np
import psutil

def print_ram_usage():
    ram = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"[Memory] Current RAM usage: {ram:.2f} GB")

def print_metric(name, values, higher_better=True):
    cat = torch.cat(values).float()
    avg, std = cat.mean().item(), cat.std().item()
    better = "higher" if higher_better else "lower"
    print(f"{name}: {avg:.4f} ± {std:.4f} ({better} is better)")

def streamed_similarity_metrics(base_prefix, finetuned_prefix, layer_idx):
    base_parts = sorted(glob.glob(f"sparse_codes/{base_prefix}_sparse_codes_layer_{layer_idx}_chunk_0_part_*.pt"))
    finetuned_parts = sorted(glob.glob(f"sparse_codes/{finetuned_prefix}_sparse_codes_layer_{layer_idx}_chunk_0_part_*.pt"))

    if len(base_parts) != len(finetuned_parts):
        raise ValueError("Mismatch in number of part files between base and finetuned.")

    cos_sims, l1_dists, jacc_sims, hamm_dists = [], [], [], []
    activation_counts_base, activation_counts_fine = [], []
    base_neuron_usage, fine_neuron_usage = set(), set()

    # Streaming stats
    dist_stats_base = {"sum": 0, "sumsq": 0, "nonzero": 0, "count": 0, "entropy_sum": 0, "entropy_count": 0}
    dist_stats_fine = {"sum": 0, "sumsq": 0, "nonzero": 0, "count": 0, "entropy_sum": 0, "entropy_count": 0}

    for b_path, f_path in tqdm(list(zip(base_parts, finetuned_parts)), desc="Comparing parts"):
        base = torch.load(b_path).to_dense()
        fine = torch.load(f_path).to_dense()

        min_len = min(base.shape[0], fine.shape[0])
        base = base[:min_len]
        fine = fine[:min_len]

        # Cosine
        cos_sims.append(cosine_similarity(base, fine, dim=1).cpu())

        # L1
        l1 = torch.sum(torch.abs(base - fine), dim=1)
        l1_dists.append(l1.cpu())

        # Binarize
        base_bin = (base != 0).int()
        fine_bin = (fine != 0).int()

        # Jaccard
        inter = torch.sum((base_bin & fine_bin), dim=1).float()
        union = torch.sum((base_bin | fine_bin), dim=1).float() + 1e-8
        jacc_sims.append((inter / union).cpu())

        # Hamming
        hamm_dists.append(torch.sum(base_bin != fine_bin, dim=1).float().cpu())

        # Activation counts
        activation_counts_base.append(torch.sum(base_bin, dim=1).cpu())
        activation_counts_fine.append(torch.sum(fine_bin, dim=1).cpu())

        # Neuron usage sets
        for row in base_bin:
            base_neuron_usage.update(torch.nonzero(row).flatten().tolist())
        for row in fine_bin:
            fine_neuron_usage.update(torch.nonzero(row).flatten().tolist())

        # Streaming stats
        for tensor, stats in [(base, dist_stats_base), (fine, dist_stats_fine)]:
            stats["sum"] += tensor.sum().item()
            stats["sumsq"] += (tensor ** 2).sum().item()
            stats["nonzero"] += (tensor != 0).sum().item()
            stats["count"] += tensor.numel()

            # Streaming entropy (row-wise average)
            probs = torch.abs(tensor) + 1e-8
            probs = probs / probs.sum(dim=1, keepdim=True)
            row_entropies = -torch.sum(probs * probs.log2(), dim=1)
            stats["entropy_sum"] += row_entropies.sum().item()
            stats["entropy_count"] += row_entropies.numel()

        print_ram_usage()

        del base, fine, base_bin, fine_bin
        torch.cuda.empty_cache()

    print("\n=== Similarity / Distance Metrics ===")
    print_metric("Cosine Similarity", cos_sims)
    print_metric("L1 Distance", l1_dists, higher_better=False)
    print_metric("Jaccard Similarity", jacc_sims)
    print_metric("Hamming Distance", hamm_dists, higher_better=False)

    print("\n=== Activation Pattern Analysis ===")
    print_metric("Base: Active Neurons per Example", activation_counts_base, higher_better=False)
    print_metric("Finetuned: Active Neurons per Example", activation_counts_fine, higher_better=False)

    print(f"\nUnique Neurons in Base: {len(base_neuron_usage)}")
    print(f"Unique Neurons in Finetuned: {len(fine_neuron_usage)}")
    shared = base_neuron_usage & fine_neuron_usage
    print(f"Shared Active Neurons: {len(shared)}")
    print(f"Jaccard Index of Neuron Usage: {len(shared) / len(base_neuron_usage | fine_neuron_usage):.4f}")

    print("\n=== Distributional Stats (Streaming) ===")

    def summarize(label, stats):
        total = stats["count"]
        mean = stats["sum"] / total
        std = (stats["sumsq"] / total - mean**2) ** 0.5
        sparsity = stats["nonzero"] / total
        print(f"\n{label}:")
        print(f"  Mean: {mean:.4f}")
        print(f"  Std: {std:.4f}")
        print(f"  Sparsity: {sparsity:.4f}")
        if stats["entropy_count"] > 0:
            entropy_mean = stats["entropy_sum"] / stats["entropy_count"]
            print(f"  Entropy (mean per vector): {entropy_mean:.4f}")

    summarize("Base Codes", dist_stats_base)
    summarize("Finetuned Codes", dist_stats_fine)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, help="Layer index for sparse code comparison.")
    args = parser.parse_args()

    print("Comparing sparse codes using streaming statistics and similarity metrics...")
    streamed_similarity_metrics("base", "finetuned", args.layer)
