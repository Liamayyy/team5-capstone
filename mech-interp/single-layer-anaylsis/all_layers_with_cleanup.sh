#!/bin/bash
# Create logs directory if it doesn't exist
mkdir -p ./layer-logs

for LAYER in $(seq 16 4 24); do
  LOG="./layer-logs/layer_${LAYER}.log"
  echo "=== Processing Layer $LAYER ===" | tee "$LOG"

  # Generate activations (log and terminal)
  echo "=== Generating Activations for Layer $LAYER ===" | tee -a "$LOG"
  if python generate_activations.py --layer $LAYER 2>&1 | tee -a "$LOG"; then
    echo "[✓] Activations generated" | tee -a "$LOG"
  else
    echo "[✗] Activations failed" | tee -a "$LOG"
    continue
  fi

  # Generate sparse codes (log and terminal)
  echo "=== Generating Sparse Codes for Layer $LAYER ===" | tee -a "$LOG"
  if python generate_sparse_codes.py --layer $LAYER 2>&1 | tee -a "$LOG"; then
    echo "[✓] Sparse codes generated" | tee -a "$LOG"
  else
    echo "[✗] Sparse codes failed" | tee -a "$LOG"
    continue
  fi

  # Delete layer-specific activations
  echo "[~] Deleting activations for layer $LAYER..." | tee -a "$LOG"
  find ./activations/ -type f -name "*layer_${LAYER}_chunk_*.pt" -delete

  # Run comparison
  echo "=== Running Comparison for Layer $LAYER ===" | tee -a "$LOG"
  if python compare_sparse_codes.py --layer $LAYER 2>&1 | tee -a "$LOG"; then
    echo "[✓] Comparison successful" | tee -a "$LOG"
  else
    echo "[✗] Comparison failed" | tee -a "$LOG"
    continue
  fi

  # Delete layer-specific sparse codes
  echo "[~] Deleting sparse codes for layer $LAYER..." | tee -a "$LOG"
  find ./sparse_codes/ -type f -name "*layer_${LAYER}_chunk_*.pt" -delete

  # Optionally, delete SAE cache
  SAE_CACHE="/home/liam23/.cache/huggingface/hub/models--google--gemma-scope-2b-pt-res"
  if [ -d "$SAE_CACHE" ]; then
    echo "[~] Deleting SAE cache..." | tee -a "$LOG"
    rm -rf "$SAE_CACHE"
  fi

  echo "=== Done with Layer $LAYER ===" | tee -a "$LOG"
  echo "" >> "$LOG"
done
