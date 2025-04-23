# Create logs directory if it doesn't exist
mkdir -p ./layer-logs

for LAYER in $(seq 8 4 24); do
  LOG="./layer-logs/layer_${LAYER}.log"
  echo "=== Processing Layer $LAYER ===" | tee -a $LOG

  if python generate_activations.py --layer $LAYER >> $LOG 2>&1; then
    echo "[✓] Activations generated" | tee -a $LOG
  else
    echo "[✗] Activations failed" | tee -a $LOG
    continue
  fi

  if python generate_sparse_codes.py --layer $LAYER >> $LOG 2>&1; then
    echo "[✓] Sparse codes generated" | tee -a $LOG
  else
    echo "[✗] Sparse codes failed" | tee -a $LOG
    continue
  fi

  if python compare_sparse_codes.py --layer $LAYER >> $LOG 2>&1; then
    echo "[✓] Comparison successful" | tee -a $LOG
  else
    echo "[✗] Comparison failed" | tee -a $LOG
    continue
  fi

  # Delete all activations for this layer
  echo "[~] Deleting activations for layer $LAYER..." | tee -a $LOG
  find activations/ -type f -name "*layer_${LAYER}_chunk_*.pt" -delete

  # Delete all sparse codes for this layer
  echo "[~] Deleting sparse codes for layer $LAYER..." | tee -a $LOG
  find sparse_codes/ -type f -name "*layer_${LAYER}_chunk_*.pt" -delete

  echo "=== Done with Layer $LAYER ===" | tee -a $LOG
  echo "" >> $LOG
done
