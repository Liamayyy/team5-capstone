for LAYER in $(seq 0 4 24); do
  LOG="layer_${LAYER}.log"
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

  echo "=== Done with Layer $LAYER ===" | tee -a $LOG
  echo "" >> $LOG
done
