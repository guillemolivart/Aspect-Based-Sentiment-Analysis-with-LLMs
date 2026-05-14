# Fine-Tuning Training Metrics

**Model:** LoRA fine-tuned on train.json  
**Weights:** FT.train.fewshot.weights  
**Checkpoint:** checkpoint-396 (epoch 3)

## Loss Curves

| Epoch | Train Loss | Val Loss | Gap (Overfitting) | Grad Norm |
|-------|------------|----------|-------------------|-----------|
| 1     | 0.6962     | 0.6284   | 0.0678 ✓          | 0.2550    |
| 2     | 0.2823     | 0.5643   | 0.2819 ❌         | 0.1793    |
| 3     | 0.2697     | 0.5562   | 0.2865 ❌         | 0.1287    |

## Training Quality

### ✅ What went well:
- **Strong convergence**: Training loss dropped 66% (0.696 → 0.270)
- **Validation loss improved**: Epoch 1 → 3: 0.628 → 0.556 (-11.4%)
- **Gradient norms decreasing**: Shows stable training
- **Completed full 3 epochs** with consistent learning

### ⚠️ Issues detected:
- **Overfitting from Epoch 2+**: Gap between train/val loss grows
  - Epoch 2: train=0.282, val=0.564 (gap=0.282)
  - Epoch 3: train=0.270, val=0.556 (gap=0.286)
- **Model memorizing train data** instead of generalizing
- **This explains low ABSA F1 (67.5%)** despite good token-level loss

## Why is ABSA F1 low despite good training loss?

The `eval_loss` is **token-level prediction loss** (next-token accuracy), not ABSA-specific metrics.
- Low token loss ≠ High aspect extraction F1
- Model may predict tokens correctly but extract wrong aspects/polarities
- ABSA requires structured output (aspect → polarity pairs), not just token prediction

## Recommendations

1. **Reduce overfitting:**
   - Add dropout during LoRA training
   - Use earlier stopping (save at epoch 1 instead of 3)
   - Try lower learning rate or fewer epochs

2. **Improve ABSA performance:**
   - Use better prompts (v10, v11, v12 instead of v6)
   - Increase training data (only 528 examples now)
   - Adjust LoRA rank (r=8 may be too conservative)
   - Fine-tune with ABSA loss, not just token prediction

3. **Current baseline to beat:**
   - Best fewshot: 72.4% F1 (dense_topk)
   - Current FT: 67.5% F1
   - **Target: >73% F1** to improve over baselines
