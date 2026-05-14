# Fine-Tuned vs Fewshot Baselines Comparison

| Model | Type | Precision | Recall | F1 | F1% |
|-------|------|-----------|--------|----|----||
| absa_mmr (k=5) | Zero-shot + In-context | 7911.6% | 6854.0% | 72.005 | 7200.5% |
| dense_topk (k=5) | Zero-shot + In-context | 8036.6% | 6836.2% | 72.394 | 7239.4% |
| hard_mix (k=5) | Zero-shot + In-context | 7450.8% | 7072.4% | 70.980 | 7098.0% |
| manual_fixed_hard (k=5) | Zero-shot + In-context | 7319.4% | 6582.4% | 67.509 | 6750.9% |
| random_dynamic (k=5) | Zero-shot + In-context | 7834.6% | 6494.7% | 69.537 | 6953.7% |
| random_fixed (k=5) | Zero-shot + In-context | 7468.4% | 6566.6% | 68.406 | 6840.6% |
| LoRA Fine-tuned (FT.train.fewshot) | Fine-tuned on train.json | 75.1% | 61.2% | 0.675 | 67.5% |

## Analysis

**Fine-tuned model performance: 67.5% F1**

- ❌ **NOT better than fewshot baselines**
- ⚠️ Equal to worst baseline (manual_fixed_hard: 67.5%)
- ❌ Below best baseline (dense_topk: 72.4%)
- ❌ Gap: -4.9% F1 from best fewshot method

## Why is FT worse than fewshot?

Possible reasons:
1. **Not enough training data** - Only 528 examples in train.json
2. **Poor hyperparameters** - Learning rate 1e-5, 3 epochs might be too conservative
3. **Wrong prompt** - Using v6, but v10/v11/v12 might be better
4. **Overfitting** - Model may have overfit to train set, generalizing poorly
5. **Recall collapse** - 61.2% recall vs 68.4% in fewshot (detected fewer aspects)
