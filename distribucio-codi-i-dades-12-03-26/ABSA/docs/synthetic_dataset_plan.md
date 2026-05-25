# Synthetic Dataset Plan for Final ABSA Fine-Tuning

This document defines the final synthetic-data strategy for the ABSA project. It is intentionally conservative because the current best model is already strong: QLoRA reaches `87.35` micro-F1 and `85.29` macro-F1 on `devel`. The goal is not to create a large artificial dataset, but to add a small set of targeted examples that address the remaining systematic errors.

## Current Best Model

The reference model is the final QLoRA run selected by generative validation:

```text
model: FT.train.absa_v6.qlora4bit.simple.all-linear.r16.lr1e-4.generative_f1.weights
best step: 400
devel M.avg F1: 85.29
devel m.avg F1: 87.35
exact match: 73 / 132
predicted labels: 439
gold labels: 454
correct labels: 390
```

The model is already well calibrated:

```text
average predicted labels/review: 3.33
average gold labels/review:      3.44
empty-gold overprediction:       0
non-empty predicted empty:       0
```

This means the synthetic dataset must not push the model toward broad over-prediction. Precision is already high; the main objective is to improve hard recall and polarity boundaries without damaging the common aspects.

## Development-Set Use Policy

`devel.json` is validation data. It may be used for aggregate error analysis and model selection, but it must not be used as a source of synthetic review text.

Allowed:

- Count which aspects, polarities, and boundaries still fail on `devel`.
- Use those counts to define synthetic generation quotas.
- Compare the final model against the current QLoRA baseline.

Not allowed:

- Put `devel` review text into the API prompt.
- Ask the API to paraphrase `devel` examples.
- Create synthetic examples directly from individual `devel` items.

All concrete seed examples shown to the API must come from `train.json`.

## Why the Plan Is Conservative

Synthetic data can help, but only when quality and distribution are controlled. Self-Instruct is a useful precedent because it generates synthetic examples and then filters invalid or overly similar ones before fine-tuning. For text classification, Li et al. report that LLM-generated synthetic data has inconsistent benefits across tasks, especially when labels are subjective. ABSA is highly subjective: price/value, neutral/conflict, ambience, and general sentiment all depend on annotation conventions.

Recent work on synthetic augmentation for imbalanced learning also warns against naive full balancing: extra synthetic data can degrade performance when generator mismatch dominates the benefit of additional minority samples. Therefore the plan below uses small, targeted augmentation rather than broad class balancing.

## Error Analysis Summary

### Aspect-Level Errors on QLoRA

| Aspect | Train count | Devel gold | Devel pred | F1 | Missing | Extra | Wrong polarity | Priority |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ambience` | 413 | 47 | 44 | 81.3 | 10 | 7 | 0 | high |
| `food_prices` | 186 | 26 | 28 | 74.1 | 5 | 7 | 1 | high |
| `restaurant_prices` | 171 | 24 | 14 | 63.2 | 10 | 0 | 2 | high |
| `food_style_options` | 250 | 24 | 24 | 79.2 | 5 | 5 | 0 | high |
| `food_quality` | 840 | 100 | 99 | 90.5 | 4 | 3 | 6 | medium |
| `restaurant_general` | 1025 | 128 | 129 | 91.8 | 0 | 1 | 10 | medium |
| `service` | 654 | 82 | 83 | 95.8 | 2 | 3 | 1 | low |
| `drinks_quality` | 71 | 10 | 8 | 77.8 | 3 | 1 | 0 | medium |
| `drinks_style_options` | 56 | 6 | 2 | 50.0 | 4 | 0 | 0 | high but rare |
| `location` | 51 | 4 | 5 | 66.7 | 1 | 2 | 0 | medium but very rare |
| `drinks_prices` | 31 | 3 | 3 | 100.0 | 0 | 0 | 0 | no direct priority |

### Main Remaining Failure Modes

1. **Restaurant price vs food price confusion**
   - The model misses `restaurant_prices` and often predicts `food_prices` instead.
   - Examples include texts where the review says the restaurant is expensive overall, "good value", "too expensive for what it offers", or "the price is worth it".
   - Target: teach the boundary between overall value and specific dish/menu prices.

2. **Food quality vs food style/options**
   - The model sometimes treats variety, portions, menu structure, originality, or presentation as `food_quality`.
   - Target: generate reviews where taste is neutral/positive but portions, menu variety, presentation, or originality are evaluated separately.

3. **Ambience vs service/location overreach**
   - `ambience` has both missing and extra errors.
   - The model sometimes adds ambience from tiny/noisy/crowded descriptions that are not annotated, or misses explicit atmosphere/decor comfort.
   - Target: create explicit ambience examples and hard negatives where location/service is mentioned without ambience.

4. **Rare drinks aspects**
   - `drinks_style_options` has very low recall.
   - The model confuses drink variety/options with drink quality.
   - Target: wine list, beer selection, cocktails, coffee quality, drink price, and beverage variety examples.

5. **Neutral and conflict polarity**
   - Train distribution is heavily positive: `positive=2659`, `negative=792`, `neutral=176`, `conflict=121`.
   - Devel is similarly skewed: `positive=326`, `negative=103`, `neutral=15`, `conflict=10`.
   - QLoRA's wrong-polarity errors concentrate on `restaurant_general` and `food_quality`, especially `neutral`/`conflict` collapsing into `positive` or `negative`.
   - Target: controlled neutral/conflict examples, but not enough to distort the overall distribution.

## Final Synthetic Dataset Size

Accepted synthetic examples:

```text
target accepted examples: 220
maximum accepted examples: 250
```

This adds about 21% to the original `train.json` size (`1056` examples). That is large enough to affect hard cases, but small enough to avoid replacing the real distribution.

The generator may create more candidates than accepted, but the final training dataset should contain only the filtered examples.

## Target Distribution

### Language

Keep the synthetic data close to the original language distribution:

```text
Spanish: 65-70%
English: 30-35%
```

The current train split is approximately 67.5% Spanish and 32.5% English.

### Labels per Review

The real train distribution is centered around 3-4 labels per review. Synthetic examples should follow that shape:

| Labels per review | Target share |
| --- | ---: |
| 1 label | 8-12% |
| 2 labels | 18-22% |
| 3 labels | 30-35% |
| 4 labels | 25-30% |
| 5 labels | 6-10% |
| 6+ labels | at most 2% |

Avoid many long synthetic reviews with 6-8 labels. That would train the model to over-predict.

### Polarity

Do not balance polarities uniformly. Enrich rare polarities, but keep the dataset recognizably ABSA-like:

```text
positive: 48-55%
negative: 27-33%
neutral:  9-13%
conflict: 7-11%
```

This intentionally oversamples `neutral` and `conflict` relative to train, but not enough to make them dominate.

## Synthetic Buckets

Use these as primary buckets. Each synthetic example belongs to exactly one primary bucket, although it may contain multiple labels.

| Bucket | Accepted examples | Purpose |
| --- | ---: | --- |
| Price/value boundary | 56 | Improve `restaurant_prices` vs `food_prices`; include positive, negative, neutral, conflict value judgments. |
| Food style/options boundary | 36 | Separate `food_style_options` from `food_quality`; focus on menu variety, portions, originality, presentation. |
| Ambience/location/service boundary | 30 | Improve explicit ambience recall and reduce ambience/location hallucination. |
| Drinks rare aspects | 24 | Improve `drinks_style_options` and `drinks_quality`; include wine list, beer/cocktail variety, coffee quality. |
| Neutral/conflict polarity | 40 | Improve `restaurant_general`, `food_quality`, and price polarity boundaries. |
| Hard negatives / empty-gold | 14 | Preserve calibration: mentions of restaurant context without explicit evaluative aspects. |
| Natural regularizers | 20 | Realistic ordinary reviews following train distribution; prevent the model from becoming too adversarial. |

Total: `220`.

## Generation Strategy

### Unit of Generation

Generate **one review per API call**.

Do not ask for batches like "generate 20 reviews". Batch generation tends to produce repeated phrasing, shallow diversity, and templated labels.

### API Model

Use a strong but cost-effective API model, preferably `gpt-5.4-mini`, because OpenAI positions it as a faster, efficient model for high-volume workloads. Use Structured Outputs so every API response follows a strict JSON schema.

### Prompt Context

Each generation call should include:

1. The target language.
2. The target `gold` label dictionary.
3. A short difficulty description.
4. Two or three style examples from `train.json`.
5. A list of aspects that must not be added unless explicitly evaluated.

The train examples are for style and annotation convention only. The prompt must explicitly forbid copying or paraphrasing them.

### Output Schema

The API response should contain metadata for filtering, but the final dataset should keep only `id`, `language`, `text`, and `gold`.

Generation-time response:

```json
{
  "text": "...",
  "language": "es",
  "gold": {
    "food_prices": "negative",
    "food_quality": "positive",
    "restaurant_general": "conflict"
  },
  "support_spans": {
    "food_prices": ["..."],
    "food_quality": ["..."],
    "restaurant_general": ["..."]
  },
  "notes": "short explanation for internal filtering only"
}
```

Final stored training example:

```json
{
  "id": "synth_v1_0001",
  "language": "es",
  "text": "...",
  "gold": {
    "food_prices": "negative",
    "food_quality": "positive",
    "restaurant_general": "conflict"
  }
}
```

`support_spans` are not an LLM verifier. They are a cheap generation-time consistency aid. They allow deterministic rejection when a claimed aspect has no visible textual evidence.

## Deterministic Filtering

No separate LLM verifier will be used. To compensate, every candidate must pass deterministic checks:

1. Valid JSON and valid schema.
2. `language` is `es` or `en`.
3. `gold` aspects are in the official aspect set.
4. `gold` polarities are in `{positive, negative, neutral, conflict}`.
5. `gold` exactly matches the planned target labels unless the plan explicitly allows optional regularizer labels.
6. Every gold aspect has at least one support span present in the generated text.
7. Text length is realistic:
   - Spanish: roughly 15-140 words.
   - English: roughly 15-140 words.
8. No markdown, bullets, explicit annotation language, or "the aspect is..." phrasing.
9. No copied train example:
   - reject near-exact string overlap;
   - reject extremely high n-gram overlap;
   - reject very high embedding similarity if embeddings are available.
10. No devel contamination:
   - deduplicate against `devel.json`;
   - never send devel text to the API.
11. Dataset-level quota checks:
   - label count distribution stays within the target range;
   - language distribution stays within the target range;
   - no aspect/polarity pair exceeds its quota.

Manual review is still recommended, but lightweight:

```text
review 25 random accepted examples
review 15 high-risk examples: neutral/conflict, restaurant_prices, drinks_style_options
```

If more than 10% of reviewed examples have questionable labels, regenerate the affected bucket before training.

## Seed Selection

Use `train.json` as the only source of examples shown to the API.

Preferred seed procedure:

1. Run the current best QLoRA model on `train.json`.
2. Compute a difficulty score for each train example:

```text
difficulty =
  2.0 * missing_gold_labels
  + 1.5 * wrong_polarity_labels
  + 1.0 * extra_predicted_labels
  + rare_pair_bonus
  + target_bucket_bonus
```

3. For each generation call, retrieve 2-3 style examples from train:
   - same language;
   - similar target bucket;
   - at least one high-difficulty seed when possible;
   - avoid using the same seed too often.

If train inference is skipped for time reasons, use train examples selected by rare aspect/polarity pairs and label-pattern similarity instead.

## Bucket Details

### 1. Price/Value Boundary

Accepted examples: `56`.

Primary targets:

- `restaurant_prices`
- `food_prices`
- `restaurant_general` with value-driven sentiment

Subcases:

- overall restaurant expensive but food quality good;
- specific dish/menu overpriced;
- good value overall despite expensive individual dishes;
- cheap menu but poor food;
- fair price but disappointing service;
- "worth it" versus "too expensive for what it offers".

Important distinction:

```text
restaurant_prices = opinion about the restaurant's prices/value overall
food_prices       = opinion about food/menu/dish prices specifically
```

### 2. Food Style/Options Boundary

Accepted examples: `36`.

Primary targets:

- `food_style_options`
- `food_quality`

Subcases:

- good taste but tiny portions;
- bland food but wide menu;
- original presentation but average taste;
- limited vegetarian options;
- tasting menu variety;
- portion size praised/criticized separately from taste.

Important distinction:

```text
food_quality       = taste, freshness, cooking, ingredients
food_style_options = variety, portions, presentation, originality, menu options
```

### 3. Ambience/Location/Service Boundary

Accepted examples: `30`.

Primary targets:

- `ambience`
- `location`
- `service`

Subcases:

- explicit atmosphere/decor/noise/comfort;
- central location without ambience;
- friendly staff without ambience;
- crowded/tiny room with or without explicit negative ambience;
- terrace/views/location versus interior atmosphere.

The goal is not simply to increase ambience recall. It is to improve boundary precision.

### 4. Drinks Rare Aspects

Accepted examples: `24`.

Primary targets:

- `drinks_style_options`
- `drinks_quality`
- a few `drinks_prices`

Subcases:

- excellent wine list but average food;
- limited beer/cocktail options;
- expensive wine bottle;
- good coffee but poor dessert;
- beverage variety versus beverage quality.

Do not create many `drinks_prices` examples because the current model already gets the three devel cases right and the train count is very small.

### 5. Neutral/Conflict Polarity

Accepted examples: `40`.

Primary targets:

- `restaurant_general`
- `food_quality`
- `restaurant_prices`
- `food_prices`
- `service`

Subcases:

- good food but bad price -> `restaurant_general: conflict`;
- average food, nothing special -> `food_quality: neutral`;
- acceptable service with no strong opinion -> `service: neutral`;
- mixed overall experience -> `restaurant_general: conflict`;
- "correct", "normal", "sin más", "acceptable", "decent" as neutral cues.

Neutral must mean explicitly evaluated but neither good nor bad. It must not mean absent.

### 6. Hard Negatives / Empty-Gold

Accepted examples: `14`.

Purpose:

- preserve calibration;
- teach that mentions are not automatically opinions;
- prevent synthetic augmentation from increasing over-prediction.

Subcases:

- factual visit description without explicit opinion;
- location or menu mentioned without evaluation;
- "I went with friends and ordered pasta" without sentiment;
- short ambiguous statements where gold should be `{}` or one minimal label.

Use this bucket sparingly.

### 7. Natural Regularizers

Accepted examples: `20`.

Purpose:

- keep the synthetic set close to the real distribution;
- avoid training only on adversarial edge cases.

These should look like ordinary train examples:

- 3-4 labels;
- mostly positive/negative;
- common aspects: `restaurant_general`, `food_quality`, `service`, `ambience`;
- realistic Spanish/English review style.

## Final Training Dataset

Create:

```text
ABSA/dataset/synthetic_v1.json
ABSA/dataset/train_plus_synthetic_v1.json
```

`synthetic_v1.json` contains only accepted synthetic examples.

`train_plus_synthetic_v1.json` is:

```text
train.json + synthetic_v1.json
```

Do not oversample synthetic examples. Do not duplicate them. Just concatenate and shuffle during training.

## Final Training Run

Only one final training run is planned. Use QLoRA, because it is currently the best method:

```text
base: Qwen/Qwen3.5-2B
method: QLoRA 4-bit
prompt: absa_v6
data: train_plus_synthetic_v1
lr: 1e-4
epochs: 5
max_length: 3072
per_device_train_batch: 2
gradient_accumulation_steps: 4
eval_steps: 50
checkpoint selection: devel generative m.avg F1
max_new_tokens: 512
few-shot in training: no
```

The only intended experimental variable is the dataset. Hyperparameters should stay fixed so the result is interpretable.

## Success Criteria

Baseline to beat:

```text
QLoRA current: M.avg F1 = 85.29, m.avg F1 = 87.35
```

The synthetic-data run is considered successful if:

1. `m.avg F1 > 87.35`, and
2. `M.avg F1 >= 85.29 - 0.30`, and
3. exact match does not decrease by more than 2 examples, and
4. precision does not collapse through over-prediction.

Secondary desired improvements:

- `restaurant_prices` recall improves.
- `food_style_options` remains at least as good and preferably improves.
- `drinks_style_options` recall improves without adding many false positives.
- `neutral` and `conflict` polarity errors decrease.

If the synthetic run is worse, keep the current QLoRA model as the final model and report synthetic augmentation as an attempted but harmful ablation. That is still scientifically valid and consistent with the literature.

## Report Wording

The report should frame this as targeted data augmentation:

> We used the development set only to identify aggregate error categories of the best fine-tuned model. Synthetic examples were generated from train-derived label plans and train-only style examples, not from development examples. The synthetic set was deliberately small and focused on high-error ABSA boundaries, because prior work shows that LLM-generated synthetic data can be inconsistent for subjective classification tasks and that naive augmentation may introduce distribution shift.

## References

- Wang et al. (2023), Self-Instruct. The method motivates generating examples with subsequent filtering of invalid or similar outputs: https://arxiv.org/abs/2212.10560
- Li et al. (2023), "Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations." The paper reports that synthetic data effectiveness is inconsistent across classification tasks and worsens with subjective instances: https://aclanthology.org/2023.emnlp-main.647/
- Zhang and Pavlick (2025), "Does Training on Synthetic Data Make Models Less Robust?" Useful cautionary framing around blindspots and synthetic-data robustness: https://arxiv.org/abs/2502.07164
- Ma and Zhang (2026), "Synthetic Augmentation in Imbalanced Learning: When It Helps, When It Hurts, and How Much to Add." The paper argues against naive full balancing and highlights generator mismatch risk: https://arxiv.org/abs/2601.16120
- OpenAI model documentation for `gpt-5.4-mini`, used as the planned high-volume API generator: https://developers.openai.com/api/docs/models/gpt-5.4-mini/
- OpenAI Structured Outputs documentation, used to enforce the generation schema without a separate LLM verifier: https://platform.openai.com/docs/guides/structured-outputs
