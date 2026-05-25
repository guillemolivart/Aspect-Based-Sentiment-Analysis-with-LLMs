# Synthetic V1 Generation Report

This run generated the final targeted synthetic ABSA augmentation set.

## Execution

- Machine: Vast.ai RTX 3090.
- Base model for train difficulty scoring: `Qwen/Qwen3.5-2B`.
- Adapter used for train scoring: `FT.train.absa_v6.qlora4bit.simple.all-linear.r16.lr1e-4.generative_f1.weights`.
- Prompt for train scoring: `prompts/absa_v6.json`.
- Train scoring mode: `model_train_errors`.
- API generation model: `gpt-5.4-mini`.
- API candidates: 220.
- Accepted synthetic examples: 220.
- Rejected candidates: 0.

## API Call Design

The API generator was called once per synthetic candidate. Since all 220 candidates
were accepted by the validator, this produced exactly 220 API calls for the final
dataset.

Each call gave the large model a controlled generation plan, not a free-form
request. The plan included:

- The target language: English or Spanish.
- The target bucket, such as price boundary, food style/options boundary,
  rare drinks aspects, neutral/conflict polarity, hard negatives, or regularizers.
- The exact target `gold` dictionary to generate.
- A natural-language difficulty instruction explaining the desired edge case.
- A list of aspects to avoid.
- The full list of non-target aspects, with an explicit instruction not to
  express sentiment about them.
- Official aspect definitions and polarity rules.
- Three style examples from `train.json`.

The large model was therefore asked to generate a realistic review whose gold
labels matched our target exactly. It also had to return one support span for
each generated label. The local validator then checked that the returned gold
matched the target exactly and that support spans were present in the generated
text.

## Style Example Selection

The three train examples included in each API call were not uniformly random.
They were selected as style/context examples using a weighted retrieval scheme:

- Same language as the target synthetic example whenever possible.
- Higher score if the train example shared target aspects or exact
  aspect-polarity pairs with the requested synthetic gold.
- Higher score if the example was difficult for the current QLoRA model on
  train, measured with missing, wrong-polarity, and extra-label errors.
- Higher score for rare or priority aspects.
- Lower score when the same train example had already been reused often.

This means the API context was similar and useful, but not a deterministic copy
of the same few hardest examples. The goal was to expose the generator to the
style and ambiguity of hard training cases while avoiding repeated synthetic
templates.

## Output Files

- `dataset/synthetic_v1.json`: final synthetic-only dataset.
- `dataset/train_plus_synthetic_v1.json`: `train.json` followed by `synthetic_v1.json`.
- `outputs/finetune/best_train_predictions_qlora.json`: QLoRA predictions on train used for difficulty scoring.
- `outputs/synthetic/synthetic_v1.accepted_with_metadata.json`: accepted examples with bucket, plan, support spans, seed ids, usage, and similarity metadata.
- `outputs/synthetic/synthetic_v1.report.json`: full run summary.
- `outputs/synthetic/synthetic_v1.plans.json`: generation plans sent to the API.
- `outputs/synthetic/synthetic_v1.rejected.jsonl`: empty in this run.
- `outputs/synthetic/synthetic_v1.run.log`: execution log.

## Dataset Size

- Original train examples: 1056.
- Synthetic examples: 220.
- Combined train+synthetic examples: 1276.

The combined file was verified to be exactly `train + synthetic` in that order.

## Bucket Distribution

| Bucket | Count |
|---|---:|
| price_value_boundary | 56 |
| food_style_options_boundary | 36 |
| ambience_location_service_boundary | 30 |
| drinks_rare_aspects | 24 |
| neutral_conflict_polarity | 40 |
| hard_negatives_empty | 14 |
| natural_regularizers | 20 |

## Language And Label Shape

- Spanish examples: 148.
- English examples: 72.

| Number of labels | Count |
|---:|---:|
| 0 | 11 |
| 1 | 3 |
| 2 | 36 |
| 3 | 143 |
| 4 | 27 |

## Train Difficulty Signal

The QLoRA train pass produced predictions for all 1056 train examples.

- Exact train matches: 788 / 1056.
- Exact train match accuracy: 74.62%.

Top train missing labels:

- `food_prices: positive`: 17.
- `food_quality: positive`: 17.
- `food_style_options: negative`: 14.
- `restaurant_prices: positive`: 13.
- `food_prices: negative`: 12.
- `food_style_options: positive`: 12.

Top train polarity confusions:

- `restaurant_general: conflict -> positive`: 13.
- `restaurant_general: neutral -> positive`: 12.
- `restaurant_general: conflict -> negative`: 11.
- `ambience: conflict -> negative`: 9.
- `restaurant_general: conflict -> neutral`: 8.
- `food_quality: conflict -> negative`: 7.

This confirms that synthetic generation was driven by actual QLoRA train errors, not by fallback rare-label heuristics.

## Quality Checks

- Maximum 5-gram Jaccard similarity to train/devel references: 0.0556.
- Maximum 5-gram Jaccard similarity to another synthetic example: 0.2447.
- Word count range: 26 to 85.
- All accepted examples passed exact target-gold validation and support-span checks.
- Manual bucket sampling showed coherent examples for price boundaries, food style/options, ambience/service/location, rare drinks aspects, neutral/conflict polarity, hard negatives, and regularizers.

## Interpretation

The generation looks usable for the final augmentation experiment. The strongest point is that it targets the current QLoRA model's real train failures while keeping devel usage limited to aggregate diagnosis and similarity filtering. The main residual risk is synthetic label noise from subtle non-target aspect sentiment, but the final prompt explicitly controls non-target aspects and the sampled examples looked clean enough to proceed.

The recommended next experiment is to fine-tune on `dataset/train_plus_synthetic_v1.json` and compare against the current QLoRA run trained only on `train.json`.
