# Few-Shot Experiment Plan

This document defines the few-shot section before implementation. The goal is to
improve the current zero-shot baseline using high-quality in-context examples while
keeping the experiment count controlled and scientifically defensible.

## Current Baseline

Use the best current no-thinking setup as the fixed generation configuration for
few-shot experiments:

```text
model: Qwen/Qwen3.5-2B
prompt: absa_v6 adapted to few-shot
thinking: disabled
temperature: 0.68
top_p: 0.72
top_k: 20
min_p: 0.0
presence_penalty: 1.8
repetition_penalty: 1.0
max_new_tokens: 512
```

The Optuna run found a small improvement over the manual sweep, but the gain is
within sampling variance. For few-shot, keep these decoding parameters fixed so
the only variable is demonstration selection.

## Embedding Model

Use:

```text
Qwen/Qwen3-Embedding-0.6B
```

Rationale:

- Strong multilingual retrieval model.
- Small enough to run cheaply compared with the generator.
- Better aligned with the Qwen family used for generation.
- Suitable for English and Spanish review retrieval.

Store it under:

```text
ABSA/embedding_model/qwen3_embedding_0_6b
```

Only use `train.json` examples as demonstrations. Never use `devel.json` examples
inside prompts.

## Error Diagnosis To Target

The best full-output baseline is:

```text
outputs/hyperparam_sweep/v6_F_refined_9_temp065_topp75.run1.json
```

It shows high precision but limited recall:

```text
predicted labels: 383
gold labels: 454
correct labels: 286
precision_micro: 74.67
recall_micro: 62.99
f1_micro: 68.34
```

Most important false negatives:

```text
restaurant_general positive: 19
ambience positive: 15
food_quality positive: 12
restaurant_prices negative: 12
food_style_options positive: 12
food_prices negative: 12
food_prices positive: 9
food_style_options negative: 9
restaurant_prices positive: 9
drinks_quality positive: 8
```

Most important false positives:

```text
ambience neutral: 16
food_quality neutral: 14
location neutral: 9
location negative: 6
ambience positive: 6
```

The few-shot examples should therefore teach:

- prices are split between `restaurant_prices`, `food_prices`, and `drinks_prices`;
- menu variety, portions, amount, and presentation map to `food_style_options`;
- drink quality, wine quality, beverage variety, and beverage prices are separate;
- neutral is a real explicit judgment, not a filler for absent aspects;
- conflict is used when the same aspect has both clear positive and negative cues;
- comments and questions may have no valid aspect-polarity label and should output `{}`.

## Retrieval Methods

### 1. `random_fixed`

Fixed random examples selected from `train`, language-balanced when possible.

Purpose:

- Baseline for the K curve.
- Shows whether few-shot helps simply because examples are present.

### 2. `dense_topk`

Retrieve examples using Qwen3 embeddings only.

Process:

1. Embed all train examples.
2. Embed the target review.
3. Select top K nearest train examples by cosine similarity.
4. Order selected demonstrations from least similar to most similar, so the most
   similar example appears closest to the target review.

Purpose:

- Strong simple RAG baseline.
- Measures how much is gained before ABSA-specific reranking.

### 3. `absa_mmr`

Main method. Retrieve top 50 by embedding similarity, then perform ABSA-aware MMR
selection.

Base score:

```text
base_score =
    0.72 * semantic_similarity
  + 0.06 * same_language
  + 0.14 * aspect_cue_coverage
  + 0.05 * rare_or_hard_label_helpfulness
  + 0.03 * length_label_count_fit
```

MMR selection:

```text
final_score(candidate) =
    lambda_K * base_score(candidate)
  - (1 - lambda_K) * max_similarity(candidate, selected_examples)
```

Use:

```text
K=1  lambda=1.00
K=2  lambda=0.88
K=4  lambda=0.80
K=6  lambda=0.76
K=8  lambda=0.72
K=10 lambda=0.70
K=12 lambda=0.68
```

Hard redundancy guard:

```text
skip candidate if max_similarity(candidate, selected_examples) > 0.92,
unless there are not enough remaining candidates.
```

Signal definitions:

- `semantic_similarity`: cosine similarity from Qwen3 embeddings, normalized within
  the top-50 candidate pool.
- `same_language`: 1 when query and candidate have the same language, 0 otherwise.
- `aspect_cue_coverage`: weak cue overlap between the target review and candidate
  gold aspects. Cues include price, menu/variety/portion, food, service, ambience,
  drinks/wine, location, and overall recommendation.
- `rare_or_hard_label_helpfulness`: small bonus for candidates containing labels
  that the baseline often misses: prices, drinks, `food_style_options`, `neutral`,
  and `conflict`.
- `length_label_count_fit`: favors candidates with a similar review length and
  number of gold labels to the target review.

### 4. `manual_fixed_hard`

Fixed bank of hard demonstrations selected from `train`, ordered as a curriculum.

Use prefixes of this list for K:

1. `1225162`
   - Covers positive food, ambience, service, food prices, drinks style, drinks
     prices, and restaurant general.
2. `1632445`
   - Separates good food/service/ambience from limited menu and high prices.
3. `1212346`
   - Covers food conflict, negative service, negative menu/amount, and negative
     food prices.
4. `es_balmes_rossello_12_LauraRamosMartinez_2015-02-23`
   - Spanish neutral/conflict example with service and ambience complications.
5. `es_cafe_kafka_40_Kadulillo_2012-03-03`
   - Short Spanish example with positive ambience, food price, and drinks quality.
6. `es_l_olive_77_Mathews_2007-12-07`
   - Separates food prices, drinks prices, drinks style, ambience, and neutral
     restaurant general.
7. `744478`
   - Covers positive location and both food/restaurant prices.
8. `es_cafe_casa_lletres_9_RicardoSantolariaPerez_2015-01-06`
   - Teaches legitimate neutral labels and negative portion/value.
9. `1459569`
   - English negative service, drinks prices, neutral food quality, and negative
     restaurant general.
10. `es_doble-uno-zaragoza_comment-4729`
    - Spanish example with restaurant conflict, service conflict, drinks style
      negative, restaurant prices negative, and ambience positive.
11. `es_puerto-de-santa-maria-zaragoza_comment-802`
    - Empty-gold question/comment example. Use only at larger K to reduce
      over-labeling.
12. `es_luis-candelas-zaragoza_comment-4486`
    - Spanish example with drinks quality negative while the overall meal remains
      positive.

### 5. `hard_mix`

Hybrid method:

1. Select 1 or 2 examples from the manual hard bank based on weak cues in the
   target review.
2. Fill the remaining slots using `absa_mmr`.
3. Deduplicate selected examples.
4. Order from least similar to most similar, with manual examples before the most
   similar retrieved example unless that hurts format clarity.

Purpose:

- Combines stable hard-case coverage with query-specific retrieval.

## Run Matrix

Target about 30 runs. Use 24 exploration runs and 6 confirmation runs.

### Exploration Runs

```text
random_fixed:
K = 1, 2, 4, 6, 8, 10, 12        -> 7 runs

dense_topk:
K = 2, 4, 6, 8                   -> 4 runs

absa_mmr:
K = 1, 2, 4, 6, 8, 10, 12        -> 7 runs

manual_fixed_hard:
K = 4, 6, 8                      -> 3 runs

hard_mix:
K = 4, 6, 8                      -> 3 runs
```

Total exploration runs: 24.

### Confirmation Runs

After the exploration phase, select the top 3 configurations by `f1_micro`.

Run each of those top 3 configurations two additional times:

```text
top_1 repeat seeds/runs: 2
top_2 repeat seeds/runs: 2
top_3 repeat seeds/runs: 2
```

Total confirmation runs: 6.

Each top configuration will then have 3 total runs: the original exploration run
plus two repeats. The final winner should be chosen by mean score, not a single
sample.

## Metrics To Report

Primary metric:

```text
f1_micro
```

Secondary metrics:

```text
f1_macro
precision_micro
recall_micro
predicted label count
valid JSON rate
empty prediction count
sec_per_example
```

Per-aspect analysis must include:

```text
restaurant_general
food_quality
service
ambience
food_prices
food_style_options
restaurant_prices
drinks_quality
drinks_style_options
drinks_prices
location
```

Special attention:

- Does few-shot improve `food_prices`, `food_style_options`, `restaurant_prices`,
  and drinks labels?
- Does it reduce false neutral labels?
- Does higher K increase recall at an unacceptable precision cost?
- Does `absa_mmr` beat `dense_topk`, proving that ABSA-aware reranking matters?
- Does `hard_mix` beat pure retrieval, proving that hard examples add value?

## Expected Outcome

Most likely winner:

```text
absa_mmr K=6 or K=8
```

Potential alternative winner:

```text
hard_mix K=6 or K=8
```

Likely behavior:

- K=1 and K=2 may improve format and obvious mappings but lack coverage.
- K=4 should be a strong low-token baseline.
- K=6 or K=8 should give the best recall/precision tradeoff.
- K=10 or K=12 may improve recall but risk over-labeling and false neutrals.

## Naming Convention

Use output prefixes that fully identify the method and K:

```text
fewshot_v6_random_fixed_k4
fewshot_v6_dense_topk_k6
fewshot_v6_absa_mmr_k8
fewshot_v6_manual_hard_k6
fewshot_v6_hard_mix_k8
```

For confirmation repeats:

```text
fewshot_v6_absa_mmr_k6.repeat1
fewshot_v6_absa_mmr_k6.repeat2
```

Store full outputs, logs, summaries, and aggregated CSVs under:

```text
ABSA/outputs/fewshot/
```

## Decision Rule

The final few-shot configuration should be selected by:

1. Highest mean `f1_micro` across repeated runs.
2. If tied within 0.5 F1 points, prefer higher `f1_macro`.
3. If still tied, prefer higher precision if recall has already improved over
   zero-shot.
4. If still tied, prefer smaller K for lower cost and lower overfitting risk.

