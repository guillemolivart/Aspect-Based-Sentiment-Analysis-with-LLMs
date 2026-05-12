# Few-Shot Results and Conclusions

This document summarizes the few-shot ABSA experiments run with Qwen3.5-2B in
non-thinking mode. The goal of this stage was to evaluate whether in-context
examples improve the best zero-shot setup, and to choose one robust few-shot
configuration for the final system.

## Experimental Setup

All confirmed few-shot runs use the same base prompt and generation settings:

- Prompt: `prompts/absa_v6.json`
- Model mode: non-thinking
- Temperature: `0.68`
- Top-p: `0.72`
- Top-k: `20`
- Min-p: `0.0`
- Presence penalty: `1.8`
- Repetition penalty: `1.0`
- Max new tokens: `512`
- Embedding model: `qwen3_embedding_0_6b`

The exploration stage tested random fixed examples, dense embedding retrieval,
ABSA-aware MMR retrieval, hard-example mixtures, and a manually selected hard
bank. The confirmation stage reran the strongest configurations with three
different seeds: `101`, `202`, and `303`.

## Top Confirmed Methods

The table reports the mean and sample standard deviation across the three
confirmation seeds.

| Rank | Method | K | F1 micro mean | F1 micro std | F1 macro mean | F1 macro std | Precision micro mean | Recall micro mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `absa_mmr` | 8 | **73.93** | **0.50** | 73.85 | 0.56 | **79.60** | 69.01 |
| 2 | `absa_mmr` | 12 | 73.81 | 0.79 | 73.92 | 0.99 | 78.00 | **70.04** |
| 3 | `dense_topk` | 12 | 73.59 | 0.98 | **74.19** | 1.11 | 77.98 | 69.67 |

The best single run was `absa_mmr K=12 seed=303` with `F1_micro = 74.71`, but
the most robust configuration is `absa_mmr K=8`.

## Interpretation

`absa_mmr K=8` is the recommended final few-shot configuration. It obtains the
best average micro-F1 and the lowest variance among the strongest methods. This
matters because the final system should not depend on a lucky seed or a specific
sample of demonstrations.

`absa_mmr K=12` is a reasonable alternative when recall is prioritized. It
recovers slightly more gold labels on average, but it also loses precision and
is less stable across seeds. This suggests that adding more examples can help
the model find more aspects, but also introduces more noise into the prompt.

`dense_topk K=12` is competitive, especially in macro-F1, but it is less stable
than MMR. The result supports the hypothesis that pure semantic similarity is
not enough for this dataset: diversity between demonstrations is useful because
many reviews share surface meaning while requiring different aspect labels.

## Error Profile

The strong configurations mostly solve the frequent and explicit aspects:

- `restaurant_general`
- `food_quality`
- `service`
- `ambience`

The remaining errors are concentrated in rare or subtle aspects:

- `restaurant_prices`
- `food_prices`
- `food_style_options`
- `drinks_quality`
- `drinks_style_options`
- `location`

The model also remains weak on minority sentiment labels. Positive labels are
handled well, negative labels are acceptable, and `neutral` or `conflict` labels
remain difficult. This is consistent with the dataset distribution and with the
semantic ambiguity of neutral/conflict annotations.

## Technical Quality

The confirmed few-shot runs are technically clean:

- JSON parse success rate: `100%`
- Token limit hits: `0`
- Average generated answer length: about `20` tokens

Therefore, the remaining gap is not caused by malformed JSON or generation
limits. The errors are mainly semantic: missing rare aspects, confusing price
or style categories, and underpredicting minority labels.

## Final Decision

The final few-shot configuration should be:

```text
method = absa_mmr
k = 8
prompt = prompts/absa_v6.json
mode = non-thinking
temperature = 0.68
top_p = 0.72
top_k = 20
presence_penalty = 1.8
max_new_tokens = 512
```

This is the best defensible choice for the report: it has the strongest mean
micro-F1, the lowest variance among the top methods, and a clear methodological
justification based on semantic retrieval plus diversity.
