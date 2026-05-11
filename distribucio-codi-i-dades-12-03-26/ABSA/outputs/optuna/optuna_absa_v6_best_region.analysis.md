# Optuna ABSA v6 Best Region Analysis

Run prefix: `optuna_absa_v6_best_region`

This run evaluated 30 Optuna trials on the full `devel` split with prompt `absa_v6`,
`max_new_tokens=512`, and the fixed no-thinking generation setup:

- `min_p=0.0`
- `repetition_penalty=1.0`
- `enable_thinking=False`

The Optuna search space was intentionally narrow around the strongest manual sweep
region:

- `temperature`: 0.60, 0.62, 0.65, 0.68, 0.70, 0.72
- `top_p`: 0.72, 0.75, 0.78, 0.80, 0.82
- `top_k`: 15, 20, 25, 30
- `presence_penalty`: 1.7, 1.8, 1.9, 2.0, 2.1, 2.2

The objective maximized:

```text
score = 0.75 * f1_micro + 0.25 * f1_macro
```

## Best Trial

Trial 15 was selected as the best configuration:

```json
{
  "temperature": 0.68,
  "top_p": 0.72,
  "top_k": 20,
  "presence_penalty": 1.8,
  "min_p": 0.0,
  "repetition_penalty": 1.0,
  "max_new_tokens": 512
}
```

Metrics:

- score: 68.809
- f1_micro: 68.957
- f1_macro: 68.363
- precision_micro: 74.615
- recall_micro: 64.097
- precision_macro: 74.342
- recall_macro: 65.611

## Comparison With Manual Sweep

Best previous manual sweep result:

- config: `v6_F_refined_9_temp065_topp75`
- temperature: 0.65
- top_p: 0.75
- top_k: 20
- presence_penalty: 2.0
- f1_micro: 68.34
- f1_macro: 67.93

Optuna best vs manual best:

- f1_micro: +0.617
- f1_macro: +0.433

This is a small improvement, not a decisive one. The duplicate trials show noticeable
sampling variance, so this should be treated as a good candidate configuration rather
than a conclusively better configuration.

## Important Observations

Optuna behaved close to a guided categorical grid search, which is expected here:
the search space was discrete, narrow, and only 30 trials were run.

There were 27 unique configurations among 30 executed trials. Some duplicates showed
large sampling variance:

- `(temperature=0.68, top_p=0.72, top_k=30, presence_penalty=1.7)`
  - trials 10 and 11
  - f1_micro: 68.653 vs 67.696
- `(temperature=0.72, top_p=0.72, top_k=30, presence_penalty=1.7)`
  - trials 12 and 20
  - f1_micro: 67.135 vs 67.376
- `(temperature=0.65, top_p=0.80, top_k=20, presence_penalty=1.7)`
  - trials 21 and 22
  - f1_micro: 68.327 vs 65.425

The last duplicate is especially important: the same parameters differed by almost
3 micro-F1 points. This means single-run differences below roughly 1 point should not
be overinterpreted.

## Parameter Trends

Group means are noisy, but the run still suggests:

- `top_p=0.72` was the strongest region on average.
- `top_p=0.82` was consistently weak.
- `presence_penalty=1.8` was the best average region.
- `top_k=20` remained a robust default.
- `temperature=0.68` produced the best individual result.

## Recommendation

Use the Optuna best configuration as the current candidate for future few-shot runs:

```text
temperature=0.68
top_p=0.72
top_k=20
presence_penalty=1.8
min_p=0.0
repetition_penalty=1.0
max_new_tokens=512
```

However, do not claim that Optuna definitively beat the manual sweep unless this
configuration wins again in repeated runs. For the next phase, it is more valuable
to spend compute on few-shot retrieval quality than on further hyperparameter search.
