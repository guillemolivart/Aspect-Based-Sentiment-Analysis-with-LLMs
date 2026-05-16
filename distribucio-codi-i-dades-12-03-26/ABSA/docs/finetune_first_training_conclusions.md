# First Fine-Tuning Run Conclusions

This document summarizes the first Qwen3.5-2B LoRA fine-tuning attempt for ABSA, how we analyzed it, and the code change that follows from the results.

## Setup

The first training run used the current supervised fine-tuning setup:

- Base model: `Qwen/Qwen3.5-2B`
- Prompt family: mainly `absa_v6`
- Training split: `train.json`
- Validation/inference split: `devel.json`
- Training format: zero-shot prompt plus assistant JSON
- Loss masking: system/user prompt masked with `-100`; only the final assistant JSON is trained
- Fine-tuning method: LoRA over Qwen linear projection modules
- Main hyperparameters: `lr=1e-4`, `epochs=5`, `max_length=3072`, no few-shot in training

The no-few-shot decision is intentional. Few-shot examples are an inference-time prompting strategy; for the first SFT run we wanted training and zero-shot inference to stay aligned. Demonstration examples, if used later, should be a separate ablation.

## Token Budget

The token-budget notebook showed that `absa_v6 + compact gold JSON` fits safely inside:

- train `full_tokens` max: `2688`
- devel `full_tokens` max: `2193`
- train `full_tokens` p95: about `1742`

Therefore `max_length=3072` is appropriate for training. It avoids truncating current train/devel examples while still keeping memory use reasonable through dynamic padding.

For generation, the dataset study showed:

- gold JSON answer tokens p50: `27`
- p95: `43`
- p99: `52`
- max: `74`

The notebook suggested `128` as a lean candidate and `256` as a conservative JSON-only budget, but also recommended keeping `512` as a debugging safety budget until generated outputs were measured. Since our zero-shot, hyperparameter, few-shot, and fine-tuned inference runs have used `max_new_tokens=512`, generative validation now also defaults to `512` for comparability.

## Observed Results

The teammate's initial fine-tuned inference outputs were valid and not a failure, but the first interpretation was misleading because they were not evaluated under the same greedy decoding setup used for the later checkpoint analysis.

| Run | M.avg F1 | m.avg F1 | Exact match |
| --- | ---: | ---: | ---: |
| `FT.devel_v6.json` | `77.9` | `80.8` | `60/132` |
| `FT.devel_v11.json` | `78.4` | `80.0` | `60/132` |
| `FT.devel_v6.greedy.json` | `81.0` | `84.0` | `67/132` |
| `FT.devel_v6.checkpoint650.greedy.json` | `83.4` | `85.4` | `69/132` |
| `FT.devel_v6.checkpoint660.greedy.json` | `83.1` | `85.0` | `68/132` |
| `FT.devel_v6.checkpoint650.absa_mmr_k8.greedy.json` | `82.7` | `84.4` | `64/132` |

The best analyzed result so far is checkpoint `650` with greedy decoding:

```text
M.avg  85.0 / 83.0 / 83.4
m.avg  86.3 / 84.6 / 85.4
exact  69 / 132
```

This is a strong improvement over prompting-only runs and shows that the training setup is fundamentally working.

## What Went Wrong

The main issue was checkpoint selection, not necessarily the fine-tuning recipe.

The training script was relying on `eval_loss` through Hugging Face `Trainer` checkpoint selection. That is useful as a sanity check, but it is not the metric we ultimately care about. In this task, the final score depends on whether the model generates a parseable JSON object with the right aspect-polarity pairs.

That means a checkpoint can have a slightly better validation loss but worse generated JSON behavior. Our manual checkpoint analysis showed exactly why this matters: checkpoint `650` outperformed both the root adapter output and checkpoint `660` under the actual ABSA metrics.

## Few-Shot with the Fine-Tuned Model

We also tested the best checkpoint with the strongest previous few-shot method, `absa_mmr K=8`, using greedy decoding.

It did not improve the fine-tuned model:

```text
checkpoint650 zero-shot greedy:   m.avg F1 = 85.4
checkpoint650 absa_mmr K=8:       m.avg F1 = 84.4
```

The likely reason is that SFT already internalizes the extraction format and decision boundary. Adding long in-context examples changes the input distribution and can push the model toward over-prediction or slightly noisier label decisions. For the current fine-tuned model, zero-shot greedy inference is the preferred default.

## Code Change

The training script now includes generative validation:

- At each validation point, it runs greedy generation on `devel.json`.
- It parses the model output with the same JSON extraction utilities used by inference.
- It computes the same macro/micro metrics as `stats.py`.
- It selects the best checkpoint by `m.avg F1`.
- If tied, it uses `M.avg F1` as the tie-breaker.
- It also evaluates the final training step if it was not already covered by `eval_steps`.

The saved output layout is:

```text
<output_dir>/
  adapter files for the best generative-F1 checkpoint
  best_generative_f1/
    mirror copy of the best adapter
  last/
    final training-step adapter
  generative_eval/
    history.jsonl
    best_devel_predictions.json
    best_metrics.json
  best_generative_f1.json
```

This keeps the default weights path convenient while still preserving the final checkpoint and the validation history.

## Current Recommended Run

The next run should repeat the same strong configuration, but now with generative-F1 checkpoint selection enabled:

```bash
cd /workspace/Aspect-Based-Sentiment-Analysis-with-LLMs
git pull
source /workspace/absa-venv/bin/activate
cd distribucio-codi-i-dades-12-03-26/ABSA/bin/finetune

python3 finetune-train.py ../../prompts/absa_v6.json train \
  --learning-rate 1e-4 \
  --num-epochs 5 \
  --max-length 3072 \
  --per-device-train-batch 2 \
  --gradient-accumulation-steps 4 \
  --eval-steps 50 \
  --no-fewshot \
  --output-dir ../../outputs/finetune/FT.train.absa_v6.lora.simple.all-linear.r16.lr1e-4.generative_f1.weights
```

No explicit `--generative-eval-max-new-tokens` flag is needed because the default is now `512`, matching the inference experiments.

## Final Decision

The first fine-tuning attempt should be considered successful but incomplete:

1. The model clearly improves when evaluated with the right checkpoint and greedy decoding.
2. The best current checkpoint reaches about `85.4` micro-F1 on `devel`.
3. Few-shot prompting is not currently useful on top of the fine-tuned model.
4. Future training must select checkpoints with generative ABSA F1, not only `eval_loss`.
5. The next experiment should rerun the same configuration with the updated script before changing learning rate, LoRA rank, QLoRA, or few-shot training.
