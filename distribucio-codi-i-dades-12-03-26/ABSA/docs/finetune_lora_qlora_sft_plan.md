# Fine-Tuning Plan: LoRA/QLoRA SFT for ABSA

This document summarizes the new fine-tuning setup implemented for the ABSA task and the reasoning behind the defaults.

## Objective

The fine-tuning stage should specialize Qwen3.5-2B for ABSA extraction without overfitting to prompt text or few-shot demonstrations. The project statement asks for fine-tuning on `train.json`, validation on `devel.json`, experimentation with training hyperparameters, and repeated experiments with the model loaded in 4-bit quantization.

The recommended interpretation is:

- **LoRA**: train adapters on the base model loaded in bf16/fp16.
- **QLoRA**: train the same kind of adapters while the base model is loaded in 4-bit NF4.

This gives a clean comparison and satisfies the 4-bit requirement.

## Core Fix

The previous training script trained on almost the whole chat sequence. That is not ideal here because the prompt is around 1.6k tokens while the gold JSON answer is usually around 20 tokens. If the prompt is not masked, most of the loss is spent learning to reproduce instructions rather than learning ABSA labels.

The new setup builds each training example as:

```text
system prompt + user review + assistant JSON
```

but masks the labels as:

```text
system prompt + user review -> -100
assistant JSON              -> trainable labels
```

For optional few-shot training, demonstration examples are also treated as context and masked. Only the final target JSON remains trainable.

## Token Budget

The token-budget notebook showed that `absa_v6` plus compact gold JSON needs:

- train `full_tokens` max: 2688
- devel `full_tokens` max: 2193
- train `full_tokens` p95: about 1742

Therefore the default is:

```text
max_length = 3072
```

This avoids truncating the current train/devel examples in the main zero-shot SFT setting. Dynamic padding is used during batching, so the model does not always pay the full 3072-token cost for every example.

## Default Training Configuration

The current default is intentionally strong but still controlled:

```text
prompt: absa_v6.json
training data: train.json
validation data: devel.json
training prompt: zero-shot, no few-shot by default
loss: assistant JSON only
max_length: 3072
LoRA targets: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
learning_rate: 1e-4
epochs: 5
effective batch size: 8 by default
scheduler: cosine
warmup_ratio: 0.05
weight_decay: 0.01
max_grad_norm: 0.3
optimizer:
  - LoRA: adamw_torch
  - QLoRA: paged_adamw_8bit
```

The default no longer trains with few-shot demonstrations. Few-shot can still be enabled with `--use-fewshot`, but the primary run should first align training and inference in the simpler zero-shot format.

## Main Runs

Run from:

```bash
cd distribucio-codi-i-dades-12-03-26/ABSA/bin/finetune
```

### 1. Main LoRA Run

```bash
python3 finetune-train.py ../prompts/absa_v6.json train \
  --learning-rate 1e-4 \
  --num-epochs 5 \
  --max-length 3072 \
  --per-device-train-batch 1 \
  --gradient-accumulation-steps 8
```

This should be the first serious run if memory allows it.

### 2. Main QLoRA 4-bit Run

```bash
python3 finetune-train.py ../prompts/absa_v6.json train \
  --load-in-4bit \
  --learning-rate 1e-4 \
  --num-epochs 5 \
  --max-length 3072 \
  --per-device-train-batch 1 \
  --gradient-accumulation-steps 8
```

This is the required 4-bit comparison run.

### 3. Higher-Capacity LoRA Run

```bash
python3 finetune-train.py ../prompts/absa_v6.json train \
  --lora-r 32 \
  --lora-alpha 64 \
  --learning-rate 5e-5 \
  --num-epochs 5 \
  --max-length 3072 \
  --per-device-train-batch 1 \
  --gradient-accumulation-steps 8
```

This tests whether more adapter capacity helps without making the learning rate too aggressive.

## Evaluation Protocol

The Trainer validation loss is useful for sanity checking, but the model must be selected using generative ABSA metrics on `devel.json`.

After each completed run:

```bash
python3 finetune-inference.py \
  --weights ../../outputs/finetune/<RUN_DIR> \
  --prompt-file ../prompts/absa_v6.json \
  --data devel \
  --keep-raw
```

Then evaluate the produced JSON with `stats.py`.

The fine-tuned model is only worth keeping if it clearly beats or complements the current best prompting/few-shot result. The current target to beat is approximately:

```text
few-shot absa_mmr K=8: F1_micro about 73.9 on devel
```

## Important Review Notes

- `devel.json` is validation/development, not final test.
- `test.json` should only be used once the final system is fixed.
- The default training path is now no-few-shot SFT, because previous few-shot training created a mismatch with zero-shot inference.
- QLoRA is not a different task from LoRA; it is LoRA with the base model quantized in 4-bit.
- The inference script now looks for the newest run under `outputs/finetune/FT.*.weights`, while still supporting legacy output names.
- Local validation completed: Python syntax compilation and `git diff --check`.
- Runtime validation still needs a GPU environment with `torch`, `transformers`, `peft`, `bitsandbytes`, and the local model files installed.

