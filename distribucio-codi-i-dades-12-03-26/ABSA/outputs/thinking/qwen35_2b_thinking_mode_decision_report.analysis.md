# Qwen3.5-2B Thinking Mode Decision Report

## Executive Summary

Thinking mode is currently not worth using as the main ABSA strategy for this
project. The experiments show that Qwen3.5-2B often spends the full generation
budget inside `<think>` without producing a final JSON answer. The issue is not
mostly malformed parsing or simple exact repetition; it is excessive
deliberation over aspect definitions and edge cases.

For now, the project should focus on improving non-thinking inference: prompt
quality, robust parsing, few-shot selection, retrieval-style examples, and
careful sampling groups. Thinking mode can be revisited later with a controlled
reasoning budget and fallback policy.

## Model And Configuration

The experiments used `Qwen/Qwen3.5-2B` through Hugging Face Transformers.
Thinking was enabled through the Qwen chat template with `enable_thinking=True`.

The official Qwen sampling recipe for text thinking mode was used as the target
configuration:

| parameter | value |
| --- | --- |
| `temperature` | `1.0` |
| `top_p` | `0.95` |
| `top_k` | `20` |
| `min_p` | `0.0` |
| `presence_penalty` | `1.5` |
| `repetition_penalty` | `1.0` |

One implementation detail matters: Transformers does not expose the same
OpenAI/vLLM-style `presence_penalty` argument in `generate()`. We therefore
implemented a local logits processor that subtracts the configured penalty from
tokens that have already appeared in the generated continuation. Prompt tokens
are intentionally ignored, because the prompt contains valid aspect labels that
the model must still be able to generate in the final JSON.

## Experiments

### 1. Initial Thinking Pilot

File stem:
`qwen35_2b_think_devel_prompt_v2_m4096_pilot8`

Setup:

- Prompt: `absa_v2`
- Dataset: 8 evenly spaced examples from `devel`
- Thinking: enabled
- `max_new_tokens`: `4096`
- `presence_penalty`: not applied

Results:

| metric | value |
| --- | --- |
| closed thinking | `2 / 8` |
| hit max tokens | `6 / 8` |
| unfinished at token limit | `6 / 8` |
| predicted / gold / ok | `2 / 22 / 1` |
| M.avg F1 | `12.5` |
| m.avg F1 | `8.3` |
| output tokens | `29014` |
| thinking tokens | `28990` |
| final-answer tokens | `22` |
| runtime | `1007.7 s` |

Interpretation:

Only two examples closed `</think>`. The other six consumed all 4096 output
tokens while still thinking, so they produced no valid final answer. The final
JSON answers were tiny when thinking closed: only 11 tokens each.

### 2. Forced Closing Diagnostic

Files:

- `qwen35_2b_forced_close_diagnostic.json`
- `qwen35_2b_forced_close_bridge_diagnostic.json`

Purpose:

We tested whether manually inserting `</think>` after a partial reasoning budget
could make the model emit a final answer. We also tested a short bridge sentence
before the closing token, for example:

`Therefore, the final answer as one complete valid JSON object, using only allowed keys and polarity values, is:`

Findings:

- An empty closed thinking block behaves similarly to non-thinking mode.
- Closing after partial reasoning can produce a final JSON, but it is not
  reliable.
- The bridge sentence helped format in the small diagnostic, but it did not
  correct wrong decisions.
- In `Excelente atención y calidad`, partial thinking improved detection of
  `service` and `food_quality`, but missed `restaurant_general`.
- In `Restaurante para no tirar cohetes pero en general sales satisfecho.`, the
  model still predicted `restaurant_general: positive` while gold was
  `restaurant_general: neutral`.

Interpretation:

Forced closing is useful as a diagnostic, not as a robust production strategy.
If the reasoning is already wrong or incomplete, closing the thinking block
mostly makes the model express that state in final-answer form.

### 3. Presence Penalty Pilot

File stem:
`qwen35_2b_think_devel_prompt_v2_pp1.5_m4096_pilot8`

Setup:

- Same 8 examples as the initial pilot
- Same `max_new_tokens=4096`
- Added local Transformers-side `presence_penalty=1.5`

Results:

| metric | no presence penalty | presence_penalty=1.5 |
| --- | --- | --- |
| closed thinking | `2 / 8` | `2 / 8` |
| hit max tokens | `6 / 8` | `6 / 8` |
| unfinished at token limit | `6 / 8` | `6 / 8` |
| predicted / gold / ok | `2 / 22 / 1` | `2 / 22 / 1` |
| M.avg F1 | `12.5` | `12.5` |
| m.avg F1 | `8.3` | `8.3` |
| output tokens | `29014` | `30957` |
| thinking tokens | `28990` | `30933` |
| final-answer tokens | `22` | `22` |
| runtime | `1007.7 s` | `1079.9 s` |

Interpretation:

Presence penalty did not fix thinking termination. The model still spent all
4096 tokens inside `<think>` for six examples. This supports the hypothesis that
the main failure mode is not exact token repetition but deliberation: the model
keeps reconsidering labels, definitions, and exceptions without deciding to
close the reasoning block.

## Qualitative Error Pattern

The unfinished generations often look close to a useful answer. For example,
the model identifies plausible aspects and polarities, then reopens the same
decision repeatedly:

- whether `calidad` should map to `food_quality` or a broader restaurant-level
  judgment;
- whether `entorno` is `ambience` or `location`;
- whether a mild criticism should be `neutral` or `negative`;
- whether a global recommendation also warrants `restaurant_general`.

This is damaging for evaluation because text inside an unclosed thinking block
is not a final answer. Evaluating JSON-like text that appears inside `<think>`
would overstate performance and would not match how Qwen reasoning parsers
separate reasoning from content.

## Decision

Thinking mode should be paused as a main evaluation path.

The current best direction is:

1. Use non-thinking mode as the main baseline.
2. Improve prompt v2 or create prompt v3 focused on dataset-specific boundary
   cases.
3. Add few-shot variants, especially retrieved/similar examples rather than
   only random examples.
4. Tune small, coherent sampling groups for non-thinking mode.
5. Revisit thinking later only with a guardrail policy.

## Future Thinking Guardrail

If thinking is revisited, the next experiment should not be another full
`max_new_tokens=4096` run. A better design is:

1. Start generation in thinking mode.
2. Stream tokens and monitor whether `</think>` appears within a small budget
   such as 512, 1024, or 2048 reasoning tokens.
3. If the model closes thinking, evaluate the final JSON.
4. If the model does not close thinking, abort and retry from the original
   prompt in non-thinking mode.
5. Store both the reasoning-budget statistics and the fallback result.

This would make thinking an optional aid for hard examples rather than a source
of unbounded cost and missing final answers.
