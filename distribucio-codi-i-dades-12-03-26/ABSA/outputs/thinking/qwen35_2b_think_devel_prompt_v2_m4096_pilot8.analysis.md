# Thinking Probe: Qwen3.5-2B Prompt v2

## Purpose

This probe measures whether Qwen3.5-2B can use thinking mode for ABSA without spending the whole output budget in reasoning. For unfinished generations, predictions are evaluated as `{}` because text inside an unclosed thinking block is not a final answer.

## Files

- `distribucio-codi-i-dades-12-03-26/ABSA/outputs/thinking/qwen35_2b_think_devel_prompt_v2_m4096_pilot8.json`
- `distribucio-codi-i-dades-12-03-26/ABSA/outputs/thinking/qwen35_2b_think_devel_prompt_v2_m4096_pilot8.summary.csv`
- `distribucio-codi-i-dades-12-03-26/ABSA/outputs/thinking/qwen35_2b_think_devel_prompt_v2_m4096_pilot8.analysis.md`
- `distribucio-codi-i-dades-12-03-26/ABSA/outputs/thinking/qwen35_2b_think_devel_prompt_v2_m4096_pilot8.run.log`
- `distribucio-codi-i-dades-12-03-26/ABSA/outputs/thinking/qwen35_2b_think_devel_prompt_v2_m512.sanity.json`
- `distribucio-codi-i-dades-12-03-26/ABSA/outputs/thinking/qwen35_2b_think_devel_prompt_v2_m512.sanity.summary.csv`
- `distribucio-codi-i-dades-12-03-26/ABSA/outputs/thinking/qwen35_2b_think_devel_prompt_v2_m512.sanity.analysis.md`

## Run Setup

- Prompt: `prompts/absa_v2.json`
- Thinking enabled via Qwen chat template `enable_thinking=True`
- Sampling: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `repetition_penalty=1.0`
- Sanity run: 2 head examples, `max_new_tokens=512`
- Pilot run: 8 evenly spaced `devel` examples, `max_new_tokens=4096`

## Headline

| system | n | P_macro | R_macro | F1_macro | P_micro | R_micro | F1_micro | pred | gold | ok | closed | unfinished limit | sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thinking_m512_sanity | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 5 | 0 | 0/2 | 2/2 | 35.9 |
| thinking_m4096_pilot8 | 8 | 12.5 | 12.5 | 12.5 | 50.0 | 4.5 | 8.3 | 2 | 22 | 1 | 2/8 | 6/8 | 1007.7 |
| qwen35_2b_nothink_devel_prompt_v1 | 132 | 47.4 | 55.8 | 46.9 | 39.8 | 56.6 | 46.7 | 646 | 454 | 257 | - | - | - |
| qwen35_2b_nothink_devel_prompt_v2 | 132 | 66.3 | 50.2 | 53.9 | 63.1 | 49.3 | 55.4 | 355 | 454 | 224 | - | - | - |

Thinking mode is not usable yet for this task without a streaming cutoff/fallback policy. With 4096 output tokens, only 2 of 8 examples closed `</think>`. The other 6 consumed the full token budget while still thinking.

## Token Cost

| metric | value |
| --- | --- |
| total output tokens | 29014 |
| total thinking tokens | 28990 |
| total final-answer tokens | 22 |
| thinking share of output | 99.9% |
| runtime total | 1007.7 s |
| runtime avg/example | 126.0 s |
| throughput | 28.8 output tokens/s |
| output tokens p50 / p95 / max | 4096 / 4096 / 4096 |
| thinking tokens p50 / p95 / max | 4096 / 4096 / 4096 |

The final answers are tiny when thinking closes: both closed examples used only 11 final tokens. The cost is almost entirely reasoning tokens, not JSON answer length.

## Per Example

| idx | id | lang | out tok | think tok | final tok | closed | hit limit | unfinished | sec | pred | gold | ok | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | es_matilde-la-zaragoza_comment-4639 | es | 4096 | 4096 | 0 | no | yes | yes | 142.1 | 0 | 3 | 0 | 0.0 |
| 19 | es_salamanca_20_EliasMansilla_2014-10-18 | es | 838 | 826 | 11 | yes | no | no | 29.2 | 1 | 1 | 1 | 100.0 |
| 37 | es_barceloneta_130_Alvaro_2008-03-21 | es | 4096 | 4096 | 0 | no | yes | yes | 142.4 | 0 | 5 | 0 | 0.0 |
| 56 | en_ProfessorThom's_478606543 | en | 4096 | 4096 | 0 | no | yes | yes | 142.7 | 0 | 4 | 0 | 0.0 |
| 75 | es_barceloneta_56_JuanLuis_2012-06-10 | es | 4096 | 4096 | 0 | no | yes | yes | 142.7 | 0 | 3 | 0 | 0.0 |
| 94 | es_doble-uno-zaragoza_comment-4741 | es | 4096 | 4096 | 0 | no | yes | yes | 142.0 | 0 | 4 | 0 | 0.0 |
| 112 | es_bal-donsera-zaragoza_comment-1753 | es | 4096 | 4096 | 0 | no | yes | yes | 141.8 | 0 | 1 | 0 | 0.0 |
| 131 | es_balmes_rossello_58_RaulHervas_2014-01-26 | es | 3600 | 3588 | 11 | yes | no | no | 124.8 | 1 | 1 | 0 | 0.0 |

## Interpretation

1. `max_new_tokens` includes thinking tokens. It does not start counting after reasoning. This run proves it operationally: unfinished examples spent all 4096 tokens before producing a final answer.
2. The model often writes long self-checking reasoning and repeatedly revisits aspect definitions instead of deciding quickly.
3. Evaluating raw JSON inside an unclosed thinking block would overstate quality, so unfinished thinking is treated as no final answer.
4. For ABSA, no-thinking prompt v2 remains the practical baseline. Thinking is currently slower and much worse until we add streaming interruption and retry/fallback.

## Next Step

Implement a streaming loop guard before any full thinking evaluation. The first guard should abort when thinking exceeds a budget such as 512-1024 tokens without `</think>`, then retry no-thinking with a direct-answer prompt. After that, test whether a small thinking budget improves hard cases without destroying throughput.
