# Thinking Probe: Qwen3.5-2B

## Run

- Output JSON: `outputs/thinking/qwen35_2b_think_devel_prompt_v2_pp1.5_m4096_pilot8.json`
- Summary CSV: `outputs/thinking/qwen35_2b_think_devel_prompt_v2_pp1.5_m4096_pilot8.summary.csv`
- Examples: 8
- Generation config: `{'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5, 'repetition_penalty': 1.0, 'max_new_tokens': 4096}`

## Metrics

| metric | value |
| --- | --- |
| M.avg P/R/F1 | 12.5 / 12.5 / 12.5 |
| m.avg P/R/F1 | 50.0 / 4.5 / 8.3 |
| predicted / gold / ok | 2 / 22 / 1 |
| closed thinking | 2 / 8 |
| hit max_new_tokens | 6 / 8 |
| unfinished at token limit | 6 / 8 |
| total output tokens | 30957 |
| total thinking tokens | 30933 |
| total final tokens | 22 |
| total runtime seconds | 1079.9 |

## Interpretation

- `closed_thinking` means the generated output contained `</think>`.
- `hit max_new_tokens` means generation consumed the full configured budget without EOS.
- `unfinished at token limit` is the first simple loop-risk flag: it hit the token budget before closing `</think>`.
- This is intentionally not a loop detector yet; it only measures whether thinking terminates under the chosen budget.
