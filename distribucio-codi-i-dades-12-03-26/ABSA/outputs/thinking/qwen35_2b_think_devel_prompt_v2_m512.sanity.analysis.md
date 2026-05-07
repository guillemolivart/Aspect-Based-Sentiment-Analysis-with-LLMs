# Thinking Probe: Qwen3.5-2B

## Run

- Output JSON: `distribucio-codi-i-dades-12-03-26/ABSA/outputs/thinking/qwen35_2b_think_devel_prompt_v2_m512.sanity.json`
- Summary CSV: `distribucio-codi-i-dades-12-03-26/ABSA/outputs/thinking/qwen35_2b_think_devel_prompt_v2_m512.sanity.summary.csv`
- Examples: 2
- Generation config: `{'thinking': True, 'max_new_tokens': 512}`

## Metrics

| metric | value |
| --- | --- |
| M.avg P/R/F1 | 0.0 / 0.0 / 0.0 |
| m.avg P/R/F1 | 0.0 / 0.0 / 0.0 |
| predicted / gold / ok | 0 / 5 / 0 |
| closed thinking | 0 / 2 |
| hit max_new_tokens | 2 / 2 |
| unfinished at token limit | 2 / 2 |
| total output tokens | 1024 |
| total thinking tokens | 1024 |
| total final tokens | 0 |
| total runtime seconds | 35.9 |

## Interpretation

- `closed_thinking` means the generated output contained `</think>`.
- `hit max_new_tokens` means generation consumed the full configured budget without EOS.
- `unfinished at token limit` is the first simple loop-risk flag: it hit the token budget before closing `</think>`.
- This is intentionally not a loop detector yet; it only measures whether thinking terminates under the chosen budget.
