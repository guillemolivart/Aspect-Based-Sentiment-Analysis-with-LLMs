# Zero-Shot Report: Qwen3.5-2B No-Thinking Prompt v1

## Run

- Output JSON: `distribucio-codi-i-dades-12-03-26/ABSA/outputs/zeroshot/qwen35_2b_nothink_devel_prompt_v1.json`
- Stats: `distribucio-codi-i-dades-12-03-26/ABSA/outputs/zeroshot/qwen35_2b_nothink_devel_prompt_v1.stats.txt`
- Run log: `distribucio-codi-i-dades-12-03-26/ABSA/outputs/zeroshot/qwen35_2b_nothink_devel_prompt_v1.run.log`
- Examples: 132 (`devel`)
- Model loading: 6.1 s
- Runtime: 294.9 s total, 2.23 s/example
- Generation config: `{'temperature': 1.0, 'top_p': 1.0, 'top_k': 20, 'min_p': 0.0, 'repetition_penalty': 1.0, 'max_new_tokens': 512}`
- Mode: no-thinking. `--keep-raw` was enabled, so every record stores the literal model output in `raw_generation`.

## Headline Result

| system | P_macro | R_macro | F1_macro | P_micro | R_micro | F1_micro | predicted | gold | ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35_2b_nothink_devel_prompt_v1 | 47.4 | 55.8 | 46.9 | 39.8 | 56.6 | 46.7 | 646 | 454 | 257 |

This first zero-shot run is not competitive yet. It reaches **46.9 macro F1**, while the strongest legal frequency baseline reaches **55.6 macro F1**. The model recovers many true labels, but predicts too many extra labels, so precision collapses.

## Baseline Comparison

| baseline | P_macro | R_macro | F1_macro | F1_micro | predicted | gold | ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_aspects_majority_polarity | 72.6 | 72.6 | 72.6 | 71.8 | 454 | 454 | 326 |
| top3_positive | 60.4 | 56.8 | 55.6 | 56.2 | 396 | 454 | 239 |
| top3_majority | 60.4 | 56.8 | 55.6 | 56.2 | 396 | 454 | 239 |
| top4_positive | 51.9 | 63.4 | 54.5 | 55.8 | 528 | 454 | 274 |
| top4_majority | 51.9 | 63.4 | 54.5 | 55.8 | 528 | 454 | 274 |
| top2_positive | 67.0 | 44.3 | 50.4 | 49.3 | 264 | 454 | 177 |
| top2_majority | 67.0 | 44.3 | 50.4 | 49.3 | 264 | 454 | 177 |
| top5_positive | 43.5 | 65.8 | 50.2 | 51.5 | 660 | 454 | 287 |

Interpretation: `top3_positive` is still the bar to beat. It predicts only `restaurant_general`, `food_quality`, and `service` as positive for every review. Qwen predicts more labels and gets higher recall than top3 in some examples, but the extra false positives make the global F1 worse.

## Label Count Shape

| labels/review | gold reviews | predicted reviews |
| --- | --- | --- |
| 0 | 3 | 14 |
| 1 | 13 | 17 |
| 2 | 14 | 11 |
| 3 | 37 | 23 |
| 4 | 35 | 10 |
| 5 | 21 | 8 |
| 6 | 7 | 7 |
| 7 | 1 | 2 |
| 8 | 1 | 7 |
| 9 | 0 | 4 |
| 10 | 0 | 9 |
| 11 | 0 | 20 |

- Average gold labels/review: 3.44
- Average predicted labels/review: 4.89
- Over-predicting reviews: 64 / 132
- Under-predicting reviews: 42 / 132
- Perfect full predictions: 12 / 132
- Zero-F1 reviews: 27 / 132

Main diagnosis: the prompt lets the model enumerate plausible restaurant aspects too freely. It frequently outputs 10-11 labels, which is almost never the gold shape. We need a stricter prompt that says to only include explicitly expressed aspects, not generally relevant restaurant categories.

The most damaging subtype is all-aspect output with `neutral` for missing categories. For example, one negative English review produces almost every aspect, with `restaurant_prices`, `food_prices`, `drinks_quality`, `drinks_prices`, `drinks_style_options`, and `location` marked neutral even though the review does not evaluate those aspects.

## By Language

| language | n | P_macro | R_macro | F1_macro | F1_micro | predicted | gold | ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 41 | 44.5 | 65.1 | 49.1 | 50.1 | 270 | 169 | 110 |
| es | 91 | 48.7 | 51.7 | 46.0 | 44.5 | 376 | 285 | 147 |

English is slightly better than Spanish in this run, but the English subset is small and both languages are below the top3 baseline.

## By Aspect

| aspect | pred | gold | ok | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| food_quality | 97 | 100 | 72 | 74.2 | 72.0 | 73.1 |
| restaurant_general | 90 | 128 | 74 | 82.2 | 57.8 | 67.9 |
| service | 75 | 82 | 53 | 70.7 | 64.6 | 67.5 |
| ambience | 58 | 47 | 20 | 34.5 | 42.6 | 38.1 |
| drinks_quality | 47 | 10 | 7 | 14.9 | 70.0 | 24.6 |
| food_prices | 41 | 26 | 8 | 19.5 | 30.8 | 23.9 |
| restaurant_prices | 53 | 24 | 9 | 17.0 | 37.5 | 23.4 |
| food_style_options | 56 | 24 | 8 | 14.3 | 33.3 | 20.0 |
| drinks_prices | 44 | 3 | 3 | 6.8 | 100.0 | 12.8 |
| drinks_style_options | 44 | 6 | 3 | 6.8 | 50.0 | 12.0 |
| location | 41 | 4 | 0 | 0.0 | 0.0 | 0.0 |

The model is useful on frequent obvious aspects, especially `restaurant_general`, `service`, and `food_quality`, but weak on price/style/location/drinks. The long tail gets many false positives and unstable polarity.

## By Polarity

| polarity | pred | gold | ok | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| positive | 304 | 326 | 203 | 66.8 | 62.3 | 64.4 |
| negative | 104 | 103 | 53 | 51.0 | 51.5 | 51.2 |
| neutral | 234 | 15 | 1 | 0.4 | 6.7 | 0.8 |
| conflict | 4 | 10 | 0 | 0.0 | 0.0 | 0.0 |

Positive is much easier because the dataset is heavily positive. Negative has reasonable recall but poor precision. Neutral/conflict are essentially not handled yet.

A key failure mode is visible here: the model often uses `neutral` to mean "not mentioned". In this task that is wrong. If an aspect is not explicitly evaluated in the review, it must be omitted from the JSON instead of emitted as neutral.

## Frequent Aspect-Polarity Pairs

| pair | pred | gold | ok | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| restaurant_general:positive | 67 | 95 | 61 | 91.0 | 64.2 | 75.3 |
| food_quality:positive | 69 | 82 | 62 | 89.9 | 75.6 | 82.1 |
| service:positive | 52 | 62 | 42 | 80.8 | 67.7 | 73.7 |
| ambience:positive | 28 | 35 | 17 | 60.7 | 48.6 | 54.0 |
| restaurant_general:negative | 13 | 25 | 12 | 92.3 | 48.0 | 63.2 |
| service:negative | 13 | 18 | 11 | 84.6 | 61.1 | 71.0 |
| restaurant_prices:negative | 11 | 13 | 6 | 54.5 | 46.2 | 50.0 |
| food_style_options:positive | 23 | 13 | 6 | 26.1 | 46.2 | 33.3 |
| food_prices:negative | 7 | 13 | 5 | 71.4 | 38.5 | 50.0 |
| food_quality:negative | 18 | 12 | 10 | 55.6 | 83.3 | 66.7 |
| food_prices:positive | 17 | 11 | 3 | 17.6 | 27.3 | 21.4 |
| food_style_options:negative | 10 | 9 | 2 | 20.0 | 22.2 | 21.1 |
| ambience:negative | 10 | 9 | 3 | 30.0 | 33.3 | 31.6 |
| restaurant_prices:positive | 12 | 9 | 3 | 25.0 | 33.3 | 28.6 |
| drinks_quality:positive | 14 | 8 | 5 | 35.7 | 62.5 | 45.5 |
| drinks_style_options:positive | 7 | 5 | 2 | 28.6 | 40.0 | 33.3 |
| restaurant_general:neutral | 8 | 5 | 1 | 12.5 | 20.0 | 15.4 |
| location:positive | 7 | 4 | 0 | 0.0 | 0.0 | 0.0 |
| food_quality:conflict | 0 | 4 | 0 | 0.0 | 0.0 | 0.0 |
| restaurant_general:conflict | 2 | 3 | 0 | 0.0 | 0.0 | 0.0 |

## False Positives

Most common exact false positives:

| pair | count |
| --- | --- |
| drinks_style_options:neutral | 32 |
| location:neutral | 29 |
| restaurant_prices:neutral | 29 |
| drinks_quality:neutral | 29 |
| drinks_prices:neutral | 28 |
| food_style_options:neutral | 23 |
| ambience:neutral | 20 |
| food_style_options:positive | 17 |
| food_prices:neutral | 16 |
| food_prices:positive | 14 |
| ambience:positive | 11 |
| service:positive | 10 |

Most common false-positive aspects:

| aspect | count |
| --- | --- |
| food_style_options | 48 |
| restaurant_prices | 44 |
| location | 41 |
| drinks_prices | 41 |
| drinks_style_options | 41 |
| drinks_quality | 40 |
| ambience | 38 |
| food_prices | 33 |
| food_quality | 25 |
| service | 22 |
| restaurant_general | 16 |

## False Negatives

Most common exact false negatives:

| pair | count |
| --- | --- |
| restaurant_general:positive | 34 |
| service:positive | 20 |
| food_quality:positive | 20 |
| ambience:positive | 18 |
| restaurant_general:negative | 13 |
| food_prices:positive | 8 |
| food_prices:negative | 8 |
| restaurant_prices:negative | 7 |
| food_style_options:positive | 7 |
| food_style_options:negative | 7 |
| service:negative | 7 |
| ambience:negative | 6 |

Most common missed aspects:

| aspect | count |
| --- | --- |
| restaurant_general | 54 |
| service | 29 |
| food_quality | 28 |
| ambience | 27 |
| food_prices | 18 |
| food_style_options | 16 |
| restaurant_prices | 15 |
| location | 4 |
| drinks_quality | 3 |
| drinks_style_options | 3 |

## Raw / Parser Health

| check | value |
| --- | --- |
| missing raw_generation | 0 |
| empty raw_generation | 0 |
| raw starts with JSON | 132 |
| raw contains JSON-like braces | 132 |
| raw contains markdown fence | 0 |
| raw contains think tags | 0 |
| empty parsed prediction | 14 |
| empty parsed prediction but non-empty gold | 14 |
| raw chars p50 | 162 |
| raw chars p95 | 342 |
| raw chars max | 717 |

Parser health is good enough for this run: raw outputs are present and mostly JSON-like. The main problem is not parsing; it is model behavior, especially over-prediction and polarity/aspect selection.

## Worst Examples

| id | lang | F1 | pred_n | gold_n | false positives | false negatives | text |
| --- | --- | --- | --- | --- | --- | --- | --- |
| es_cullera_de_boix_90_EvaGonzalez_2011-03-25 | es | 0.0 | 0 | 1 | - | restaurant_general:positive | sorpresa agradable. |
| es_colette-zaragoza_comment-2660 | es | 0.0 | 1 | 0 | service:neutral | - | Que en alusion a los comentarios de alguien que se hace pasar por mi persona utilizando mi nombre y haciend... |
| es_cullera_de_boix_110_ElDefensor_2008-08-15 | es | 0.0 | 1 | 0 | ambience:negative | - | para meri,no te pongas asi mujer,que no es para tanto,seguro que tu eres de las que te gusta que te hagan l... |
| es_matilde-la-zaragoza_comment-4338 | es | 0.0 | 0 | 1 | - | restaurant_general:positive | Genial!!Todo estuvo muy bien |
| es_kupela-la-sidreria-zaragoza_comment-3017 | es | 0.0 | 0 | 2 | - | food_quality:positive, restaurant_general:positive | Un restaurante exceccional, con una calidad de productos impresionantes, el pescado recien traido de San SE... |
| es_cullera_de_boix_72_RicardGrane_2012-02-27 | es | 0.0 | 0 | 2 | - | food_quality:neutral, restaurant_general:positive | Esperaba un poco más del arroz, por lo demás, todo muy correcto. |
| es_balmes_rossello_58_RaulHervas_2014-01-26 | es | 0.0 | 1 | 1 | restaurant_general:positive | restaurant_general:neutral | Restaurante para no tirar cohetes pero en general sales satisfecho. |
| es_cullera_de_boix_48_Anna_2013-02-12 | es | 0.0 | 0 | 3 | - | food_quality:negative, restaurant_general:negative, service:positive | Personal amable pero la comida muy floja.Sin ninguna gracia, arroz poco sabroso.No lo recomendaría!!! |
| es_maur_muntaner_3_CarlosReinleinFarre_2015-03-16 | es | 0.0 | 0 | 3 | - | food_quality:positive, restaurant_general:positive, service:positive | este domingo 15 de marzo me desplace desde galicia con unos amigos para competir en la maraton de bcn.Como ... |
| es_luis-candelas-zaragoza_comment-4780 | es | 0.0 | 1 | 2 | food_quality:negative | food_quality:neutral, restaurant_general:positive | Todo muy bien, hubiera sido perfecto si alguno de los chuletones estuviera mas en su punto, pues nos lo sir... |
| 447697 | en | 0.0 | 0 | 3 | - | food_quality:positive, restaurant_general:positive, restaurant_prices:positive | Great sushi experience.Nice value.Unique apppetizers.Try sushimi cucumber roll. |
| es_colette-zaragoza_comment-4024 | es | 0.0 | 0 | 3 | - | ambience:negative, restaurant_general:negative, service:negative | Sábado por la noche,  reservo para 4 personas en el colette, y al llegar no sólo no tenemos reserva sino qu... |

## Strong Examples

| id | lang | gold_n | text |
| --- | --- | --- | --- |
| es_l_olive_33_Pepe_2011-04-24 | es | 3 | Me ha causado una gratisima impresión, tanto en la calidad de sus productos como en el exquisito trato de s... |
| es_salamanca_36_Jenny_2013-02-28 | es | 3 | Muy extrañada con los comentarios abajo, este restaurante es uno de mis favoritos, siempre que vienen amigo... |
| 1543345 | en | 3 | I can't believe people complain about no cheese sticks?Who has room for Cheesesticks with the best pizza in... |
| es_9reinas_36_LuisAlbertoReyesFigueroa_2013-03-24 | es | 2 | La mejor carne que he comido en Barcelona. |
| es_9reinas_58_NuriaCasas_2011-09-17 | es | 2 | Comida y Servicio estupendo |

## Conclusions

1. The current zero-shot prompt is a valid technical baseline, but it is below the simple `top3_positive` baseline. We should not move to fine-tuning from this prompt yet.
2. The failure mode is clear: Qwen often predicts many plausible categories instead of only the aspects explicitly supported by the review.
3. The next prompt iteration should explicitly penalize over-prediction: include an aspect only if the text directly evaluates it; do not infer price, ambience, drinks, style, or location unless mentioned.
4. Add 2-3 minimal counterexamples in the prompt or few-shot pool where short positive reviews only output `restaurant_general`/`service`, not every frequent aspect.
5. Before testing thinking mode, improve no-thinking prompt v2. Thinking will not fix over-prediction by itself and may increase verbosity/loop risk.
6. Keep `max_new_tokens=512` for debugging raw outputs. Once prompt is stable and outputs remain compact JSON, reduce to 128 or 256 for no-thinking runs.

## Next Experiment

Recommended next run: `prompt_v2` no-thinking on `devel`, same decoding params, same `--keep-raw`. Success criterion: beat `top3_positive` macro F1 55.6 and reduce predicted labels/review closer to the gold average of 3.44.
