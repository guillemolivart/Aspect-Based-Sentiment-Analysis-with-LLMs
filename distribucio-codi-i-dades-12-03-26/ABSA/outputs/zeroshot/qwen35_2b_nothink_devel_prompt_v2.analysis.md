# Zero-Shot Report: Qwen3.5-2B No-Thinking Prompt v2

## Run

- Output JSON: `distribucio-codi-i-dades-12-03-26/ABSA/outputs/zeroshot/qwen35_2b_nothink_devel_prompt_v2.json`
- Stats: `distribucio-codi-i-dades-12-03-26/ABSA/outputs/zeroshot/qwen35_2b_nothink_devel_prompt_v2.stats.txt`
- Run log: `distribucio-codi-i-dades-12-03-26/ABSA/outputs/zeroshot/qwen35_2b_nothink_devel_prompt_v2.run.log`
- Summary CSV: `distribucio-codi-i-dades-12-03-26/ABSA/outputs/zeroshot/summary.zero_shot.csv`
- Examples: 132 (`devel`)
- Runtime: 177.3 s total, 1.34 s/example
- Model loading: 6.0 s
- Generation config: `{'temperature': 1.0, 'top_p': 1.0, 'top_k': 20, 'min_p': 0.0, 'repetition_penalty': 1.0, 'max_new_tokens': 512}`
- Mode: no-thinking. `--keep-raw` was enabled.
- Prompt change: v2 explicitly forbids neutral-as-absence, requires quoted JSON strings, and clarifies when to emit `restaurant_general`.
- Parser change used in this run: common malformed JSON with unquoted polarity values is repaired before normalization.

## Headline Result

| system | P_macro | R_macro | F1_macro | P_micro | R_micro | F1_micro | predicted | gold | ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35_2b_nothink_devel_prompt_v1 | 47.4 | 55.8 | 46.9 | 39.8 | 56.6 | 46.7 | 646 | 454 | 257 |
| qwen35_2b_nothink_devel_prompt_v2 | 66.3 | 50.2 | 53.9 | 63.1 | 49.3 | 55.4 | 355 | 454 | 224 |
| oracle_aspects_majority_polarity | 72.6 | 72.6 | 72.6 | 71.8 | 71.8 | 71.8 | 454 | 454 | 326 |
| top3_positive | 60.4 | 56.8 | 55.6 | 60.4 | 52.6 | 56.2 | 396 | 454 | 239 |
| top3_majority | 60.4 | 56.8 | 55.6 | 60.4 | 52.6 | 56.2 | 396 | 454 | 239 |
| top4_positive | 51.9 | 63.4 | 54.5 | 51.9 | 60.4 | 55.8 | 528 | 454 | 274 |
| top4_majority | 51.9 | 63.4 | 54.5 | 51.9 | 60.4 | 55.8 | 528 | 454 | 274 |

Prompt v2 is a clear improvement over v1: macro F1 rises from 46.9 to **53.9**, and micro F1 rises from 46.7 to **55.4**. It still does not beat the strongest legal baseline, `top3_positive` at 55.6 macro F1, but it is now close enough that a small prompt/few-shot or decoding improvement may pass it.

## Label Count Shape

| labels/review | gold reviews | predicted reviews |
| --- | --- | --- |
| 0 | 3 | 4 |
| 1 | 13 | 44 |
| 2 | 14 | 20 |
| 3 | 37 | 28 |
| 4 | 35 | 16 |
| 5 | 21 | 10 |
| 6 | 7 | 4 |
| 7 | 1 | 3 |
| 8 | 1 | 0 |
| 9 | 0 | 2 |
| 10 | 0 | 1 |

- Average gold labels/review: 3.44
- Average predicted labels/review: 2.69
- Over-predicting reviews: 22 / 132
- Under-predicting reviews: 81 / 132
- Perfect full predictions: 16 / 132
- Zero-F1 reviews: 20 / 132

The main improvement is precision: v1 predicted 646 labels, v2 predicts 355 labels, much closer to the 454 gold labels. The cost is lower recall, so the next step should recover common missing positives without reopening the all-category neutral failure.

## By Language

| language | n | P_macro | R_macro | F1_macro | F1_micro | predicted | gold | ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 41 | 65.7 | 53.4 | 55.3 | 57.2 | 142 | 169 | 89 |
| es | 91 | 66.6 | 48.8 | 53.2 | 54.2 | 213 | 285 | 135 |

## By Aspect

| aspect | pred | gold | ok | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| restaurant_general | 117 | 128 | 99 | 84.6 | 77.3 | 80.8 |
| service | 55 | 82 | 45 | 81.8 | 54.9 | 65.7 |
| food_quality | 59 | 100 | 49 | 83.1 | 49.0 | 61.6 |
| restaurant_prices | 27 | 24 | 10 | 37.0 | 41.7 | 39.2 |
| ambience | 23 | 47 | 11 | 47.8 | 23.4 | 31.4 |
| drinks_quality | 12 | 10 | 3 | 25.0 | 30.0 | 27.3 |
| food_prices | 15 | 26 | 3 | 20.0 | 11.5 | 14.6 |
| drinks_style_options | 8 | 6 | 1 | 12.5 | 16.7 | 14.3 |
| location | 14 | 4 | 1 | 7.1 | 25.0 | 11.1 |
| food_style_options | 20 | 24 | 2 | 10.0 | 8.3 | 9.1 |
| drinks_prices | 5 | 3 | 0 | 0.0 | 0.0 | 0.0 |

The frequent aspects are now strong: `food_quality`, `restaurant_general`, and `service` are useful. The long-tail aspects remain unstable, especially `location`, drinks, and style/options categories.

## By Polarity

| polarity | pred | gold | ok | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| positive | 230 | 326 | 181 | 78.7 | 55.5 | 65.1 |
| negative | 75 | 103 | 43 | 57.3 | 41.7 | 48.3 |
| neutral | 50 | 15 | 0 | 0.0 | 0.0 | 0.0 |
| conflict | 0 | 10 | 0 | 0.0 | 0.0 | 0.0 |

The neutral-as-absence problem is reduced substantially: neutral predictions drop from 234 in v1 to 50 in v2. That is still high compared with only 15 neutral gold labels, but it is no longer dominating the output.

## Frequent Aspect-Polarity Pairs

| pair | pred | gold | ok | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| restaurant_general:positive | 89 | 95 | 84 | 94.4 | 88.4 | 91.3 |
| food_quality:positive | 43 | 82 | 41 | 95.3 | 50.0 | 65.6 |
| service:positive | 40 | 62 | 34 | 85.0 | 54.8 | 66.7 |
| ambience:positive | 15 | 35 | 10 | 66.7 | 28.6 | 40.0 |
| restaurant_general:negative | 22 | 25 | 15 | 68.2 | 60.0 | 63.8 |
| service:negative | 12 | 18 | 11 | 91.7 | 61.1 | 73.3 |
| restaurant_prices:negative | 10 | 13 | 5 | 50.0 | 38.5 | 43.5 |
| food_style_options:positive | 9 | 13 | 1 | 11.1 | 7.7 | 9.1 |
| food_prices:negative | 6 | 13 | 2 | 33.3 | 15.4 | 21.1 |
| food_quality:negative | 12 | 12 | 8 | 66.7 | 66.7 | 66.7 |
| food_prices:positive | 6 | 11 | 1 | 16.7 | 9.1 | 11.8 |
| ambience:negative | 2 | 9 | 1 | 50.0 | 11.1 | 18.2 |
| restaurant_prices:positive | 14 | 9 | 5 | 35.7 | 55.6 | 43.5 |
| food_style_options:negative | 4 | 9 | 1 | 25.0 | 11.1 | 15.4 |
| drinks_quality:positive | 7 | 8 | 3 | 42.9 | 37.5 | 40.0 |
| drinks_style_options:positive | 4 | 5 | 1 | 25.0 | 20.0 | 22.2 |
| restaurant_general:neutral | 6 | 5 | 0 | 0.0 | 0.0 | 0.0 |
| location:positive | 2 | 4 | 1 | 50.0 | 25.0 | 33.3 |
| food_quality:conflict | 0 | 4 | 0 | 0.0 | 0.0 | 0.0 |
| restaurant_general:conflict | 0 | 3 | 0 | 0.0 | 0.0 | 0.0 |

## False Positives

| pair | count |
| --- | --- |
| restaurant_prices:positive | 9 |
| food_style_options:positive | 8 |
| location:neutral | 8 |
| food_style_options:neutral | 7 |
| restaurant_general:negative | 7 |
| restaurant_general:neutral | 6 |
| service:positive | 6 |
| ambience:neutral | 6 |
| ambience:positive | 5 |
| drinks_quality:neutral | 5 |
| restaurant_prices:negative | 5 |
| food_prices:positive | 5 |

## False Negatives

| pair | count |
| --- | --- |
| food_quality:positive | 41 |
| service:positive | 28 |
| ambience:positive | 25 |
| food_style_options:positive | 12 |
| food_prices:negative | 11 |
| restaurant_general:positive | 11 |
| restaurant_general:negative | 10 |
| food_prices:positive | 10 |
| ambience:negative | 8 |
| food_style_options:negative | 8 |
| restaurant_prices:negative | 8 |
| service:negative | 7 |

## Parser / Raw Health

| check | value |
| --- | --- |
| missing raw_generation | 0 |
| empty raw_generation | 0 |
| raw starts with JSON | 132 |
| raw contains JSON-like braces | 131 |
| raw contains markdown fence | 0 |
| raw contains think tags | 1 |
| empty parsed prediction | 4 |
| empty parsed prediction but non-empty gold | 4 |
| raw chars p50 | 90 |
| raw chars p95 | 224 |
| raw chars max | 2378 |

## Biggest Improvements From v1

| id | lang | delta F1 | v1 F1 | v2 F1 | gold_n | v1 pred_n | v2 pred_n | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| es_matilde-la-zaragoza_comment-4338 | es | +100.0 | 0.0 | 100.0 | 1 | 0 | 1 | Genial!!Todo estuvo muy bien |
| es_cullera_de_boix_90_EvaGonzalez_2011-03-25 | es | +100.0 | 0.0 | 100.0 | 1 | 0 | 1 | sorpresa agradable. |
| es_cullera_de_boix_48_Anna_2013-02-12 | es | +100.0 | 0.0 | 100.0 | 3 | 0 | 3 | Personal amable pero la comida muy floja.Sin ninguna gracia, arroz poco sabroso.No lo reco... |
| es_colette-zaragoza_comment-574 | es | +85.7 | 0.0 | 85.7 | 4 | 0 | 3 | Unico en Zaragoza, Ya era hora!!!.Excelentes istalaciones, muy buena atención e inmejorabl... |
| 447697 | en | +85.7 | 0.0 | 85.7 | 3 | 0 | 4 | Great sushi experience.Nice value.Unique apppetizers.Try sushimi cucumber roll. |
| es_balmes_rossello_70_SoniaFernandezCaras_2013-04-29 | es | +83.3 | 16.7 | 100.0 | 1 | 11 | 1 | Recomendable. |
| es_cullera_de_boix_14_Ju_2014-12-14 | es | +75.0 | 0.0 | 75.0 | 5 | 0 | 3 | He ido varias veces a este restaurante por varios motivos: buena relación calidad precio, ... |
| 444818 | en | +75.0 | 0.0 | 75.0 | 4 | 0 | 4 | Seriously, this place kicks ass.The atmosphere is unheralded, the service impecible, and t... |

## Biggest Regressions From v1

| id | lang | delta F1 | v1 F1 | v2 F1 | gold_n | v1 pred_n | v2 pred_n | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| es_9reinas_58_NuriaCasas_2011-09-17 | es | -100.0 | 100.0 | 0.0 | 2 | 2 | 1 | Comida y Servicio estupendo |
| 1642666 | en | -66.7 | 66.7 | 0.0 | 2 | 1 | 0 | We ate at this Thai place following the reviews but very unhappy with the foods.We thought... |
| es_bodega-de-chema-la-zaragoza_comment-4292 | es | -57.1 | 57.1 | 0.0 | 3 | 4 | 0 | La comida la verdad muy buena pero la oferta q sale aqui luego no es la mismo de alli me s... |
| es_cafe_kafka_21_GuillermoFerrer_2013-10-02 | es | -52.4 | 85.7 | 33.3 | 4 | 3 | 2 | Tuvimos una experiencia magnífica, la atención inmejorable, las recomendaciones del encarg... |
| es_balmes_rossello_96_Marivi_2012-02-10 | es | -50.0 | 50.0 | 0.0 | 3 | 1 | 1 | Servicio muy bueno, quedamos contentos de la calçotada. |
| es_barceloneta_25_JoseOrtego_2013-10-27 | es | -46.4 | 75.0 | 28.6 | 5 | 3 | 9 | Una comida deliciosa.Excelente arroz y buen trato.La localizacion es muy buena.Si tienes o... |
| es_l_olive_11_ClaudiaVelutini_2014-01-01 | es | -45.7 | 85.7 | 40.0 | 4 | 3 | 1 | Su atención muy buena, la comida excelente, un restaurante muy bello. |
| 1131595 | en | -42.9 | 42.9 | 0.0 | 3 | 11 | 0 | This place is incredibly tiny.They refuse to seat parties of 3 or more on weekends.The hos... |

## Worst Remaining Examples

| id | lang | F1 | pred_n | gold_n | false positives | false negatives | text |
| --- | --- | --- | --- | --- | --- | --- | --- |
| es_colette-zaragoza_comment-2660 | es | 0.0 | 1 | 0 | restaurant_general:negative | - | Que en alusion a los comentarios de alguien que se hace pasar por mi persona utilizando mi nombre y ha... |
| es_cullera_de_boix_110_ElDefensor_2008-08-15 | es | 0.0 | 1 | 0 | restaurant_general:negative | - | para meri,no te pongas asi mujer,que no es para tanto,seguro que tu eres de las que te gusta que te ha... |
| es_bal-donsera-zaragoza_comment-2198 | es | 0.0 | 1 | 0 | restaurant_general:positive | - | Enhorabuena!!, ahora sí lo has entendido. |
| 1642666 | en | 0.0 | 0 | 2 | - | food_quality:negative, restaurant_general:negative | We ate at this Thai place following the reviews but very unhappy with the foods.We thought that this p... |
| es_bal-donsera-zaragoza_comment-1753 | es | 0.0 | 1 | 1 | restaurant_general:neutral | restaurant_general:positive | Yo comí este martes, y de maravilla; como siempre.Fiel a mi tradición en este restaurante, no pedí y d... |
| es_balmes_rossello_58_RaulHervas_2014-01-26 | es | 0.0 | 1 | 1 | restaurant_general:positive | restaurant_general:neutral | Restaurante para no tirar cohetes pero en general sales satisfecho. |
| 1131595 | en | 0.0 | 0 | 3 | - | food_quality:negative, restaurant_general:negative, service:negative | This place is incredibly tiny.They refuse to seat parties of 3 or more on weekends.The hostess is rude... |
| es_maur_muntaner_3_CarlosReinleinFarre_2015-03-16 | es | 0.0 | 0 | 3 | - | food_quality:positive, restaurant_general:positive, service:positive | este domingo 15 de marzo me desplace desde galicia con unos amigos para competir en la maraton de bcn.... |
| es_luis-candelas-zaragoza_comment-4780 | es | 0.0 | 1 | 2 | restaurant_general:neutral | food_quality:neutral, restaurant_general:positive | Todo muy bien, hubiera sido perfecto si alguno de los chuletones estuviera mas en su punto, pues nos l... |
| es_bodega-de-chema-la-zaragoza_comment-4292 | es | 0.0 | 0 | 3 | - | food_prices:negative, food_quality:positive, restaurant_general:conflict | La comida la verdad muy buena pero la oferta q sale aqui luego no es la mismo de alli me salio  5€ mas... |
| es_9reinas_58_NuriaCasas_2011-09-17 | es | 0.0 | 1 | 2 | restaurant_general:positive | food_quality:positive, service:positive | Comida y Servicio estupendo |
| es_balmes_rossello_96_Marivi_2012-02-10 | es | 0.0 | 1 | 3 | restaurant_general:negative | food_quality:positive, restaurant_general:positive, service:positive | Servicio muy bueno, quedamos contentos de la calçotada. |

## Conclusions

1. V2 is directionally correct: it fixes much of the over-prediction problem and gives a large precision gain.
2. V2 is not enough yet: it is still 1.7 macro F1 below `top3_positive`, mainly because recall is too low on common positive aspects.
3. The parser repair is worth keeping. It recovers common Qwen outputs with unquoted polarity values without accepting invalid aspect or polarity names.
4. The next prompt/few-shot iteration should add a small number of examples that teach two things at once: do not emit absent aspects as neutral, but do emit `restaurant_general`, `food_quality`, and `service` when explicitly praised or criticized.
5. For the next experiment, keep no-thinking mode and run a targeted few-shot/RAG selection before testing thinking. Thinking is unlikely to solve this extraction threshold issue by itself.
