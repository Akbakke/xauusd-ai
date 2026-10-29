# Fire eldre kronologiske retningsmodeller og D1-dekning

De fire allerede trente trehodersmodellene er målt mot korrigert fasit i hver sin Q1–Q4 2025-utviklingsperiode. Dette er beskrivende gjenbruk, ikke en ren test av årsak eller en ny uberørt evaluering. Modellene ble trent på den opprinnelige flyttallsfasiten. Ingen modellforward eller trening ble kjørt i denne kontrollen.

| Senere kvartal | TRAIN-beslutninger | Snitt D1-GC-dager i TRAIN / 252 | Snitt D1-GC-dager i senere periode / 252 | 5 min treff | 5 min alltid opp | 5 min AUC |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 47,602 | 82.89 | 199.39 | 48.848 % | 51.582 % | 0.502978 |
| Q2 | 64,890 | 113.93 | 244.70 | 50.367 % | 50.682 % | 0.507236 |
| Q3 | 82,427 | 141.75 | 247.09 | 50.626 % | 50.832 % | 0.499927 |
| Q4 | 100,499 | 160.69 | 247.06 | 51.322 % | 51.322 % | 0.505880 |

Mer komplett D1-GC-historikk sammenfaller med voksende treningsprefiks, andre modeller og ulike markedsperioder. Tallene kan derfor ikke isolere effekt av dekning eller ekstra trening. D1-tallet gjelder observert GC-volum i et 252-dagers vindu, ikke hele D1 eller makro. Resultatene beviser verken feilårsak, generalisering eller økonomisk edge. Q4-modellen predikerte opp på alle kvalifiserte femminuttersrader.

En uavhengig avspilling med sklearn for 1/5/15 minutter og direkte summasjon av de 252 D1-maskeradene gjenskapte alle fire kvartaler; største absolutte differanse var 1,11e-16. TEST og handel ble ikke brukt.

RESULT.json: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CHRONOLOGICAL_COVERAGE_REVIEW_20261030_001/OUTPUT/RESULT.json` SHA-256 `cf6723204e521ee0e985064ce5161e2b8dd00d567e44754efb0c326490c7a328`.
TERMINAL.json: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CHRONOLOGICAL_COVERAGE_REVIEW_20261030_001/OUTPUT/TERMINAL.json` SHA-256 `85bdd15c1daf2e246619f6265f33d81be3440219e35a10e15f54ef6fea74b658`.
INDEPENDENT_REPLAY.json: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CHRONOLOGICAL_COVERAGE_REVIEW_20261030_001/OUTPUT/INDEPENDENT_REPLAY.json` SHA-256 `2fb2b080ef78e7a52e2e57d4a8ca47b5ba9bcc8db387e58328e36d2c9fa8612f`.
