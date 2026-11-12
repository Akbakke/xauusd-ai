# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_ACTIVITY_SHORT_EFFECT_VERIFIED_RESIDUAL_CALIBRATION_LOCALIZED.
Målet11.11 er aktivt: avklar målte årsaker til svak retningsverdi i indikatorer,
sammensetting og læring. Korte runder er bevisst feilsøking, ikke kapasitetsdom.
Full kvalifisert TRAIN-historikk og30–40epoker er senere skalering.
Alle tidligere engangsplaner er konsumert; nyeste er COMPLETE/CONSUMED. Training,
native_training/native_launch/full_epoch er false. GX1TFActivityShortLearning001,
GX1NonM5ShortLearning001 og begge c171-oppgavene er Disabled.
TEST, Entry/Exit-evaluering, broker, handel, spending og promotion stengt; GC på pause.
Ingen ny modell-/treningsjobb er innrømmet. Ingen operasjonell blokkering.

## Indikatorer og sammensetting

Alle8genuine TAfamilier og M5/M15/H1/H4/D1 når faktisk modellkonsum, med makrokontekst.
Numerisk dekning, varierende TRAIN-støtter og oppdatering i11encodere/228tensorer
beviser mekanikk, ikke prediktiv familienytte. Mean+last kansellerer ikke generelt;
fast MTF-intervensjon ga ingen bedre rangering. Trendlinjeslots mangler entydig
samme-linje-identitet/ankere/endringsmarkør; prediktiv kostnad er UNKNOWN.
Alle genuine features/TF og kausalitetsvakter er bevart. Ingen generell normaliserings-
eller konstantinputfeil er vist. [Samlet oversikt](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).

Tidligere felles nonM5-sideforskyvning ble målt. Separate sidehoder endret læringen,
men ga ingen ekstra M5gevinst og forverret aktivitetstapet. Retained-gradientkontroll
bekreftet motstridende TF-bidrag til den delte aktivitetsraden; ingen tapsformelfeil.
Sparse batchvekter blåser ikke opp D1s optimale konstante aktivitetssannsynlighet.
Se [sideforsøket](docs/TA_NON_M5_SHORT_LEARNING_REVIEW.json) og
[aktivitetskontrollen](docs/TA_ACTIVITY_PRESSURE_REVIEW.json).

## Nyeste faktiske læringsmåling

[Resultat, alle10celler og klasse-/kvartalsregnskap](docs/TA_TF_ACTIVITY_SHORT_LEARNING_REVIEW.json).
ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_SHORT_LEARNING_20261112_001.
Kilde7dcddb37586cdcf297ffac2e8b1f643912f4c50a uendret under kjøring; boot642.
RECIPE4ae0082f, PLAN63b7702e, faktisk vindua4d7d93e, RESULTaca65dc4,
campaignterminala2f80817, cursor2f7b7593. COMPLETE/CONSUMED, alle exit0.

Samme4096kronologiske TRAIN-rader/256updates som originalshort002,1.–21.mai2024.
Originale startvekter og INITIAL-/FINAL-trenings-RNG, fersk tom AdamW, uendrede
hyperparametre/features/targets/tapsvekter. Bare ti separate aktivitetshoder på
M5/M15/H1/H4/D1 forS/F:20klonede tensorer/1290parametre. Separate nonM5-sidehoder av.
Hele4096initial før første update: akser/targets/alle sidekanaler byteeksakte;
aktivitetsavvik maksS9,5367e-7/F2,3842e-7, tapsavvik2,3842e-7 innen faste toleranser.
Hele4096målt igjen final.256initial+256train+256final=768forwards,0skips.
Worker948,98s; faktisk fit350,19s. Uendret20GiB/512MiB,CPU0–18/pids128,8tråder,
250Wlimit/310Wdraw/85Ccore/80Cmemory/GPU12288MiB;1800smodell/3600svindu/3900sytre.
792telemetripunkter, topper55C/64C/133,59W/2172MiB. Ingen vaktbrudd.

| Samme brukte TRAIN | Initial | Original final | Sidehoder final | Aktivitetshoder final |
| --- | ---: | ---: | ---: | ---: |
| Opprinnelig totaltap | 1,878003 | 1,892871 | 1,884922 | 1,884324 |
| S/M5 side-AUC | 0,546185 | 0,551466 | 0,551231 | 0,551352 |
| F/M5 side-AUC | 0,546125 | 0,553079 | 0,553037 | 0,553147 |
| S/M5 side-Brier | 0,247999 | 0,247924 | 0,248301 | 0,248027 |
| F/M5 side-Brier | 0,248788 | 0,248193 | 0,248318 | 0,248218 |

AUC er rangering, ikke treffprosent eller økonomi. Alle tre totaltap forverres fra
initial. Aktivitetsvarianten gir mindre forverring enn original med0,00854651 og
sidevarianten med0,00059749. Ingen entydig samlet M5gevinst mot originalfinal:
S/M5AUC litt lavere, F/M5AUC litt høyere; beggeM5Brier/log litt dårligere.
Aktivitets-Brier bedre enn originalfinal i alle10celler, aktivitets-log i9av10.

Δoriginalt tap for aktivitetsvarianten=+0,006321593: aktivitet+0,000967350 og
side+0,005354243. Netto aktivitetsregresjon er87,23% mindre enn originalens,
men ingen87% forbedring i prediksjon eller edge. Sidevarianten hadde i stedet
aktivitet+0,013960765 ogside−0,007041688. Separate armer beviser ikke additiv gevinst
fra en kombinert modell. Første to kronologiske kvartaler forverres, siste to bedres.

## Gjenværende målt feil og neste kontroll

S/D1aktivitet:2011AKTIV-/215FLAT-rader. AKTIV-bidraget bedres−0,020664163,
FLAT forverres+0,066652219 før/5; netto+0,009197611 i totaltapet.
På faktiskFLAT øker P(aktivitet)88,49→93,35%. Q2 rommer207av215FLAT-rader:
observert aktivandel68,87%, mens gjennomsnittlig P(aktivitet)89,82→94,00%.
Q2forverres, de tre andre delene bedres. F/M15-H1-side trekkes også opp i alle
kvartaler; Q1–Q3ender72–77%UP mot51–57%observertUP, med verre Brier/log.

S/D1s isolerte eval-biasderivert med original/5 er+0,007071016initial og
+0,011873531final: gjennomsnittstapet ønsker lavere logit. Samtidig har175/174batcher
negativ derivert,49/50positiv og32tom. Færre positive bidrag veier mer samlet.
Dette er eval-endepunkter, ikke de historiske treningsgradientene eller AdamW-stegene.

Neste avgrensede arbeid er retained rekonstruksjon av S/D1-aktivitetsbiasens
treningsderivert/klipping og AdamW-retning. TRACE har alleTFrawlogits[256,16,5,2],
masker og256gradientnormer; direkte parametergradienter/momenter gjelder M5side.
Kvalifiser aritmetikken mot disse fangede M5gradientene og eksisterende retained
optimizer-replay før D1tolkning. Skill gjennomsnittlig tapspress fra oppdateringsretning
og felles representasjon; ikke velg en ny arkitektur på antagelse. Ingen ny modellfit.

## Kvalifisering og ubeviste grenser

302fokuserte tester og obligatoriske commit-vakter bestod før launch.92kildeeiere
var uendret etter trygg fysisk boot641→642. Preflight og faktisk immutablevindu
hadde likt innhold, ulik JSON-formatering; faktiske bytes fikk egen cap-/oppstartskontroll
før claim. Ingen endring av plan/kilde. Originale feilkvitteringer er bevart.

Uavhengig læringskontrollf79b828f:5NPZ/60akse-targetfelt/600scoreceller/6tapsvektorer,
0modeller/checkpoints/forwards/updates. Maks replayavvik total3,90e-7/komponent4,64e-7.
S/D1klassekontroll7ccf551f:2NPZ/0modeller, originalB16-nevner/tommebatcher og/5;
konserveringsavvik2,22e-16. RNG/checkpoint-identitet bygger på bundne nativekvitteringer.
Windowsoppgaven er lesende bekreftet Disabled/LastTaskResult0 etter vellykket disable;
kvitteringsformattering av tom NextRunTime feilet først, uten å endre treningsresultat.

Brukt avhengig TRAIN og ett seed/256updates. Ingen signifikans, senere generalisering,
kapasitetsdom, økonomisk edge eller promotion er etablert. Familienytte og trendlinje-
identitet er fortsatt åpne spørsmål; fullhistorikk/30–40epoker krever senere kvalifisering.
