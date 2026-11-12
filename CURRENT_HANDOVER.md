# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_MATCHED_HORIZON_EFFECT_MIXED_ACTIVITY_REGRESSION_LOCALIZED.
Målet11.11 er aktivt: avklar målte årsaker til svak retningsverdi i indikatorer,
sammensetting og læring. [Samlet status](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte runder er bevisst feilsøking, ikke kapasitetsdom. Full kvalifisert TRAIN-historikk
og30–40epoker er senere skalering, ikke et krav ved hver rettelse.
Alle engangsplaner er konsumert. Training/native_training/native_launch/full_epoch er false.
Ny GX1NonM5ShortLearning001 og begge gamle c171 Windowsoppgaver er Disabled.
Ingen ny native kjøring er innrømmet. GC på pause; TEST, Entry/Exit-evaluering, broker,
handel, spending og promotion stengt. Checkpoints er forskningsevidens, ikke promotion.

## Indikatorer og sammensetting

Alle8TAfamilier og M5/M15/H1/H4/D1 når faktisk modellkonsum, med makrokontekst.
Numerisk dekning/ikke-konstante TRAIN-støtter og oppdatering i11encodere/228tensorer
beviser mekanikk, ikke prediktiv familienytte. Mean+last kansellerer ikke systematisk;
fast MTF-intervensjon ga ingen bedre rangering. Trendlinjeslots mangler entydig
samme-linje-identitet/ankere/endringsmarkør; prediktiv kostnad er UNKNOWN.
Alle genuine features/TF er bevart. Ingen generell normaliserings-/konstantinputfeil er vist.

Tidligere matched-mode og decoderkontroll fant nesten felles nonM5-sideendring og
følsomhet. Bias alene forklarte ikke forskyvningen; mye endring ligger før sluttlaget.
Se [matched-mode](docs/TA_C171_MATCHED_TRAIN_MODE_REVIEW.json),
[faktorregnskap](docs/TA_C171_ALL_TF_TAIL_FACTOR_REVIEW.json) og
[decoderkontroll](docs/TA_C171_DECODER_CONDITIONING_REVIEW.json).
Minste opt-in forsøk:8uavhengige nonM5-sidehoder,16tensorer/1032parametre, klonet
fra originalen. M5, aktivitet, backbone/query, targets, tapsvekter og numeriske
innstillinger beholdes. Standardmodellen er fortsatt uendret.
[Implementasjon](docs/TA_NON_M5_SIDE_DECODER_REPAIR.json).

## Siste faktiske trening er fullført

[Fullt resultat og alle10oppgave/TF-celler](docs/TA_NON_M5_SHORT_LEARNING_REVIEW.json).
ROOT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_NON_M5_SHORT_LEARNING_20261112_001.
Kilde e28ed537836ae68fb3d6180a531c2cffe5cb3ad0 uendret under kjøring; boot641.
RECIPE0dfe1a15, PLAN0622b877, RESULTbabe029c, campaignterminal9dc798ab,
cursorce528a1a. COMPLETE/CONSUMED; guard/progress-observer/task exit0.
348fokuserte tester og obligatoriske commit-vakter bestod før launch.

Samme4096kronologiske TRAIN-rader/256updates som originalshort002,1.–21.mai2024.
Fersk tom AdamW, samme hyperparametre og originalt tap.256initial+256train+256final=
768forwards. Original INITIAL-/FINAL-trenings-RNG matchet.0skips.
Før første update stemte alle4096akser/targets/M5/aktivitetslogits byteeksakt;
nonM5-sideavvik maks3,5763e-7 og tapsavvik2,3842e-7 innen faste toleranser.
Alle4096rader ble målt igjen etter trening. Den gamle48raders paneloverlappen avgjør ikke resultatet.
Worker962,62s. Uendret trainer20GiB/512MiB,CPU0–18/pids128,8tråder,
250Wlimit/310Wdraw/85Ccore/80Cmemory/GPU12288MiB;1800smodell/3600svindu/3900sytre.
Målte topper55Ccore/64Cmemory/136,16W/2172MiB. Ingen vaktbrudd.

## Hva sammenligningen faktisk viser

| Mål på samme brukte TRAIN | Initial | Original final | Uavhengige sidehoder final |
| --- | ---: | ---: | ---: |
| Originalt samlet tap | 1,878003 | 1,892871 | 1,884922 |
| S/M5 AUC | 0,546185 | 0,551466 | 0,551231 |
| F/M5 AUC | 0,546125 | 0,553079 | 0,553037 |
| S/M5 Brier | 0,247999 | 0,247924 | 0,248301 |
| F/M5 Brier | 0,248788 | 0,248193 | 0,248318 |

AUC er rangering, ikke treffprosent/økonomi. Samlet tap blir mindre forverret enn
originalen, men beggeM5 er svakere enn originalfinal på Brier/log/AUC.
S/H4,D1 og F/M15,H1,D1 har bedre Brier/log enn originalfinal; S/M5,M15,H1 og
F/M5,H4 dårligere. F/M15,H1 blir fortsatt dårligere enn egen initial:
P(UP)68,01%/69,31% mot observert52,08%/53,97%. Alle10AUC øker fra initial,
men dette er avhengige brukte TRAIN-data, ingen senere generalisering.

Originalt tap øker0,00691908 mot originalens0,01486811. Retning bidrar−0,00704169,
aktivitet+0,01396077. S/D1 aktivitet bidrar+0,01091411,78,18% av NETTO aktivitetsøkning;
regnskapsandel, ikke kausal gradientandel. P(aktivitet) stiger91,37→94,56% mot90,34%
aktive labels på2226informative S/D1-rader. Første to kronologiske kvartaler forverres
fortsatt, siste to forbedres; alle fire har bedre totalendring enn originalen.

Felles nonM5-sideendringsenergi som andel: S99,92→77,85%,F99,97→97,00%.
F sin absolutte felles energi øker; lavere brøk beviser ikke mindre skadelig overføring.
Endringen påvirker læringen, men er ikke en ferdig løsning for M5 eller bevist edge.

Uavhengig analyse c1913ebd bruker3NPZ/0modeller/checkpoints/updates; tapsreplayavvik
maks4,64e-7. Retained aktivitetsanalyse c772f0291 bruker3NPZ/0modeller og bekrefter
alle40fullprefix-retnings-AUC med annen parvis metode, eksakt avvik0,0.
Originale råkvitteringer/checkpoints og prelaunch-feil er bevart. Ingen ny modelljobb
ble kjørt for resultatregnskapet. Høy D1 TRAIN-AUC er ingen handels-/kapasitetsdom.

## Neste konkrete avklaring

Bruk beholdte outputs og TRAIN-trace til å isolere felles aktivitetsskift og original
vekting av batcher med sparsom støtte, først S/D1, samt vedvarende F/M15-H1 UP-skjevhet.
Mål peroppgave/TF-trykk før ny modell-/tapsendring. Ikke fjern indikatorer eller D1,
ikke velg nye tap eller treningsmengde fra denne ene runden. Ingen ny native plan er
innrømmet. Full historikk/30–40epoker vurderes senere når grunnlaget er kvalifisert.
