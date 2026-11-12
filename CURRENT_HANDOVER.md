# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_ACTIVITY_INITIAL_COMPLETE_LEARNING_EFFECT_UNMEASURED.
Målet11.11 er aktivt: avklar målte årsaker til svak retningsverdi i indikatorer,
sammensetting og læring. [Samlet status](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte runder er bevisst feilsøking, ikke kapasitetsdom. Full kvalifisert TRAIN-historikk
og30–40epoker er senere skalering, ikke et krav ved hver rettelse.
Alle engangsplaner er konsumert. Training/native_training/native_launch/full_epoch er false.
Ny GX1NonM5ShortLearning001 og begge gamle c171 Windowsoppgaver er Disabled.
Aktivitetsinitialkontrollen er fullført; ingen ny native kjøring/trening er innrømmet. GC på pause; TEST, Entry/Exit-evaluering, broker,
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

## Retained aktivitetskontroll og neste rettede forsøk

[Ny kontroll](docs/TA_ACTIVITY_PRESSURE_REVIEW.json): fem beholdte NPZ,0modeller/checkpoints/
forwards/autograd/updates. Originale masker, perbatch-nevnere,/5 ogF-vekt1 beholdt.
Sentraldifferanse bekrefter15360TF-biasgradientceller innen1,94e-11; begge originale
TRACE-tap gjenskapes innen3,14e-7. Uavhengig kontroll723a68d0 består; maksimal
gradientrest5,55e-17, største publiserte feltavvik9,09e-13.

Alle5TF deler aktivitetens sluttrad innen hver S/F-gren, M5 inkludert. I den nye
kjøringen har175/256 S-batcher og220/256 F-batcher motstridende TF-bidrag til denne
biasen. M5-bidraget har motsatt fortegn av summen i88S-/86F-batcher. Dette er ett
koordinat, ikke antall skadelige oppdateringer eller full gradient-/Adam-attribusjon.
S/D1 har2226informative rader i224batcher, radkoeffisienter varierer16ganger. Den
batchvektede aktive labelandelen87,186% er LAVERE enn pooled90,341%; vektingen
forklarer ikke høy aktivitet ved å heve konstantprediktorens optimum.
S/D1-tapsendringen før/5 består av−0,02455363 på aktive rader og+0,07912416 påFLAT.
252/256gradientnormer klippes i begge kjøringer; positiv skalering bevarer fortegn
innen batch, men endrer batchers relative bidrag. Ingen numerisk tapsfeil er påvist.

[Valgt aktivitetsforsøk](docs/TA_TF_ACTIVITY_DECODER_REPAIR.json) er mekanisk kvalifisert:
fem egne aktivitetshoder perS/F, klonet fra row0;20tensorer/1290parametre ved128bredde.
Originalc171-sidekonfigurasjon beholdes, den separate nonM5-sidevarianten er av.
Dette endrer aktivitetsvekter og bias; det er ingen ren biasintervensjon. Bias-only
var mindre matematisk, mens full radkloning gjenbruker det etablerte head-mønsteret.
DefaultFalse bevarer eksisterende profiler, features/TF og tapsformler.
226fokuserte tester bestod, exit0, alle16kildebindinger uendret under kjøring.
Dette omfatter kloning/RNG, retningsparitet, aktivitetsinitialtoleranse, alleTF/S/F-
gradientruter, avvisning av ugyldige hoder og avvisning fra gamle numeriske oppskrifter.
Den første bølgen kvalifiserte bare mekanikk. Neste bølge nedenfor kvalifiserer
binding/migrasjon og åpner én lesende initialbaseline. Full historikk/30–40epoker
er fortsatt senere skalering.

Streng weights-only-migrasjon og factory/view/numerical/batch-binding er kvalifisert.
198fokuserte tilfeller:196bestod først;2feilet kun på forventet feilmelding for
non-finite backbone og bestod etter presis testrettelse. Produksjonskode var uendret.
Originale feilkvitteringer er bevart. Uavhengig kildegjennomgang fant ingen blokkering.
Eksakt module-preflight bestod:0modeller/checkpoints/arrays/claims/updates.

Lesende initialkontroll COMPLETE/CONSUMED, kilde a921f317 uendret:
ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_INITIAL_PARITY_20261112_001.
PLAN754a891b, RESULT53a92201, TERMINALd8b41784, controllerexit0 på128,23s.
2modeller,256faste TRAIN-rader per modell,32forwards,20klonede aktivitetstensorer.
0optimizer/backward/updates/checkpoint-skriving/RNG-import. Kilden gjenskaper alle
12lagrede initialfelt og tap byteeksakt. Alle5TF/S/F-sideutganger er byteeksakte;
aktivitetsavvik maks S9,5367e-7/F2,3842e-7 og tapsavvik2,3842e-7, innen faste
atol2e-6/rtol1e-6/tap5e-6. Modi/vekt-/buffer-/RNG-signaturer uendret, ingen gradienter.
Ny initialmodelldigest730409e615aa fra originalc171, kun klonede aktivitetshoder.
101telemetripunkter; topper44Ccore/46Cmemory/132,6W/592MiB, guardexit0.
Uavhengig retained-kontroll78329d9e bestod:1NPZ/0modeller/checkpoints/forwards,
10akse-/targetfelt og alle sidekanaler byteeksakte. Alle10aktivitetsceller og16lagrede
tapsdifferanser bekreftet. Historisk originalparitet/fullmodellpassivitet er bundne
kjøringskvitteringer; uavhengig kontroll åpnet ikke checkpoint/originalNPZ på nytt.

Neste steg er én separat kvalifisert kort aktivitetssammenligning på originalens
4096TRAIN-rader/256updates, sammeRNG/ferskAdamW/features/targets/tap. Separate
nonM5-sidehoder forblir av. Hele bruktprefixets initial må samsvare før førsteupdate.
Læringseffekt, senere generalisering og økonomisk edge er fortsatt ubevist.
Ingen ny trening/fullhistorikk/30–40epoker er innrømmet.
