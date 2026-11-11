# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste maskinlesbare arbeidsstatus; prosess og
originale terminalkvitteringer overstyrer tekst. Historikken finnes i Git.

## Gjeldende mål og port

Lær framtidig OPP/NED/FLATT markedstilstand før Entry. M5 er prognosetakten;
kausalt lukket M15/H1/H4/D1 og genuine indikatorer beholdes. Alle fem TF har
egne mål. GC er på pause; TEST, Entry/Exit, broker, handel og promotion stengt.
TA_M5_R1_TWO_ALPHA_COMPLETE_DIRECTION_COMPARISON_FAILED.
Alle trenings-/launchporter er false; training_enabled_scope=null.
Ingen CURRENT-jobb ved lukking. Konsumerte planer relanseres aldri.
Målet er fortsatt uoppnådd.

## Siste målte kontroll

[To-alpha-kontroll](docs/TA_M5_R1_INNER_L2_BOUNDARY_REVIEW.json) er COMPLETE/
konsumert, worker/prosess exit0, kilde43510f4b uendret.
PLANd7c507a8, RESULT4c4a9703, SELECTION33194ec9, LOCK9c04ef9b.
Kun alpha.01/.1, eksisterende tre kronologiske indre TRAIN-vinduer,
prefixberegnede equal-observation-vekter, samme solver og frossen R1.
12 indre fits valgte.1; pooled logtap .691730659→.691566559.
To fullTRAIN-fits ble låst før senere arrays ble åpnet.
14fits totalt;3700indre+5918full B16-affine kall. Ingen encoder/modell/
checkpoint/autograd/native update. Helekommando4,697s, worker3,793s.
Uavhengig32skårceller/192felt, maksavvik1,11e-16,0,342s,ingen refit/
forward/matriseprojeksjon. Valg, koeffisientlås og fullfit-objectiver stemmer.

Brier INITIAL → alpha.1 / fryst prior:
TRAIN S .248473364 → .248516138 / .249651008.
TRAIN F .249327011 → .249418521 / .249876211.
Senere S .249652488 → .249671196 / .249827988.
Senere F .249855791 → .249926401 / .249833416.
Logtap gir samme hovedkonklusjon. Alpha.1 taper INITIAL i alle fire celler;
senere F taper også prior. Ingen FP32/FP64-grensetvetydighet.
Senere F meanUP .512987537 mot observert .513049546, men AUC .505557377.
Nesten riktig gjennomsnitt er ikke nok til gode enkeltprognoser.

## Gjenbrukbar forbedring og grenser

Alpha.01 W2 på R1 reparerer fullTRAIN S/F:
Brier .248194763/.249065517, bedre enn INITIAL og prior.
Native kvalifisering: MARKET_STATE_TA_M5_REVERSED_FIXED_NATIVE_SCORE_20261110_001,
RESULTe5be50b5,kandidate94348ac. Alle øvrige tensorer/buffere/RNG/modi bevart.
Samme-instans weights-only diskroundtrip/row0 kvalifisert; fresh constructor/
optimizer-resume utestet. Full senere17753: S .249822360/F .249942166;
begge taper INITIAL. F AUC .510747753. Originalvekter/capturer bevart.

W2 på INITIAL-representasjon svekket senere S/F; R1-input under sammeW2
bedret begge. Dette er en ordnet funksjonell kontrast, ingen isolert
optimizerårsak. Månedlig F-AUC oktober/november/desember .5253/.5273/.5028.
MeanUP-driften finnes også på alle rader; ikke bare known-utvalg.

Indre head-score er IKKE encoder-OOF: encoder/normalisering så hele TRAIN.
Senere oktober–desember2024 er gjenbrukt utvikling. Ingen urørt generalisering,
kalibreringsgaranti, unik treningsårsak eller handelsedge er dokumentert.
Ingen kontrollert hastighetsgevinst for en hel trening er målt.

## Neste direkte spørsmål

Kvalifiser én rangbevarende shrink fra alpha.01 mot TRAIN-prior, tilpasset
kun lagrede indre TRAIN-prognoser. Uavhengig metodevurdering støtter denne
avgrensede testen av amplitude, ikke rangering. Positiv slope bevarer rangering;
slope0 er kollaps til prior. Prefixprognosene blir kalibrator-fitdata.
Transfer fra prefixhoder til fullTRAIN-hode må måles, aldri antas.
Den gamle FINAL-kalibratoren brukte in-sample fullTRAIN og er ikke samme test.
Ingen ny fit/native import er innrømmet ennå; eksakt plan/preflight/admisjon
kreves. Ingen automatisk videre alpha-søk eller encodertrening.
