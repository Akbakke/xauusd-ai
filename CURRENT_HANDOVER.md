# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_FOLD2_INPUTS_NORMALIZED_CURRENT_VIEW_PENDING. Edge-målet er uoppnådd.
Fold2s fysiske span-inputs og egen normalisering er kvalifisert. Alle treningsflagg er false.
Begge engangsjobber er COMPLETE/CONSUMED; ingen relaunch eller ny jobb er admittert.
Samlet rapport: docs/TA_FOLD2_INPUT_REVIEW.json. Eksakte referanser/SHA står i policy.

Neste konkrete sperre er den fortsatt fold1-bundne inputvisningen:
entry_anchor_span_feature_view_v1._parent/build_anchor_span_prefix_packet/proveniens,
og span-dispatch i entry_native_direction_feature_view_v1 som krever enhetsprofil.
Bind bare eksakt fold2-parent32f917bb, fysisk resultat8f2acfaf og TA-normalisering3528af4e
(kontrakt53e57991), med eksplisitt None for enhetsprofil. Bevar fold1-rekonstruksjonen.
Deretter må faktisk TRAIN-gather og senere januar–mars2025-dekning kvalifiseres før egen
fersk native initialbaseline/fit. TRAIN-adapteren bruker allerede viewets cutoff.
Ingen modelljobb eller native trening er åpnet av denne overleveringen.

## Fullført og verifisert

SPAN_REUSE002: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_REUSE_20261113_002.
Kilde084d6c51; RESULT8f2acfaf, TERMINAL3de8d27a, VALIDATION9313ef0d. Exit0,264,72s.
116668rader=116197gjenbrukte+471nye. Alle seks rad-/tilgjengelighetsbevis matcher.
413overlapprader er bitlike for tid, native-posisjon og FP32spans. Tre gap: lokal181,
M5261, M1529; alle31gamle felt bitlike på nye rader (14601celler). Tre produsentkall,
15sidecar-dekodinger,6klokker; null targets/modeller/checkpoints/oppdateringer.
Rekonstruksjon fra kildestart beregnet2473843historiebars;471 er nye outputrader.
Eksplisitt subset er planbundet; tidligere seksflate-standard er uendret.11tester PASS.

SPAN_NORMALIZATION001: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_NORMALIZATION_20261113_001.
Kilde8541e352; RESULT372eaa86, TERMINALb0928b65. Exit0,3,56s under4G/512M.
Seks tofeltsfit på eget fold2-prefix; ingen gamle felt refittet. Alle6532gamle feltstatsett,
masker, kategorier, support, aliaser og fysisk lineage er bevart. Ingen fold1-enhetsunntak.
317signal/132kontekst/255perTF er projisert uten ny fit. Alle observerte spanverdier er
monotone og gjenfinnes eksakt etter heltallsavrunding; største inverse feil9,44e-6.
Basekontrakt7bd0f896; TA-kontrakt53e57991. Full feature-view/gather og baseline er ubevist.
27fokuserte tester, faktisk CLI-preflight og påkrevde commit-kontrakttester PASS.
Uavhengig kilde-/JSON-kontroll PASS. Fold1s beholdte normaliseringsartefakt er identisk.

Fold2 har47338TRAIN-beslutninger fra2024-05-01 til2024-12-30 23:50UTC. Tilgjengelighet
må være strengt før2024-12-31UTC; fysisk MTF-historikk omfatter også eldre vindusrader.
Original hashkontrollfeil og preflightfeil er bevart i001-root/policy. ROW_AXIS_CORRECTION
fastslår at rå aksehash var feil sammenligningsmetode; alle seks namespacede radakser stemmer.
Dette endret ingen accuracy-resultater eller rådata. Ikke gjenta denne metadataavklaringen.

## Hvorfor periodeoverføringen undersøkes

Siste prognoserapport: docs/TA_PERIOD_TRANSFER_REVIEW.json.
M5 F senere: accuracy51,21%, balansert50,39%, AUC0,507776; alltidUP51,92%.
UP-recall71,72%, DOWN-recall29,06%. Treklasse-argmax er alltidFLAT:58,76%, balansert1/3.
Kontrollert regneverk stemmer. M5 oktober AUC0,534610→0,565820 mens accuracy55,14→54,00%:
terskelaccuracy kan skjule bedre rangering i punktestimatet, men treningsgevinsten er usikker.
FINAL oktober M5/H1 AUC er over0,5 i begge punktvise dag-/ukeintervaller. Senere overføring
svikter i flere mål: H1 desember−oktober AUC−0,202292 utelukker0 i begge intervaller;
M5s månedsfall gjør ikke det. Brier forverres etter trening i november M5/M15/H1 og
 desember M5/M15/H1/H4. M15 november viser at én enkel gjennomsnittsbias ikke forklarer alt.
Alle fem TF og alle tre måneder er beholdt. H4/D1-unntak og usikkerhet står i rapporten.

PREPOOL_CAPTURE001/READOUT001 er ferdige; ikke gjenta dem. Se docs/TA_PREPOOL_PREDICTIVE_REVIEW.json.
Alle fem PREPOOL−SHARED AUC-/accuracy-intervaller inkluderer0; dette isolerer ikke all
informasjon eller pooling alene. Nære TF-rangeringer begrunner ingen ny arkitekturendring.

## Bevarte mål og ubeviste grenser

Alle genuine317/132/255features, fem TF, åtte familier og D1/makro beholdes. Tidligere
fullTA-konsum/paritet gjenbrukes; ingen ny feltcensus er nødvendig for disse to span-feltene.
COT2024-vintage forblir UNKNOWN; ingen konkret lekkasje er påvist. Juni2026 er gjenbrukt
utviklings-VAL. Alle tidligere statistiske intervaller er punktvise på gjenbrukt utvikling.
INITIAL arver eldre backbone-trening; eksponerte rader blir ikke uavhengig OOF ved ny splitt.
Kjent-side-nevner er ikke valgbar Entry-dekning. Årsak, uavhengig generalisering, Entry/Exit-
kvalitet og økonomisk edge er ubevist. Korte forsøk dømmer ikke full historikk/kapasitet.
Full kvalifisert historikk/30–40epoker er senere retning, ikke en åpnet kjøring.
GC er på pause; M5/MTF/26månedersmålet består. TEST forseglet. Ingen broker, handel,
spending, promotion eller DATA/RUNS-opprydding.
