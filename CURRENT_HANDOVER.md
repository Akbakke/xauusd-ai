# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_FOLD2_FULL_TRAIN_INPUTS_QUALIFIED_LATER_COVERAGE_PENDING. Edge-målet er uoppnådd.
Fold2s fulle TRAIN-inputvisning/gather og egen normalisering er nå kvalifisert.
Alle tre inputjobber er COMPLETE/CONSUMED. Ingen relaunch eller ny jobb er admittert.
Alle treningsflagg er false. Ingen native modell/baseline er konstruert i denne bølgen.

Neste konkrete sperre er fold1-bindingen i entry_anchor_span_feature_inputs_v1._require_later_coverage
og qualify_market_state_ta_benchmark_later_inputs_v1s utvalg/klokkegrenser. Utvid bare til
eksakt fold2 januar–mars2025 med uendret fold1-adferd. Bind et forhåndsvalgt panel og
utled seks fysiske vindusomslutninger fra klokker; gjenbruk overlappende spans og eksisterende
native vindus-/31felt-kontroller. Ingen nye targets eller TRAIN-normaliseringsfit.
Senere coverage og egen fersk native initialbaseline må kvalifiseres før eventuell fit.
Samlet rapport: docs/TA_FOLD2_INPUT_REVIEW.json. Eksakte referanser/SHA står i policy.

## Fullført og verifisert

SPAN_VIEW001: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_VIEW_20261113_001.
Kilde148a1505; RESULTa108db7b, exit0;744,71s.47338kronologiske TRAIN-rader,2959batcher
med siste10;23672normaliseringskall. Alle åtte inputflater har korrekt bredde/vindu/feltnavn,
alle rå native klokke-/alias-/finitetssjekker består, alle normaliserte verdier er endelige.
Besøkt foreldrerekkefølge matcher eget fold2-prefix. Null modeller/forwards/targets/fit/
checkpoints/oppdateringer.7,787milliarder skalare vinduspresentasjoner er overlappende
inputverdier, ikke selvstendige observasjoner. Kilde/policy/plan/backing-bookends PASS.
VALIDATION binder originalterminalens manglende TEST-felt til resultatets eksisterende false;
originalterminalen er bevart. Uavhengig JSON-/hashkontroll PASS. Ingen array-rerun.

Samme aktivitet-only-funksjon som FULL_PREFIX_ORDER: non-M5-side=False, uendret constructor,
ruting,317/132/255bredder,96lokale bars og16/64/96/96/252MTF-vinduer.51fokuserte tester,
CLI-preflight, materialisering og påkrevde commit-kontrakttester PASS. Fold1-rekonstruksjon
identisk. Fold2 har ingen enhetsunntak. Eksisterende GC bridge binder historisk ef64a40a
separat til dagens25881da8; ingen GC-input eller ny GC-kode. Mislykket frakoblet start kom
aldri til wrapper/CLAIM/data; samme uclaimede plan ble startet via vedvarende SSH-håndtak.
Denne kvalifikasjonen gjelder TRAIN-inputs; senere inputs/modellbaseline/prognoseverdi er ubevist.


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
Basekontrakt7bd0f896; TA-kontrakt53e57991. TRAIN-gather er nå kvalifisert nedenfor; modellbaseline er ubevist.
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
