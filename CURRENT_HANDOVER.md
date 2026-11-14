# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_FOLD2_M5_DIRECTION_FIT_003_ONE_USE_ADMITTED_PREPARATION_PENDING. Edge-målet er
uoppnådd. TARGET_COUNTS001 er COMPLETE/CONSUMED, exit0,3,78s. RESULT ba50eb96,
VALIDATION dd45bd05, TERMINAL dccda6f6; ingen relansering. På eksakt47338 modne
TRAIN-rader /2959 B16 med siste10 er M5 kjent retning S N16101/q2890 og F N19072/q2950.
Sampled S-M5 er lik native M5-endepunktet; alle fit-targets er modne. Egne S/F-priorer
er fit på samme TRAIN-rader.9arkivdekodinger,2referanseprior-fit, null modeller,
forwards, feature-gather, optimizer, checkpoint, senere fit eller TEST.

Den eksisterende M5-equal-taps-/batch-eieren binder nå fold2s egne q/N, akser,
inputview og målkvitteringer; fold1-profil og default kontrakt er uendret. Forholdet
mellom fold1- og fold2-koeffisientene er1,00127(S) og1,00106(F), omtrent0,1%.
Dette er ikke en målt forklaring på svak ACC eller bevis for bedre AUC. En gammel
testhash ble korrigert mot uendret b248deda-baseline; den var foreldet før fold2-bindingen.
Fokuserte kildekontroller består. Fersk fold2 INITIAL er målt nedenfor.

Fersk fold2 INITIAL 004 er COMPLETE/CONSUMED, exit0,506,2s, plan6fe63d5e,
RESULT ebaa2de9, TERMINAL 7a44ec02, uavhengig VALIDATION 0c6abfbf.
512 B16-forwards på fast4096 TRAIN +4096 senere, én fersk modell, null checkpoint,
backward/optimizer/oppdatering/TEST. Modelltilstand uendret; signed GPU-vakt exit0.
Senere F M5 kjent-side N1591: AUC0,4868, ACC48,40%, alltid-UP53,74%, Brier0,2504.
Januar/februar/mars F M5 AUC0,4921/0,4980/0,4547. Dette er nullstegsprognoser
på gjenbrukt utvikling, ingen lært edge eller økonomisk kvalifikasjon. S H1/D1
AUC0,5612/0,5628 til tross for svak ACC; rangering og terskel-ACC skilles,
men tilfeldig initialisering er ikke bevis på nyttig modellrangering. Alle fem
TF og månedsceller er bevart i RESULT; 130 celler validert mot uavhengig AUC,
Brier og eksisterende ACC-eier. Plan002/003 er FAILED_CONSUMED før første forward,
med respektive normaliserings-/dualmodellhash-kontrollfeil; aldri relanser.
Plan001 hadde bare metadata-preflight, ingen claim.

Eksisterende trent oktober–desember-readout er gjenlest: oktober M5 AUC-gevinst
+0,0312, men paret dag- og ukeintervall omfatter0; november/desember FINAL AUC
0,5118/0,5051, samlet0,5078 med ukeintervall0,4844–0,5324. Oktober-ACC skjuler
lokal rangering, men ingen stabil M5-edge. D1-samlet AUC0,5499 har bredt
ukeintervall0,3881–0,7204 og er ikke M5 Entry-kvalifikasjon.
Fold2 M5-retning-alene-tapet har nå kilde-/gradientreview i
`docs/TA_FOLD2_INPUT_REVIEW.json`: kontrakt da042c0f, egen q/N, null direkte
gradient i aktivitet og andre TF. Én foreslått kronologisk pass bruker alle47338
TRAIN-rader/maks2959 B16-oppdateringer fra samme ferske INITIAL og skårer de samme
faste senere panelene, uten checkpoint-/terskelvalg. Tidligere fold1 fulltap ga
F M5 senere AUC0,5113→0,5101→0,5091 etter pass1/pass2 og dårligere Brier;
det er begrunnelse, ikke matchet fold2-kontroll. Eksisterende fit-runner importerer
gammelt checkpoint og binder fold1. Fold2 har fast B16/B10-kildekontrakt,
fit-kjerne og sju fokuserte syntetiske tester PASS, samt read-only metadata-
preflight og slutt-scorer. Scoreren replayet alle130 INITIAL-celler eksakt;
identisk INITIAL-par ga null AUC-gevinst. Bootstrap-seed er dag20260911,
mandagsuke20260912. Arbeidskoden binder alle17 uendrede INITIAL-inputkvitteringer;
design001/002 er fryst separat. Indre livssyklus er kildebundet og avviser
lukket/full-fit-svak policy og manglende GPU-vakt før payload. Den krever fersk startdigest,
fast pass og parvis FINAL-readout, men faktisk payload/modell/fit er ikke kjørt.
Fersk INITIAL-design er bitlik originalkilden og nå fryst i egen kildebundet
reseptfil; metadata-resepten binder 47338 TRAIN, maks2959 oppdateringer, null
checkpoint/resume og de 17 gjenbrukte inputkvitteringene. Campaignens
metadataautoritet kjenner fold2-resepten, 2959 besøkte batcher og mulige
hoppede oppdateringer i terminal cursor/progress. Materialisering, dispatch og
ytre worker er kildebundet og syntetisk rutet; syntetisk COMPLETE og midt-i-step FAILED publiserer
immutable resultat/cursor/progress, mens gjentak avvises. Plan001 bandt immutable kandidat-resept SHA4dac6a56 og Windows-boot649.
Den ble innrømmet og materialisert på ren kilde; fysisk boot650 startet
signert trainer. Runtime feilet før payload/modell/forward med RESULT SHAa331cbbc,
TERMINAL SHA2aab677b og kvittering SHA54ca0175:0 batchbesøk,0 updates.
Runtime leste toppnivå-`fold_id`, mens fryst resept har
`numerical_recipe.fold_id=2`.21 av22 kontrollerte adgangsfelt var sanne.
Plan001 er FAILED/CONSUMED uten relansering; Windows-oppgaven er deaktivert.
Eksisterende runtime-eier leser nå riktig schemafelt. Den gamle syntetiske
fixturen er rettet, og en negativ/positiv gatetest er lagt til. Design002
er bitlik design001 utenom ny engangsidentitet.13 fokuserte tester, syntaks,
lesende metadata-preflight og faktisk resept gjennom runtime-gaten før
payload består. Resept002 SHA31960926 er publisert fra ren commit0e365422, med målt
boot650 SHA94f48f81. Lesende preflight SHA27a5fecf og faktisk bundet
feature-/normaliserings-/initialidentitet SHAf21ee0dc er PASS; eksakt
resept når GPU-vakt før payload, og moduloppstart er røykprøvd uten claim.
Plan002 ble materialisert og startet etter fysisk boot651, men feilet med
RESULT SHA0babbcac, TERMINAL SHA2e4feb7b og kvittering SHAa41571f0:
0 batchbesøk,0 updates, ingen FINAL. Faktiske S/F-targeteiere har118249
rader, mens TRAIN-prefikset har47338; hele aksen ble feilaktig sammenlignet
med prefikset. Begge faktiske prefiks er eksakt like. Koden sjekker nå
prefikset, fullakse-regresjonen og14 fokuserte tester består; lesende
reell aksekontroll når første batch uten payload/update. Plan002 er
FAILED/CONSUMED uten relansering, Windows-oppgaven deaktivert. Design003
er identisk med002 utenom ny engangsidentitet. Resept003 SHAee16c4c4 er publisert fra ren commit853ddadc og boot651
SHA14a06a41. Lesende PREFLIGHT SHA27a5fecf og faktisk resept/
fullakse-gate SHA693ce1af passerer før forward/update. Én separat
12000s/250W-admisjon003 med høyst2959 updates og null checkpoint er
åpnet; faktisk fit er ikke startet. Neste steg er ren commit/push,
capped materialisering, maskinvid idle-kontroll og fysisk boot>651. TEST, Entry, Exit, broker og ordre er stengt.
Edge er fortsatt ubevist.
Fokuserte fold2/campaign-tester PASS. Eldre fold1-fit-testfil har19 isolerte
fixture-feil NATIVE_F_PRIVATE_TA_M5_EXACT_INPUT_IDENTITY i uendrede kilder/refs;
ikke bruk den som fold2-kvittering eller gjenta uten konkret behov.

LATER_INPUTS002 er COMPLETE/CONSUMED, exit0,215,18s; terminal1297d154,
RESULT2cab8b60/COVERAGE5abd47c4/GATHER_RESULTe6aacfe9/VALIDATIONdf45d60f.
4096faste senere rader og alle åtte inputflater består.17353lokale rader gjenbrukt,
25028nye MTF,17944samlet gjenbrukte,42972fysiske rader,414bitlike overlapper.
Alle775868gamle MTF-celler bitlike.2025delen fjernet alle17343manglende M5stempler.
Dette rettet privat filliste, ingen påvist modellinputfeil eller prediktiv gevinst.
Full47338TRAIN-gather og89tester gjenbrukes. Aktivitetsfunksjon/egen normalisering/view
f7979c85 er uendret. TEST forseglet. Rapport: docs/TA_FOLD2_INPUT_REVIEW.json.

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
Basekontrakt7bd0f896; TA-kontrakt53e57991. TRAIN-gather er nå kvalifisert nedenfor; framtidig fit er ubevist.
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
Tidligere fold1 INITIAL arvet eldre backbone-trening; fold2 INITIAL er fersk. Eksponerte rader blir ikke uavhengig OOF ved ny splitt.
Kjent-side-nevner er ikke valgbar Entry-dekning. Årsak, uavhengig generalisering, Entry/Exit-
kvalitet og økonomisk edge er ubevist. Korte forsøk dømmer ikke full historikk/kapasitet.
Full kvalifisert historikk/30–40epoker er senere retning, ikke en åpnet kjøring.
GC er på pause; M5/MTF/26månedersmålet består. TEST forseglet. Ingen broker, handel,
spending, promotion eller DATA/RUNS-opprydding.
