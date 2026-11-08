# CURRENT_HANDOVER — GX1

Arbeidssted: `/home/andre2/src/GX1_CURRENT`, gren `work/gx1-current`. Les
[GX1_RULES.md](GX1_RULES.md) før arbeid og kjør `bash scripts/gx1_handover.sh --check`.
[NEXT_RUN_POLICY.json](NEXT_RUN_POLICY.json) er eneste maskinlesbare arbeidsstatus;
terminalkvitteringer og faktiske prosesser overstyrer tekst her. Eldre arbeid,
feil og kvitteringer finnes i Git og på sine opprinnelige artefaktstier. Åpne dem
bare ved et konkret avvik eller en ny nødvendig gatekontroll.

## Mål og gjeldende port

Lær løpende framtidig OPP/NED/FLATT markedstilstand **før** Entry-plassering.
M5 er prognosetakten; kausalt lukkede M15/H1/H4/D1 og eksisterende genuine
indikatorhistorikk er input. Alle fem TF har egne framtidige mål. Nåværende
forskningsfokus er SMC og teknisk analyse etter brukerens vedtak 08.11; GC-planen
er utsatt. Entry, Exit, ordre, broker, handel, spending og forseglet TEST er stengt.

Policystatus: `TA_SF_CHECKPOINT_RECOVERY_ADMITTED_WITH_CORRECTED_CODE`.
[Fullføringen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_INPUT_COMPLETION_20261108_001/RESULT.json)
er COMPLETE: 600 ekstra historikkrader × 94 felt = 56 400 sammenligninger,
null avvik. M5/M15/H1/H4/D1-gapene var 94/63/95/96/252 rader. Alle ni
modellflater har null ikke-endelige verdier på unike konsumerte rader.
315 lokale felt og 132 kontekstfelt er ikke konstante i dette utvalget;
noen tilstedeværelses-/sesjonsfelt er konstante på lengre TF.
Frosne S/F-priorer bruker eksakt 29 574 modne fit-rader, null score-fit.
385,86 s og 5,74 GB topp-RSS; null nye modeller, gather, normalisering,
lokal31-replay, backward eller optimizer. Én tidligere forward gjenbrukes
eksplisitt som evidens. Fullføringsplanen er konsumert.

Den tidligere lokal31-diagnosen er ACTUAL_SOURCE_MATCH: 47 697 rader ×
31 felt, 1 478 607 sammenligninger, null oracle/cache-/producer-avvik.
Den gamle auditreferansen startet i 2009, mens faktisk produsent startet
i juni 2010. Ingen produksjonsfeil ble demonstrert; ulike state-historikker
og absolutte koordinatstarter ble ikke isolert fra hverandre.

Ny [TA S/F-mekanikk](/home/andre2/GX1_RUNS/MARKET_STATE_TA_SF_MECHANICAL_20261108_001/RESULT.json)
er PASS: 32 modne fit-rader, seks forwards, fire backwards/vektoppdateringer,
fersk baseline og bitlik gjenopptakelse. 154,96 s, 5,75 GB topp-RSS og
1,79 GB toppallokert GPU-minne. Trener/observatør/kontroller avsluttet med 0.
Oppgaven er deaktivert og planen konsumert. Dette er ikke markedspresisjon.

Brukeren ba om å rette den målte ytelsesflaskehalsen straks. Ettpass-fit
ble derfor avbrutt før kildeendringen; planen er ABORTED_CONSUMED.
Full INITIAL på 29 574 rader og checkpoint ved 4 096 rader / 256 updates
er bevart. Checkpointets hele state- og komponenthasher er kontrollert.
Oppdateringer etter siste checkpoint er UNKNOWN; FINAL og senere TRAIN-score
er ikke fullført. Startoppsettet ignorerte SIGINT; SIGTERM ga guard-exit143
og Windows-task-exit1. Ingen vellykket trenerterminal eller resume-cursor
ble publisert. Oppgaven er deaktivert, GPU/prosess/lås er ledige.
[Operatørterminalen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_FOLD1_FIT_20261108_001/OPERATOR_INTERRUPTION_TERMINAL.json)
bevarer dette som avbrudd, ikke PASS. Ingen gammel plan relanseres.

Den minste ytelsesrettelsen erstatter 2 700 gjentatte tuple.count-skanninger
med Counter. I 25 like metadata-målinger sank feltvalideringen fra
66,994 til 0,834 ms (80,34 ganger); hele TA-profilbyggingen fra
117,099 til 50,161 ms (2,33 ganger). Gyldige profiler, kontrakter,
feltrekkefølge og seks kontrollerte normal-/feiltilfeller er identiske.
Alle validatorer kjøres fortsatt; modell, features og regning er uendret.
Tre fokuserte TA-profil-/normaliserings-/konsumtester består. Tre nye
duplikatfeiltester følger den eksisterende obligatoriske pre-commit-gaten.
[Resultatet](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DUPLICATE_CHECK_PERF_20261108_001/RESULT.json)
binder før/etter-målingene. Gevinsten for hele treningsløpet er ikke målt.

Den separate [inputfasekvitteringen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_INPUT_20261108_003/INPUT_TENSOR_PHASE.json)
er komplett innen sitt scope: 47 327 beslutningsrader og alle overlappende
inputvinduer, 7 726 606 020 råverdier bitlike, null normaliseringsavvik og
én faktisk forward med åtte familier/fem TF.
Disse bevisene er gjenbrukt i fullføringskvitteringen ovenfor.
003 stoppet i M5-lokalreplay; de øvrige MTF-historikkgapene er nå kvalifisert.
Alle originale `CLAIM/FAILED/TERMINAL` og logger for input001–003 og
diagnosen er bevart. Ingen konsumert plan relanseres.

Brukeren presiserte læring av berøring→brudd→retest på samme trendlinje.
Dagens representasjon har to nærmeste-linjeplasser uten linje-ID, ankere
eller byttemarkør; hendelsesfelt kan gjelde andre registerlinjer.
Tapet har S/F-framtidsmål, men ingen separat fasit/evaluering av samme-linje-
reaksjoner. Dette er en konkret representasjons-/målegrense; effekten på
prediksjonsevne er ikke målt. Ingen ny modellarkitektur er innført.
GC-registerplanen er utsatt og uklamet. Juni 2026 er brukt utviklings-VAL;
forseglet periode følger bare forward-journalregelen i GX1_RULES.md.

Ny brukerbestilling åpner én eksakt gjenopptakelse med rettet kode:
[recipe](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_FOLD1_RECOVERY_20261108_001/RECIPE.json).
Den starter fra bevart rad 4 096 / update 256, med 25 478 rader / 1 593
oppdateringer igjen i samme pass. Original full INITIAL gjenbrukes.
Samme modell, targets, normalisering, feltrekkefølge og optimizer beholdes.
85 kildeeiere er bundet; bare Counter og seks nødvendige recovery-eiere
avviker fra originalen. Metadata/TA-loader/faktisk view er kvalifisert.
Original state/AdamW/RNG/order må bestå full restore og fersk initialparitet
før første nye oppdatering; importen lagres deretter under ny kilde.
Input- og mekanikkjobbene gjentas ikke.

Historisk evalueringsbane brukte 3 260,23 s for 29 574 rader / 1 849 batcher.
Første bevarte fitblokk brukte 565,66 s for 256 oppdateringer. Dette er
filtid-intervaller med dokumenterte publiseringsgrenser. Ny kjøring måler
oppstart/restore, fitblokker, FINAL, senere score og total monotont.
Gammel fullført total finnes ikke: samlet speedup kan bare anslås, mens
fasegjennomstrømning og faktisk ny total kan måles. Vektene er forskjellige
ved gammel INITIAL og ny FINAL, men populasjon og evalueringsbane er like.
[Tidsgrunnlaget](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_FOLD1_RECOVERY_20261108_001/HISTORICAL_TIMING_BASELINE.json)
binder sammenligningen. Ingen ny epoke eller lengre historikk åpnes.

## Det som faktisk er målt

- **Eksplisitt TA-kandidat og korrigert læringsmåling:** Profilen
  `full_xau_ta_mtf_entry_research_v1` beholder 315 signal-, 132 kontekst- og
  253 MTF-felt, alle fem TF, åtte familier og 61 makrofelt. GC er fjernet
  før modellinput; modne fold1-statistikker projiseres uten refit.
  Faktisk FeatureView/checkpoint-identitet er kontrollert lesende:
  `3e2761afaf1d29e6537843e9e59b592a7c149a9cd99c97f866a2cd1e88611906`.
  Ny rute bruker korrigert S/F og hele samme 29 574-raders TRAIN-populasjon
  før/etter fit, med globale aktivitets-/sidemasker og frosne priorer.
  Senere 17 753 TRAIN-rader scores separat. Gammel initialkvittering
  med null forwards er ikke en læringsmåling. Inputkvalifikasjonen er komplett innen deklarert scope;
  ny fireoppdateringsmekanikk består. Fit ble avbrutt etter brukerens
  ytelsesbestilling; 256 oppdateringer er bevart, men full læringsmåling gjenstår.

- **Funksjonsgjennomgang:** EMA bruker SMA-seed og alfa `2/(p+1)`;
  ATR bruker klassisk TR og Wilder14. SMC-pivot blir synlig etter bekreftelse,
  BOS bruker strict close-brudd, og alle inputbarene skal være lukket.
  Ingen ny produksjonsfeil er påvist i disse formlene ved kildekontroll.
  Modellen får indikatoravstander/helninger og candle-geometri.
  Prisoppdateringsantall er ikke omsatt volum.
  En reell makrodatofeil rundt høstens DST ble reprodusert og rettet:
  tilgjengelighet følger nå kalenderen D+2 ved NY-midnatt, ikke 48 elapsed timer.
  Alle 9 000 bundne kildedatoer fram til 2024-12-31 har identisk tilgjengelighet
  før/etter rettelsen; kandidaten trenger ingen ny materialisering for denne
  rettelsen. Kalender-/driverkontroll: 11 beståtte tester.

- **Test- og formelbevisgrense:** Ny TA-gather mistet C-kontiguitet og
  signaturrefleksjon mistet de fem gamle GC-argumentnavnene. Begge er rettet;
  sju berørte tester består.
  [Kilde-/testkvittering](/home/andre2/GX1_RUNS/MARKET_STATE_TA_SOURCE_QUALIFICATION_20261108_001/REVIEW_20261108T131458787986Z.json) SHA 68fcf96b270bf5c944fae4b3f6724ce25ad92c2ed25aaa3e1cdc329b55288a6d. Øvrige korrigerte taps-/batch-/state- og
  initial/final-tester bestod første fokuserte kjøring.
  Én eldre GC-relative-kildetest feiler på en uendret, foreldet
  `gc_order_flow_v1.py`-SHA; ingen GC-reparasjon åpnes i denne bølgen.
  186 lokale felt har eksakt M5-cache-alias. De 31 lokale trendlinjefeltene
  bruker egne 96-barsparametere og kontrolleres separat. Auditen inkluderer
  også 37 HTF/session-felt med eksplisitte asof-/avrundings-/klokkekontroller;
  den nye auditeierens kontroller er utvidet til 16 beståtte tilfeller.
  Fysisk tensorfase bestod; lokal31 er nå kvalifisert mot faktisk kilde som beskrevet over.
  61 makrofelt har eldre samme-eier-paritet/asof-bevis, ikke et uavhengig
  formelorakel; dette er en uttrykkelig bevisgrense.
   COT i 2024 bruker as-of +7 kalenderdager som tilgjengelighetsproxy.
   Faktiske publiseringstidspunkt og historisk datavintage er ikke kvalifisert
   per rapport. Ingen konkret 2024-lekkasje er påvist; full historisk
   tilgjengelighet er UNKNOWN og blir ikke bevist av teknisk input-PASS.
  Tidligere 253-feltsunion 158+31+1+63 og asof/postnorm-bevis gjenbrukes
  på deklarerte rader; de eldre faktisk konsumerte historikkgapene er nå også målt.
  [Tidligere kvitteringskontroll](/home/andre2/GX1_RUNS/MARKET_STATE_TA_INPUT_LOSS_REUSE_REVIEW_20261108_001/REVIEW_20261108T120132104946Z.json).

- **Input og mål:** 118 249 modellkoblede TRAIN-rader, 26 måneder GC MBP-1 og
  kausale fem-TF-flater er bevart. Foldprefikser, masker og normalisering er
  kvalifisert i sine deklarerte vinduer, ikke som bevis for prediktiv verdi.
  Hver genuine indikator trenger fortsatt kontroll helt til faktisk nytte;
  numerisk paritet og gradient på én rad er ikke edge. Lokal framtidstilstand
  `S_s(v)` bruker valgt aktivitet `R≥0,5 ATR`; i 0,5–1,0 ATR er side uavklart
  `{NED,OPP}`. Beslutningsforankret prisforskjell `F_s(t,v)` er et **separat**
  retningsmål med egen maske, ikke en ommerking av S-gråsonen. I senere TRAIN
  hadde 52 931 M5-rader S-side uavklart og gyldig F-endepunkt; bare 13 947
  hadde `|F|≥0,5 ATR(t)` og dermed meningsfull F-side.
  [F-dekning](/home/andre2/GX1_RUNS/MARKET_STATE_F_SUPERVISION_COVERAGE_20261108_001/RESULT.json)
  SHA `61b1e21a2dac1b86c8263e60fdcec5cd12b652135fcb991e231607493c942312`.

- **SMC/teknisk grunnlag:** Eksisterende 26-måneders M5-kilde og den eksakte
  modellklokken har 118 249 felles TRAIN-rader. Kausalt lukket M5 BOS og
  EMA20-helning gir 6 737 sidejusterte hendelser totalt; fem senere TRAIN-folds
  har henholdsvis 1 006/953/1 017/1 038/1 032 modne kandidater før noe
  framtidsutfall ble lest. [Kildedekning](/home/andre2/GX1_RUNS/MARKET_STATE_SMC_EMA_BOS_20261108_001/PREFLIGHT.json)
  SHA `f87a834f985cc53f1e5517f76753f6b00b4ef66d07735cc1d51bcc56594681b7`.
  Eldre SMC-familiefit kvalifiserte ikke retningsverdi på sitt tidligere mål;
  enkel EMA50/200-referanse ga ikke stabil sidegevinst på nytt S-mål. Ingen
  av dem testet den forhåndsbundne BOS+EMA20-sammenhengen ved 60 minutter.

- **Fast SMC+EMA20-forsøk, negativ retningsport:** Engangsplanen
  [MARKET_STATE_SMC_EMA_BOS_20261108_001](configs/research/MARKET_STATE_SMC_EMA_BOS_20261108_001.json)
  er konsumert. På 4 876 modne TRAIN-hendelser var ekstra signert
  60-minutters F-forskyvning mot EMA-kontroll i samme uke/side `−0,006385`
  H1-ATR, 95 % ukeblokkintervall `[−0,022610,+0,010314]`; bare 1/5 fold
  var positiv. Halv-ATR-kandidatens event-hit-rate var `0,488753`; det er
  ikke en låst markedstilstand eller handelsmåling. Uavhengig pandas-gruppering
  ga samme 4 876 rader og samme fem foldeffekter. Første auditforsøk tok med
  tre rader over foldcutoff; dette var auditens egen indeksfeil, rettet før
  kvittering, og påvirket ikke engangskjøringen. [Resultat](/home/andre2/GX1_RUNS/MARKET_STATE_SMC_EMA_BOS_20261108_001/RESULT.json)
  SHA `79fd9bccd08c78af6f4fb2de59f0416258e73546b0a302969ce152fd4822e40b`;
  [uavhengig kontroll](/home/andre2/GX1_RUNS/MARKET_STATE_SMC_EMA_BOS_20261108_001/INDEPENDENT_CHECK.json) SHA
  `d7a036c2065a29ea4f1a98c1c87cf4f62f2abc3ca51ae652243551ce5c36729d`.

- **Native tilstand, gammel v1:** Én fold-1-fit brukte 29 574 modne TRAIN-rader
  og 1 849 AdamW-steg. På 6 096 senere sertifiserte M5-siderader var
  side-Brier `0,277631` mot fryst prior `0,249828`; ingen retningsgevinst.
  Kilde/algebra viste at v1-tapet ga direkte skjev sidegradient når fasiten
  bare sa AKTIV; OPP-skjevheten er målt, men årsaken er ikke isolert.
  [Fit-resultat](/home/andre2/GX1_RUNS/MARKET_STATE_TIME_QUERY_FOLD1_FIT_20261107_002/RUNTIME/prefix-fit/window-0001/RESULT.json)
  SHA `08172fc2c17f8eaf4f0457e38bdd3aa750bfbbf9ae719cb587aabd4d93d21254`.
  Ingen ny v1-fit.

- **Ny S/F-rute:** Faktorisert v2-S og separat F-hode deler én native enkoding.
  På to ekte TRAIN-rader besto ett fysisk forward/backward: begge hoder og
  backbone fikk gradient, mens ukjent S-side og liten F-bevegelse fikk null
  direkte sidegradient. Vektene var bitlike før/etter; null optimizer/TEST.
  [S/F-nullsteg](/home/andre2/GX1_RUNS/MARKET_STATE_NATIVE_F_CPU_ZERO_STEP_20261108_001/RESULT.json)
  SHA `a2bb6cb8f586ea85f18ca4b313ffe28117dee67659216d00641d0a25cb0ccc7f`.
  Dette er teknisk kobling, ikke fit eller prognoseevne.

- **Enkle kronologiske referanser:** Femfolds S-LightGBM ga M5 side-Brier
  `0,249819` mot prior `0,249440`, men bedre aktivitet. En separat femfolds
  F-M5-LightGBM med alle 3 991 tabellfelt versus samme oppskrift uten de
  2 410 GC-navngitte feltene fikk **FAILED-terminal etter alle 20 fits** på
  grunn av skygget kilde-HEAD i siste bokendekontroll. Originalplan og
  terminal er konsumert; ingen RESULT eller omkjøring. Lesende hash-/mål-/
  prognosekontroll av de fem foldartefaktene ga på 34 760 meningsfulle
  M5-siderader Brier prior/full/uten GC `0,249755/0,250705/0,250821`.
  Prior slo full i alle fem folder; GC-tillegget var `+0,000115` Brier,
  3/5 positive folder og 95 % UTC-ukeintervall
  `[−0,000183,+0,000425]`. Den låste retningsporten feiler. På 88 397
  aktivitetsrader var prior/full/uten GC `0,238637/0,230894/0,231863`;
  aktivitet og GC-tillegget var positive i 5/5 folder. Dette er diagnostikk
  fra en feilet engangsjobb, ikke et fullført kampanjeresultat eller dom over
  native tidssekvenser.
  [Feilterminal](/home/andre2/GX1_RUNS/MARKET_STATE_F_M5_GC_ABLATION_20261108_001/TERMINAL.json)
  SHA `ce1b1862d1808fc67d63720b20e5c4075bf82b478ec5afbd2391b71d06f4e4a3`;
  eksakte foldreferanser, skår og begrensning står i policyens gate
  `market_state_F_m5_gc_ablation_20261108_001`. Den konkrete variabelskyggen
  i beholdt kilde er rettet etter kjøringen.

- **GC-tolkning:** Kjent aggressivt kjøps-/salgsvolum, sidevis gjennomsnittlig
  handelsstørrelse, OFI og GC-prisrespons når inputruten. Største registrerte
  handel har ikke egen sideetikett; dagens felt identifiserer ikke presist ett
  stort kjøp i aktiv modellvisning. De nye sidevise maksimumsfeltene er fysisk
  bygget, men ikke registrert eller konsumert av modellen. En enkel positiv
  GC-kjøpsubalanse ga ingen stabil framtidig XAU-retningsverdi; det tester ikke full native sekvensmodell eller absorpsjon
  ved struktur. [GC-flytreferanse](/home/andre2/GX1_RUNS/MARKET_STATE_GC_FLOW_REFERENCE_20261107_001/RESULT.json)
  SHA `256710c697cba9324cb4635aa323e99f4319ddbd1c418fa54ec549f04fde4946`.

- **GC-side-dekning på faktisk modellakse:** Av 118 249 M5 TRAIN-rader har
  117 626 gyldig GC-observasjon og global største handelsstørrelse; 117 541
  har både kjente aggressive kjøps- og salgsregistreringer. Sidevis
  størrelsesmaske finnes, men ingen sidevis største handel i den fysiske
  tabellen. Dette er presist informasjonstap for store handler, ikke mangel
  på GC-side generelt og ikke målt prediktiv verdi. 97 917 rader har også
  minst én handel med ukjent aggressorside (`N`); det kan sameksistere med
  kjente kjøp og salg. Kun eksakt M5-modellakse er kontrollert her.
  [Dekningskvittering](/home/andre2/GX1_RUNS/MARKET_STATE_GC_SIDE_LARGE_TRADE_COVERAGE_20261108_001/RESULT.json)
  SHA `7fdf2adb1138e0d777f85181c59d4272a50c6e832c6391d40ee9519ac684d184`.

- **Sidevis storhandel, kilde og fysisk data:** Eksisterende GC-samler bevarer nå
  største betrodde `T`-kjøp (`B`) og -salg (`A`) separat per M1 og reduserer
  dem kausalt til M5/M15/H1/H4/D1. Ti nye kandidater har samme sidevise
  observasjonsmasker og flow-eier; `N` gjøres ikke om til kjøp/salg. 77
  fokuserte kilde-/syntetiske tester består. Gamle fysiske GC-tabeller og
  nåværende native register/visning mangler fortsatt de nye feltene; nye
  fysiske tabeller er kvalifisert nedenfor. Prediktiv verdi og Entry er ubevist. Policyens
  `market_state_gc_side_max_source_20261108_001` binder dette kildesteget.
  Ny engangsplan [sidevis GC-bygg](configs/research/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001.json)
  SHA `57c076ffe2797297fa502a6fc32a0f0d38f4f347eaa71c9517e0d9c3b350bd47`
  er kjørt én gang og konsumert. Den gjenbrukte kvalifiserte råkilder og
  samme byggeeier: 66 filer/892 100 712 poster, 1 603 s under 1 800 s/4 GiB-
  kapsel. [Terminal](/home/andre2/GX1_RUNS/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001/RESULTS/TERMINAL.json)
  SHA `664d7b9aca3d969cfd7a72c34f1b5a186d2c37fd1f3a079a0bcacd5446abebd6`.
  Alle gamle 270 TF-felt er numerisk like på fem TF. Nye sidevise maksverdier
  har eksakte masker og likhet med maksimum over godkjente M1-minutter;
  [fysisk kontroll](/home/andre2/GX1_RUNS/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001/PHYSICAL_SIDE_MAX_AUDIT.json)
  SHA `9cefe83b23b7c02d848308ce756deeb8575205edfb89551478ad740db28c4dcc`,
  [M1–TF-paritet](/home/andre2/GX1_RUNS/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001/PHYSICAL_SIDE_MAX_M1_TF_PARITY.json)
  SHA `9bbcaf4bec20cd1e4bc642c8e60b8c4ee0cfd41ccfa2b803d51247a36942d7a0`.
  På de eksakte 118 249 M5-modellklokkene har 117 541 begge nye felt;
  [klokkedekning](/home/andre2/GX1_RUNS/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001/MODEL_M5_SIDE_MAX_CLOCK_COVERAGE.json)
  SHA `3f8fcaa460b5f8e7f90d93cd84c504df914d1717bc1422bb8eb67f676d19289b`.
  Native register/layout/visning har ennå ikke konsumert de nye feltene.

- **Ny retningsvekt, kun kilde:** På de fem senere TRAIN-foldenes faste
  endepunkter hadde S entydig side i 33–44 % og F i 37–41 % av informative
  rader. Det gamle v2-tapet delte sidefeilen på alle informative rader. En
  separat side-normalisert tapsmodul deler nå på faktisk sidekjente rader;
  S/F-ruten bruker den for begge hoder. Den gamle v2-filen er byteidentisk
  med kvalifisert featurevisning.
  Neste kandidats F-vekt er bundet til 1,0 mot S; ingen scorevalgt vektsøk.
  11 fokuserte tester består. Ny fysisk nullstegskjøring på to ekte
  TRAIN-rader besto: S-/F-hode og backbone fikk gradient; ukjent S-side og
  FLATT F-side fikk null direkte sidegradient. Vektene var bitlike før/etter;
  én forward/backward, null optimizer, null TEST. 87,6 s og 5,35 GiB topp-RSS
  under 12 GiB kapsel. [Ny nullstegs-PASS](/home/andre2/GX1_RUNS/MARKET_STATE_NATIVE_F_CPU_ZERO_STEP_20261108_002/RESULT.json)
  SHA `7302d760741ae0329bd936a3bc0461dfedc2ada3a337fb23700bb2be7207b84b`.
  Dette beviser teknisk kobling, ikke læring eller retning. Policyens
  `market_state_native_F_side_normalized_design_20261108_001` binder designet.

- **Optimizer/gjenopptakelse, faktisk teknisk PASS:** Én guarded native S/F-prøve
  på de tidligere valgte TRAIN-radene 2–33 fullførte med seks forwards,
  fire backwards, fire AdamW-oppdateringer og tre modellkonstruksjoner.
  Avbrutt/gjenopptatt forløp var bitlikt referanseforløpet; trener- og
  observatørutgang var 0. 132,8 s, 5,76 GB topp-RSS og 1,98 GB toppallokert
  GPU-minne under 20 GiB/250 W-vakter. Planen er konsumert, og ingen
  prediktor-fit eller retningsverdi følger av dette.
  [Resultat](/home/andre2/GX1_RUNS/MARKET_STATE_NATIVE_F_PRIVATE_4STEP_20261108_001/RUNTIME/private-research/RESULT.json)
  SHA `55936ccfb06bddcebebfb10a6a591de29143fae3f64036594463f04985ccd97e`;
  [kampanjekvittering](/home/andre2/GX1_RUNS/MARKET_STATE_NATIVE_F_PRIVATE_4STEP_20261108_001/RUNTIME/receipts/invocation-0001.json)
  SHA `d274e2f0ea3d27842645105199b92761562965562cb2752d85169cd6fbc08f13`.
  Policyens `market_state_native_F_private_4step_20261108_001` binder
  engangsutfallet.

## Ikke gjort, hindring og neste steg

Native S/F har **ingen** full kronologisk fit eller senere femfolds OOF.
GC-merverdi i den native tidssekvensmodellen er ikke målt. Verken framtidig
markedstilstand, senere Entry-beslutningsverdi eller økonomisk edge er påvist.
Den enkle tabellmodellens negative retning er ikke en generell dom over GC,
indikatorene eller native modell.

**Neste steg:** Fullfør ny recovery-campaigns preflight og maskinvid
idle-kontroll; behold kravet om fersk Windows-boot. Kjør den nye planen
med full state-restore og fortsett kun resterende pass, FINAL og senere
TRAIN-score. Kontroller stabil drift hver 15.–30. minutt og bruk publiserte
fasetider til den avgrensede hastighetssammenligningen. Kilden og policyen
fryses gjennom kjøringen; original avbrutt plan relanseres aldri.
Brukerens spørsmål om 30 epoker og fem år beholdes som åpne hypoteser.
Dagens fit dekker omtrent mai–september2024, senere score oktober–desember2024.
Ett kort pass er ingen generell dom over lengre læring eller historikk.
Ingen slik utvidelse eller etterfølgende utvalg på forseglet TEST er innført.
Native femfolds generalisering, samme-linje-reaksjonsverdi, Entry-verdi og
økonomisk edge er ubevist.

Den tidligere [GC-registerplanen](configs/research/MARKET_STATE_GC_SIDE_MAX_REGISTRY_20261108_001.json)
forblir bevart, men er utsatt og uklamet etter brukerens nye prioritering.
`training_enabled=true` gjelder bare den nye eksplisitte recovery-planen.

Kilde/metode for denne fasen: [docs/NATIVE_LEARNING.md](docs/NATIVE_LEARNING.md).
Detaljert historikk og eldre beslutninger: Git, ikke nye arkivmapper her.
