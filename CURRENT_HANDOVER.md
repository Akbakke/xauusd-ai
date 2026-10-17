# CURRENT HANDOVER — 17.10.2026

Autoritet: /home/andre2/src/GX1_CURRENT, work/gx1-current. NEXT_RUN_POLICY.json er
eneste arbeidsstatus; prosess og terminal overstyrer prosa. Historikk per forsøk
ligger i Git og i policyens bindinger, ikke her.

## Nå: uavhengig Entry-target-audit fullført før videre P0

Audit 010 fullført én gang, exit 0, kilde uendret, TEST=false. Formler, kodeeiere,
klokker, kostnader, faktiske eksempler og grenser: docs/ENTRY_TARGET_AUDIT.md.
652 552 TRAIN-rader er uavhengig rårekonstruert ved 1/5/20 observerte sesjoner;
bitlik brutto/elapsed/netto på alle gyldige rader. VAL: bare 1 792 råstøttede
prefiksrader kontrollert. Original M1s siste row-group krysser TEST og er ikke
dekodet; VAL-halen er ikke rårekonstruert. 008/009 feilet i auditoren, er
konsumert og bevares; 010 rettet kun kanoniske bindinger og delvis råsesjon.

Faktisk gjennomført native RECENT_YEAR-recipe binder Entry-only h20d, uten
lært Exit; primært tap predikerer netto terminal markout LONG/SHORT/FLAT=0.
1/5 sesjoner er observasjon; gammel 95m-auxiliary og joint Exit-teacher-fase er
andre baner. Native brutto har korrekt inngangsnotional og bid/ask-spread én gang.
Kostnad 4 bps + veggklokke-finansiering LONG 569 / SHORT 0 årlige bps (klippet
kreditt); aktuell autoritet, ikke historisk swapfasit. Tillatt sluttforsinkelse
når 50 340 s; komplette fraværssesjoner hoppes over og indre M1-gap avviser ikke
terminaleieren. Kalenderstengt versus leveransegap er ikke klassifisert.
Korrekt rekonstruksjon beviser derfor ikke god inngang eller generalisering.

GC/M5 lærer kostnadsfri mid-logretur ved nominelt 30/120/360 minutter og 60 s
inngangslatens. Alle 17 565 labels er rårekonstruert bitlikt. Separat SHORT-
regnskap bruker feil utgangsnevner mot native inngangsnotional: gjennomsnittlig
hypotetisk overvurdering 0,0257 / 0,1153 / 0,4016 bps, maksimum 19,22 bps.
Dette endrer ikke publisert 007 MSE og er ikke målte strategiresultater.
Finansierings-/utførelseskonvensjonene er ulike mellom sporene. Ingen target,
Exit, produksjonsmodell eller fryst baseline er endret under auditen.

11,81 % er nå eksakt forklart på faktisk M5-serie: 141 806 → 88 834 usignert
kvalifiserte → 16 745 helt signerte. Hele 72 089 ellers gyldige vinduer har noe
ukjent aggressor, selv om 97,62785 % av observert volum har kjent side.
Duplikatsignatur fjerner 50 936 i deklarert waterfall; fullrecord-duplikat er
ikke bevist fordi gammel signatur utelater BBO/depth/ts_in_delta. Gapflagg fjerner
135 i samme rekkefølge, rull inne i M5 10; overlappende marginaler ligger i RESULT.
Reset/bokkontinuitet er ikke egne eksisterende Trade-maskekrav og forklarer
ikke automatisk denne dekningen. Separat L1-tilstandsaudit gjenstår. Ingen masker
svekket, ingen N-side gjettet, ingen rå GC-passering i audit 010.

Neste: avgrenset L1-kvalifisering på valgt kontrakt, deretter eksisterende parser
og ekte GC VWAP/POC/VAH/VAL med egne masker. ATR/RV/session gjenbrukes. Ny P0-
recipe må binde symmetrisk handelsregnskap og matched armer før fit. Ingen ny
modelltrening/strategiøkonomi her; netto, Sharpe og drawdown er ikke målt.
Native TEST, live, nye data/kostnader og relansering av konsumerte planer er stengt.

## Nå: strategisk revisjon og P0-prioritet — 17.10

Ny bestilling prioriterer dynamiske GC-volumnivåer × L1-flow × volatilitet,
med P1–P5-familieablasjoner før eventuell Transformer/RL-utvidelse. Analyse og
baselinefrysing er fullført i docs/FEATURE_AUDIT.md. Felt-/kildeinventar:
docs/FEATURE_AUDIT_INVENTORY_20261017.json; 707 filer, 583 Python AST-PASS,
315 native felt, åtte faktiske spesialister, 132 aliaser, 190 MTF-felt/klokke.
configs/research/XAU_M5_INFORMATION_BASELINE_20261017.json binder eksisterende
007-A/inputs/prediksjoner/folder og separat eldre økonomireferanse. Ingen rerun.

Operatøren oppgir 36 kjøpte måneder; lokale nedlastingskvitteringer og 66 MBP-1-
gzipfiler (20,04 GB) dokumenterer 26 måneder mai 2024–juni 2026. Ingen ekstra
leveranse er verifisert; ikke kjøp/hent data for å fylle differansen. L1-BBO/
OFI er hovedgapet; trade-masker kvalifiserer ikke bokhendelser. Makro-as-of bruker
deklarerte lag, ikke per-utgivelse-/vintagebevis. Alle dokumenterte GC-måneder
har utviklingsbruk; ny splitting skaper ingen urørt test. Native TEST er lukket.

Ny forskningsvisning pauser 27 deklarerte candle-/geomchan-felt; bare planlagt
view, ingen native felt eller kode slettet. Behold original full baseline for
attribusjon. Eksisterende ATR/RV/SMC/EMA og 61 makrofelt gjenbrukes; P3/P5 må
ablateres uten dubletter. XGBoost finnes ikke i miljøet; tidligere GBDT er
LightGBM/HGB. Åtte P0-armer er skissert, ingen execution-recipe eller fit bundet.
Ingen nye features, eksterne datapakker, installasjoner, økonomikjøringer eller
kostnader i revisjonen. training_enabled=false og tidligere sperrer består.
Neste konkrete arbeid er en liten forhåndsbundet L1-kvalitets-/dekningsplan i
eksisterende GC-eier før P1-featurekode. Gratis ressurser/lisenser og forventet
arbeidskostnad er vurdert i rapporten; ny modellarkitektur er ikke foreslått.

## Fullførte M5/MTF-forsøk før strategisk revisjon


Brukeren avklarte «Vi holder oss til M5 intradag (med mtf så klart).» GC-
Trade-featurebygger 006 er fullført på innrømmede 004-masker; bare eksisterende
lokale bytes, ingen nye Databento-kall/kostnader. Eksisterende M5/M15/H1/H4/D1-
klokker og native familier beholdes. Byggeren lager observerte volum/profil/
footprint/delta/relativ CVD og kausale SMC-order-block-/pris-sweep-proksier.
Institusjonell ordresannhet/multinivå-sweeps og ukvalifisert book-OFI bygges
ikke. Plan/forhåndsformler: configs/research/GC_M5_FEATURE_BUILD_20261017_005.json,
docs/GC_M5_FEATURE_RECIPE.md. 100 fokuserte mekanikk-/SMC-/handover-tester og 41 kilde-/disposisjonstester
består. En gammel kilde-test krevde fortsatt at alle steg var uferdige etter
004; den skiller nå maskert input-admission fra komplett tape-kvalifisering.
005 fullførte råpassering/avstemming og beregning, men feilet på rapportens
NumPy-int64 → JSON før publisering. Ingen ferdig featurefil ble publisert av 005.
Feil, logg og delrapport er bevart; 005 relanseres ikke. 006 er ny, claimet
minimal gjenopptakelse: samme formler/masker, gjenbruk av 15 hash-bundne
ladderfiler (ca. 56 MB), gjenoppbygging av ikke-persisted OHLC-rekkefølge og
ny eksakt avstemming. Syntetisk JSON-/gjenbrukskontroll består. Ekte 006-
publisering er fullført (006 exit 0). 141 806 native M5-rader, 255 kolonner,
11,8 MB; 60 råreferansebarer, eksakte avstemminger og Parquet readback/hasher
består. M5 volum/profil 88 834 rader; signert footprint/delta/relativ CVD 16 745,
begge alle 26 måneder. M5 OB-pivotmasker 9 211/9 476 rader, alle 26 måneder.
M15 volum/profil og H1 volum/profil har 26-måneders støtte; M15 signert bare
25 måneder, H1 signert tre unike barer, H4 volum én bar, D1 ingen. Ny GC-flow
på H4/D1 er derfor ikke treningsgrunnlag. H4-prefix har suffix=0 (triviell),
D1 har ingen gyldig komplett bar. Eksisterende native MTF beholdes.
RESULTS/M5_FEATURES.parquet sha256 9d5f035b43e9b3889a8492ad0d855adc42a3c0a2b1376b6568433713d296469f.
Ingen outcomes, fit, TEST eller økonomi er lest/kjørt av bygget. Book-OFI og
native integrasjon/paritet er ubevist; hele opprinnelige GC-protokollens steg 2
er derfor ikke feilaktig markert fullført. Matched A/B/C-forskningsfit 007 er
fullført med eksisterende native snapshots og MTF-kontekst (resultat under).
Native trening/normalisering, TEST, økonomi og launch er fortsatt stengt;
native GC-representasjon/paritet og konkret M5-mål-/recipebinding gjenstår.
Opprydding/komprimering er fullført; dette er ikke lenger en blokkering.

Historisk: minimum native læring på D1-horisontmålet (vedtak 12.10) er gjennomført
P0–P3 og gjennomgått: REJECT_ENTRY_QUALIFICATION. Modellen lærer de trente
radene og generaliserer ikke til CONTROL4096 ved 4 096 steg. Gjennomgangsbølgen
14.10 (seks leserettede agenter) er bokført og fiksbølgen er brukt (kode,
retention-eier, repo-opprydding, policy/dokumenter). Neste læringsfase krever
operatørvedtak: brukeren foretrekker et mindre kronologisk vindu (ca. ett år)
før noen full epoke: fasen «recent_year» (siste 365 dager før TRAIN-kuttet,
65 148 kvalifiserte rader, 4 071 steg) er kjørt og gjennomgått 14.10:
REJECT etter forhåndsregelen, men med et retningssignal kurven ikke hadde
(se under). Neste fase krever operatørvedtak. Retention-
bølge 1 er utført gjennom eieren (35 mål, 78,3 GiB, DELETE_COMPLETE). Ingen
dokumentert Entry-edge, ingen prediktor innrømmet, training_enabled = false,
TEST forseglet.

007-originalterminalen mangler `source_unchanged` og er bevart som den er.
Nyeste faktiske terminal er nå audit 010 med eksplisitt `source_unchanged=true`
og `test_data_used=false`. Dette er auditbevis, ingen native launchautoritet;
training_enabled=false og øvrige læringssperrer består.

## Avgrenset M5/MTF Trade-probe 007 før native adopsjon

Ny frosset plan GC_M5_TRADE_FLOW_PROBE_20261017_007 viderefører brukerens
GC-/M5-bestilling og den enkle ridge-referansen fra forskningsprotokollen.
Samme 16 745 strengt kvalifiserte M5-rader, før utfall: A = alle 315 native
felt (132 MTF/ctx-aliaser kontrolleres og telles én gang) + TRAIN-domene for
session. B = ekstra GC-pris/SMC-prisproksier; C = ekstra observerte volum-/
profil-/footprint-/delta-/CVD-felt. M5, M15 og H1 brukes bare der kilden har
26-måneders støtte for feltgruppen; native H4/D1-kontekst beholdes.

30/120/360 minutter og ridge-oppsett gjenbrukes fra den eksisterende M5-
eieren; de fem ekspanderende GC-foldene er uendret. Full fysisk TRAIN eier
scaleren; indre 20 % brukes til de seks allerede deklarerte alphaene. 45
endelige modeller maksimalt. Feltvalg fra bare initial-TRAIN-inputs og kohort
låses før utfall. C−B og B−A er én seks-kontrast-familie med eksisterende
ukeblokk-bootstrap. Ingen strategi-/økonomisimulering, TEST eller native fit.
Dette er snapshot-/kontekstforskning, ikke native sekvens/attention-bevis eller
fullføring av opprinnelig C-arm med book-OFI. 007 er fullført én gang (exit 0, 120,4 s, 45 modeller). A/B/C har 325/393/473
felt etter initial-TRAIN-inputlås; alle 315 native felt og 132 aliaser beholdt,
GC-diagnostikkene ukjent=0/kjentandel=1 er konstante i denne selektive kohorten
og brukes ikke som nye lærte felt. Ingen indikator er fjernet fra native.

Utviklings-MSE i bps² (A / B / C / TRAIN-konstant):
30m 253,844 / 253,944 / 254,072 / 253,795 (5 935 rader);
120m 1 156,887 / 1 157,811 / 1 158,956 / 1 151,942 (5 894);
360m 4 052,491 / 4 054,063 / 4 056,442 / 4 041,438 (5 736).
TRAIN-fit faller litt med nye felt, mens senere feil øker litt. Alle seks
familiekorrigerte C−B/B−A-intervaller dekker null: INCONCLUSIVE, ingen
positiv inkrementell prediksjonsstøtte. Dette er ingen falsifisering av native
M5-footprint eller sekvens/attention; slik modell er ikke trent her.

RESULT.json sha256 4976db1e0e8b658894684d179536f092947eea75213d8f55726f00eff22a18f4;
TERMINAL.json sha256 40e5528f4e83d55d3fc699998aa817c0aab07f92e1f6ee6162869ce8f260bcce.
COMPLETION_REVIEW rekonstruerte alle 17 565 observerte mål uavhengig fra
M1-kvotene: 0,0 bps avvik; samme armer/rader, MSE og split-exitgrenser består.
Modeller/prediksjoner ligger bare i RUNS, ingen vekter/rådata pushes.

M5-footprint er bevart og kan vises prisnivå for prisnivå. Reelt eksempel i
run-root M5_FOOTPRINT_EXAMPLE.{csv,png,json}: GCQ6, 30.06.2026 09:15–09:20 UTC,
19 observerte prisnivåer, 75 kontrakter (22 B-kjøp, 53 A-salg, 0 ukjent), eksakt
avstemt mot maskert M5-volum. Eksemplet velges for kompakt fullnivåvisning,
ikke etter utfall. Ingen TEST, økonomi, API/spending eller native fit.

Native GC-representasjon/masker og ekte train/serve-paritet gjenstår før egen
kildebundet M5-læringsplan kan vurderes. Ingen blind GPU-fase eller relansering
av konsumerte planer; tidligere D1-recipe er ikke denne M5-bestillingen.

## Minimum native læring (vedtak 12.10): gjennomført og gjennomgått

Generasjon HISTORY2009W_NATIVE_D1_20261011_003 (315 signalfelt med den niende
D1-makro-/posisjoneringsfamilien, 132 kontekstfelt, 1/5/20-handelsdagers
observerte mål i TRAIN/VAL, TEST forseglet). Alle tall er målt på ekte
deklarerte bytes; policyblokk native_d1_minimum_learning_20261012 binder hvert
steg med hash.

P1 (CPU, run-root NATIVE_D1_REBUILD_20261011_003): NATIVE_PREPARATION_003
(frosset design; horisont-kutt 2025-05-02T19:20Z, 644 255 kvalifiserte av
652 552 TRAIN-rader, 3 284 sensurert; VAL 65 575 komplette av 70 880) →
SAMPLER_BENCHMARK_001 stoppet (projisert over vinduet, lukket) →
SAMPLER_BENCHMARK_002 (57 822 s; valgt budsjett 32 768, batch 16; målt
CPU-epoke 7 906,5 s) → NATIVE_METADATA_HANDOFF_001 → NATIVE_PREPROCESSING_001
(labels/normalisering begge splitter) → NATIVE_COORDINATES_001 (epoke-0-
rekkefølge, TRAIN256, CONTROL256, TRAIN4096/CONTROL4096 uten utfallslesing).

P2 (GPU, fersk fysisk Windows-boot per invokasjon, brukerens omstarter):
- Fit forsøk 1 feilet i preflight (CANDIDATE_PREFIX_PHYSICAL_LABEL_MISMATCH →
  trenerfix 528b754d: label-rader vs kvalifisert delmengde). Forsøk 2 (root
  ENTRY_D1_HORIZON_LEARNING_20261012_002/FIT) fullførte 1 024 steg på TRAIN256:
  kontrast-rank-IC ved 20 d 0,07 → 0,97 (memorering; mekanikk og tapsfall).
- Kurve forsøk 1 feilet i preflight (kontrollvisningens scope → dispatcher
  f72a430b). Forsøk 2 (root …_003/CURVE) fullførte 4 096 steg på 65 536 unike
  TRAIN-rader i frossen rekkefølge. TRAIN_PROBE4096 kontrast-MSE 725 499 →
  116 061 (rank-IC 0,96). CONTROL4096 kontrast-MSE 2 032 815 → 2 157 135 mot
  TRAIN-konstant 1 997 171; rank-IC 0,039 → 0,040 ved 20 d, 0,001 → 0,085 ved
  5 d, 0,026 → 0,013 ved 1 d; handlinger LONG/SHORT/FLAT 822/192/3 082 →
  2 893/997/206; valgt-handling-snitt +50 → −3 bps netto mot alltid-LONG
  +148,5 (alltid-SHORT −204,4).

P3 (STUDY_REVIEW.json v2 i root _003, REVIEW_STUDY.py, REVIEW_SUPPLEMENT.json
forhåndsregistrert): alle mål gjenoppbygget uavhengig fra parquet + kostautoritet
(float32-eksakt); seks familiekorrigerte parvise intervaller på CONTROL4096 ved
20 d dekker alle null (kontrast vs initial, ukesnitt +112 537, [−776 321,
+861 594]; vs TRAIN-konstant +146 307, [−674 657, +830 246]); 6-ukers
blokksensitivitet og per-måned-instabilitet rapportert, ikke besluttende.
Vedtak etter forhåndsregel: REJECT_ENTRY_QUALIFICATION. Grenser: CONTROL er
gjenbrukt utviklingsdata; komplette utfall til 2026-06-02; overlappende
20-dagers markouts, ikke portefølje; ett seed; dagens kostpolicy er ikke
historisk kostsannhet; ingen økonomi, Exit-rollout eller edge.

## Gjennomgangsbølge 14.10 og fiksbølge

Funn og evidens: …_003/REVIEW_WAVE_FINDINGS.md (bundet i policyblokken
fix_wave_20261014). Brukt:
- Epoke-sti (A): observasjonskadens-feil rettet (35270857); epoke = 40 266 steg
  i fire 12 000 s-vinduer, resume ubevist på ekte data; ikke lansert.
- Dokument/policy (B): stale skalarer, feilgenerasjons-bindinger, sha-avvik,
  smoke-/kontrollsteg-ordlyd, plan-/CTX-/regeldokumenter rettet i denne commiten.
- P3-verktøy (C): referansehjelpere og REVIEW_STUDY v2 (35270857, 91e26fb7).
- Repo-opprydding (D, 270cd8c7): 38 filer, −61 191 linjer (frakoblede wrappere,
  keeper-skript, 1,98 MB feltregister, døde rolling-varianter). Beholdt med
  bevist levende importør: v12_trade_state, model-free/macro-event/pattern-
  instrumentene (importeres av research_ta_campaign via intraday), OANDA-
  backfill/credentials (xau_tape_provenance), Exit-wrappere med tester,
  pretest-recipe (seq513-beredskap), kildemanifester i configs/research.
- Retention (E, 35270857): launch-kontrakten beskytter 22 deklarerte leaves,
  bare deklarerte nøkler er referanser, mål navngitt i gjeldende dokumenter
  nektes, CLI tillater DATA+RUNS; hendelsesvitnene (juli-inventarene) bindes
  uten traversering (511a668f). Bølge 1 UTFØRT 14.10 (vedtak GX1_RETENTION_
  LEGACY_PREBUILT_WAVE1_20261014, policyblokk retention_wave1_20261014): plan
  → validert tørrkjøring → godkjenning → utførelse, 35 eksakte mål, 8 138
  filer, 78,3 GiB (LOCAL_PILOT, V42, NATIVE_EARLY, 2006/2009-bootstraps,
  LIFECYCLE_V2_FULL_TRAIN-barn ≥ 100 MiB; PROSPECTIVE_COST_POLICY_V1 beholdt).
  Disk 524 → 456 GB brukt. Bølge 2 krever dokumentlukking: V38 ×2 (64),
  FEATURE_REPAIR (34), D1-feilkjerner (10), V46 utenom beskyttet leaf (117).
- Datasveip (F, målt på TRAIN, bundet som data_sweep_20261014): 0 døde/ikke-
  endelige/mettede felt; 132 ctx-par bitidentiske; ett udeklarert duplikat
  (rsi14_centered ≡ m5_rsi14_canon_v2); smc_choch = up + down; 11 felt IQR-
  disjunkte 2012 vs 2024 (rente-/GLD-nivåer, spread, rv, COT) = epoke-proxy-
  risiko for nivåfelt; COT ukentlig som ventet.
- Runner (F): én eksklusiv flock for alle klasser + krav om 20 GiB ledig;
  vedtak 11.10 (b) er ikke implementert. Åpent.

## Swing-program og forward-journal (status uendret siden 11.10)

Frossen 68-feltsflate (A, COT, panel, drivere) med swing_ridge h20 på XAU D1:
treff-familiesignifikant selektivt (009: 0,872/0,831/0,784/0,698 ved
2/5/10/25 %, familie-p 0,022/0,0085/0,007/0,026); diagnosen er NÅR en 20-
dagers long betaler, ikke retning; forward 2026H1 støttende men INKONKLUSIV.
Journalen (run-swing-forward-journal, blokk forward_journal_20261011) scoret
juli–september 2026 én gang hver: alle LONG, null flagg; rank-IC innen måned
0,50/0,39/−0,10. Årsregelen krever 12 løste måneder. Ingen prediktor innrømmet.

## Baselinjer (ingen læring)

FRESH_INITIAL_BASELINE_003 (akseptert generasjon, 254 felt, 9 637 663 param.)
og FRESH_INITIAL_BASELINE_001 i D1-generasjonen (315 felt, 9 661 087 param.,
385 s, 512 ONLINE-forwards + TARGET-paritet bitidentisk, null optimizersteg):
utrent initialtilstand på samme 4096/4096-koordinater; kontrast-MSE 3 398
TRAIN / 7 363 CONTROL i D1-generasjonen. Ikke læring eller edge.

## Bevart avvik og grenser

- SAMPLER_BENCHMARK_001 ble stoppet på projeksjon; i ettertid ville forsøket
  trolig rukket vinduet (rate-degradering mindre enn V38). FAILURE/TERMINAL bevart.
- Kurve/fit forsøk 1 feilet i preflight og er bevart i egne røtter (_001, _002).
- Studiespesifikasjonen binder _001/.specs-kopien av horisontdesignet (hash OK);
  designdokumentet er hash-bundet med stale ordlyd (dokumentert i policy).
- Retention-nektelsen 12.10 (stagekatalog uten manifest) er avløst av eierfiksen
  14.10; de to feilede D1-kjernegenerasjonene navngis fortsatt av sine
  configs/research-spesifikasjoner og går i bølge 2. Ingenting er slettet ennå.
- GLD fra Internet Archive slutter 2026-02-13 (carry-forward i 008–015/010);
  SPDR-arbeidsboken brukes av journalen. Kommandomelding 18856469 oppgir feil
  ubetinget referanse (+29 bps); riktig er +63 bps.
- Verten frøs 14.10 05:17 CEST (Kernel-Power 41, ingen dump) uten GX1-last;
  WHEA-korrigerte feil i september. Lange kjøringer forblir segment-/resume-
  sikret.
- Ukjente rådatagap er ukjente; ingen imputering. Deklarerte kostvilkår er ikke
  historisk kostsannhet. CONTROL er gjenbrukt utviklingsdata.

## Fasen «recent_year» (vedtak 14.10 b): kjørt og gjennomgått

Resultat (målt på ekte bytes; root …/ENTRY_D1_HORIZON_LEARNING_20261014_001,
STUDY_REVIEW_RECENT_YEAR.json). Invokasjon på boot 498, 18:20–19:27Z, exit 0,
toppeffekt 138 W, kjerne 58 °C. CONTROL4096 ved 20 d (samme rader som kurven):
- Kontrast-MSE 2 032 815 → 1 877 306; konstant fra de trente vindusradene
  1 929 010. Alle seks punktestimater er forbedringer, men alle seks
  familiekorrigerte intervaller dekker null → REJECT_ENTRY_QUALIFICATION.
- Rank-IC 0,039 → 0,189 ved 20 d, 0,001 → 0,160 ved 5 d (kurven: 0,040/0,085).
- Handlinger LONG/SHORT/FLAT 822/192/3 082 → 2 324/1 382/390; valgt-handling-
  snitt +50 → +128 bps netto mot alltid-LONG +148,5 (kurven: −3).
- 9 av 13 måneder er verre enn minst én referanse (ustabilt over tid).
Lesning: siste-års-rader gir et retningssignal på utviklings-CONTROL som hele
historikken ikke ga, men det er ikke statistisk kvalifisert på 4 096 rader
med overlappende 20-dagersutfall. Gjenbrukt CONTROL, ett seed, ingen økonomi.

Avvik: confirm-reboot med nonce ble ikke kjørt før omstarten; controlleren
startet likevel (reboot-pending/after-0000.json bevart). Etter kjøringen sto
GPU-en fast i P0 på ~139 W uten eier: ingen /dev/dxg-holder i WSL, bare
C+G-skrivebordsprosesser i Windows, GX1-klokkelåsen bevist frigjort, og
driveren rapporterer «Idle: Active» mens den blir i P0 (WDDM 591.86,
sporadisk — boot 497 gikk til ~34 W etter samme syklus). Diagnose hash-bundet
i run-rotens HOST_GPU_IDLE_P0_STUCK_20261014.json; ryddet av operatørbestilt
omstart. Stående praksis: verifiser tomgangstrekk etter hver GPU-fase; fast
P0 uten eier blokkerer neste invokasjon til maskinen er omstartet.

Implementasjon (6f058c5d, dc2f4f74): vindusrader og stegbudsjett
avledes av kontrakteieren fra den bundne TRAIN-klokken (365 dager før kuttet:
65 148 rader → 4 071 steg, 65 136 trente rader; nesten samme antall som kurven,
men siste år); frossen rekkefølge begrenset til vinduet først; TRAIN-probe =
første 4 096 vindusrader ved 0/256/1 024/4 071; CONTROL4096 uendret ved 0 og
4 071; samme seks-grense-regel. Spec: configs/research/ENTRY_D1_HORIZON_
LEARNING_STUDY_20261014.json (binder forgjengeren); root
/home/andre2/GX1_RUNS/ENTRY_D1_HORIZON_LEARNING_20261014_001 (RECENT_YEAR/).
Kjørebok: P2_RUNBOOK.md i roten.

## To-års-vinduet (operatør «kjør» 14.10): kjørt og gjennomgått — recency-forfall målt

Resultat (root …/ENTRY_D1_HORIZON_LEARNING_20261015_001, STUDY_REVIEW_RECENT_
YEAR.json; invokasjon boot 500, 20:08–21:55Z, exit 0, topp 146 W; GPU-tomgang
ren etterpå, 36,9 W). CONTROL4096 ved 20 d, samme rader som alle gjennomganger:

| | utrent | 365 d-vindu | 730 d-vindu | kurve (full) |
|---|---|---|---|---|
| kontrast-MSE | 2 032 815 | 1 877 306 | 2 327 431 | 2 157 135 |
| rank-IC 20 d | 0,04 | 0,19 | −0,02 | 0,04 |
| valgt handling, bps | +50 | +128 | +96 | −3 |

REJECT etter forhåndsregelen (fire av seks intervaller dekker null; mot
vinduskonstanten er nedre grense nær null på feil side). Den forhåndsregistrerte
lesningen slår til i informativ retning: å doble vinduet gjorde generaliseringen
VERRE enn utrent — signalet bor i det siste året, og eldre nær-regime-data
skader aktivt. 11 av 13 måneder verre enn en referanse. Beskrivende: gjenbrukt
CONTROL, ett seed, ingen økonomi. Ingen fase er kvalifisert; full epoke består
stengt (vedtak 14.10 b).

## M5 intradag-recheck 15.10 (operatør: M5 er ikke falsifisert på ny data)

Data, målt fra manifester og artefakter (run root …/M5_RECHECK_20261015_001/audit):
- V38 og FEATURE_REPAIR har samme 254 felt; reparasjonen endret bare 5 snap-felt
  (geomline, ≤ 0,8 % av radene) — resten ligger i MTF-sekvenser, M1-Exit og etiketter.
- NATIVE_D1_003 = REPAIR bit-likt + 61 nye D1-felt (4 COT, 44 panel, 13 drivere).
- Treninger: 95-min-studien kjørte på V38; ingen på REPAIR; på D1_003 bare
  20-dagersmål. Ingen intradagtrening fantes på ny data før 15.10.
- Tilgjengelighet: uavhengig as-of-gjenoppbygging av drivere/COT = 0 avvik på
  723 432 rader; COT brukt ≥ 99,6 t etter fredagspublisering; panel-D1-barer
  inneholder ingen H1 etter stempelet; MTF-kontekst endrer seg på egne lukkinger.

Studie (spec M5_INTRADAY_RECHECK_20261015_001, forhåndsregistrert og committet
7246795e før første tilpasning): 30/120/360 min etter lukket M5-bar, ridge +
LightGBM (XGBoost-stand-in, se avvik), sett A (254) mot B (+61 D1), 11
walk-forward-perioder 2015–2026H1 med 3-års vindu, purge + embargo, utførelse på
M1 bid/ask med 60 s latens, 2 bps slippage per utførelse, finansiering per
rollover, én posisjon. Resultat: 10 av 12 celler NEGATIVE, 2 INKONKLUSIVE bare
pga. få handler, ingen LOVENDE. Netto per handel −4,6 til −10,5 bps; brutto
(spread inkludert, før slippage) negativ i alle celler; alltid-LONG/SHORT ≈ −6.
Rank-IC per periode i snitt −0,014 til +0,012. B er ikke bedre enn A. Handler
verifisert uavhengig mot rå tape (0,00 bps avvik, ingen overlapp).

Ikke testet: transformer-utfordrer (krever minutthorisont-målkontrakt, og
walk-forward i TRAIN kolliderer med regel 18), sett C (GC-kjøp), ekte XGBoost
(miljøendring). GC: 26M-kilden fikk maskert input-admission i 004; Trade-features 006 og avgrenset forskningsfit 007 er nå fullført. Ingen komplett flow, native trening eller prediktiv kvalifisering er bevist.
Databento: Kun kildehenting innen samlet hardgrense USD 120.00 var autorisert. Kjøringen ble fullført 2026-10-16 06:56:38 UTC: 79/79 jobber og 63 nye datafiler. DOWNLOAD_SUMMARY.json rapporterer ALL_BATCH_JOBS_COMPLETE_FILES_DOWNLOADED og actual_or_quoted_completed_job_cost_usd USD 119.65511028468609. Ingen nye batch-submit eller pay-as-you-go-forespørsler. Brukeren rapporterer USD 5 remaining credits; saldoen er ikke uavhengig verifisert. Databento opplyser at batch-jobben belastes én gang ved innsending, og at samme data kan lastes ned igjen uten ekstra kostnad i 30 dager. Den ufullstendige .part-filen ble pensjonert gjennom retention-eieren; kanoniske gzipfiler/kvitteringer beholdes. Det gamle hentevedtaket åpnet ingen fit/featurebygging; senere 004/006/007-evidens er angitt ovenfor. Native trening, TEST, broker/live/paper og handel forblir stengt. 180 d-fasen forblir desarmert.

## GC-order-flow: plan 002 strukturrevidert; kildekvalifisering fortsatt åpen

Den forhåndsregistrerte studiedelen er [2024-05-01, 2026-07-01) UTC, nøyaktig 26 kalendermåneder. Den omfatter 12 måneders første TRAIN og fem ekspanderende utviklings-VAL-vinduer (3+3+3+3+2 måneder). Supporting definition- og ohlcv-1d-data starter 2024-04-01 for én måneds kausal volum-warmup. Ingen slutt-TEST-fold; TEST starter 2026-07-01 og er ikke åpnet.

Databento-hentingen ble fullført 2026-10-16 06:56:38 UTC: 79/79 scope-jobber og rapportert total USD 119.65511028468609 mot USD 120.00. Dette er de allerede nedlastede filene; ingen nye Databento-kall, Pay-as-you-go eller kortbelastning er autorisert. Brukeren rapporterte USD 5 remaining credits, saldoen er ikke uavhengig verifisert. Kildeinventaret avstemmer 66 MBP-1, 23 definition- og 23 ohlcv-1d gzip-filer; MBP-1-dataene er 20038896092 byte. Definisjonsauditen bandt 15 outright-symboler til unike instrument_id/publisher_id-par.

Auditforsøk 001 brukte 4-GiB auditvakt og feilet på GCG5-rad 100: action F, side B. Alle 66 fil- og kvitteringshasher var kontrollert før parsing; ingen delresultater ble publisert. Databentos dokumentasjon definerer F som en passiv resting-order-fill, ikke en ekstra aggressorhandel ([common fields and flags](https://databento.com/docs/standards-and-conventions/common-fields-enums-types)). Eierkoden ble rettet til å telle F som passiv handling; 36 fokuserte tester passerer. Plan 001 relanseres ikke.

Plan 002, [GC_SOURCE_AUDIT_20261016_002.json](configs/research/GC_SOURCE_AUDIT_20261016_002.json), ble godkjent for én kjøring («Ja», 2026-10-16), kvittering /home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_SOURCE_QUALIFICATION_20261016_002/OPERATOR_APPROVAL.json (sha256 799008e5c49192d83886705ca174b901386439771988e3742d30676c41498376). Den ble kjørt én gang via scripts/gx1_capped_run.sh --class audit og fullførte med exit 0. Resultatet er STRUCTURAL_AUDIT_COMPLETE_NOT_ADMITTED: /home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_SOURCE_QUALIFICATION_20261016_002/AUDIT_RESULTS/RESULT.json (sha256 c869e3d1395c5892fab19d7c3056faaca1ca4aaac41f70d3f05943d4e63dcbab); terminalen er COMPLETE_NOT_ADMITTED (sha256 e513dc5f58f360cf52e06e53d280f3ce1b854a808a020c224159223dcfc0c1f8). Auditresultatet hash-verifiserte 66/66 kilde- og kvitteringspar og telte 892100712 rader, inkludert 62783740 Trade-rader; ingen nye data, fits, TEST eller modelladgang.

Etterkontrollen /home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_SOURCE_QUALIFICATION_20261016_002/POST_AUDIT_RECONCILIATION.json (sha256 14288c053ad5ca972577c854183fee93abe972565ba92cc1a21ae90b75f41cda) bekrefter at de 15 definisjonsmappingene matcher alle 66 fildeklarasjonene. Receive-aktivitet finnes på 678 av 791 kalenderdatoer; alle 113 datoer uten aktivitet er lørdager. Dette samsvarer på dagsnivå med CME Golds publiserte søndag–fredag-sesjon med daglig pause, men beviser ikke intraday-dekning eller historiske helligdagslukkinger ([CME GC hours](https://www.cmegroup.com/trading/metals/files/fact-card-gold-futures-options.pdf)).

Kvalitetsmålingene inkluderer 2391688 trades med ukjent side (3.81%), 3630329 kontrakter ukjent aggressorvolum, 182999 F_MAYBE_BAD_BOOK-flagg, 1130 F_BAD_TS_RECV-flagg, 12180 event-tider etter receive-tid, 16582 crossed BBO, 1101 locked BBO, 529 utilgjengelig tosidig BBO, 286 event-tidsreverseringer og 594 sekvensreverseringer. Sekvenssprang er ikke gapbevis. Databento beskriver F_MAYBE_BAD_BOOK som et uopprettelig kanalhull og F_BAD_TS_RECV som upålitelig receive-tid ved klokke-/rekkefølgefeil ([Databento flags](https://databento.com/docs/standards-and-conventions/common-fields-enums-types)); håndtering av flaggene og ukjent side er ennå ikke frosset.

Eksakt tidsstempelsoverlapp mot eksisterende pre-TEST OANDA XAU_USD M1 er nå
avgrenset i plan 003 til tidskolonnen `time` med 60 sekunders
beslutningstilgjengelighet. Planen bruker bare de hash-bundne GC MBP-1-filene
og tidskolonnen i OANDA M1; ingen priser eller utfall.

Brukeren godkjente én lokal kjøring av
[GC_SOURCE_DISPOSITION_20261016_003.json](configs/research/GC_SOURCE_DISPOSITION_20261016_003.json)
(sha256 7b3c5614307d7849edd7694f6b5297dd1dfb8b66cab400e590bd1d089f5d5082),
kvittering /home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_SOURCE_DISPOSITION_20261016_003/OPERATOR_APPROVAL.json
(sha256 6bd49d90767016c3baa7237f23e9e13a02dff06489543071d1630013d9769406).
Engangsgodkjenningen gjaldt bare plan 003 og er konsumert. Koden ble pushed
på `work/gx1-current` i commits 96c3111d (implementasjon) og 7af27af3
(engangskrav). Fem fokuserte syntetiske tester passerer. Auditten fullførte
2026-10-16 18:43:54Z som invocation 1 av 1 via
`scripts/gx1_capped_run.sh --class audit`: MemoryMax/High 4 GiB, swap 512 MiB,
CPU 0-7, åtte parsearbeidere med én numerisk tråd hver, TasksMax 64.

Resultatet
/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_SOURCE_DISPOSITION_20261016_003/AUDIT_RESULTS/RESULT.json
(sha256 8f1072f9cadfa060e6748722d1cc775af6bf80fed529274ffb400e121594e4a1)
er SOURCE_DISPOSITION_COMPLETE_NOT_ADMITTED. Terminalen
(/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_SOURCE_DISPOSITION_20261016_003/AUDIT_RESULTS/TERMINAL.json,
sha256 86ccde635551227c1953e573f5bcfc814d3f385dec7e9de2ced7b6f8288717db)
og etterkontrollen
(/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_SOURCE_DISPOSITION_20261016_003/POST_AUDIT_RECONCILIATION.json,
sha256 a736023ced431292a25af61a8e87b41a7fec2dd56874399be43e2e0587e2925c)
bekrefter exit 0 og ny hashverifisering av 66/66 GC-filer og kvitteringer,
OANDA-manifestet og alle tre Parquet-partisjonene. Samlet ble 892100712 rader
og 62783740 trade-rader telt hver for seg per outright.

I 26-måneders studieperioden hadde 765111 av 765840 OANDA M1-starttider minst
én matchende god GC ts_recv; 764277 hadde minst én T-handel. Siste M1-bar ved
grensen ble ekskludert fordi beslutningstidspunktet ikke var strengt før
2026-07-01. Unionen teller unike OANDA-minutter berørt av minst én kontrakt;
den er en øvre grense, ikke en rullet GC-serie. 1130 F_BAD_TS_RECV-rader ble
talt og ekskludert fra kausal matching; 182999 F_MAYBE_BAD_BOOK-flagg ble
telt uten å hevde gjenopprettet bok/gyldig OFI. 392 regelmal-åpne timebøtter
uten god receive-hendelse er event-stille; historiske helligdagstilpasninger er
SCHEDULE_UNKNOWN, og dette er ikke bevis på pakketap.

Ingen Databento-kall eller utgifter, OANDA-priser/volum/utfall, featurebygging,
fit, TEST, modelladgang, native endringer, broker/live/paper eller handel.
Kildekvalifisering er fortsatt ufullstendig og kilden er ikke innrømmet.
Ingen relansering av plan 003; videre kildearbeid krever et nytt vedtak.

## GC kontinuerlig kilde: nytt operatørvedtak og plan 004

Brukeren bestilte implementasjon av eksisterende GC-data med volum/order blocks
og senere ny trening. Den siterte rekkefølgen respekteres: først kausal serie,
masker, faktisk dekning og ADMIT/REJECT per indikator; TEST, fit og økonomi er
fortsatt stengt i dette trinnet. Dette er et nytt vedtak, ikke gjenbruk av 003.

Plan `configs/research/GC_CONTINUOUS_SOURCE_ADMISSION_20261016_004.json` (sha256 8eab7e5457a6b9e88899674bf24de551b346ee70b251f2eed4a44f8138a09aaf) ble kjørt én gang med exit 0 via
4-GiB auditvakt. Kildeeier: `gx1/contracts/gc_continuous_source_v1.py`;
runner: `gx1/scripts/gc_continuous_source_v1.py`. Den eksisterende 003-eierens
hash-/OANDA-/cgroup-kontroller gjenbrukes. Ingen Databento-kall eller kostnad.

Kontrakt velges ved 00:01 UTC fra sist avsluttede UTC-døgns observerte rå
T-volum, aldri samme dags sluttvolum og aldri bakover i forfall. Første døgn
har ingen valgt kontrakt; april-OHLCV brukes ikke som oppfunnet intraday-
warmup eller historisk publiseringstid. Det faste universet er de 15 hentede
kontraktene; ingen full-univers-/komplett-feed-påstand. Ukjent side bevares.
Gap, ugyldige handler, snapshots, gjentatte T-signaturer og klokkeproblemer
maskeres. Upålitelig T-klokke i en potensielt brukt fil gir global REJECT
for kausale handelsindikatorer uten gjettet lokalisering. Hver av de 26
månedene må ha gyldige rader; ingen prosentterskel er tilpasset.

Separat resultat for volum/profil, footprint, delta, relativ segment-CVD,
sweeps og order blocks. MBP-1 kan ikke bevise flernivå-/enkeltordresweeps
eller institusjonelle order blocks. Prisbaserte mønstre krever senere egen
kausal formel i eksisterende SMC-eier. Ingen indikatorverdier bygges før
kildeavgjørelsen. OANDA leses fortsatt bare som M1-tid; native radparitet,
læring, generalisering og økonomi er ikke bevist.

49 fokuserte kilde-/rulletester passerer. To gamle handover-tester krevde
GC-status «paused» allerede før denne bølgen; de kontrollerer nå de beholdte
trenings-/TEST-/relanseringssperrene under gjeldende kildearbeid.

Plan 004 fullførte 19:48:48Z: 892 100 712 kilderader avstemt fil for fil mot
002, 66 GC-kilde-/kvitteringspar, 23 definisjonspar og OANDA-inputs rehashet.
765 839 strengt pre-TEST OANDA M1-rader; valgt kontrakt på 764 462, valgt GC-
handel på 761 878. ADMIT gjelder bare maskerte observerte inputs: volum/profil
682 945 M1-rader (98 175 komplette M5-vinduer); footprint/delta/relativ CVD
372 005 M1-rader (19 073 komplette M5-vinduer), med støtte i alle 26 måneder.
Ingen upålitelig Trade-klokke i potensielt valgte filer; 133 987 gjentatte
T-signaturer og 153 190 gapflagg i valgt serie er maskert. Ingen deduplisert
markedssannhet eller komplett flow hevdes. Sweeps over flere boknivåer og
institusjonell order-block-sannhet: REJECT. Kausal rulle-prefix og eksakt
heltallsvolumavstemming består. `RESULTS/` er atomisk publisert med
Parquet-paritet/hasher i run-roten GC_CONTINUOUS_SOURCE_ADMISSION_20261016_004.
Ingen indikatorverdier, fit, TEST, økonomi eller native integrasjon er bygget.

Brukerens hard-slettevedtak 16.10 er i arbeid gjennom retention-eieren.
Første bølge DELETE_COMPLETE: 1 333 eksakte mål, 21 906 filer, 219,3 GiB;
alle originalmål er borte, ingen feil. WSL gikk fra ca. 499 til 280 GiB.
To ubrukte miljøkopier i lagrings-worktrees og pip-cachen frigjorde ytterligere
13,0 GiB; CURRENT-miljøet/importstien er verifisert beholdt.

Andre bølge har ferskt bevis: alle 112 zstd/gzip-par er hashverifisert og
har identiske utpakkede byte. Bare zstd-dobbeltkopier, én ubrukelig .part og
sju frakoblede V9-datasett-Parquets pensjoneres. De to brede beskyttelsesrøttene
er presisert til beskyttede blader/undertrær med bevis for at hver øvrig fil
fortsatt er beskyttet. Ingen launch-/treningsfelt eller guardkode endres.
Andre bølge er nå DELETE_COMPLETE uten feil: 120 eksakte filer, 31,9 GiB.
Begge bølger har null gjenværende originalmål og intakte kanoniske GC-kilder.
Totalt 251,2 GiB DATA-payload og 13,0 GiB miljø/cache er hard-slettet.
WSL-bruk er nå ca. 235 GiB. Fysisk VHDX-komprimering er FULLFØRT:
616 961 146 880 → 268 416 581 632 byte, DiskPart exit 0 og eksklusiv filtilgang
verifisert før operasjonen. Windows har ca. 494 GiB ledig (før ca. 173 GiB).
WSL ble stoppet ved tom prosjekt-/GPU-/låsegrense; Windows ble ikke omstartet.
Autentisert Windows-SSH → WSL-tilkobling er bekreftet etterpå. Alle 112
beholdte GC-filer og 112 kvitteringer er rehashet uten avvik; GC-maskfilens
hash er uendret. TRAIN/VAL-footere er lesbare (652 552 / 70 880 rader);
full rehash av native Parquet-payloads ble ikke gjentatt. TEST ble ikke åpnet
i denne etterkontrollen. Første komprimeringsforsøk etter bare distro-stopp
ble blokkert av en filholder; begge forsøk/loggene er bevart.

Målt GX1_DATA 139,7 + GX1_RUNS 18,7 + CURRENT-kildemiljø 5,7 = ca. 164 GiB.
Hele WSL er større fordi andre prosjekter, verktøy og editorhistorikk også
ligger der. Ingen nye Databento-kall, kostnader, fits eller økonomitest.
Neste treningsmål er avklart 17.10: M5/intradag med MTF. Kausale indikator-
verdier bygges i ny plan 005; native trening er ennå ikke bundet.
Gamle generasjonskvitteringer er historisk evidens, ikke lenger en påstand
om at pensjonerte payloads fortsatt er komplette på disk.
Alle kanoniske GC-gzipfiler/kvitteringer, gjeldende D1-inputs, modellvekter,
checkpoints og terminalkvitteringer beholdes. Trade-features 006 er ferdige; native GC-integrasjon gjenstår.

## Seed-/180d-bølgen (operatør «Ja, 1 og 2» 15.10)

Tre forhåndsregistrerte vindusfaser i spec ENTRY_D1_HORIZON_LEARNING_STUDY_
20261015_002 (root …_20261015_002), én om gangen, hver med egen omstart:
1. recent_365d_s20261015 — KJØRT OG GJENNOMGÅTT 15.10 (boot 07:06Z, exit 0):
   REJECT etter regelen, men rank-IC-signalet REPRODUSERTE: CONTROL kontrast-
   rank-IC 20 d endte +0,218 (base-seed +0,189) fra −0,165 ved egen init; 5 d
   +0,118. MSE-nivået reproduserte IKKE (2 041 073, hårfint verre enn egen
   init; base forbedret til 1 877 306); valgt handling +72,6 bps (base +128).
   Rangeringen er seed-uavhengig, nivåkalibreringen er det ikke. 6/13 mnd verre.
2. recent_365d_s20261016 — KJØRT OG GJENNOMGÅTT 15.10 (09:48–10:55Z, exit 0):
   REJECT etter regelen, men tredje seed reproduserte rangeringen OG nivået:
   rank-IC 20 d +0,277 (fra +0,117 ved egen init), 5 d +0,209; kontrast-MSE
   2 034 166 → 1 889 226, under vinduskonstanten 1 929 010; alle seks
   punktestimater forbedret; valgt handling +83,9 bps mot alltid-LONG +148,5;
   7/13 mnd verre. Tre av tre seeds: +0,19 / +0,22 / +0,28.
3. recent_180d (2 010 steg, 32 174 rader, design-seed).
Seed-variantene er navngitte kontraktkonstanter (RECENT_WINDOW_SEED_VARIANTS,
datokonvensjon) åpnet gjennom prefix-/komponent-/trenergaten bare for
vindusfaser (30f0b5ba). Regel per fase: seks grenser ved 20 d mot initial
(samme seed) og vinduskonstanten; tidligere faser rapporteres ved siden av.
Avgjør om rank-IC 0,19 fra 365 d overlever initialisering, og om skarpere
recency styrker signalet. Full epoke er utelukket av 730-dagersmålingen.
Alternativ 3 (lukke native, satse på journalen) er parkert av valget.

Ingen automatisk trening, forlengelse eller relansering av konsumerte planer.
training_enabled=false; TEST er forseglet. Ingen live/paper, broker, ordre,
spending eller promotion. DATA/RUNS ryddes bare gjennom retention-eieren.
Bevar én agent, én tung CURRENT-jobb, alle genuine features, åtte(+én)
familier, kausalitet, vakter, originalkvitteringer og checkpoints. Ingen fast
tapsgrense eller maksimal holdetid.
