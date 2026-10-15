# CURRENT HANDOVER — 14.10.2026

Autoritet: /home/andre2/src/GX1_CURRENT, work/gx1-current. NEXT_RUN_POLICY.json er
eneste arbeidsstatus; prosess og terminal overstyrer prosa. Historikk per forsøk
ligger i Git og i policyens bindinger, ikke her.

Nå: minimum native læring på D1-horisontmålet (vedtak 12.10) er gjennomført
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
(miljøendring). GC: ingen brukbare data på disk; forespørselsutkast og et
gratis pris-sjekk-skript (Databento metadata) ligger i run root gc/; verifisert
pris krever Databento-nøkkel. 180 d-fasen er desarmert og ikke re-armert.

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
