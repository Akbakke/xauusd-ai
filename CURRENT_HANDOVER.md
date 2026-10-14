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
før noen full epoke. Retention-bølge 1 (ureferert prebuilt-evidens) går gjennom
eieren i bakgrunnen. Ingen dokumentert Entry-edge, ingen prediktor innrømmet,
training_enabled = false, TEST forseglet.

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
  nektes, CLI tillater DATA+RUNS. Bølge 1 = ureferert prebuilt (LIFECYCLE_V2_
  LOCAL_PILOT 19 GiB, V42 13, NATIVE_EARLY 5, LIFECYCLE_V2_FULL_TRAIN 46 minus
  det eieren når) via plan → godkjenn → utfør i bakgrunnen. Bølge 2 krever
  dokumentlukking: V38 ×2 (64), FEATURE_REPAIR (34), D1-feilkjerner (10), V46
  utenom beskyttet leaf (117) etter rekkeviddesjekk.
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

## Neste beslutning

Alternativ 1 (anbefalt, brukerens preferanse 14.10): studie-v2-fase
«recent_year» — kronologisk vindu av kvalifiserte TRAIN-rader siste 365 dager
før TRAIN-kutt (ca. 46 000 rader, ≈2 900 steg, én GPU-invokasjon), CONTROL4096
ved 0 og slutt, samme seed/tap/batch; krever kontraktutvidelse, ny spec-fil,
loader-filtrering og policybinding. Alternativ 2: full epoke (40 266 steg, fire
vinduer) bare etter læringssignal i alternativ 1. Ingen av delene er lansert.

Ingen automatisk trening, forlengelse eller relansering av konsumerte planer.
training_enabled=false; TEST er forseglet. Ingen live/paper, broker, ordre,
spending eller promotion. DATA/RUNS ryddes bare gjennom retention-eieren.
Bevar én agent, én tung CURRENT-jobb, alle genuine features, åtte(+én)
familier, kausalitet, vakter, originalkvitteringer og checkpoints. Ingen fast
tapsgrense eller maksimal holdetid.
