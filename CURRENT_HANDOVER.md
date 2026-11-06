# CURRENT HANDOVER — GX1 M5 Entry

Eneste arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les `GX1_RULES.md` helt ved første overtakelse og kjør
`bash scripts/gx1_handover.sh --check`. `NEXT_RUN_POLICY.json` er eneste
arbeidsstatus. Terminalkvitteringer overstyrer denne teksten. Tidligere
statusnotater er bevart i Git og originale kvitteringer.

## Gjort og målt

- Begge armers pre-oktober TRAIN-fit er avsluttet på 29 849 foreldre og
  1 866 optimizersteg hver. De egne oktober-scorekjøringene er konsumert:
  6 323 senere foreldre, 396 forwards per arm, null nye optimizersteg og
  signert vakt PASS. Oppstartsoppgaven er deaktivert. Ingen TEST ble brukt.
- [Kontrollens terminalreview](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CONTROL_SCORE_003/ACTUAL_ROOT_REVIEW.json)
  SHA-256 `d37c387736b8578fc3ff228aff7eba09d75b8380dcd102624061c6f4d4e9d932`;
  [kandidatens terminalreview](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CANDIDATE_SCORE_002/ACTUAL_ROOT_REVIEW.json)
  SHA-256 `02043546ea5da35437843164724d9107cdc0a994042e99b8b07a332b0e71c47d`.
- [Paret oktoberutviklingsrapport](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_PAIRED_DESCRIPTIVE_20261106_001/REPORT.json)
  SHA-256 `82988bb594c31c5d77ac1108b79befb6497dbaac65277b3e6babc8770f17e258`.
  M5 +5 har 3 359 observerte av 6 323 foreldre. Brier for kandidat /
  kontroll / fryst TRAIN-prior / Markov er 0,614672 / 0,615091 / 0,612714 /
  0,615325. Kandidat-minus-kontroll har radvektet 95 % UTC-dag-intervall
  [−0,000861, +0,000032]. OPP/NED-AUC er 0,4898 for kandidat og 0,4897
  for kontroll. M5 fullgrid Brier er 0,686345 mot kontroll 0,685498 og
  prior 0,624951. De andre fire tidsrammene er sekundære i rapporten.
- Av 2 964 ukjente M5 +5-fasiter har 2 950 årsakskode 24:
  intraminutt-rekkefølge ukjent og retning uavklart under M1-OHLC. Ukjent
  er ikke FLAT. Dette er målkontraktens konservative kildegrense, ingen
  påvist kodefeil.

## Hindring og neste minste steg

Aldersinput kvalifiserte ikke M5-retningsverdi. Oktober er utviklingsdata,
og den opprinnelige planen manglet en konkret beslutningsregel bundet før
scoring. Rapporten er derfor deskriptiv og gir ingen retrospektiv PASS.
Ingen uavhengig retningsedge eller økonomisk edge er påvist.

[Entry-path TRAIN-kvitteringen](/home/andre2/GX1_RUNS/NATIVE_M5_ACTUAL_NATIVE_BINDINGS_20261019_057/PHYSICAL_TARGET_RECEIPT.json) SHA-256
`dd95d9ecb43f309635e61ba268ec4db6518a3ee23550aa0c94330ff303a5063d` og [oktoberaksen](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_INPUT_ONLY_20261105_003/OCTOBER_AXIS.json) SHA-256
`2aa5bad2804d6635211ceac21eb9935639198f147d7d1a821188d3a07d6c17c5` har samme 118 249-raders M5-barstartklokkehash
`e5d528ff8fcec2b23f0099834b1c5b42a12489dadfada0a6fc4aeab3b6ff4c5d`. Oktober er radene [29 849, 36 172), 6 323
foreldre. Beslutningstid er barstart pluss lukket M5-bar i den eksisterende
eieren. Dette kvalifiserer rad-/klokkematch, ikke observerbarheten av
framtidige executable endepunkter eller MFE/MAE-stier; tette framtidsbaner
er ikke lagret. [Dekningsrapporten](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_ENTRY_MASK_COVERAGE_20261106_001/REPORT.json) SHA-256
`17943a62ef36020120878c3e72a124c99304fe2f60ea7d7c638926b5175a0b37` er fullført og engangsplanen konsumert. Den målte alle
tette 1..480 minutters horisonter innen oktobercutoff med 0 modellkall,
0 oppdateringer, 121,6 sekunder og maksimalt 2,94 GiB RSS under 4 GiB.
Ved +5 var 6 322 rader foldmodne (én høyresensur); 6 297 hadde executable
terminal-open og observerbar terminal-open bane på begge sider, og 6 322
hadde observerbar closed-prefix-bane. Dette er andre mål enn tilstandsfasiten
på 3 359 observerte rader; bedre dekning er ikke lært retningsedge.
Ved +480 hadde 5 607 av 6 240 foldmodne executable terminal og 4 079
observerbar terminal-open bane. Den eksisterende kildeleseren åpnet hele
pre-2026 TRAIN-kilden ved oppsett, men rapporterte bare spørringer før
oktobercutoff; ingen TEST. Den forhåndsbundne, modellfrie M5 +5-joinen er nå fullført én gang.
[CLAIM](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_ENTRY_DIRECTION_JOIN_20261106_001/CLAIM.json) SHA-256
`03dbcfb1e5836502ed02bc129cbc2b4cc6e98251eb38369642090fca3bf12c34` og
[REPORT](/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_ENTRY_DIRECTION_JOIN_20261106_001/REPORT.json) SHA-256
`078f9ea44a096a64d5001799773b9ae8d6e508756c332868fa74dce236101ad0`.
Av 6 323 foreldre var 6 322 innen cutoff og 6 297 hadde executable
terminalutfall på begge sider. Ingen eksakte LONG/SHORT-ties. Kandidatens
radvektede AUC var 0,490519 mot kontrollens 0,490249. Paret forskjell
hadde 95 % UTC-dag-bootstrapintervall [−0,001306, +0,001950]. Ved lik
vekt per UTC-utfallsdag var AUC 0,477911 mot 0,477474. Kandidat mot
0,5 hadde lik-dag-intervall [−0,043802, −0,004016]; radvektet
mot 0,5 omfattet null. 27 utfallsdager. 0 modellkall, 0 oppdateringer,
0 TEST, 0 seleksjon; 116,0 sekunder, 2,72 GiB maksimal RSS under 4 GiB.
Erklærte historiske kostnader er ikke sannhetskvalifiserte. Dette er bare
utviklingsdiagnostikk og gir ingen Entry-gate, uavhengig eller økonomisk edge.
Planen er konsumert og kan ikke kjøres på nytt.

Stans +5-tilstandsscore som surrogat for Entry-sidegate. Neste minste steg
er en smal, lesende designgjennomgang av eksisterende direkte executable
Entry-target og allerede prøvde baseliner. Tidligere rene retningsreferanser
(247/249/252/259 og sekskvartalsreferanse 215) er konsumert eller avvist;
ikke gjenta dem. Gjennomgangen skal finne én særskilt hypotese for direkte
Entry-kvalitet på kronologiske data, eller stanse videre eksperimenter. Gjenbruk eierne i
`gx1/contracts/entry_observed_market_v1.py` og
`gx1/contracts/entry_causal_m1_outcomes_v1.py`; ikke lag ny targeteier eller
start nye modellrunder. Bare en ny, særskilt bundet kronologisk plan med
konkret akseptregel kan åpne et nytt eksperiment. Oktober er allerede
undersøkt utviklingsperiode og er ikke uavhengig bekreftelse. FLAT-valg,
pre-MFE-MAE og realisert økonomi gjenstår.

M5 Entry er hovedmål med kausal lukket M15/H1/H4/D1-kontekst, åtte
spesialister og niende D1/makro. `training_enabled=false`. TEST er forseglet;
bare forward-journalen scorer sin periode på forhåndsbundet måte. Exit,
broker, ordre, live/paper, spending og promotion er stengt. Ingen konsumert
engangsplan relanseres. Stabil langkjøring kontrolleres ca. hver 15.–30.
minutt, gjerne sjeldnere ved stabil drift.

## Bevarte eldre artefaktankere

Disse stistrengene sto bare i tidligere overlevering og beholdes for
retention-eierens navngitte sti-vern. De gir ingen arbeids- eller kjørestatus.

/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2
/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_BPS_HEAD_RANK_DESIGN_20261027_001.json.
/home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19.
/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_INPUT_ONLY_PREP_20261105_003/TEST_TERMINAL.json
/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001
/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001/CONTROL_PRIVATE_001/CAMPAIGN/CAMPAIGN_PLAN.json
/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/
/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_CURRENT_INPUT_LINEAGE_PARITY_REVIEW.json
/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_LOSS_COMPOSITION_REVIEW.json
/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63
/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/ORDER_BATCH_GEOMETRY_ROOT_REVIEW_20261028_002.json.
/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/ORDER_NATIVE_B16_SUPPORTED_QUALIFICATION_20261028_003/ACTUAL_ROOT_REVIEW_001.json.
/home/andre2/GX1_RUNS/XAU_INPUT_PHYSICAL_CLOCK_20261105_003/CAPTURE/INPUT_TIMING.json
