# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_COMBINED_DECODER_COMPLETE_RETAINED_REVIEW_ADMITTED. Edge-målet er fortsatt uoppnådd.
Native sammenligning er COMPLETE/CONSUMED; controller har exit0 og oppgaven er deaktivert.
Global training/native er false. Bare én separat lesende etterkontroll av fire lagrede
prognosearkiver er admittert. Kjør eksakt preflight før claim, deretter eksisterende4G/512M-audit.
Ingen ny modell, checkpoint-/featurelesing, fit, TEST, Entry/Exit, broker eller handel.

Root: /home/andre2/GX1_RUNS/MARKET_STATE_TA_COMBINED_DECODER_BENCHMARK_20261112_001.
REVIEW_COMBINED_DECODER.py gjenbruker kvalifiserte scorere/bootstrap fra COVERAGE.
Sammenlign begge armers faste pass3 TRAIN/LATER: alle5TF S/F, sannsynlighetsfeil, AUC,
UP-skjevhet og punktvise dag-/ukeintervaller. Kontrollpunkter og gamle intervaller må replays
nøyaktig.4arkiver,2000trekk per gruppeskjema,0modeller. Ingen utvelgelse av pass/gruppe/terskel.

## Fullført matched native sammenligning

Frossen kilde a23c9346, RECIPE826d89a3, plan2fd275e9, fersk fysisk boot647.
RESULT31868249 og TERMINAL93cb7f17 er COMPLETE. COMPLETION_VERIFICATION.json kontrollerer
plan, receipt, cursor, progress, ren kilde, identiske eksponerings-/RNG-planer og faktiske tellere.
768updates/0skip;2304forwards(768TRAIN+1536eval),2modeller/1optimizer/1checkpointlesing,
0checkpointskrivinger og1evidence-kapsel. Native målt3129,08s. Originalkvitteringer bevares.
Vakt: exit0; peak57Ckjerne/60Cminne/167,15W/2206MiB. Ingen jobb ble stoppet for omstart.

Begge INITIAL-paneler bestod før updates. Akser/targets/aktivitet/M5 er bitlike;
non-M5-side maksavvik3,58e-7 innen forhåndsbundet toleranse, RNG lik etter hver pass.
TRAIN har ingen lagrede representasjoner og hevder ingen slik likhet; LATER kontrollerer
begge lagrede M5-representasjoner. Alle genuine features/8familier/5TF/D1, normalisering,
delt encoder/query og tapsaritmetikk er beholdt;16nye sidetensorer/1032parametre.
Separate utganger fjerner ikke all påvirkning mellom oppgaver gjennom den delte encoderen.

Foreløpige publiserte pass3-punktmål: TRAIN-loss1,832358→1,823358; senere samletloss
2,360385→2,360441. M15F-side Brier0,279244→0,269589, mens H1/H4/D1F blir dårligere.
Dette er ennå ikke ferdig etterkontroll eller bevis for samlet senere retningsgevinst.

## Gjenbrukt evidens og grenser

COVERAGE er fullført i /home/andre2/GX1_RUNS/MARKET_STATE_TA_COVERAGE_BENCHMARK_20261112_001.
Begge armer bruker12288unike TRAIN-rader/768updates og samme4096TRAIN/LATER-paneler.
COVERAGE ga senere M5F AUC0,528606; AUC er ikke winrate. Flere non-M5-prognoser var
fortsatt dårligere enn frossen prior. Effektiv vektaudit89c0a1f5 viste ingen universell
UP-drift fra vektingen og begrunnet ingen tapsrettelse.

Kilde-/migreringskvalifikasjon og beståtte fokuserte tester gjenbrukes.224unike benchmark-
kontroller var kvalifisert før kjøring; ingen modellkode er endret etter native kjøring.
Lagrede utviklingsperioder er gjenbrukt. Ingen uavhengig generalisering, Entry-kvalitet eller
økonomisk edge er bevist. Korte kontroller dømmer ikke endelig kapasitet. Full kvalifisert
historikk og30–40epoker er senere skalering. TEST er forseglet; GC er på pause.
