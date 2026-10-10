# CURRENT HANDOVER — 10.10.2026

Ny firestegs Entry-læringsstudie er forhåndsregistrert etter brukerens
«Lag deg et nytt mål og kjør disse 4». Arbeidssted er
/home/andre2/src/GX1_CURRENT, branch work/gx1-current. Én agent, én tung jobb.
NEXT_RUN_POLICY.json eier aktuell status og eksakt kjøreautorisasjon.

## Gjort og neste steg

Kode for endelig fit-/kurvebudsjett og tilstandsuavhengige målinger er lagt
til eksisterende native eiere. 95 fokuserte syntetiske kontrakttester bestod:
mål, masker, scope, observasjon, checkpoint/resume og bevart RNG. Dette
beviser mekanikk, ikke læring. FIT1024 er nå fullført og kontrollert; CURVE er neste separate kjøring.

1. Liten fit: fersk komplett modell, 1024 optimizersteg på 256 faste ekte
   TRAIN-rader. Målinger0/64/256/1024; ingen CONTROL. Bare fit-diagnose.
2. Læringskurve: separat fersk initialisering, 16384 steg /262144 unike
   Entry-rader fra opprinnelig epoch-order. TRAIN-probe4096 ved
   0/256/1024/4096/8192/16384. Enkel HGB-referanse får de samme fit-radene
   og alle254 rå snapshotfelt; indre valg skjer kronologisk med purge.
3. Senere CONTROL4096 måles bare ved0 og16384. Gjenbrukt utviklingsdata,
   ingen urørt OOS-påstand eller CONTROL-valg. TEST forblir forseglet.
4. Sammenlign LONG/SHORT/kontrast mot initial og samme-TRAIN-konstant,
   med seks parvise ukeblokkintervaller, sentrert feil, handlinger og alle
   måneder. Bare kvalifisert Entry kan åpne Entry-bevarende Exit og
   full økonomi. Uklart eller negativt utfall lukker denne hypotesen.

Plan: configs/research/ENTRY_LEARNING_STUDY_20261010.json.
Run-root: /home/andre2/GX1_RUNS/ENTRY_LEARNING_CURVE_20261010_001.
CPU-baseline er fullført på kilde4f92781e:262144 fit-rader,254 snapshotfelt,
to HGB-fits. Ingen av100 tretrinn slo samme-indre-TRAIN-konstant på noen
side; begge referanser ble konstante. RESULT/TERMINAL er bundet i policy.
FIT1024 fullførte på kildeab8f1368 etter fysisk boot486, native guard PASS
og Windows-task0. PHASE_REVIEW binder1024 optimizer-/EMA-steg, uendrede
targets og150 bit-like Exit-eide statefelter. På de256 gjentatte TRAIN-radene:
kontrast-MSE3096.696→600.928, sentrert3092.499→241.162, korrelasjon0.96094,
handlinger45 LONG/146 SHORT/65 FLAT. Dette er tilpasning på fit-rader.
Fit-planen er konsumert og Windows-tasken deaktivert; aldri relanser den.
Neste jobb er bare den allerede registrerte CURVE16384 fra fersk initialisering,
med egen recipe/campaign og fysisk ny boot. Maksimalt to native vinduer. training_enabled=false gjelder
generisk trening; full epoch/full VAL er stengt.

## Bevarte data, funksjon og vakter

TRAIN652552 er fysisk2011–mai2025, CONTROL-kilden70880 er juni2025–juni2026.
Normalisering er allerede tilpasset én gang på hele fysisk TRAIN.
Alle254 genuine felt, åtte familier, tidsrammer, targets, optimizer og
treningssløyfens tap består. Entry-target er observert95min BID/ASK-nettoutfall
med gjeldende bundet kostpolicy og FLAT0; ingen Exit-modell eller ny horisont.
95min er beregningshorisont, aldri maksimal holdetid.
Kostpolicy er ikke historisk kostfasit.

Eksisterende native campaign,12000s vindu/13800s guard,20G/512M cgroup,
CPU-/GPU-/effekt-/temperaturvakter og atomisk to-slot-resume består.
Checkpoint hvert64 steg og ved pause. Ingen kildeendring under kjøring.
Kontroller stabil langkjøring sjelden; ingen minuttvis polling.
Fysisk reboot krever ferskt maskinvidt prosess-, writer-, lås- og GPU-idle-bevis.

## Tidligere evidens og grenser

Forrige native smoke256 steg/4096 rader ga FLAT på alle256 TRAIN og256 CONTROL;
CONTROL slo ikke TRAIN-konstantene. Ekte192→256-resume ble bitvis verifisert
etter fysisk reboot. Begge planer er konsumert og skal aldri relanseres.
Siste tidligere Windows-boot var485; oppgaven var deaktivert etter terminal.
Eksisterende Windows-staging kan bare gjenbrukes ved fersk bytekontroll.

Den forrige firestegs gradient-/ridge-undersøkelsen er også lukket:
Entry-kontrast når538 delte tensorer; svakere hjelpegradient var ikke
motrettet. Ridge-samspill ga bare0.007524% samlet MSE-forbedring mot konstant
og intervallet inkluderer null.529545/529545 FLAT, ingen kvalifisert Entry.
Se docs/ENTRY_EDGE_RESEARCH_20261010.md og lukkede policyfelt.
Dette begrunner en avgrenset kapasitetsmåling, ingen garanti om edge.

Ingen live/paper, broker, ordre, spending, promotion eller TEST.
Ingen DATA/RUNS-opprydding er bestilt; behold gammel evidens og checkpoints.
Modellens stabile generalisering, Entry-bevarende Exit og profitt er ubevist.
