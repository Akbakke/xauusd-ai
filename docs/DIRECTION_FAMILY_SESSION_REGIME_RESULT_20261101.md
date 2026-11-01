# Matchet variant med nøytraliserte session/regime-verdier

Varianten er fullført og kontrollert. Session/regime-familiens verdier
erstattes med bundne TRAIN-referanser; masker, øvrige verdier, alle aktive
tidsrammer og arkitektur beholdes. Dette er familiens betingede bidrag
under den faste læringsoppskriften, ingen session-only-modell.

Original trening fullførte ett pass, 47 602 TRAIN-foreldre og 2 976
oppdateringer. Windows-krasj 0x1e avbrøt den etterfølgende evalueringen.
Original W2 har ingen kjent native sluttkode og er aldri relansert.
Trent modell, initial og cursor ble bevart. En ny, separat scoringkampanje
fullførte full TRAIN og 17 288 senere Q1-foreldre: 4 057 forwards,
null backward og null nye optimizersteg. Q1 er gjenbrukt utvikling.

| Femminutters måling | TRAIN | Senere Q1 |
|---|---:|---:|
| Gyldige utfall | 47 282 | 17 192 |
| Treff | 51,3028 % | 50,4886 % |
| AUC | 0,5155539582 | 0,5056503406 |
| Logloss | 0,6926514480 | 0,6935262951 |
| Brier | 0,2497521956 | 0,2501888030 |
| Treff, fast TRAIN-prior | 50,9052 % | 51,5821 % |
| Logloss, fast TRAIN-prior | 0,6929832916 | 0,6927381622 |
| Brier, fast TRAIN-prior | 0,2499180600 | 0,2497955087 |

Senere treff og AUC er høyere punktverdier enn familiebaseline, men treff
og begge sannsynlighetsskårene er svakere enn fast TRAIN-prior. Parrede
familieintervaller er fortsatt ikke beregnet. Tallene dokumenterer ingen
fordel ved å fjerne familien, ingen generell mangel på indikatorinformasjon
og ingen økonomisk edge. Alle 60 horisonter beholdes: deskriptivt slo
49 TRAIN-paneler og to Q1-paneler prioren på logloss og Brier; AUC var
over 0,5 i henholdsvis 55 og 51 paneler.

Kanonisk etterkontroll bekreftet ekte original trening og separat scoring,
original initialidentitet, eksakte antall og uendret semantisk full-state-segl:
acd03406a41c968b2346b49edf9995b29ea679b3f710d3637ad889ac0e75097f.
Den nye serialiserte checkpointfilen har annen filsjekksum; innholdets
semantiske tilstandssegl er identisk. TRAIN/Q1-bevaring av modell, buffere,
RNG og modus samsvarer. Etterkontrollen deserialiserte ikke checkpoints.

Separat capped 4 GiB-avspilling kontrollerte nøyaktig tre prediksjonsarkiver:
original INITIAL 256, gjenopprettet full TRAIN 47 602 og senere 17 288. Eksakte
kanoniske foreldre-/beslutningsakser, familieegen initialparitet, fast
TRAIN-prior og alle 60 modell-/priorpaneler bestod; ingen nektede åpninger,
modellkall eller fit-kall. Eksisterende metrikkode ble gjenbrukt. Dette er
ikke uavhengig metrikkalgebra, råfasit-replay eller konfidensintervaller.
Initial256 og full TRAIN 47 602 er ulike populasjoner og utgjør ingen matchet
før/etter-læringskurve. Generisk OOF-scopeprosa sier TRAIN-utvikling; rolle,
fold og radantall identifiserer Q1. Ingen urørt TEST er brukt.

Treningskilde: 78436e71296b8ef815e15fe9998562b956b19430, boot 589.
Scoringkilde: 346e3a58cfcde838a98a044b3218976ecbaa3e59, boot 590.
Ny scoring sluttet 07:11:20 UTC med native sluttkode 0 og vakt PASS. Topper:
57 grader kjerne,66 grader GPU-minne,130,84 W og 858 MiB. Begge oppgaver er
deaktivert, 250 W beholdt. Ingen ny omstart, TEST, Exit, handel eller promotion.
Krasjets underliggende årsak er fortsatt ukjent.

Syv av ti varianter er fullført. Chart geometry, price action/candle og
macro samt 54 parrede primærkontraster gjenstår. Originale kvitteringer,
avbruddsmarkører og checkpoints er bevart som to ærlige kjøringskjeder.

Evidensrot:
/home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_SESSION_REGIME_SCORING_RECOVERY_001

- VERIFIED_TERMINAL.json:45b14620ffaa411fba4f2fd2bdf9c82ce896267a704c9afbae5cdf0287ae3adb
- ACTUAL_ROOT_REVIEW_001.json:47f2e66661d3184ae1c2aeacc64bef8efa0a5310976ba5eebe917605934ab75f
- SAVED_PREDICTION_VERIFY_001/RESULT.json:761bb4fc2d1b8453d684a5f705ca8df6fa8c220d1701d2db74e4e4f7dd570278
- SAVED_PREDICTION_VERIFY_001/TERMINAL.json:9191e8c4d9a48820cc292bcb8d2143091682cee8b5dc40dcc63e7bb1889ca10c
- ACTUAL_COMPARISON_METADATA_REVIEW_001.json:7712126038f0677cf4ffb3082219a12ac97a60daa50b9935085b5340d75dcc23
