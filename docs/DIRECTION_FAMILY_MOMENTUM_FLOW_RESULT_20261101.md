# Matchet variant med nøytraliserte momentum/flow-verdier

Varianten er fullført og kontrollert. Den erstatter momentum/flow-familiens
verdier med de bundne TRAIN-referansene, og beholder observasjonsmasker,
andre verdier og arkitektur. Dette er ikke en momentum-only-modell.

Samme start, datarekkefølge, numerikk og protokoll som familiebaseline:
ett pass med 47 602 TRAIN-foreldre og 2 976 oppdateringer. Et separat
evalueringsvindu skåret full TRAIN og 17 288 senere Q1-foreldre med
4 057 forwards og null backward-/optimizersteg. Q1 er gjenbrukt utvikling.

| Femminutters måling | TRAIN | Senere Q1 |
|---|---:|---:|
| Gyldige utfall | 47 282 | 17 192 |
| Treff | 51,4170 % | 51,0412 % |
| AUC | 0,5063382078 | 0,5048791521 |
| Logloss | 0,6927807083 | 0,6930371104 |
| Brier | 0,2498168128 | 0,2499448305 |
| Treff, fast TRAIN-prior | 50,9052 % | 51,5821 % |
| Logloss, fast TRAIN-prior | 0,6929832916 | 0,6927381622 |
| Brier, fast TRAIN-prior | 0,2499180600 | 0,2497955087 |

Senere femminutters treff og AUC er høyere punktverdier enn familiebaseline,
men treff og begge sannsynlighetsskårene er svakere enn fast TRAIN-prior.
Parrede intervaller er ikke beregnet; dette fastslår ingen fordel ved å
fjerne familien, ingen manglende indikatorinformasjon og ingen økonomisk edge.
Alle 60 horisonter beholdes. Deskriptivt slo 51 TRAIN-paneler og ni Q1-paneler
prioren på både logloss og Brier; AUC var over 0,5 på henholdsvis 55 og 54.

Kanonisk kontroll bandt begge native kvitteringer, checkpoints, inputlineage
og prediksjonsarkiver. W1/W2 rapporterer identisk semantisk tilstandssegl
3d67cdebc98dddf3250aa927e8f2197f503833f6e5364f7746b12bde17e9d0f4.
TRAIN/Q1-bevaring av modell, buffere, RNG og modus er identisk.
Denne etterkontrollen deserialiserte ikke checkpoints.

Separat capped 4 GiB-avspilling verifiserte arkivdomener, eksakte
kanoniske foreldre-/beslutningsakser, familieegen initialparitet, fast
TRAIN-prior og alle 60 modell-/priorpaneler. Eksisterende metrikkfunksjon
ble gjenbrukt; dette er ikke uavhengig metrikkalgebra, ny råfasitrevisjon
eller konfidensintervaller. Initialutvalget på 256 og full TRAIN på 47 602
er ulike populasjoner og må ikke presenteres som matchet før/etter-læring.

Frossen kjøringskilde: dbcabf489686a1ec680bb5ca1bed2fee2f928797.
Begge vinduer brukte fysisk boot 589; ingen omstart.
Vakt PASS, topper trening 57/66 grader og 143,59 W; evaluering
58/66 grader og 133,39 W. Task er deaktivert, 250 W beholdt,
målt ettertrekk 37,35 W. Ingen TEST, Exit, handel eller promotion.

Seks av ti varianter er ferdige. Fire familier og alle forhåndsbestemte
parrede sammenligninger gjenstår. Alle native porter for denne varianten er
stengt; originalkvitteringer og checkpoints er bevart.

Evidensrot: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_MOMENTUM_FLOW_001

- VERIFIED_TERMINAL.json: 3e2959559fb673e53968c5314b6ad6ddd8e8b079f7c2f3eba60bd7698216dea6
- ACTUAL_ROOT_REVIEW_001.json: 1ec2a019fe3d607255ab48e9b6fd1d3b0ce4ec8214d547fbf4714ace35c06de4
- SAVED_PREDICTION_VERIFY_001/RESULT.json: 56cc23cc793acc14fa9645165635b98997aa3736cc5f7be83abf579eb1f11787
- SAVED_PREDICTION_VERIFY_001/TERMINAL.json: d4733586b493cab11bb2868b832b458390f948e616d384127d6145c0080912d5
