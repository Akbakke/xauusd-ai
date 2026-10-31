# Volatilitet/kompresjon: ett pass og full evaluering

Familiens valgte verdier ble erstattet med bundne TRAIN-sentre/referansekategorier; masker, øvrige verdier og arkitektur ble bevart. Målingen gjelder betinget verdi under den faste læringsoppskriften, ikke indikatorenes fulle potensial.

Trening: 47 602 TRAIN-rader, ett pass og 2 976 oppdateringer. Separat evaluering: hele TRAIN og 17 288 senere Q1-rader, 4 057 forwards og null nye backward-/optimizersteg. Begge vinduer brukte Windows-boot 589, uten omstart.

| Datasett/variant | Treff | AUC | Logloss | Brier |
|---|---:|---:|---:|---:|
| TRAIN: Baseline | 49,5347 % | 0,502346697 | 0,693568065 | 0,250210386 |
| TRAIN: Volatilitet-/kompresjonsverdier erstattet | 51,3409 % | 0,508303757 | 0,692994601 | 0,249923711 |
| Q1: Baseline | 48,6156 % | 0,496961028 | 0,694381428 | 0,250616422 |
| Q1: Volatilitet-/kompresjonsverdier erstattet | 50,4770 % | 0,507702267 | 0,693136403 | 0,249994600 |

Fem minutter er forhåndsvalgt hovedhorisont, med 47 282 gyldige TRAIN-utfall og 17 192 Q1-utfall. Fast TRAIN-prior/alltid OPP gir 50,9052 % og 51,5821 % treff. Q1-punktresultatene er bedre enn baseline, men treff/logloss/Brier er svakere enn fast prior. Parrede usikkerhetsintervaller gjenstår; familieverdi er ikke fastslått.

Alle 60 horisonter beholdes. Logloss/Brier er bedre enn prior på 52/60 TRAIN- og 2/60 Q1-horisonter; AUC over 0,5 på 58/60 og 53/60. De øvrige 59 horisontene er beskrivende og gir ikke autorisasjon til nytt utvalg.

## Verifikasjon og begrensninger

Kanonisk kvitteringskjede, bundne checkpoints, initialparitet og evalueringsbevaring bestod. Separat 4 GiB-kontroll av lagrede prediksjoner gjenskapte alle 60 modell-/priorpaneler, kontrollerte kanoniske radakser og rekonstruerte den faste prioren fra TRAIN. Dette er samme kvalifiserte metrikkode, ikke ny uavhengig metrikkalgebra eller råfasit-replay.

En separat kilde-/JSON-gjennomgang bekreftet samme semantiske fulltilstandssegl i begge vinduer: 2c627bf7c1ef2cbf3630bcfc61023e3b7fd8d4981f45ad6a842dc3fb18720a6d. TRAIN/OOF-bevaring og begge ferske tomgangsbevis ved faktisk start stemte. Gjennomgangen leste ikke checkpoint-/prediksjonspayload.

Maskinvarevaktene bestod. Treningens topper var 58/66 grader og 131,62 W; evalueringens 58/66 grader og 129,13 W. Engangsoppgaven er deaktivert, begge keeper-/telemetrioppgaver kjører og konfigurert 250 W beholdes; ettertrekk 41,28 W.

Fem av ti varianter er fullført. Momentum/flyt, sesjon/regime, chart-geometri, prislys og makro gjenstår, sammen med alle parrede familieintervaller. Q1 er gjenbrukt utvikling, TEST/Exit/handel er urørt og ingen netto handelsverdi er målt.

Brukeren utfordret hvorfor flere tester fortsatt gir svake resultater. Før neste familie prioriteres en avgrenset diagnose av svak funksjonell påvirkning, med gjenbruk av eksisterende resultater. Kildekontrollen fant ingen price_only-frakobling eller manglende optimizer-medlemskap. Dagens profil og ettpassoppskrift tester ikke alle genuine indikatorers potensial. Ny diagnose må ikke gjenta fullførte head-gradient-, vekt-, ekstra-pass- eller momentum-nullingsforsøk.

Terminal: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_VOL_COMPRESSION_001/VERIFIED_TERMINAL.json, SHA-256 bad38b76fd8efa901141bdfe500b73790e213c104479dbbb181502f8d65e1aa0.

Native kontroll: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_VOL_COMPRESSION_001/ACTUAL_ROOT_REVIEW_001.json, SHA-256 bd6fa7ee0f8048d5c1714ebbc56c6bce534abd43ca64628db21a1a3b010f3c4d.

Lagret-prediksjonskontroll: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_VOL_COMPRESSION_001/SAVED_PREDICTION_VERIFY_001/RESULT.json, SHA-256 e6e2f49437abae709565bcb910c197935853988ac124a37c6ce114ed3c9cd81b.
