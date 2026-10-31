# Trend-/EMA-familie: ett pass og full evaluering

Trend-/EMA-intervensjonen er fullført og konsumert. Familiens valgte verdier ble erstattet med bundne TRAIN-sentre/referansekategorier; observasjonsmasker, øvrige verdier og arkitektur ble bevart. Dette måler betinget verdi gitt tilgjengeligheten, ikke all informasjon i familien.

Trening: 47 602 TRAIN-foreldre, ett pass og 2 976 oppdateringer. Separat evaluering: hele TRAIN og 17 288 senere Q1-foreldre, 4 057 forwards og null nye backward-/optimizersteg. Samme startoppskrift, rekkefølge og numerikk som baseline. Begge vinduer brukte Windows-boot 589, uten omstart.

| Datasett/variant | Treff | AUC | Logloss | Brier |
|---|---:|---:|---:|---:|
| TRAIN: Baseline | 49,5347 % | 0,502346697 | 0,693568065 | 0,250210386 |
| TRAIN: Trend-/EMA-verdier erstattet | 51,6433 % | 0,511190554 | 0,692886789 | 0,249869809 |
| Q1: Baseline | 48,6156 % | 0,496961028 | 0,694381428 | 0,250616422 |
| Q1: Trend-/EMA-verdier erstattet | 50,2327 % | 0,501631272 | 0,693278107 | 0,250065440 |

Fem minutter er forhåndsvalgt hovedhorisont; 47 282 gyldige TRAIN-utfall og 17 192 Q1-utfall. Fast TRAIN-prior/alltid OPP gir 50,9052 % og 51,5821 % treff. På Q1 er logloss/Brier bedre enn baseline, men dårligere enn fast prior. Punktestimatene fastslår ikke familieverdi; parrede dag-/ukeintervaller gjenstår.

Alle 60 horisonter beholdes. Logloss/Brier er bedre enn fast prior på 49/60 TRAIN- og 5/60 Q1-horisonter; AUC over 0,5 på 55/60 og 48/60. De øvrige 59 horisontene er beskrivende og brukes ikke til nytt utvalg.

## Kontroll og grenser

Kanonisk kvitteringskjede, bundne checkpoints og alle rapportreferanser bestod 4 GiB-kontroll. En separat kontroll leste initial256, fullTRAIN og Q1-prediksjonsarkiver; dimensjoner, domener, klokker og kanoniske radakser bestod. Fast prior ble rekonstruert fra TRAIN og brukt uendret senere. Alle 60 modell-/priorpaneler ble gjenskapt med eksisterende metrikkode. Dette er lagret-prediksjonskontroll, ikke uavhengig metrikkalgebra eller råfasit-replay.

W1/W2 rapporterer samme semantiske fulltilstandssegl: f2b01ff1a1f7e897b42b95361055f7d9e76214351b88c23d049319b23dfcae1e. TRAIN/OOF-preservasjon er identisk. Uavhengig kilde-/JSON-gjennomgang bekreftet dette, null oppdateringer i W2, eksakte forgjengerreferanser og gyldige tomgangsbevis ved begge starter. Denne gjennomgangen leste ikke checkpoint-/prediksjonspayload.

Begge maskinvarevakter avsluttet normalt. Treningens topper: 58/66 grader og 132,11 W; evalueringens topper: 58/66 grader og 129,06 W. Engangstasken er deaktivert; 250 W konfigurert grense beholdes. Ved etterkontroll var faktisk trekk 37,50 W og GPU-utnyttelse 3 %.

Privatkvalifiseringens første etterkontroll manglet keeper-status i effektkvitteringen. Originalen er bevart; fersk POWER_RETAIN_002 og etterkontroll002 bestod uten native relansering. Full-fit og evaluering brukte uendret modellkode.

Fire av ti matchede varianter er ferdige: baseline, struktur/sving, SMC/likviditet og trend/EMA. Seks øvrige familie-fit og alle parrede kontraster gjenstår. Q1 er gjenbrukt utviklingsdata; TEST, Exit og handel er urørt. Ingen netto handelsverdi er målt.

## Bevis

- Terminal: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_TREND_EMA_001/VERIFIED_TERMINAL.json SHA-256 6c6f2797e3898242fa9c86c7219512e639e801cbbebfa9ced5289a3b22d082a8
- Native terminalkontroll: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_TREND_EMA_001/ACTUAL_ROOT_REVIEW_001.json SHA-256 642de2cf3c281250e6eb63bf048d005f767023ae407af9e39f7a48ce8a8ff09b
- Lagret-prediksjonskontroll: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_TREND_EMA_001/SAVED_PREDICTION_VERIFY_001/RESULT.json SHA-256 0d1bb7cb59af012b9be4df717e7d42a19ea76f0352e2bfa2a3360fce1d0c3605
