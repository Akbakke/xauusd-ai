# SMC-/likviditetsfamilie: ett pass og full evaluering

SMC-/likviditetsintervensjonen er fullført og konsumert. Familiens valgte verdier ble erstattet med bundne TRAIN-sentre/referansekategorier; observasjonsmasker, øvrige verdier og arkitektur ble bevart. Dette måler betinget verdi gitt tilgjengeligheten, ikke all informasjon i familien.

Trening: 47 602 TRAIN-foreldre, ett pass og 2 976 oppdateringer. Separat evaluering: hele TRAIN og 17 288 senere Q1-foreldre, 4 057 forwards og null nye backward-/optimizersteg. Samme startoppskrift, rekkefølge og numerikk som baseline. Begge vinduer brukte Windows-boot 589, uten omstart.

| Datasett/variant | Treff | AUC | Logloss | Brier |
|---|---:|---:|---:|---:|
| TRAIN: Baseline | 49,5347 % | 0,502346697 | 0,693568065 | 0,250210386 |
| TRAIN: SMC-/likviditetsverdier erstattet | 50,6133 % | 0,521577438 | 0,692862640 | 0,249857848 |
| Q1: Baseline | 48,6156 % | 0,496961028 | 0,694381428 | 0,250616422 |
| Q1: SMC-/likviditetsverdier erstattet | 48,9588 % | 0,503966980 | 0,694578428 | 0,250714387 |

Fem minutter er forhåndsvalgt hovedhorisont; 47 282 gyldige TRAIN-utfall og 17 192 Q1-utfall. Fast TRAIN-prior/alltid OPP gir 50,9052 % og 51,5821 % treff. På Q1 er logloss/Brier dårligere enn både baseline og fast prior. Punktestimatene fastslår ikke familieverdi; parrede dag-/ukeintervaller gjenstår.

Alle 60 horisonter beholdes. Logloss/Brier er bedre enn fast prior på 55/60 TRAIN- og 4/60 Q1-horisonter; AUC over 0,5 på 58/60 og 56/60. De øvrige 59 horisontene er beskrivende og brukes ikke til nytt utvalg.

## Kontroll og grenser

Kanonisk kvitteringskjede, bundne checkpoints og alle rapportreferanser bestod 4 GiB-kontroll. En separat kontroll leste initial256, fullTRAIN og Q1-prediksjonsarkiver; dimensjoner, domener, klokker og kanoniske radakser bestod. Fast prior ble rekonstruert fra TRAIN og brukt uendret senere. Alle 60 modell-/priorpaneler ble gjenskapt med eksisterende metrikkode. Dette er lagret-prediksjonskontroll, ikke uavhengig metrikkalgebra eller råfasit-replay.

W1/W2 rapporterer samme semantiske fulltilstandssegl: 61dd25d4af36af5583c2f7df6b3e9ed0706dbecb9bc124388351f7c163d5a89d. TRAIN/OOF-preservasjon er identisk. Uavhengig kilde-/JSON-gjennomgang bekreftet dette, null oppdateringer i W2, eksakte forgjengerreferanser og gyldige tomgangsbevis ved begge starter. Denne gjennomgangen leste ikke checkpoint-/prediksjonspayload.

Begge maskinvarevakter avsluttet normalt. Treningens topper: 58/66 grader og 135,34 W; evalueringens topper: 58/66 grader og 128,99 W. Engangstasken er deaktivert; 250 W konfigurert grense beholdes. Ved etterkontroll var faktisk trekk 37,46 W og GPU-utnyttelse 3 %.

Etterkontrollen observerte først task Ready etter terminal og stoppet før kontrollartefakt ble opprettet. Den fullførte tasken ble deretter deaktivert og tilstanden kontrollert. Native kjøring ble ikke relansert.

Baseline, struktur/sving og SMC/likviditet er ferdige. Sju øvrige familie-fit og alle parrede kontraster gjenstår. Q1 er gjenbrukt utviklingsdata; TEST, Exit og handel er urørt. Ingen netto handelsverdi er målt.

## Bevis

- Terminal: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_SMC_LIQUIDITY_001/VERIFIED_TERMINAL.json SHA-256 31e1e980ae09cf1ff9859c504f09fec38c1bda4f7f99cadd324a903485bd6c5d
- Native terminalkontroll: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_SMC_LIQUIDITY_001/ACTUAL_ROOT_REVIEW_001.json SHA-256 160bc4fcf563d51e2fa3fe8de74a8e0bdc99b03fd21f2256279d0523e3168092
- Lagret-prediksjonskontroll: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_SMC_LIQUIDITY_001/SAVED_PREDICTION_VERIFY_001/RESULT.json SHA-256 6bed8beaf5f2ca8d17232ba4511b37f89825acc53442d02d8b7c25177de7d9ea
