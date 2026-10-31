# Struktur-/svingfamilie: ettpass og full evaluering

Struktur-/svingintervensjonen er fullført og konsumert. Valgte genuine verdier ble erstattet med bundne TRAIN-sentre/referansekategorier; alle observasjonsmasker, øvrige verdier og modellarkitekturen beholdes. Dette måler betinget verdi ved samme tilgjengelighet, ikke all informasjon i familien.

Ett pass på 47 602 TRAIN-foreldre, 2 976 oppdateringer. Separat evaluering av alle 47 602 TRAIN- og 17 288 senere 2025-Q1-foreldre brukte 0 nye oppdateringer og 4 057 forwards. Samme forhåndsvalgte fold, startoppskrift, rekkefølge og numerikk som baseline. Begge vinduer på Windows-boot 589; ingen omstart.

## Fem minutter, beskrivende punktestimater

| Datasett/variant | Treff | AUC | Logloss | Brier |
|---|---:|---:|---:|---:|
| TRAIN: Baseline | 49.53470665% | 0.502346697 | 0.693568065 | 0.250210386 |
| TRAIN: Struktur-/svingverdier erstattet | 50.67044541% | 0.508623020 | 0.692971383 | 0.249912049 |
| Q1: Baseline | 48.61563518% | 0.496961028 | 0.694381428 | 0.250616422 |
| Q1: Struktur-/svingverdier erstattet | 49.40670079% | 0.499686767 | 0.694803894 | 0.250824402 |

Femminutters gyldige antall: TRAIN 47 282, Q1 17 192. Fast TRAIN-prior/alltid OPP gir henholdsvis 50,9052 % og 51,5821 % treff. På Q1 er AUC fortsatt nær 0,5 og sannsynlighetsfeilene dårligere enn både baseline og fast prior. Ingen parrede dag-/ukeintervaller er beregnet ennå; dette fastslår verken positiv eller negativ familieverdi.

Alle 60 horisonter er beholdt. Strukturintervensjonens logloss/Brier er bedre enn fast prior på 51/60 TRAIN- og 2/60 Q1-horisonter; AUC er over 0,5 på 59/60 TRAIN og 46/60 Q1. De øvrige 59 horisontene er beskrivende, ikke grunnlag for nytt utvalg.

## Kontroll og grenser

Original kjede, checkpoints, egen initialmåling, alle 60 rapportpaneler og full raddekning er kontrollert. Separat 4 GiB CPU-kontroll av lagrede prediksjoner kontrollerte typer, former, klokker, kanoniske foreldre/beslutningsakser og rekonstruert fast TRAIN-prior. Alle modell-/priorpaneler ble gjenskapt med samme kvalifiserte metrikkode. Ingen ny modell, target-kildereplay eller uavhengig metrikkalgebra ble brukt.

W1/W2s rapporterte semantiske fulltilstandssegl er identisk: 5e087ab164bb5e7c7b2660d2cd76c5ad6c5fd25fbef4ce7567a7ba4f932ad3af. Terminalkontroll hashet bundne checkpoints, men deserialiserte dem ikke på nytt. Uavhengig kilde-/JSON-kontroll bekreftet samme-boot-kjede og at tomgangsbevis var gyldige ved hver faktisk start.

Signerte maskinvarevakter bestod; oppgaven er deaktivert og 250 W beholdt. TEST/Exit/broker/handel er ikke åpnet. Q1 er gjenbrukt utviklingsperiode. Parret familieanalyse, forhåndsvalgte intervaller og åtte øvrige ikke-baseline-familier gjenstår.

Et foreldet NEXT_ACTION-format i den eksterne W1-etterkontrollen ble rettet i v2 før bestått kontroll; originalen er bevart. Native kjøring ble ikke relansert eller endret.

## Bevis

- Terminal: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_STRUCTURE_SWING_001/VERIFIED_TERMINAL.json SHA-256 c694a47e4f9c37c9dc896d4fba57af0bd26487b9b8d060844e387a79f0f43349
- Native terminalkontroll: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_STRUCTURE_SWING_001/ACTUAL_ROOT_REVIEW_001.json SHA-256 6d426238ed7e2a6f5dbb08ba3f0d5cbf11e413ac12b5fb1d1afb324ccd00d56d
- Lagrede prediksjoner: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_STRUCTURE_SWING_001/SAVED_PREDICTION_VERIFY_001/RESULT.json SHA-256 82d0f41420ede7573e0121d3bf061ecca56a7fd5ddbf179e672756540db485d6
