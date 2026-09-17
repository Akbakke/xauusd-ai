# Veien videre — mål det nye utgangspunktet

## Nøyaktig stoppunkt for neste agent

Modellrettelse, bevart lærer og TRAIN-only-måleport er kontrollert. Kontrakttester:
44 bestått / 3 utelatt; handover: ni bestått. Ikke gjenta disse kontrollene.
Én plan er bundet: NATIVE_MAIN_ENCODER_INITIAL_MEASUREMENT_20260918.

Følg docs/MAIN_ENCODER_INITIAL_MEASUREMENT_20260918.md og bundne operatører.
PREPARE.py og aktivering utføres én gang. Se først PREPARATION_RESULT, faktisk
prosess og terminal receipt; ikke relanser en aktiv eller avsluttet jobb.
Kun null optimizersteg og TRAIN 256/256/1024 er tillatt.

Etter målingen: kontroller nye ONLINE-startprediksjoner, samme frosne lærer og
kausale targets, uendret state/optimizer/EMA/RNG og koordinater. Gamle prediksjoner
er ikke baseline for ny funksjon. Deaktiver brukt Windows-task, steng scope og
oppdater handover etter terminalt resultat. Først da kan én separat256-prøve
bindes. Ingen automatisk trening. Krev bedre tilstandsavhengig Entry OG Exit,
per side/måned; lavere bias, større variasjon eller all-FLAT/all-HOLD er ikke PASS.
TRAIN-fit, senere kronologisk kvalitet og samlet økonomi er separate porter.
Ingen full epoch/full VAL, CONTROL/TEST, live/paper eller spending.
