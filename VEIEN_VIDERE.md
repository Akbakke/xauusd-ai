# Veien videre — finn hvorfor bedre signal ikke gir bedre valg

## Nøyaktig stoppunkt for neste agent

Hovedencoder256 er ferdig analysert. Paret review og verdict finnes; ikke kjør
REVIEW_OPERATOR.py eller DECISION_GAP_OPERATOR.py på nytt. Entry viser mer
sammenheng med targets innen månedene, men alle256 valg er FLAT. Exit har
uendrede, sidefaste valg og svakere sentrert feil enn residual256. Porten er ikke bestått.

1. Start med handover --check, CURRENT_HANDOVER.md, NEXT_RUN_POLICY.json og
   docs/MAIN_ENCODER_FIXED256_REVIEW_20260919.md. Ny kjøring er ikke bundet.
2. Det konkrete uavklarte spørsmålet: er tidligere representasjonskollaps
   borte etter final LayerNorm, eller undertrykkes variasjonen fortsatt senere?
   Gjenbruk cached TRAIN16 og eksisterende initial/final-representasjonsmåler
   fra NATIVE_RESIDUAL_REPRESENTATION_20260918. Ikke lag en ny måler eller bredt søk.
3. Kontroller først eksakt input-/koordinat-/targetparitet og hvilke bevis den
   eksisterende kontrakten krever. Den gamle representasjonsporten binder
   residual-verdict og gammel initial. Den kan ikke kjøres uendret for dagens
   hovedencoder-verdict. NY native initial og dagens finalprediksjoner må brukes.
   Begrunn eventuelt minste nødvendige målebindingstilpasning; modellkode uendret.
4. Bind deretter, dersom kontrakten er kontrollert, én native inference-only
   initial/final-måling på samme16 TRAIN-inputs: seq_pool, main_fuse, joint-
   normalisert Entry-input og Entry-hidden. Ingen backward/optimizer eller
   CONTROL/VAL/TEST. Krev samsvar med allerede lagrede native prediksjoner.
5. Bruk målingen til å velge én konkret videre handling. Ingen ytterligere
   normalisering, tapsendring, terskeljustering eller ekstra trening på antagelse.
   Oppdater handover og commit/push når ferdig arbeid er kontrollert.

Eksisterende bevis ligger under BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
NATIVE_MAIN_ENCODER_NORMALIZED_FIXED256_20260918 har PAIRED_TRAIN_REVIEW.json,
DECISION_GAP_AUDIT.json, VERDICT.json og COMPLETION_REVIEWED.json.
NATIVE_MAIN_ENCODER_INITIAL_MEASUREMENT_20260918 har riktig startbaseline.
Originale checkpoints og fullførte målinger bevares. Ingen relansering av brukt plan.
Targetvariasjon er ikke nødvendigvis predikerbar; ikke skaler opp for å tvinge handler.
Først samlet læring, så separat bundet kronologisk kvalitet, deretter full økonomi.
