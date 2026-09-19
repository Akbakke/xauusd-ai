# Veien videre — målrettet fuse-kandidat

## Nøyaktig stoppunkt for neste agent

Native representasjonsmåling og saved-state-review er ferdige. Gjenbruk dem.
Start med handover --check og docs/MAIN_ENCODER_REPRESENTATION_REVIEW_20260919.md.
Ingen native kjøring er bundet. Læringsporten er ikke bestått.

1. Fuse-kandidat og reference-copy er implementert. Fem tester bestod; gammel
   Entry/token/Exit-lærerfunksjon og RNG/vekter er eksakt bevart. Produksjonsaudit
   bekreftet 9 617 497 parametere og originale initialvekter. Ikke gjenta disse.
2. Gjenbruk OPERATOR_HANDOVER/BIND.py og PREPARE.py fra
   NATIVE_MAIN_ENCODER_INITIAL_MEASUREMENT_20260918. Bind nye navn
   ENTRY_FUSE_INITIALIZATION_20260919 og NATIVE_ENTRY_FUSE_INITIAL_MEASUREMENT_20260919.
   Audit/testbevis: BASE/ENTRY_FUSE_NORMALIZATION_20260919.
3. Mål NY native ONLINE-initialbaseline på TRAIN, null optimizersteg. Sammenlign
   gammel lærer/targets/koordinater og lagret optimizer/EMA/RNG eksakt. Først etter
   bestått audit kan én separat fixed256-prøve bindes. Ingen gammel baseline som erstatning.
4. Krev bedre tilstandsavhengige Entry/Exit-valg og samlet læringsport. Deretter
   separat kronologisk vurdering og økonomi inklusive alle handler/åpne posisjoner.

Ingen automatisk trening, full epoch/VAL/CONTROL/TEST eller skalering for å tvinge handler.

BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
