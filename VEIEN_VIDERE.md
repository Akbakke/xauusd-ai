# Veien videre — ny Entry fuse-baseline

## Nøyaktig stoppunkt for neste agent

NATIVE_ENTRY_FUSE_INITIAL_MEASUREMENT_20260919 er bundet, ikke forberedt/startet.
Modellrettelse, fem tester og produksjonsinitialisering er ferdige; gjenbruk dem.

1. Etter ren commit/push: kjør /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_ENTRY_FUSE_INITIAL_MEASUREMENT_20260919/OPERATOR_HANDOVER/PREPARE.py via audit-vakt én gang.
2. Bind faktisk campaign-hash/kildecommit i ACTIVATE_TEMPLATE.ps1 og aktiver én gang
   via eksisterende Windows native launcher. Ikke relanser aktiv eller brukt plan.
3. Kilden fryses under kjøring. Oppdater operatørkopien; native receipt og faktisk
   prosess avgjør status. TRAIN256, Exit-ankre256/samplede1024, null optimizersteg.
4. Gjenbruk AUDIT_INITIAL.py fra tidligere hovedencoder-initialmåling. Krev uendrede
   originale teacher-targets/masker/cohort og optimizer/EMA/RNG. Mål NY ONLINE-funksjon.
5. Steng brukt scope. Først ved bestått initialaudit kan separat native fixed256 bindes
   på samme inputs/targets. Ingen automatisk trening, full epoch/VAL/CONTROL/TEST.

Målbar bedre Entry/Exit-beslutning er fortsatt ikke påvist. Den samlede læringsporten
må bestås før kronologisk vurdering og økonomi med alle handler/åpne posisjoner.
