# Veien videre — mål gjenværende representasjonsproblem

## Nøyaktig stoppunkt for neste agent

NATIVE_MAIN_ENCODER_REPRESENTATION_20260919 er bundet, ikke forberedt/startet.
Bruk kun /home/andre2/src/GX1_CURRENT, work/gx1-current. Start med handover --check.
CPU-audit bekreftet identiske TRAIN16-inputs/rader/targets/masker, også mot NY
initialmåling. 16 fokuserte kontrakt-/dispatchtester bestod. Gammel initialfunksjon
avvises selv ved samme vekter. Modell, treningsmatematikk og inferensmåler er uendret.

1. Commit/push ferdig binding. Gjenbruk OPERATOR_HANDOVER/PREPARE.py via audit-vakt
   én gang på ren kilde. Deretter bind faktisk campaign-hash/kildecommit i eksisterende
   ACTIVATE_TEMPLATE.ps1 og aktiver én gang via Windows native launcher og guards.
2. Mål bare to inferensforwards, initial/final, på eksisterende TRAIN16-cache.
   Krev lagrede native prediksjoner innen uendret toleranse og identiske valg.
   Sammenlign seq_pool, main_fuse, normalisert joint-input og Entry-hidden.
3. Under native kjøring fryses kilden. Kontroller faktisk runtime/receipt; ikke
   relanser aktiv eller brukt plan. Ingen backward, optimizer, Exit, CONTROL/VAL/TEST.
4. Etter terminal måling: vurder representasjonene, steng brukt scope, dokumenter
   én målt årsak og minste nødvendige videre rettelse. Ingen ny trening automatisk.

BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
Plan/operators: BASE/NATIVE_MAIN_ENCODER_REPRESENTATION_20260919.
Input-/testbevis: BASE/MAIN_ENCODER_REPRESENTATION_INPUT_AUDIT_20260919.
Fullført review: BASE/NATIVE_MAIN_ENCODER_NORMALIZED_FIXED256_20260918.
Det reviewet gjenbrukes: begrenset Entry-signal, FLAT256/256 og sidefast Exit;
samlet læringsport ikke bestått. Ingen antagelse om lønnsomhet eller generalisering.
Først målbar læring, så separat bundet kronologisk kvalitet, deretter full økonomi.
Alle valgte handler og åpne posisjoner skal med; TEST forblir forseglet.
