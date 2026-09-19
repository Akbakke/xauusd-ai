# Neste handling — ett fast konvergensforsøk

NATIVE_ENTRY_EXIT_CONVERGENCE512_20260919 er bundet, ikke startet. Gjenbruk beståtte
14 tester, faktiske kilde-/rekkefølgebindinger og alle tidligere målinger.

1. Etter ren commit/push, kjør /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_ENTRY_EXIT_CONVERGENCE512_20260919/OPERATOR_HANDOVER/PREPARE.py via audit-vakt én gang.
2. Bind kilde/campaign-hash i ACTIVATE_TEMPLATE.ps1 og aktiver eksisterende
   Windows native controller én gang. Fysisk reboot og vakter beholdes.
3. Frys kilden. Native skal laste bevart256-state i ny session, fortsette den
   eksakte rekkefølgen til512 og måle samme TRAIN256 og1024Exit-samples.
   Ikke start ny initialmåling, lærer, tester eller alternativ kjører.
4. Ved stabil drift kontroller omtrent hver time. Ved terminal status, deaktiver
   task og vurder512 mot lagret256/initial/konstanter med eksisterende review.
   Samme mål, begge sider, alle ni måneder, handlingsvalg og referanseverdi.
5. Steng brukt scope. Svakt eller uklart resultat gir ingen automatisk utvidelse.
   Læringsporten krever bedre Entry og Exit, ikke bare lavere loss eller flere handler.

Modell-/treningsmatematikk, features og alle originale checkpoints bevares.
Ingen full epoch/full VAL, CONTROL/TEST, live/paper/spending. Senere kronologisk
vurdering og full økonomi inklusive åpne posisjoner krever læringsport først.
