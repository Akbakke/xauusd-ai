# Én residualnormaliseringsprøve — bundet, ikke startet

Kandidaten normaliserer input per rad før specialist_out, cross_tf_out og
family_tf_cooperation_out. Ingen nye parametere eller RNG-trekk; ingen blanding
av observasjoner/tidspunkter. Alle features/familier/tidsrammer, kostnader,
targets, tapsvekter, optimizer og faste fusjonsskalaer er beholdt.

Fem målrettede CPU-tester består: skalakontroll, radisolasjon, gradientflyt,
eksakte ferske Entry/token/Exit-outputs i inferens og gradientmodus og bevarte
forbindelser fra alle åtte familier. Faktisk produksjonskonstruktør gir samme
RNG og alle 9 617 497 parametere som originalen og lagret online-/lærertilstand
f1e8691d. Tre projeksjoner har nullvekter/nullbias og uendrede nullkorreksjoner.
Kontrollen omfatter ingen native data-forward; eksakt targetlikhet kreves
fortsatt i native sluttmåling. Testene er ikke læringsbevis.

Én prøve: samme ferske initialisering,4096 TRAIN-rader/orden,TRAIN16,256 steg,
frossen lærer og slutt-ONLINE. Mål TRAIN256,256 Exit-ankere og1024 samplede
Exit-states. Sammenlign initial,kausal256,connected256 og konstanter på samme
korrigerte mål. Begge sider og ni måneder skal med: MSE/sentrert feil,
LONG–SHORT-kontrast, handlinger, referanseverdi/regret og Exit-kvalitet.
Bare fellesnivåbedring/all-FLAT/sidekonstant Exit består ikke læringsporten.

Dette er gjenbrukt TRAIN med fitted-overlapp, ikke generalisering eller profitt.
Ingen CONTROL/VAL/TEST, læreroppdatering, søk, automatisk gjentakelse, full
epoch/full VAL eller live/paper/spending. Bruk bare etablerte native vakter.

Artefaktmappe:
/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_RESIDUAL_NORMALIZED_FIXED256_20260917.
Se PLAN.json, INITIALIZATION_AUDIT.json, TESTS.log og BINDING_RESULT.json.
Etter ren commit/push: OPERATOR_HANDOVER/PREPARE.py via audit-vakt. Bind
campaign-filhash/kildecommit i ny kopi av ACTIVATE_TEMPLATE.ps1 før aktivering.
Ingen relansering av brukt plan. Observer faktisk prosess/receipt; ved fullføring
deaktiver task, steng scope og gjenbruk kausal256-vurderingen med nye outputs.
