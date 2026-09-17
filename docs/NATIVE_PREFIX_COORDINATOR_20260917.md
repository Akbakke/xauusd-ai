# Prefix-utvalg og gjenopptak i native koordinator — 2026-09-17

Ferdige prefix-artefakter var koblet til komponentoppstart, men koordinatoren
manglet den samme populasjonen og datogrensen i sin lagrede sesjonskontrakt.
Den eksisterende koordinatoren binder nå originale TRAIN-ID-er, frosset native
rekkefølge, rollevis fasit, policy, cutoff, normalisering og fersk modellidentitet.
Eksisterende treningsløkke, targetkopi, optimizer, EMA og checkpoint brukes.

Omfanget er maksimalt256 oppdateringer med TRAIN16. Prefix-populasjonen er
større enn4096, så forsøket kan ikke fullføre en epoch eller fornye læreren.
Kontrollutvalget er rapportering ved det faste ONLINE-sluttpunktet; vanlig
EMA-validering og checkpointutvelgelse etter epoch er eksplisitt avvist her.
Kontrollens sammenhengende Entry/Exit-referansemåling må fortsatt kobles inn.

25 fokuserte syntetiske CPU-tester består på første testforsøk. Fire oppdateringer
sammenhengende er eksakt lik to, ny prosessmodell og to til: online, frossen
lærer, AdamW, EMA, scheduler, RNG, rekkefølge og progresjon. Endret initialisering,
blandede fasiter/normaliseringer, utvidet budsjett og full VAL avvises. To berørte
legacy-kontroller består. Overføringsscriptet hadde en syntaksfeil før installasjon;
ingen repoendring eller test skjedde før den ble rettet.

Dette er teknisk bevis med små testmodeller, ikke målbar markedslæring.
Native recipe/campaign, immutable transformreferanser og kontrollmål gjenstår
før separat bundet klargjøring og det ene frosne treningsforsøket. Ingen faktisk
modellkjøring, ny normaliseringsfit, TEST eller større trening er åpnet.

Bevis: handover_snapshot/NATIVE_PREFIX_COORDINATOR_SYNTHETIC_20260917.json.
