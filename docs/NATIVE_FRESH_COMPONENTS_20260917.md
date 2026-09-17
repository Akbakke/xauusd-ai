# Fersk native komponentoppstart — 2026-09-17

Den eksisterende full-TRAIN-eieren lastet alltid tidligere EMA-vekter og gammel
normalisering. En eksplisitt chronological_prefix-gren bruker nå samme hele
modellkonstruktør med den ferdige prefix-transformen og avviser alle gamle
seed-/checkpointargumenter. Ingen annen arkitektur, treningsløkke eller runner.

Frosset design, normalisering, policy-klargjøring og fasiter må ha samme
cutoff/populasjon/proveniens. Gamle eller blandede transformasjoner avvises.
TRAIN-data opprettes én gang; en separat datasettinstans deler uendrede inputs
før rollevis fasitbinding og TRAIN-lifecycle bindes. Kontrollinstansen bruker
CONTROL256-fasiter og fysisk TRAIN med samme sekvensbevis, uten gammel juni.

Hele native rekkefølgen filtreres med eksisterende eier og sammenlignes eksakt
mot den låste prefix-rekkefølgen. Ingen sampler eller successor-counts endres.
Reference-policy og cutoff sendes til eksisterende TRAIN-factory. Modellen
starter ferskt med nøytrale oppgavevekter; AdamW er tom, EMA er en eksakt fersk
kopi med eksisterende regel beregnet fra prefix-populasjonen. Scheduler beholdes.

20 syntetiske kontroller består på første forsøk. Det inkluderer den komplette
komponentkoblingen med små testmodeller og kontrollerte datasett-/I/O-dobler,
reell native rekkefølgefiltrering og optimizer/EMA, samt avvisning av gamle
vekter, feil normalisering, framtidig fit og endret plan. Dette er ikke faktisk
modellkjøring på markedsdata. Legacy seed-/checkpoint-/normaliseringsblokk er
AST-lik bortsett fra lokalt variabelnavn; alle andre eksisterende toppnivå-
funksjoner i oppstartseieren er uendret. Modellkonstruktøren er ikke endret.

Komponentgrenen er ennå ikke admitted av native recipe/campaign. Koordinatoren
må binde prefix-populasjon, cutoff og resume-proveniens; indeks-/bridge-/bundle-
referanser må peke på nye transforms. Faktisk fersk targetkopi og frosne
målinger må fortsatt klargjøres under separat bundet tillatelse. Senere
kontrollmåling må bruke sammenhengende referansemål også ved Entry-state0.
Ingen faktisk konstruksjon/forward/optimizer, refit, fasitberegning eller TEST.
Læringsporten er fortsatt uavklart. Ikke gjenta beståtte tester eller fits.

Bevis: handover_snapshot/NATIVE_FRESH_COMPONENTS_SYNTHETIC_20260917.json.
