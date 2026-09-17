# Felles eier for policyavhengige fasiter

Den eksisterende datasetbuilderen eksponerer nå to avgrensede funksjoner:
build_policy_dependent_entry_auxiliary_targets og
materialize_policy_dependent_entry_auxiliary_targets. Førstnevnte gjenbruker
de opprinnelige formlene og float32-konverteringene for10 aktive targets og4
relaterte diagnostikkfelt. Full builder bruker den samme funksjonen. Sistnevnte
bruker eksisterende eksakt-M1- og trendlinje-eiere, og avviser støtte som mangler
eller overskrider den oppgitte datagrensen. Ingen nye fasitregler eller fit.

12 målrettede syntetiske tester består på første forsøk. Alle14 felter er
bitlike originalkoden for213 syntetiske rader i tre scenarier: observerte
prisforløp, terskler/blandede events og ingen events.42 øvrige toppfunksjoner
er AST-identiske, inklusive37 faste hjelpefasiter. Endringer på+5000 i alle
senere M1/M5-priser endrer ingen tidligere labels. Feil klokker, policybinding,
framtidig fitscope, manglende M1/M5-støtte og ugyldige eventmasker avvises.

Én label-only CPU-plan er bundet. Frosne47814 TRAIN-ID-er og CONTROL256 brukes
uten nye valg. Én registry-gjennomgang gjenbrukes for begge; hver TRAIN-rads
utfallsstøtte må slutte senest mars1. Kontrollstøtte slutter senest juni1.
Original Group-A-manifest binder registry-konteksten til464244 M5-rader fra
2019-11-11T00Z til2026-05-31T23:50Z. Nøyaktig original konteksthash kontrolleres
før beregning. Dette hindrer en skjult endring av registry-initialiseringen.

Jobben skriver separate labelartefakter. Opprinnelige datasett, modellinputs,
policyer, features og checkpoints endres ikke. Nye labels er ennå ikke koblet
til native dataset.37 faste targets beholdes; øvrige gamle diagnostikkfelt skal
ikke presenteres som oppdatert. Ingen policy-/normaliseringsfit, modell-forward,
optimizer eller TEST. Teknisk labelkontroll er ikke læringsbevis.

Syntetisk bevis:handover_snapshot/PREFIX_DEPENDENT_LABELS_SYNTHETIC_20260917.json.
Plan:GX1_DATA/.../PREFIX_DEPENDENT_LABELS_20260917/PLAN.json.
