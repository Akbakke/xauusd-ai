# GX1 — frossen TRAIN-evaluering bundet, ikke startet

NATIVE_ENTRY_EXIT_CONVERGENCE512_20260919 er ferdig, vurdert og deaktivert.
Exit har delvis bedre tilstandsavhengige TRAIN-valg. Entry er fortsatt FLAT256/256;
læringsport, kronologisk generalisering og samlet strategiprofitt er ikke bevist.
Originale checkpoints, lærer, features og kostnader er bevart.

FROZEN_EXIT_TRAIN_POLICY_20260919 har nå egen ONLINE512-checkpointbinding og
nullstegs TRAIN-gren i eksisterende native campaign. 24 målrettede testtilfeller
bestått; de 13 allerede beståtte cutoff-tilfellene gjenbrukes. Den faktiske
512-cursoren, kohorten og fire modellfunksjonskilder er verifisert uten forwards.
EVALUATION_PLAN.json binder sammenligninger, kostnader, åpne posisjoner og én
invokasjon. Forbered campaign fra ren pushet kilde; ingen ny jobb er startet.

Tre underagenter er brukt etter brukerens uttrykkelige bestilling. En liten
cachet audit finner bedre hybridutfall i høyere Entry-rangert halvdel:
−0,3998 mot −5,7213 Bps, bedre i6/9 måneder. Øvre halvdel er fortsatt negativ
og alle130 velger LONG. Dette er retrospektiv gruppering på brukt TRAIN og
ett Exit-valg fulgt av referansepolicy, ikke en gjennomførbar handelsregel
eller full Exit-policy. Ingen terskel skal flyttes på dette grunnlaget.
FLAT-Q satt eksakt0 endrer0/256 valg; lange forecast-signaler er konstant
positive. Disse to raske rettelsene er avkreftet uten modellkjøring.

Følg VEIEN_VIDERE.md og NEXT_RUN_POLICY.json. Handover prioriterer nå korrekt
nytt evalueringsomfang og bevarer originalt treningscheckpoint. Ikke relanser512
eller gjenta cohort-, gradient- eller testarbeidet uten konkret ny feil.
Kun /home/andre2/src/GX1_CURRENT, work/gx1-current. Mac er overleveringskopi.
Én tung jobb samtidig; underagentenes avgrensede analyser er ferdige.
Ingen full epoch/full VAL, CONTROL/TEST, live/paper/spending. Målet er aktivt.
