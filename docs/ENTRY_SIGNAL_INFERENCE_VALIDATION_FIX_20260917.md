# Entry-signaldiagnose: verifiser inferens i riktig modus

Den opprinnelige signaldiagnosen sammenlignet gradientaktivert eval direkte
med lagret inferens og stoppet. Den fullførte fire-forward-målingen viser
at vanlig inferens matcher begge frosne modeller eksakt, mens gradientmodus
avviker med maksimalt 0,0002992153 / 0,0001640320 Bps uten handlingsbytter.

Minste rettelse er implementert i eksisterende native diagnostikk. For hver
modell verifiseres først vanlig inferens mot lagret inferens med uendret
0,0001 Bps-grense og identiske handlinger. Deretter beregnes gradientene fra
deres faktiske gradientaktiverte output. Avviket rapporteres, og handlingsbytter
avvises. Eksakt samsvar mellom beregningsmodus påstås ikke. Ingen modell,
treningsmål, tapsvekt, clipping, optimizer eller produksjonsregel er endret.

Fem målrettede tester består: feil inferens avvises, modusavvik rapporteres,
gradienter måles uten optimizer/akkumulering, korrekt fire-forward-omfang og
frosset tidligere parity-bevis kreves. Logg ligger i den fullførte parity-
artefaktmappen som SIGNAL_INFERENCE_FIX_TEST.log. Gjenbruk dette beviset.

## Planbinding — separat plan er nå bundet

Se ENTRY_SIGNAL_INFERENCE_CHECK_20260917.md for faktisk run-id og oppstart.

Fortsett det opprinnelige signalspørsmålet: hvor svekkes representasjonens
variasjon, og motarbeider hjelpetapenes gradienter LONG–SHORT-komponenten?
Bevar samme cachede TRAIN16, frosne initial-/sluttmodeller og korrigerte mål.
Ingen ny optimizeroppdatering, CONTROL/VAL/TEST eller Exit-forward.

En separat plan skal bruke diagnostic_kind
`initial_final_entry_signal_inference_checked`, schema
`gx1_entry_signal_diagnostic_plan_v2`, model_forwards=4 og variants initial/final.
Bind forward_parity_result til den fullførte native RESULT.json med filhash
`dc7a50ebff2747d4cd22f648326e58a3199c05ba3ca30cf7088d86264315b45f`.
Kontrakten krever samme cache, checkpoints, modeller og eksakt tidligere
inferenssamsvar. Øvrige bindinger gjenbrukes fra den bevarte signalplanen.

Bruk en ny artefaktmappe, oppdater avgrenset policy og kildebinding, og gjenbruk
eksisterende ADMISSION_CHECK.py og native forberedelses-/aktiveringsoperatører.
Forrige Windows-plan er nå NATIVE_ENTRY_FORWARD_PARITY_20260917 med campaign-
filhash 186abf50379c585b25eeccf632c8197570de8124acb4fcda0995a65b9272641e.
Gamle operatører har gamle run-id-er og skal ikke relanseres uendret.
Forbered bare fra ren, pushet kilde; én native jobb, fire forwards, null
optimizersteg. Ingen automatisk utvidelse eller ny trening etter rapporten.

Dette er fortsatt en eval-diagnose på én gjenbrukt TRAIN16. Den viser ikke en
full felles klippet Adam-oppdatering, framtidig læring eller lønnsomhet.
