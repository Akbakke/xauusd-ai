# Residualkandidatens hovedbane — fullført diagnose 18. september 2026

To native inferensforwards er fullført med guard PASS, null backward/optimizer-
steg og eksakt samsvar med lagrede initial-/sluttprediksjoner. Ingen aktiv jobb.
Windows-task er Disabled og brukt scope er stengt. Kjøringen er ikke læringsbevis.

| Målepunkt | Initial | Slutt |
|---|---:|---:|
| seq_pool, gjennomsnittlig L2-norm | 13,5393 | 156,8660 |
| main_fuse, gjennomsnittlig L2-norm | 1,0458 | 117,7130 |
| main_fuse, RMS feature-standardavvik | 0,05593 | 0,15194 |
| rå MTF, RMS feature-standardavvik | 0,10583 | 0,31511 |
| rå context, RMS feature-standardavvik | 0,24734 | 0,32667 |
| joint-normalisert, RMS feature-standardavvik | 0,33926 | 0,03569 |
| Entry-hidden, RMS feature-standardavvik | 0,28316 | 0,04003 |

Main-fuse-utgangene er nesten parallelle mellom de16 radene: gjennomsnittlig
parvis cosinus øker fra0,6154 til0,999878. Absolutt variasjon i rå MTF/context
øker; det er derfor feil å si at all upstream variasjon er borte. Stor nesten
felles hovedbane dominerer inngangen til felles normalisering. Variasjonen
etter denne er9,51 ganger mindre enn initialt, og Entry-hidden er7,07 ganger
mindre. Dette lokaliserer svekkelsen; det beviser ikke en rettelses effekt.

Raw Entry-MSE er343,7502→308,0916. LONG–SHORT-kontrastdelen er308,3224→307,9635
og står for99,9584% av sluttfeilen. Det meste av forbedringen er felles bias.
Siste treningsforsøk forblir avvist: ingen bedre Entry/Exit-beslutninger.

Koden bekrefter pre-norm TransformerEncoder uten avsluttende norm og en hoved-
fuse uten normalisering. De tre tidligere endrede residualportene begrenser
ikke denne banen. Samtidig opprettes læreren ved deepcopy av online-modellen.
En direkte arkitekturendring vil derfor også endre lærerfunksjonen, selv om
de lagrede vektene er identiske. Det må løses eksplisitt før et nytt forsøk.

Neste: avklar minste binding av uendret lærerfunksjon og bootstrap-targets før
én rettelse av den målte hovedbanen. Ikke stable normaliseringer eller åpne
ny trening før sammenligningsgrunnlaget er bevart. Ingen ny måle-/treningsplan
er bundet. Ingen bevis på generalisering eller positiv økonomi foreligger.

BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
Bevis: BASE/NATIVE_RESIDUAL_REPRESENTATION_20260918/{REVIEW.json,entry_gradient_diagnostic/RESULT.json}.
Kilde bff19fbe3d4c1b037c5ec159b31adb79db6dd9b8. Originale checkpoints/cache og
brukte operatører bevares; ikke relanser dem. TRAIN16 er gjenbrukt utvikling,
ikke en uavhengig validering. TEST forblir forseglet.
