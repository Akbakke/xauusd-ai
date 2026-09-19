# Neste: koble frossen ONLINE512 til én native TRAIN-vurdering

Observasjonsgrensen og det faste utvalget er ferdige. Gjenbruk
train_observation_cutoff_review og frozen_exit_train_footprint i
NEXT_RUN_POLICY.json. 13 målrettede tester er bestått; ikke gjenta dem uten
berørte endringer. Ingen kjøring er bundet eller startet.

Neste konkrete rettelse er egen frossen ONLINE512-checkpointbinding og et
nullstegs TRAIN-evalueringsomfang i eksisterende native campaign/dispatcher.
Gjenbruk prefix TRAIN-state factory, train_probe_ds, full evaluator og alle
vakter. Den gamle fitted-readout-veien er bundet til VAL5508 og er ingen
fallback. Ingen ny runner, modell, target, normalisering eller treningsendring.

Eksisterende utvidelsespunkter er require_native_run_scope i
unified_exit_native_candidate_campaign_v1.py, checkpointvalidering i
unified_exit_random_access_val_checkpoint_v1.py og dispatch før trening i
run_unified_exit_random_access_full_train_v1.py. evaluate_bound_full_val_v1
og build_chronological_train_rollout_cohort håndterer allerede det nye utvalget.
Bevar gjeldende measurement_only-sperre og originalt 512-checkpoint/lærer.

De samme 256 TRAIN-identitetene er bundet i kronologisk rekkefølge, med begge
kontrafaktiske sider. Observasjonsgrense er 2026-03-01T00:00:00Z. Siste
observerbare beslutning før grensen er 2026-02-27T22:00:00Z; de tre berørte
forløpene beholder åpne posisjoner til siste observerbare verdi. Grensen er
en datagrense, ingen maksimal holdetid eller modellbestemt EXIT.

Øvre omfang hvis alle holder: 3375234 delte tilstander og 46573 native
policyforwards ved batch256. Dette er en øvre arbeidsmengde, ikke målt
kjøretid. Bind én invokasjon, uendrede kostnader, åtte CPU-arbeidere og
etablert tre timers vindu før oppstart. Ressursstopp er ufullstendig vurdering.
Bind resultatkriterier: begge sider og alle måneder, alle åpne posisjoner,
modellens faktiske Entry-valg separat fra kontrafaktiske Exit-forløp.

Spørsmålet er om hele den frosne Exit-policyens forbedring overlever flere
påfølgende beslutninger under samme kostnader. Et positivt kontrafaktisk
resultat er ikke en dokumentert kausal Entry-strategi. Negative sidegjennomsnitt
avkrefter heller ikke alene betinget edge. Entry er fortsatt FLAT256/256;
læringsport, kronologisk overføring og samlet strategiprofitt er ubevist.

Separate quote/ordre/fill-logger kan undersøkes når brukeren oppgir dem.
Manglende logger hindrer ikke denne sammenligningen med uendrede kostnader.
Ingen blind ekstra trening, terskelsøk, full epoch/full VAL, CONTROL/TEST,
live/paper eller spending. Målet er aktivt.
