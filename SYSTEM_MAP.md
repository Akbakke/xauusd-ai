# SYSTEM MAP — beholdt offline kjede

Gjeldende bindinger eies av NEXT_RUN_POLICY.json; dokumentet er ikke en recipe.

Aktiv fase er prognose av åpnings-/sluttretning (lavere/lik/høyere)
på M5, M15, H1, H4 og D1
ved hver lukkede M5-bar. Fasit sammenligner hver TF-bars åpning med
forrige åpning og slutt med forrige slutt, med kjente likhetsutfall. Entry, Entry-Q og Exit
kommer senere. Siste kjørte kontroll brukte TA-profilen med åtte
spesialistfamilier og D1/makro via global kontekst. Native GC-retning
har en egen beholdt modellrute. Barprognosen bruker egne treklassehoder
for åpning og slutt på hver TF; den arver ingen S/F-aktivitetsmaske.
Full-TA-profilen omfatter de kanoniske tekniske feltene og makro, mens
felles fysisk GC MBP-1-komposisjon krever egen kvalifisering.

```text
native XAUUSD M5-priser og kausalt lukkede M15/H1/H4/D1-barer
  → indikatorer, observasjonsmasker, native signalmanifest og FIT-statistikk
  → åtte spesialistfamilier og D1/makro-kontekst
  → familie- og tidsrammeoppmerksomhet, separat blanding innen hver tidsramme
  → per-TF representasjon og egne prognoser for åpnings- og sluttretning
  → egen skåring av framtidig sluttretning og allerede observerbar åpning
```

Retningsinputs, fasit, masker, akser, priorer, tap og evaluering bygges
fra native M5/høyere med egen kildeidentitet. M1 er for senere oppfølging
av åpne handler. Eldre M1-baserte mål og dataklargjøring er sperret.

GC-kjeden beholder MBP-1 trades/BBO, prisgrid, footprint, kontrakt og faktisk
dekning i egen native feature- og modellrute. Åpnings-/sluttfasit og observerbarhet
går til tap og skåring.

## Eiere

- Barfasit og framtidige TF-koblinger: market_state_native_m5_targets_v1.
  build_native_m5_direction_targets_v1 kontrollerer hele tillatte native
  historikken med eksplisitte første-forgjenger- og grensemasker.
- Native M5-featurebygging, egne akser og fysiske normaliseringspopulasjoner:
  market_state_native_m5_feature_inputs_v1. Ingen gammel M1-view eller statistikk.

- Feltorden/familier: gx1/contracts/entry_model_native_signal_v1.py og
  gx1/features/entry_model_native_feature_layers_v1.py.
- GC-kilde/flow/profil: gc_continuous_source_v1, gc_order_flow_v1,
  rebuild_native_gc_inputs_v1 og gc_native_context_v1. Komplette og observerte
  domener holdes atskilt; ukjent aggressor og faktisk coverage følger verdien.
- Featureformler: gx1/features; HTF beregnes på lukkede native candles, ikke
  ved resampling av ferdige indikatorer. SMC sweep-AVWAP er markedsevidens,
  aldri en separat handelsregel.
- Normalisering: entry_model_native_input_normalization_v1 og eksisterende
  unified_exit-native/composite normalization-eiere.
- Aktiv barprognose: NativeM5BarDirectionModel i market_state_ta_integrated_model_v1,
  med native_bar_direction_loss/score_sums i market_state_ta_integrated_objective_v1.
  Felles encoder beholder fem TF, åtte familier og makro; gamle minuttqueryer
  konstrueres ikke. Eldre S/F-eiere er historiske kontrakter.
- Fysisk ny native rute: qualify_market_state_native_m5_pipeline_v1,
  med eksakte ferske parameterplaner og uavhengig priskontroll før modellbaseline.
- Entry-observerte markedsmål: entry_observed_market_horizons_v1; aktuelt
  h20-terminalmål må ikke beskrives som en måling av M5-MAE/MFE-path-quality.
- Fitted-Q/handlinger: entry_fitted_q_v1, unified_exit_fitted_q_v1 og
  modellens eksisterende native trenings-/forwardeiere under gx1/models/entry_v10.
- Kostnader, kapitalhurdle og lazy steg: unified_exit_*economic*,
  unified_exit_prospective_cost_policy_v1 og hash-bundne splitautoritetene.
- Datasett/indekser/sampler: eksisterende materialize/build/benchmark-eiere
  under gx1/scripts; fryste outputs gjenbrukes, ikke parallelle implementasjoner.
- Native kjøreautoritet: run_unified_exit_native_candidate_window_v1 og
  native campaign-eiere; training_enabled=false holder dem lukket.
- Kapasitet: scripts/gx1_capped_run.sh, gx1_guarded_trainer_exec.sh,
  signert host-telemetri og den native Windows clock/profile-launcheren.
- Nåstatus: scripts/collect_gx1_handover_readonly.py og gx1_handover.sh;
  ingen historisk checkpointfallback.
- Retention: gx1/contracts/evidence_retention_v1.py og
  gx1.scripts.cleanup_gx1_evidence_v1. .env/.venv/.git er ikke oppryddingsfyll.

Offline serving/paritets- og persistenseiere beholdes for samme bundletilstand;
de gir ingen adgang til live/paper/broker. Fokuserte tester ligger under tests/.
Eksisterende research_ta_campaign-eier beholdes for det separate full-B-målet.
En beholdt CLI er ikke autorisert bare fordi filen finnes.
