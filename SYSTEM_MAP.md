# SYSTEM MAP — beholdt offline kjede

Gjeldende bindinger eies av NEXT_RUN_POLICY.json; dokumentet er ikke en recipe.

```text
hash-bundne native M1/M5-priser og lukket MTF-kontekst
  → eksisterende feature-eiere / åtte familier
  → native signalmanifest og uforanderlig TRAIN-normalisering
  → Entry M5-vinduer + separate M1-views + økonomi/tilstandsindekser
  → delte encoders og Entry-/Exit-fitted-Q i én bundle
  → initial-/læringsmåling og fryste TRAIN/CONTROL-koordinater
  → egne lærings-, generaliserings-, paritets- og økonomiporter
```

## Eiere

- Feltorden/familier: gx1/contracts/entry_model_native_signal_v1.py og
  gx1/features/entry_model_native_feature_layers_v1.py.
- Featureformler: gx1/features; HTF beregnes på lukkede native candles, ikke
  ved resampling av ferdige indikatorer. SMC sweep-AVWAP er markedsevidens,
  aldri en separat handelsregel.
- Normalisering: entry_model_native_input_normalization_v1 og eksisterende
  unified_exit-native/composite normalization-eiere.
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
