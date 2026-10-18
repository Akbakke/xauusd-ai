# SYSTEM MAP — beholdt offline kjede

Gjeldende bindinger eies av NEXT_RUN_POLICY.json; dokumentet er ikke en recipe.

Aktuell bølge bygger bare M5 Entry. Samme encoder lærer separat netto
endepunktsmarkout, signert MFE og konservativ MAE før første MFE fra1..480min;
dagshorisonter er maskert støtte. Et delt indeks over originale M1 BID/ASK
og target-only radpekere erstatter lagrede tette framtidsbaner. Endepunkt og
prisforløp har uavhengige masker. Fremtidige utfall/masker brukes aldri som
inputs. Ingen Exit-tap er aktive. Kvalitetsratio fra modellens prognoser er en
kandidat uten native beslutningsautoritet inntil kvalifisert etterfølger er
bundet; publiserte022/026features kvalifiserer ikke den nye modellen.

```text
hash-bundne XAUUSD M1/M5-priser + GC MBP-1 trades/BBO/order counts
  → GC M1-observationer + bevaret prisgrid/footprint + kontrakt/coverage-metadata
  → eksisterende feature-eiere / åtte spesialister + niende D1/makro-kontekst
  → kausale M5/M15/H1/H4/D1-ruter med eksplisitt kilde-/observasjonsmaske
  → native signalmanifest og uforanderlig TRAIN-normalisering
  → Entry M5-vinduer + separate M1-views + økonomi/tilstandsindekser
  → delte encoders og Entry-/Exit-fitted-Q i én bundle
  → initial-/læringsmåling og fryste TRAIN/CONTROL-koordinater
  → egne lærings-, generaliserings-, paritets- og økonomiporter
```

## Eiere

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
