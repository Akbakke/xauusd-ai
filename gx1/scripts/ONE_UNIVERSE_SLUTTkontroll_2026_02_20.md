# ONE UNIVERSE – Sluttkontroll (E) 2026-02-20

## Grep-rapport (aktive stier; _quarantine og _archive utelatt)

| Mønster | Treff i aktiv kode | Kommentar |
|---------|--------------------|-----------|
| `EntryV10HybridTransformer` | 0 | Kun i `gx1/_quarantine/legacy_entry_2026_02_20/` og `_archive_artifacts/`. |
| `entry_v10_hybrid_transformer` | 1 | Kun i `gx1/utils/truth_banlist.py` som **banned module** (krevende). |
| `16/88` eller seq 16 / snap 88 | Se under | Kommente referanser («legacy 16/88 removed») er OK. Noen filer refererer fortsatt til 16/88 som *støtte* (se «Gjenstående»). |
| `--legacy` / `--allow-legacy` (entry) | 0 | Fjernet fra trainer. Andre scripts har andre `--legacy*` (f.eks. trade journal, XGB) – ikke entry. |
| `model_variant.*"v10"` (som default/konfig) | 0 | Default er `v10_ctx`; ingen aktiv kode setter variant til `"v10"`. |
| `ctx_cont_dim` 2/4 eller `ctx_cat_dim` 5 (4/5-stier) | Flere | `run_truth_e2e_sanity`, `signal_bridge_v1`, `exit_transformer_io_v2` er strammet til 6/6. Runner, chunk_bootstrap, entry_context_features, add_ctx_cont_columns, rebuild_prebuilt, exit_transformer_v0 og noen tester har fortsatt defaults/fallbacks 2/4/5 – se «Gjenstående». |

## What changed (kort)

### Slettet / flyttet
- **Flyttet til `gx1/_quarantine/legacy_entry_2026_02_20/`:**  
  `evaluate_entry_v10_offline.py`, `thresholds_v10.py`, `analyze_entry_v10_label_quality.py` (med LEGACY-header).  
  `ENTRY_V10_HYBRID_STACK_DESIGN.md` (stub med header i quarantine).  
  `TRANSFORMER_SESSION_INFO_TRUTH.md` (allerede i quarantine med header).
- **Trainer:** `train_entry_v10_legacy_transformer()`, CLI `--ctx`, `--legacy`, `--allow-legacy`, `--variant`, `--model-out`, `--meta-out` fjernet. Kun ctx-path.
- **Dataset:** `EntryV10Dataset` (legacy), `EntryV10RowSchema` og legacy-doc fjernet. Kun `EntryV10CtxDataset` 6/6.
- **Runner:** Legacy forward-gren (session/vol/trend som egne args) fjernet; mangler ctx → hard fail.
- **model_loader_worker:** Legacy v10-gren fjernet; kun `v10_ctx`; ellers `ONE_UNIVERSE`-feil.
- **model_worker:** Erstattet med hard fail (ONE_UNIVERSE); ingen checkpoint-basert legacy-load.
- **build_v10_1_edge_buckets:** Stub som feiler med RuntimeError (peker på quarantine).

### Oppdatert (6/6, 7/7, ctx-only)
- **run_truth_e2e_sanity:** Kun 6/6 (konstanter `CTX_CONT_DIM`/`CTX_CAT_DIM`); `--ctx-cont-dim`/`--ctx-cat-dim` og `--exit-ctx-*-dim` fjernet; preflight og postrun krever 6/6; footer ≠ 6/6 → hard fail.
- **signal_bridge_v1:** `ALLOWED_CTX_CONT_DIMS = (6,)`, `ALLOWED_CTX_CAT_DIMS = (6,)`.
- **exit_transformer_io_v2:** `DEFAULT_CTX_CONT_DIM`/`DEFAULT_CTX_CAT_DIM` = 6; doc oppdatert.
- **Tester:** `test_entry_v10_ctx_model_shapes.py` og `test_entry_v10_ctx_runtime.py` oppdatert til 6/6 og 7/7, ctx-only forward; legacy backward-compat-test fjernet. `test_model_loader_worker.py`: variant `v10_ctx`, klassename `EntryV10CtxHybridTransformer`, test for ONE_UNIVERSE ved variant `v10`.
- **truth_banlist:** `gx1.models.entry_v10.entry_v10_hybrid_transformer` lagt til som forbudt modul.

### Gjenstående (utenfor denne runden)
- Filer som fortsatt nevner eller støtter 16/88 (f.eks. `validate_entry_v10_features.py`, `feature_contract_v10_ctx.py`, `build_entry_v10_dataset.py`, `runtime/feature_fingerprint.py`) – enten quarantined eller oppdatert i senere runde.
- Runner / chunk_bootstrap / entry_context_features / add_ctx_cont / rebuild_prebuilt / exit_transformer_v0: defaults eller fallbacks 2/4/5 – kan strammes til 6/6 i TRUTH-path i en senere endring.
