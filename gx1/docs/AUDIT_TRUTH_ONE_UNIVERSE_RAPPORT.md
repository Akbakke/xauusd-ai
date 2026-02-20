# TRUTH/ONE-UNIVERSE audit – rapport (fakta med kildehenvisning)

**Dato:** 2026-02-20  
**Regler:** Kun lest kode; ingen endringer. Kilde = fil + linjeområde. Ved usikkerhet: "ukjent" + hva som mangler.

---

## 1) Entry transformer – faktiske inputs (bevis)

### 7 signaler: navn + kontrakt + runtime bygging + validering

**Navn (alle 7), i rekkefølge:**  
`gx1/contracts/signal_bridge_v1.py:45-51`

```text
ORDERED_FIELDS = [
    "p_long", "p_short", "p_flat",
    "p_hat", "uncertainty_score", "margin_top1_top2", "entropy",
]
```

- **Kontrakt:** `signal_bridge_v1.py:45-56` (ORDERED_FIELDS, SEQ_SIGNAL_DIM = SNAP_SIGNAL_DIM = 7).
- **Runtime bygging (entry):** `oanda_demo_runner.py:9400-9417` – `field_to_seq` / `field_to_snap` mappes til `ORDERED_FIELDS`; `for j, fname in enumerate(ORDERED_FIELDS)` fyller `seq_data[:, j]` og `snap_data[j]`. Importerer ORDERED_FIELDS fra `gx1.contracts.signal_bridge_v1` (linje 9375, 9410).
- **Validering:** `validate_seq_signal` og `validate_snap_signal` kalles i `oanda_demo_runner.py:9421-9422` (kontekst `"entry_v10_hybrid.signal_only"`). Definert i `signal_bridge_v1.py:129-185` (dtype, ndim, shape[-1] == SEQ_SIGNAL_DIM/SNAP_SIGNAL_DIM, finite).

**Konklusjon:** Entry transformer får de 7 signalene; navn og rekkefølge er som i kontrakten; runtime bygges fra ORDERED_FIELDS og valideres med signal_bridge_v1.

---

### ctx_cont 6: navn (alle 6) + kontrakt + runtime

**Navn (alle 6), rekkefølge:**  
`gx1/contracts/signal_bridge_v1.py:62-76`

- Baseline (2): `atr_bps`, `spread_bps`
- Slow core (4): `D1_dist_from_ema200_atr`, `H1_range_compression_ratio`, `D1_atr_percentile_252`, `M15_range_compression_ratio`
- `ORDERED_CTX_CONT_NAMES_EXTENDED` = baseline + slow_core → 6 navn.

**Kontrakt:** `signal_bridge_v1.py:74-78` (ORDERED_CTX_CONT_NAMES_EXTENDED, N_CTX_CONT_EXTENDED = 6).

**Runtime (entry):**  
- Bygges i `entry_context_features.py`: `EntryContextFeatures` med feltene atr_bps, spread_bps, d1_dist_from_ema200_atr, h1_range_compression_ratio, d1_atr_percentile_252, m15_range_compression_ratio (`entry_context_features.py:90-96`).  
- Eksport til tensor: `to_tensor_continuous()` (`entry_context_features.py:119-133`) – returnerer array i rekkefølge (atr_bps, spread_bps, d1_dist..., h1_range..., d1_atr_percentile..., m15_range...).  
- Runner bruker ctx: `oanda_demo_runner.py:9523-9526` og `9574-9577` – henter `expected_ctx_cont_dim` / `expected_ctx_cat_dim` fra `bundle_meta.get("expected_ctx_cont_dim", 2)` og `bundle_meta.get("expected_ctx_cat_dim", 6)`, deretter `entry_context_features.to_tensor_continuous(expected_ctx_cont_dim)` og `to_tensor_categorical(expected_ctx_cat_dim)`.

**Merknad (signatur):** I `entry_context_features.py` tar `to_tensor_continuous()` og `to_tensor_categorical()` kun `self` (linje 103, 119). Runner kaller med ett argument (`expected_ctx_cat_dim` / `expected_ctx_cont_dim`). Dersom disse metodene ikke har ekstra parameter, vil kallene gi TypeError. **RED FLAG:** Mulig signaturmismatch – verifiser at EntryContextFeatures faktisk aksepterer (og ev. ignorerer) dim-argumentet.

---

### ctx_cat 6: navn (alle 6) + kontrakt + runtime

**Navn (alle 6), rekkefølge:**  
`gx1/contracts/signal_bridge_v1.py:85-98`

- Baseline (5): `session_id`, `trend_regime_id`, `vol_regime_id`, `atr_bucket`, `spread_bucket`
- Extended (+1): `H4_trend_sign_cat`
- `ORDERED_CTX_CAT_NAMES_EXTENDED` = baseline + [CTX_CAT_COL_H4_TREND_SIGN] → 6 navn.

**Kontrakt:** `signal_bridge_v1.py:94-99` (ORDERED_CTX_CAT_NAMES_EXTENDED, N_CTX_CAT_EXTENDED = 6).

**Runtime (entry):**  
- Bygges i `entry_context_features.py`: `EntryContextFeatures` med session_id, trend_regime_id, vol_regime_id, atr_bucket, spread_bucket, h4_trend_sign_cat (`entry_context_features.py:82-87`).  
- Eksport: `to_tensor_categorical()` (`entry_context_features.py:103-117`) – rekkefølge som over.  
- Runner: samme steder som for ctx_cont; ctx_cat brukes med `expected_ctx_cat_dim` fra bundle_meta.

---

### Dim enforcement: hvor hard-fails skjer

- **Kontrakt (TRUTH/SMOKE):** `signal_bridge_v1.py:214-285` – `validate_bundle_ctx_contract_in_strict`: expected_ctx_cont_dim må være i `ALLOWED_CTX_CONT_DIMS` (6,), expected_ctx_cat_dim i `ALLOWED_CTX_CAT_DIMS` (6,); ellers RuntimeError. Meta-lister må matche kontraktens prefix.
- **Entry bundle load:** `oanda_demo_runner.py:2899-2907` – ctx_cont_dim/ctx_cat_dim satt fra metadata; hvis ikke 6/6 → RuntimeError "ONE_UNIVERSE_CTX_DIM_MISMATCH".
- **Chunk bootstrap:** `chunk_bootstrap.py:357-377` – MASTER_TRANSFORMER_LOCK må ha ctx_cont_dim=6 og ctx_cat_dim=6; ellers RuntimeError "NON_CANON_TRANSFORMER".
- **E2E postrun:** `run_truth_e2e_sanity.py:458-466` – footer ctx_cont_dim/ctx_cat_dim må være 6/6; ellers gates_failed "ctx_dims".
- **Entry hot path:** `oanda_demo_runner.py:9529-9553` – ctx_cat/ctx_cont shape sjekkes mot expected_ctx_cat_dim/expected_ctx_cont_dim; ved mismatch RuntimeError.

**RED FLAG (default i hot path):**  
`oanda_demo_runner.py:9524` – `expected_ctx_cont_dim = bundle_meta.get("expected_ctx_cont_dim", 2)`. Default er **2**, ikke 6. Entry-bundle skriver `ctx_cont_dim`/`ctx_cat_dim` i `bundle_metadata.json` (`train_entry_transformer_v10.py:418-439`), ikke nødvendigvis `expected_ctx_cont_dim`/`expected_ctx_cat_dim`. Hvis metadata mangler `expected_ctx_cont_dim`, brukes 2 i hot path. Load-stien bruker `metadata.get("expected_ctx_cont_dim", metadata.get("ctx_cont_dim", 6))` (2899), så der er fallback til ctx_cont_dim og deretter 6.

---

## 2) Exit transformer – faktiske inputs (bevis)

### 7 signaler (fra row/signals)

**Kontrakt (IOV1/IOV2/IOV3):**  
`exit_transformer_io_v1.py:19-27` – ORDERED_SIGNAL_FIELDS:  
`p_long`, `p_short`, `p_flat`, `p_hat`, `uncertainty_score`, `margin_top1_top2`, `entropy`.

**Runtime (exit row):**  
`oanda_demo_runner.py:16439-16445` – signals-dict med p_long_now, p_short_now, p_hat_now, uncertainty_score, margin_top1_top2, entropy (p_flat implisitt/0 der det trengs i IOV1-slice).  
IOV2/IOV3 bygger vektor via `_row_to_iov1_slice` / `row_to_feature_vector_v2` / `row_to_feature_vector_v3` som bruker samme signal-rekkefølge fra row["signals"].

### entry_snapshot 5 (navn)

**Kontrakt:** `exit_transformer_io_v1.py:28-35`  
ORDERED_ENTRY_SNAPSHOT_FIELDS:  
`p_long_entry`, `p_hat_entry`, `uncertainty_entry`, `entropy_entry`, `margin_entry`.

**Runtime:**  
`exit_transformer_io_v2.py:102-116` – _row_to_iov1_slice henter entry_snapshot fra row og fyller indeks 7–11 med disse feltene.

### trade_state (navn)

**Kontrakt:** `exit_transformer_io_v1.py:37-45`  
ORDERED_TRADE_STATE_FIELDS:  
`pnl_bps_now`, `mfe_bps`, `mae_bps`, `dd_from_mfe_bps`, `bars_held`, `time_since_mfe_bars`, `atr_bps_now`.

**Runtime:**  
`exit_transformer_io_v2.py:99-124` – state brukes for pnl_bps, mfe_bps, mae_bps, dd_from_mfe_bps, bars_held, time_since_mfe_bars, atr_bps_now (indeks 12–18).

### ctx_cont 6 og ctx_cat 6 (navn) – exit

Exit bruker **samme** ctx-semantikk som entry (ONE UNIVERSE 6/6):  
- IOV2/IOV3 forventer `row["context"]["ctx_cont"]` og `row["context"]["ctx_cat"]` med lengde 6 hver.  
- IOV2 navngir dem som `ctx_cont_0..ctx_cont_5` og `ctx_cat_0..ctx_cat_5` i `ordered_feature_names_v2` (`exit_transformer_io_v2.py:36-46`).  
- Semantisk rekkefølge skal matche signal_bridge_v1 ORDERED_CTX_CONT_NAMES_EXTENDED / ORDERED_CTX_CAT_NAMES_EXTENDED (ikke eksplisitt listet i exit_transformer_io_v2; avtalt ONE UNIVERSE).

**Eksakte 12 ctx-navn (rekkefølge):**  
- ctx_cont: som i signal_bridge_v1 ORDERED_CTX_CONT_NAMES_EXTENDED (atr_bps, spread_bps, D1_dist..., H1_range..., D1_atr_percentile..., M15_range...).  
- ctx_cat: som i signal_bridge_v1 ORDERED_CTX_CAT_NAMES_EXTENDED (session_id, trend_regime_id, vol_regime_id, atr_bucket, spread_bucket, H4_trend_sign_cat).  
I IOV2 feature_names brukes generiske `ctx_cont_i` / `ctx_cat_i`; innholdet i row["context"] bygges i runner med candle_row ctx_cont/ctx_cat (6/6) (`oanda_demo_runner.py:16463-16472`).

### IOV2 dim 31 og IOV3 dim 37: bevis for mapping

- **IOV2:** `exit_transformer_io_v2.py:29-30` – ORDERED_IOV1_NAMES (19) + ctx_cont (6) + ctx_cat (6) → `feature_dim_v2()` = 19+6+6 = 31 (`exit_transformer_io_v2.py:49-53`).
- **IOV3:** `exit_transformer_io_v3.py:54-68` – IOV2_DIM = 31, ORDERED_EXIT_EXTRA_FIELDS_V3 (6 ekstra), FEATURE_DIM_V3 = 31+6 = 37.
- **Append av ctx i IOV2:** `exit_transformer_io_v2.py:126-167` – `row_to_feature_vector_v2`: først _row_to_iov1_slice (19), deretter vec[IOV1_DIM:IOV1_DIM+ctx_cont_dim] fra context.ctx_cont, deretter ctx_cat (`exit_transformer_io_v2.py:161-166`).
- **Append av ctx og V3-ekstra i IOV3:** `exit_transformer_io_v3.py:230-289` – `row_to_feature_vector_v3`: kaller `row_to_feature_vector_v2` for prefix, deretter out[IOV2_DIM:] = V3-extras (drawdown_from_peak_bps, pnl_over_atr, spread_over_atr, bars_since_mfe, price_dist_from_entry_bps, price_dist_from_entry_over_atr).

---

## 3) IO-versjon – hva som faktisk kjøres

### Er input_dim 19/31/37? Hvor kommer den fra?

- **Kilde:** `exit_transformer_config.json` i modellmappen som lastes.  
- **Leses:** `exit_transformer_v0.py:261` – `ExitTransformerDecider.__init__`: `self.input_dim = int(config.get("input_dim", FEATURE_DIM))`. FEATURE_DIM er IOV1 = 19 (`exit_transformer_v0.py:25, 116`).  
- **Skrives ved trening:** `exit_transformer_v0.py:176-185` – `save_exit_transformer_artifacts` skriver config med `"input_dim": in_dim` (in_dim = input_dim argument eller FEATURE_DIM).

Så **input_dim** er 19 (IOV1), 31 (IOV2) eller 37 (IOV3) avhengig av hva som ble skrevet i exit_transformer_config.json for den mappen som policy peker på.

### V2 vs V3 mapping: hvor velges det i runner/decider?

- **Runner:** `oanda_demo_runner.py:16480-16486` – `expected_dim = getattr(decider, "input_dim", None)`. Hvis `expected_dim == FEATURE_DIM_V3` (37) → `row_to_feature_vector_v3(...)`, ellers → `row_to_feature_vector_v2(...)`.
- **Decider predict:** `exit_transformer_v0.py:236-241` – validering: hvis input_dim == FEATURE_DIM_V3 brukes validate_window_v3, ellers validate_window_v2 (eller IOV1 validate_window).

Valget er altså **kun** basert på `config["input_dim"]` i den lastede exit-transformer-configen: 37 → IOV3, 31 → IOV2, 19 → IOV1.

### Footer / exits jsonl: felter

- **Footer:** `run_truth_e2e_sanity.py` postrun leser `chunk_footer.json`; exit-strategi felter inkluderer exit_type, exit_ml_enabled, exit_ml_decision_mode, exit_ml_config_hash (runner setter disse). Footer har **ikke** eksplisitt `io_version` eller `input_dim` i det som er vist i audit-koden; disse kommer fra exits jsonl.
- **exits jsonl:** `oanda_demo_runner.py:16527-16536` – hver linje får `computed.io_version` og `computed.input_dim`:  
  `"io_version": getattr(self, "exit_ml_io_version", None) or ("IOV3" if expected_dim == FEATURE_DIM_V3 else "IOV2")`,  
  `"input_dim": int(expected_dim)`.

### Hvor exit_ml_io_version settes

`oanda_demo_runner.py:2096-2102`: Etter lasting av decider: `_cfg = getattr(self.exit_transformer_decider, "config", None) or {}`; `self.exit_ml_input_dim = _cfg.get("input_dim")`; hvis `_cfg.get("exit_ml_io_version")` satt brukes det, ellers: `"IOV3" if _in_dim == 37 else ("IOV2" if _in_dim == 31 else ("IOV1" if _in_dim == 19 else None))`.

**Konklusjon IO-versjon:**  
- **IOV3 er aktiv** hvis og bare hvis den lastede exit-transformer-mappen har `exit_transformer_config.json` med `input_dim: 37`.  
- **IOV2 er aktiv** hvis `input_dim: 31`.  
- Policy peker på `model_path: "models/exit_v0_ctx/EXIT_TRANSFORMER_V0_IOV2"` (`EXIT_TRANSFORMER_ONLY_V0.yaml:17`). Navnet antyder IOV2; **faktisk versjon avhenger av innholdet i den mappens exit_transformer_config.json**. Uten å lese den filen på disk: **ukjent** om IOV2 eller IOV3 kjører. Bevis: valget er kun `expected_dim == FEATURE_DIM_V3` (37) vs else (31/19) i runner og decider.

---

## 4) Policy/Config – hva som faktisk styrer TRUTH

### Hvilken policy YAML brukes

- **Kanonisk:** `truth_banlist.py:132` – `TRUTH_CANONICAL_POLICY_RELATIVE = "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"`.  
- **E2E:** `run_truth_e2e_sanity.py:786-791` – policy_path fra `GX1_CANONICAL_POLICY_PATH` eller default ENGINE-relativ sti til samme fil.  
- **Banlist:** `assert_truth_policy_path_canonical` (truth_banlist.py:196-262) – hard-fail hvis loaded path ikke er lik (engine_root / TRUTH_CANONICAL_POLICY_RELATIVE).resolve().

### Hvilken exit YAML brukes

- **Policy:** `GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml:63` – `exit_config: gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/EXIT_TRANSFORMER_ONLY_V0.yaml`.  
- **Innhold exit-config:** `EXIT_TRANSFORMER_ONLY_V0.yaml` – exit.type EXIT_TRANSFORMER_V0; exit_ml.exit_transformer.model_path: `"models/exit_v0_ctx/EXIT_TRANSFORMER_V0_IOV2"`; enabled: true; threshold 0.5; window_len 64.

### Er exit transformer enabled i YAML og krever model_path?

Ja. Exit-config har `exit_transformer.enabled: true` og `model_path` satt. Runner feiler med RuntimeError ved manglende eller ugyldig modell (`oanda_demo_runner.py:2108-2112`).

---

## 5) Legacy / "we think we run X but run Y" – funn

### Import-risiko

- **Banlist (banned_modules):** `truth_banlist.py:28-39` – bl.a. feature_contract_v13_core, feature_contract_v10_ctx, entry_v10_hybrid_transformer, exit_master_v1, exit_farm_v2_rules*, exit_fixed_bar.  
- **Sjekk:** `assert_truth_banlist_clean` (truth_banlist.py:64-86) – i TRUTH/SMOKE: hvis noen av disse er i sys.modules eller forbidden env satt → RuntimeError "[TRUTH_BANLIST_HIT]".  
- **Forbidden env:** `GX1_REPLAY_PREBUILT_FEATURES_PATH` (truth_banlist.py:42-43).

### Fallback defaults

- **Entry ctx (hot path):** `oanda_demo_runner.py:9524` – `expected_ctx_cont_dim = bundle_meta.get("expected_ctx_cont_dim", 2)`. Default **2** – risiko hvis bundle_metadata ikke har expected_ctx_cont_dim (entry-bundle skriver ctx_cont_dim, ikke expected_ctx_cont_dim).  
- **Entry ctx (load):** `oanda_demo_runner.py:2899-2900` – `metadata.get("expected_ctx_cont_dim", metadata.get("ctx_cont_dim", 6))` – her er fallback 6 via ctx_cont_dim.  
- **Exit:** Ingen fallback på io_version/input_dim; decider bruker config.input_dim; ved mismatch mellom vektorlengde og input_dim → RuntimeError (oanda_demo_runner.py:16486-16495).

### Uønskede policy-paths

- Kun kanonisk policy tillatt i TRUTH (truth_banlist.assert_truth_policy_path_canonical). Ingen andre policy-filer tillatt.

### Quarantine / legacy replay

- `run_truth_e2e_sanity.py:114-126` – sjekk at `gx1.scripts.replay_eval_gated_parallel` ikke er importert og at legacy script-fil ikke finnes på disk; ellers RuntimeError.

### Banlist hard-fail status

- **assert_truth_banlist_clean:** Kaller `raise RuntimeError(...)` ved hit (truth_banlist.py:86).  
- **assert_truth_exit_policy_clean:** RuntimeError ved forbudt exit-type eller forbudte nøkler (truth_banlist.py:88-128).  
- **assert_truth_policy_path_canonical:** RuntimeError ved feil policy path (truth_banlist.py:196-262).

---

## 6) Training/Artifacts sanity

### Hvor expected_ctx_cont_dim / expected_ctx_cat_dim kommer fra (bundle metadata/lock)

- **Entry transformer:**  
  - **MASTER_TRANSFORMER_LOCK:** `train_entry_transformer_v10.py:401-412` – skriver `ctx_cont_dim`, `ctx_cat_dim` (ikke expected_ctx_cont_dim/expected_ctx_cat_dim).  
  - **bundle_metadata.json:** `train_entry_transformer_v10.py:418-439` – skriver `ctx_cont_dim`, `ctx_cat_dim`, `seq_input_dim`, `snap_input_dim`, osv. **Ikke** expected_ctx_cont_dim/expected_ctx_cat_dim.  
- **Runner** leser `expected_ctx_cont_dim`/`expected_ctx_cat_dim` med fallbacks (jf. §1 og §5). Så **expected_ctx_cont_dim/expected_ctx_cat_dim** kommer ikke fra entry transformer lock/metadata som skrives i train_entry_transformer_v10; de forventes å finnes i metadata eller fallback brukes.  
- **Chunk bootstrap:** `chunk_bootstrap.py:353-359` – leser MASTER_TRANSFORMER_LOCK og forlanger `ctx_cont_dim` og `ctx_cat_dim` (begge 6). SSoT for replay bootstrap er lock, ikke expected_*-navn.

### Exit transformer config: input_dim + io_version

- **Skriving:** `exit_transformer_v0.py:176-185` – save_exit_transformer_artifacts skriver `exit_transformer_config.json` med window_len, d_model, n_layers, input_dim, feature_names_hash. **exit_ml_io_version skrives ikke** til config.  
- **Trening:** `train_from_exits_jsonl` bruker `load_dataset_from_exits_jsonl` fra exit_transformer_dataset_v1 – som **kun** støtter IOV1 og IOV2 (use_io_v2, ctx_cont_dim, ctx_cat_dim). **IOV3 brukes ikke i load_dataset_from_exits_jsonl** (ingen row_to_feature_vector_v3 eller FEATURE_DIM_V3 der). Så exit-modell trent med nåværende script er enten 19 eller 31 (IOV2 6/6), ikke 37 (IOV3), med mindre annen treningssti brukes.

### input_dim 7+ctx / 31 / 37

- Entry: 7 signaler + 6 ctx_cont + 6 ctx_cat (separate innganger til modellen; ikke én flat 19/31/37).  
- Exit: 19 (IOV1), 31 (IOV2), 37 (IOV3) er de faktiske input_dim i exit_transformer_config.json.

**Konklusjon trening/artifacts:**  
- **Entry:** Lock og bundle_metadata bruker ctx_cont_dim/ctx_cat_dim (6/6). Runner leser også expected_ctx_* med fallbacks; hot-path default 2 for expected_ctx_cont_dim er en risiko.  
- **Exit:** Treningspipeline (train_from_exits_jsonl + exit_transformer_dataset_v1) produserer IOV1 eller IOV2 (31 med 6/6). IOV3 (37) er ikke støttet i load_dataset_from_exits_jsonl; for at IOV3 skal kjøre må modell være trent med annen metode og config.input_dim=37.  
- **Indikasjon på mulig mismatch:** (1) Entry: bundle_metadata har ikke expected_ctx_cont_dim; default 2 i hot path. (2) Exit: YAML model_path-navn "EXIT_TRANSFORMER_V0_IOV2" vs. ukjent faktisk input_dim i config. (3) Exit-trening fra E2E/LAST_GO er IOV2 (--require-io-v2), ikke IOV3.

---

*Rapport slutt. Ingen kode endret; kun filer åpnet og lest. Kildehenvisninger er fil + linjeområde som oppgitt.*
