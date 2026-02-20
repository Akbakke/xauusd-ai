# MASTER_EXIT_V1 + score_v1 ML-decider – How it works (code-based)

Dokumentasjon av den faktiske flyten, dataflyt og felter. Ingen nye scripts eller refaktorering – kun gjennomgang av eksisterende kode.

---

## Adapter vs fallback (TRUTH doctrine)

- **Runtime/audit** kan bruke en **adapter** for å pakke context (ctx_cont/ctx_cat) fra eksisterende flate kolonner (ctx_cont_0.., ctx_cat_0..) eller kontraktnavn (atr_bps, session_id, osv.). Dette er ikke en fallback: hvis kolonnene mangler, utelates context; ingen beslutningslogikk er avhengig av adapteren.
- **Adapter-guards (GX1-riktig):** (1) **All-or-none:** pakk ctx_cont bare hvis alle forventede cont-keys finnes og er ikke-null/ikke-NaN; samme for ctx_cat – hvis én mangler/NaN utelates hele vektoren. (2) **Hard type-cast:** cont som float, cat som int; ved cast-feil utelates hele vektoren (adapteren raiser aldri). (3) **Kontrakt-rekkefølge:** rekkefølge er eksakt listen i ORDERED_CTX_*_NAMES_EXTENDED[:dim]; ingen sortering eller «ta de som finnes».
- **Trening med IOV2** er **hard**: context må finnes i data; `use_io_v2=True` eller `require_io_v2=True` uten context gir tidlig feil med klar melding.
- Ingen exit-beslutningslogikk er avhengig av adapteren; context er best-effort for audit og treningsdata.

---

## 1) Kall-kjede (høy nivå)

**replay_chunk.process_chunk() → runner → run_replay → on_bar → ML-decider → HOLD/EXIT → audit**

| Steg | Fil | Funksjon / område | Detalj |
|------|-----|-------------------|--------|
| 1 | `gx1/execution/replay_chunk.py` | `process_chunk()` ca. 364–418 | PHASE 0: `bootstrap_chunk_environment()`; PHASE 1: `runner = GX1DemoRunner(policy_path, replay_mode=True, output_dir=chunk_output_dir)`; `runner.run_id = run_id`. |
| 2 | `gx1/execution/oanda_demo_runner.py` | `GX1DemoRunner.__init__()` | Laster policy, deretter exit-config fra `policy["exit_config"]` → `load_yaml_config(exit_cfg_path)` → `exit_cfg`. |
| 3 | `oanda_demo_runner.py` | Samme `__init__`, exit-boot | `exit_type = exit_cfg.get("exit", {}).get("type")` (linje 1941). Ved `exit_type == "MASTER_EXIT_V1"` (ca. 2059): leser `exit_params = exit_cfg.get("exit", {}).get("params", {})`, bygger `_exit_ml_decider`, `_exit_ml_config`, setter `self.exit_ml_enabled`, `self.exit_ml_decision_mode`, `self.exit_ml_config_hash`, definerer `_build_exit_master_v1_policy()` som kaller `get_exit_policy_master_v1(..., exit_ml_decider_enabled=..., exit_ml_log_path=None, exit_ml_config=...)`, lagrer som `self.exit_master_v1_factory`. |
| 4 | `replay_chunk.py` | ca. 491 | `runner.run_replay(chunk_data_path_abs)` |
| 5 | `oanda_demo_runner.py` | `run_replay()` → `_run_replay_impl()` | `run_replay` (ca. 16680) delegater til `_run_replay_impl(csv_path)` (ca. 12941). Replay-løkka itererer bars og kaller exit-branch per åpen trade. |
| 6 | `oanda_demo_runner.py` | Replay exit-branch MASTER_EXIT_V1 | Ca. 17029–17117: `policy = self.exit_master_v1_states.get(trade.trade_id)`; ved behov `_init_master_v1_state(trade, context="replay_exit_loop")`. Bygger `ctx` fra `candle_row` (ml_key + signal-bridge). `result = policy.on_bar(price_bid=..., price_ask=..., ts=bar_ts, atr_bps=atr_bps, **ctx)`. Ved exit: `request_close(..., reason=decision.reason, ...)`. |
| 7 | `oanda_demo_runner.py` | `_init_master_v1_state()` | Ca. 4086–4124: `policy = self.exit_master_v1_factory()` ved første gang for trade_id; setter `policy.run_id`, `policy.exit_ml_log_path = Path(self.output_dir) / "logs" / "exits" / f"exits_{self.run_id}.jsonl"` når ML på; `policy.reset_on_entry(..., p_long_frozen=extra.get("entry_p_long"))`. |
| 8 | `gx1/policy/exit_master_v1.py` | `ExitMasterV1.on_bar()` | Ca. 299–592: Oppdaterer trade state; sjekker HARD_STOP, TRAIL, STAGNATION (linje 473–514). Deretter ML-blokk (516–591): bygger `_signal_history` kun når `exit_ml_decider_enabled`; beregner deltas med `len(hist) >= 2/3/5`-guard; bygger `ExitMLContext`; kaller `compute_exit_score_and_decision(ml_ctx, self.exit_ml_config)`; ved EXIT returnerer `ExitDecision(reason="ML_SCORE_EXIT", ...)`; skriver audit via `_write_exit_ml_audit()`. |
| 9 | `gx1/policy/exit_ml_score_v1.py` | `compute_exit_score_and_decision()` | Ca. 112–149: `score = w0 + w1*dd_atr + w2*entropy_slope + w3*delta_p_long + w4*uncertainty + w5*conviction_drop`; `if score > threshold` → `("EXIT", "ML_SCORE_EXIT")`, ellers `("HOLD", "")`. |
| 10 | `exit_master_v1.py` | `_write_exit_ml_audit()` | Ca. 594–651: Best-effort append til `self.exit_ml_log_path` én JSONL-rad med run_id, trade_id, ts, side, state, signals, entry_snapshot, deltas, computed. |

---

## 2) Policy, env og “who wins?” (precedence)

**Kilde:** `oanda_demo_runner.py` ca. 2069–2089.

- **Policy YAML (SSoT):** Exit-config lastes fra filen gitt av `policy["exit_config"]` (f.eks. `MASTER_EXIT_V1_A.yaml`). Innholdet leses med `load_yaml_config(exit_cfg_path)` → `exit_cfg`. Struktur: `exit_cfg["exit"]["params"]` → `exit_params`; `exit_params["exit_ml"]` → `exit_ml`; `exit_ml["decider_enabled"]`, `exit_ml["score_v1"]` (w0–w5, threshold).
- **Tolkning:** Ingen egen parser-klasse; dict-aksess direkte i runner boot (samme sted som over).
- **GX1_EXIT_ML_DECIDER:**  
  `_exit_ml_decider = (os.environ.get("GX1_EXIT_ML_DECIDER", "").strip() == "1") or bool(exit_ml.get("decider_enabled", False))`.  
  **Env vinner:** Hvis `GX1_EXIT_ML_DECIDER=1` → decider på, uavhengig av YAML. Hvis env ikke er `"1"`, brukes YAML `decider_enabled`.
- **Hvor settes runtime-feltene:**
  - `exit_ml_decider_enabled`: Satt på **policy-instansen** via factory-argument `exit_ml_decider_enabled=_exit_ml_decider` (og i `ExitMasterV1.__init__`).
  - `exit_ml_decision_mode`: Satt på **runner**: `self.exit_ml_decision_mode = "score_v1" if _exit_ml_decider else ""` (ca. 2084).
  - `exit_ml_config_hash`: Satt på **runner**: `self.exit_ml_config_hash = exit_ml_config_hash(_exit_ml_config or {})` når decider på (ca. 2086–2088).

**Precedence (kort):**

| Parameter | 1. (vinner) | 2. | 3. (default) |
|-----------|--------------|-----|--------------|
| ML decider on/off | `GX1_EXIT_ML_DECIDER=1` (env) | `exit_ml.decider_enabled` (YAML) | `False` |
| w0–w5, threshold | YAML `exit_ml.score_v1` (keys w0..w5, threshold) | – | `exit_ml_score_v1.DEFAULT_EXIT_ML_SCORE_V1` |

---

## 3) “Signals inn” – hva exit får per bar

**Kilde:** `oanda_demo_runner.py` ca. 17054–17075 (MASTER_EXIT_V1-branch i replay-løkka).

- **ctx bygges slik:**
  - `uncertainty_score`, `margin_top1_top2`, `entropy`: fra `candle_row` hvis kolonnen finnes og er ikke-NaN.
  - Signal-bridge: for `(p_long, prob_long)`, `(p_short, prob_short)`, `(p_flat, None)`, `(p_hat, None)` – bruk `candle_row[sig_key]` hvis present, ellers `candle_row[alt_key]` for p_long/p_short.
  - **p_hat:** Hvis `p_hat` ikke satt fra rad: `ctx["p_hat"] = max(ctx["p_long"], ctx["p_short"])` (linje 17064–17065).
- **Andre felt exit får:** `on_bar(price_bid, price_ask, ts, atr_bps=atr_bps, **ctx)`. `atr_bps` kommer fra `candle_row["atr_bps"]` eller `trade.atr_bps`. Trade state (pnl_bps, mfe_bps, mae_bps, bars_held, osv.) oppdateres **innenfor** `ExitMasterV1.on_bar()` fra pris og tidligere state (ikke sendt eksplisitt i ctx fra runner).

---

## 4) Entry snapshot / frozen signals

- **Hvor _last_entry_ml_snapshot settes:**  
  `oanda_demo_runner.py` ca. 10770–10781 (i entry-prediction-path). Når `uncertainty_score`, `margin_snap`, `entropy_snap` er tilgjengelige, settes:
  `self._last_entry_ml_snapshot = { "uncertainty_score", "margin_top1_top2", "entropy", "p_hat", "p_long", "p_short", "p_flat": 0.0 }`.

- **Hvordan entry_manager fyller trade.extra:**  
  `entry_manager.py` ca. 5151–5167: Ved trade-opprettelse leses `snapshot = getattr(self._runner, "_last_entry_ml_snapshot", None)`. Hvis `snapshot` er dict, settes `trade.extra["entry_uncertainty_score"]`, `entry_margin_top1_top2`, `entry_entropy`, `entry_p_hat`, `entry_p_long`, `entry_p_short`, `entry_p_flat` fra snapshot.

- **Hvordan reset_on_entry(..., p_long_frozen=...) kalles:**  
  `oanda_demo_runner.py` `_init_master_v1_state()` ca. 4103–4115: `policy.reset_on_entry(..., entry_score_frozen=extra.get("entry_p_hat"), uncertainty_frozen=..., margin_top1_top2_frozen=..., entropy_frozen=..., p_long_frozen=extra.get("entry_p_long"))`.

- **Hva som lagres/brukes i score_v1:**  
  I `exit_master_v1.py` (ML-blokk): `p_long_entry=self.p_long_entry` (satt i `reset_on_entry` fra `p_long_frozen`), `p_hat_entry`, `uncertainty_entry`, `entropy_entry`, `margin_entry` fra frozen-feltene. Deltas (dp_long_1/3/5, dentropy_3, duncertainty_3, entropy_slope) beregnes fra `_signal_history` (kun når `exit_ml_decider_enabled`), med guards `len(hist) >= 2`, `>= 3`, `>= 5` så tom/kort historikk gir `None` og ingen IndexError.

---

## 5) Exit ML score_v1 – intern logikk, inputs, outputs

**Fil:** `gx1/policy/exit_ml_score_v1.py`.

- **Funksjon som beregner score og beslutning:** `compute_exit_score_and_decision(ctx: ExitMLContext, config)` (ca. 112–149).
- **Komponenter (w0–w5):**
  - **w0:** Konstant (default 0.0).
  - **w1:** `drawdown_from_mfe_atr` (dd_atr) – hvor mye prisen har gått ned fra MFE, i ATR.
  - **w2:** `entropy_slope` – endring i entropy (f.eks. dentropy_3).
  - **w3:** `delta_p_long` – dp_long_3 eller dp_long_1 (endring i p_long).
  - **w4:** `uncertainty_score` (nåværende).
  - **w5:** `conviction_drop` = `p_long_now - p_long_entry`.
- **Kode:**  
  `score = w0 + w1*dd_atr + w2*entropy_slope + w3*delta_p_long + w4*uncertainty + w5*conviction_drop`  
  `if score > threshold` → return `(score, "EXIT", "ML_SCORE_EXIT")`, ellers `(score, "HOLD", "")`.

- **Beslutning → reason:**  
  I `exit_master_v1.py` (ca. 575–589): Hvis `decision == "EXIT"` og `reason == "ML_SCORE_EXIT"` returneres `ExitDecision(..., reason="ML_SCORE_EXIT")`; ellers returneres `(None, None)` (HOLD).

- **Signalhistorikk og guard:**  
  I `exit_master_v1.py` ca. 523–537: `_signal_history` append kun når `getattr(self, "exit_ml_decider_enabled", False)`. Deltas bruker `len(hist) >= 2`, `>= 3`, `>= 5` før indeksering (hist[-2], hist[-3], hist[-5]), så tom historikk gir ingen IndexError.

---

## Exit ML audit – keys og “big picture”-gap

**Eksisterende audit-keys (per jsonl-linje):**

| Nøkkel | Innhold |
|-------|--------|
| `run_id` | Run-id. |
| `trade_id` | Trade-id. |
| `ts` | Tidsstempel for baren. |
| `side` | long/short. |
| `state` | `pnl_bps`, `mfe_bps`, `mae_bps`, `bars_held`, `dd_from_mfe_bps`, `time_since_mfe_bars`. |
| `signals` | `p_long_now`, `p_hat_now`, `entropy`, `uncertainty_score`, `margin_top1_top2` (XGB bridge). |
| `entry_snapshot` | `p_long_entry`, `p_hat_entry`, `entropy_entry`, `uncertainty_entry` (margin_entry kan mangle). |
| `deltas` | `dp_long_1`, `dp_long_3`, `dp_long_5`, `dentropy_3`, `duncertainty_3`. |
| `computed` | `exit_score`, `threshold`, `decision`, `reason`; ved transformer også `mode`, `exit_prob`, `model_sha`. |

**Mangler for “big picture” (per i dag):**

- **ctx_cont / ctx_cat** – slow-context (samme som entry bruker: ctx_cont dim 2/4/6, ctx_cat dim 5/6) logges **ikke** i exits-audit i dag. Exit Transformer ser dermed ikke makro/regime (session, trend, vol, spread, ATR-bucket, osv.) i treningsdata.  
- **Valgfri lang-kontekst** – ingen ekstra summary-features (f.eks. D1 ATR percentile, H4 trend) i kontrakten ennå.

**Konsekvens:** Exit Transformer V0 (IOV1) er rent mikro: sekvens av signal bridge + trade-state + entry snapshot. For V1 (IOV2) må audit utvides med `context: { ctx_cont: [...], ctx_cat: [...] }` og runner må sende ctx_cont/ctx_cat inn i `on_bar`-ctx.

---

## 6) Audit / logging: exits_<run_id>.jsonl

- **Hvor exit_ml_log_path settes:**  
  `oanda_demo_runner.py` `_init_master_v1_state()` ca. 4098–4100:  
  `policy.exit_ml_log_path = Path(self.output_dir) / "logs" / "exits" / f"exits_{self.run_id}.jsonl"`  
  (kun når policy nettopp opprettes og `exit_ml_decider_enabled` og `output_dir` og `run_id` finnes).  
  `self.output_dir` er chunk-dir (f.eks. `run_root/replay/chunk_0`), så full path: `run_root/replay/chunk_0/logs/exits/exits_<run_id>.jsonl`.

- **Writer-funksjon og callsite:**  
  `ExitMasterV1._write_exit_ml_audit(self, ts, exit_score, threshold, decision, ml_ctx)` i `exit_master_v1.py` ca. 594–651. Kalles: (1) ved EXIT: ca. 576, (2) ved HOLD når ML på og `exit_ml_log_path`: ca. 590.

- **JSON-struktur (keys alltid med i rec):**  
  `run_id`, `trade_id`, `ts`, `side`, `state` (pnl_bps, mfe_bps, mae_bps, bars_held, dd_from_mfe_bps, time_since_mfe_bars), `signals` (p_long_now, p_hat_now, entropy, uncertainty_score, margin_top1_top2), `entry_snapshot` (p_long_entry, p_hat_entry, entropy_entry, uncertainty_entry), `deltas` (dp_long_1, dp_long_3, dp_long_5, dentropy_3, duncertainty_3), `computed` (exit_score, threshold, decision, reason).

- **EXIT vs HOLD:**  
  Samme struktur; `computed.decision` er `"EXIT"` eller `"HOLD"`; `computed.reason` er `"ML_SCORE_EXIT"` ved EXIT, ellers `""`.

---

## 7) Footer + TRUTH gates

- **Hvor footer-feltene skrives:**  
  `gx1/execution/replay_chunk.py` ca. 803–962: `payload` dict bygges og sendes til `write_chunk_footer(footer_ctx)`. Exit ML-feltene (ca. 941–943):
  - `"exit_ml_enabled": getattr(runner, "exit_ml_enabled", False)`
  - `"exit_ml_decision_mode": getattr(runner, "exit_ml_decision_mode", "") or None`
  - `"exit_ml_config_hash": getattr(runner, "exit_ml_config_hash", "") or None`

- **TRUTH postrun (når ML enabled):**  
  `gx1/scripts/run_truth_e2e_sanity.py` `_run_postrun_checks()` ca. 815–829:
  - Hvis `footer.get("exit_ml_enabled") is True`:  
    - Krav: `footer.get("exit_ml_decision_mode") == "score_v1"`; ellers `gates_failed.append("exit_ml_decision_mode")`, feilmelding inkl. `"exit_ml_decision_mode must be 'score_v1' when exit_ml_enabled=True..."`.  
    - Krav: minst én fil som matcher `chunk_dir / "logs" / "exits" / "exits_*.jsonl"`; ellers `gates_failed.append("exit_ml_exits_jsonl")`, feilmelding inkl. `"exit_ml_enabled=True requires exits jsonl in ..."`.  
  - `chunk_dir = run_root / "replay" / "chunk_0"` (satt tidligere i samme funksjon).

---

## 8) A/B harness: hva måles, og hvor?

- **A = ML av, B = ML på:**  
  `gx1/scripts/run_ab_fullyear_2025_exit_ml.py` `_build_truth_env()` ca. 86–89:  
  `is_baseline` True → `env["GX1_EXIT_ML_DECIDER"] = "0"`; False → `env["GX1_EXIT_ML_DECIDER"] = "1"`.

- **Hvor AB_SUMMARY genereres:**  
  Samme fil ca. 429–458: `report_path = report_dir / "AB_SUMMARY.md"`, skriving med `open(report_path, "w", ...)`.

- **Metrics som brukes:**  
  - **Exit reason counts:** `exit_stats_a` / `exit_stats_b` fra `_exit_reason_counts_and_means(run_dir, run_id)` (ca. 424–425). Teller `primary_exit_reason` eller `exit_reason` fra parquet; i rapporten skrives dict og `ML_SCORE_EXIT` count for A og B.  
  - **Timing/PnL means:** Fra samme helper: `n_trades`, `mean_pnl_bps`, `mean_duration_bars` (fra `trade_outcomes_<run_id>.parquet`: kolonner `pnl_bps`, `duration_bars`).

- **Hjelpefunksjon:**  
  `_exit_reason_counts_and_means(run_dir, run_id)` (ca. 224–246): Leser `Path(run_dir) / "replay" / "chunk_0" / f"trade_outcomes_{run_id}.parquet"`, returnerer dict med `exit_reason_counts`, `n_trades`, `mean_pnl_bps`, `mean_duration_bars`.

---

## 9) Tester som beviser mekanismen

**Fil:** `gx1/tests/test_exit_master_v1.py`.

- **ML av → ingen ML_SCORE_EXIT:**  
  `test_ml_exit_decider_off_no_ml_score_exit()` (ca. 294–311): Oppretter policy med `exit_ml_decider_enabled=False`, trigger hard stop; assert `decision.reason == "MASTER_HARD_STOP"` og `decision.reason != "ML_SCORE_EXIT"`. Garantier: Når ML-decider er av, kommer ikke `ML_SCORE_EXIT` fra denne policy-instansen; reglene kan fortsatt gi andre reasons.

- **Audit jsonl finnes og har required fields:**  
  `test_ml_exit_audit_jsonl_has_required_fields(tmp_path)` (ca. 314–366): Policy med `exit_ml_decider_enabled=True`, `exit_ml_log_path=tmp_path/exits_test.jsonl`, én `on_bar()` med ctx som ikke trigger regler. Assert: fil eksisterer, minst én linje, første rad er valid JSON med top-level keys `run_id`, `trade_id`, `ts`, `side`, `state`, `signals`, `entry_snapshot`, `deltas`, `computed`, og `computed` inneholder `exit_score`, `threshold`, `decision`, `reason`. Garantier: Når ML-decider er på og log path satt, skrives minst én audit-rad med den kontrakten; tester ikke innholdsverdi eller at EXIT-rad skrives.

---

## 10) Trace map (bullet) + Verify locally

**Trace map (kallkjede + fil)**

- `replay_chunk.process_chunk()` → bootstrap, opprett `GX1DemoRunner`, `runner.run_replay(csv_path)`  
  **replay_chunk.py**
- `GX1DemoRunner.__init__()` → last policy, last exit YAML → `exit_cfg` → ved MASTER_EXIT_V1: les exit_ml, sett exit_ml_enabled/decision_mode/config_hash, bygg factory  
  **oanda_demo_runner.py**
- `run_replay()` → `_run_replay_impl()` → replay-løkke over bars  
  **oanda_demo_runner.py**
- Per åpen trade: `_init_master_v1_state(trade)` ved behov → factory() → `policy.reset_on_entry(..., p_long_frozen=extra.get("entry_p_long"))`; sette policy.run_id, policy.exit_ml_log_path  
  **oanda_demo_runner.py**
- Bygg ctx fra candle_row (uncertainty_score, margin_top1_top2, entropy, p_long/p_short/p_flat/p_hat) → `policy.on_bar(price_bid, price_ask, ts, atr_bps=atr_bps, **ctx)`  
  **oanda_demo_runner.py**
- `ExitMasterV1.on_bar()` → regler (HARD_STOP, TRAIL, STAGNATION); deretter ML-blokk: _signal_history, deltas, ExitMLContext, compute_exit_score_and_decision(), _write_exit_ml_audit(), evt. return ExitDecision( reason="ML_SCORE_EXIT" )  
  **exit_master_v1.py**
- `compute_exit_score_and_decision(ctx, config)` → score, threshold → EXIT/HOLD, reason  
  **exit_ml_score_v1.py**
- `_write_exit_ml_audit()` → append én linje til exit_ml_log_path  
  **exit_master_v1.py**
- Footer: replay_chunk leser runner.exit_ml_enabled / exit_ml_decision_mode / exit_ml_config_hash → payload → write_chunk_footer()  
  **replay_chunk.py**
- TRUTH postrun: les chunk_footer.json; hvis exit_ml_enabled, sjekk decision_mode og exits_*.jsonl  
  **run_truth_e2e_sanity.py**

**Verify locally (sjekkliste 3–5 filer i en run-dir)**

1. **chunk_footer.json** (under `run_root/replay/chunk_0/chunk_footer.json`):  
   - `exit_type` == `"MASTER_EXIT_V1"`  
   - `exit_ml_enabled` == `true` når ML var på  
   - `exit_ml_decision_mode` == `"score_v1"` når ML var på  
   - `exit_ml_config_hash` streng (16 tegn hex) når ML var på  

2. **exits jsonl** (under `run_root/replay/chunk_0/logs/exits/`):  
   - Fil `exits_<run_id>.jsonl` finnes når ML var på  
   - Minst én linje per bar der ML ble evaluert (HOLD eller EXIT); hver linje har `run_id`, `trade_id`, `ts`, `side`, `state`, `signals`, `entry_snapshot`, `deltas`, `computed` med `exit_score`, `threshold`, `decision`, `reason`  

3. **POSTRUN_E2E.json** (under `run_root/`):  
   - `passed: true` og ingen `exit_ml_decision_mode` / `exit_ml_exits_jsonl` i `gates_failed` når ML er på  

4. **trade_outcomes_<run_id>.parquet** (under `run_root/replay/chunk_0/`):  
   - Kolonne `primary_exit_reason` eller `exit_reason`; ved ML på kan noen rader ha `"ML_SCORE_EXIT"`  

5. **Policy SSoT:**  
   - Exit-config (f.eks. `MASTER_EXIT_V1_A.yaml`): `exit.params.exit_ml.decider_enabled`, evt. `exit.params.exit_ml.score_v1`; ved A/B: env `GX1_EXIT_ML_DECIDER=0` eller `1` overstyrer decider on/off  

Dette er nok til å verifisere at ML-exit-wiring var aktiv og at config og audit er konsistent med kjørselen.

---

## Runbook: Tren Exit Transformer V0 fra exits jsonl

**Forutsetning:** En kjørsel med MASTER_EXIT_V1 og score_v1 har produsert SSoT-audit:  
`<run_root>/replay/chunk_0/logs/exits/exits_<run_id>.jsonl`.

1. **Plasser én eller flere jsonl-filer**  
   Enten bruk én fil, eller slå sammen flere (f.eks. fra flere run_id) til én fil. Modulen leser alle linjer og grupperer per `trade_id`.

2. **Kall den gjenbrukbare treningsfunksjonen** (ingen ad hoc-script):
   ```python
   from pathlib import Path
   from gx1.policy.exit_transformer_v0 import train_from_exits_jsonl

   exits_path = Path("/path/to/exits_<run_id>.jsonl")  # eller bruk get_last_go_exits_dataset() for LAST_GO
   result = train_from_exits_jsonl(
       exits_path,
       out_dir=None,  # None → GX1_DATA/models/exit_transformer_v0/<dataset_sha256>/
       window_len=64,
       d_model=128,
       n_layers=2,
       epochs=20,
       val_fold=9,
       seed=42,
   )
   # result: model_path, config_path, model_sha256, dataset_sha256, train_report_path
   print("model_sha", result["model_sha256"])
   ```

3. **Konfigurer policy og runner**  
   I exit YAML (f.eks. `MASTER_EXIT_V1_A.yaml`): `exit_ml.mode: exit_transformer_v0`, `exit_ml.exit_transformer.enabled: true`, `model_path` (relativ til GX1_DATA eller absolutt), `model_sha`, `threshold`, `window_len`.  
   Evt. env: `GX1_EXIT_ML_DECIDER=1`, `GX1_EXIT_ML_MODE=exit_transformer_v0`.

4. **Verifiser**  
   Kjør TRUTH E2E (1W1C) med `exit_transformer_v0`. Sjekk at `chunk_footer.json` har `exit_ml_model_sha`, og at minst én linje i `exits_<run_id>.jsonl` har `computed.mode == "exit_transformer_v0"`.

---

## Runbook: Full-year audit + Exit Transformer V1 (stort datasett)

**Mål:** Tren på et stort, representativt sett (full-year) med optional IOV2 (context) for “big picture”.

1. **Produser full-year score_v1-audit (med optional context for IOV2)**  
   Kjør TRUTH full-year med ML exit på og mode score_v1. For IOV2 må replay-row ha `ctx_cont`/`ctx_cat` (f.eks. fra prebuilt-kolonner eller bygget før exit-branch), slik at audit-linjene får `context: { ctx_cont: [...], ctx_cat: [...] }`.

   ```bash
   export GX1_DATA=/path/to/GX1_DATA
   export GX1_CANONICAL_TRUTH_FILE=/path/to/gx1/configs/canonical_truth_signal_only.json
   export GX1_EXIT_ML_DECIDER=1
   export GX1_EXIT_ML_MODE=score_v1
   /path/to/venvs/gx1/bin/python -m gx1.scripts.run_truth_e2e_sanity --full-year
   ```

2. **LAST_GO**  
   `LAST_GO.txt` skrives **kun etter at alle gates er bestått** (preflight, replay, postrun, zero-trades-contract hvis aktuelt). Scriptet `run_truth_e2e_sanity` oppdaterer `GX1_DATA/reports/truth_e2e_sanity/LAST_GO.txt` rett før exit 0 – dermed er LAST_GO alltid «siste ekte GO». Ved behov kan du peke manuelt: `echo /path/to/.../truth_e2e_sanity/<run_id> > $GX1_DATA/reports/truth_e2e_sanity/LAST_GO.txt`.

3. **Train + verify (én kommando)**  
   ```bash
   /path/to/venvs/gx1/bin/python -m gx1.scripts.run_truth_e2e_sanity --train-exit-transformer-v0-from-last-go
   ```  
   Skriver artifacts til `GX1_DATA/models/exit_transformer_v0/<dataset_sha256>/`. Ved suksess: `VERIFY_REPORT.json` med `passed: true`.

4. **IOV2 (context)**  
   Hvis datasettet har `context.ctx_cont`/`ctx_cat` i jsonl, kan trening bruke `use_io_v2=True` (evt. `require_io_v2=True` for hard gate). Da blir `io_version: "IOV2"` i TRAIN_REPORT og feature_count inkluderer ctx-dims.
