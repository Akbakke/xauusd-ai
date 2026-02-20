# Policy trace audit – fakta (kun observasjon)

**Dato:** 2026-02-20

---

## DEL A — POLICY LOAD SITES

**POLICY_LOAD_SITES:**

- **file:** `gx1/execution/oanda_demo_runner.py`  
  **function:** `load_yaml_config` (linje 539); kalles fra `__init__` (linje 1624).  
  **returns:** `dict` (yaml.safe_load(handle)).

- **file:** `gx1/execution/oanda_demo_runner.py`  
  **function:** `__init__` – `self.policy = load_yaml_config(policy_path)` (linje 1624).  
  **returns:** Policy er `self.policy` (dict); ikke deepcopy, direkte retur fra safe_load.

- **file:** `gx1/prod/run_header.py`  
  **function:** `generate_run_header` – ved manglende policy_dict leses policy fra disk: `policy_dict = yaml.safe_load(f)` (linje 125).  
  **returns:** dict.

- **file:** `gx1/policy/trial160_loader.py`  
  **function:** `load_policy` (linje 65).  
  **returns:** PolicyConfig (dataclass/object).

- **file:** `gx1/analysis/diff_runs.py`  
  **function:** `load_policy_bundle` (linje 105) – `load_yaml_config(policy_yaml_path)`.  
  **returns:** dict.

- **file:** `gx1/utils/policy_identity_gate.py`  
  **function:** `_read_policy_id` – `yaml.safe_load(f)` (linje 109).  
  **returns:** policy_data dict (brukes kun for policy_id).

- **file:** `gx1/analysis/parity_audit.py`  
  **function:** `load_yaml_config` (linje 90); brukes lokalt.  
  **returns:** dict.

(Andre treff: entry_manager, prod_baseline_proof, _quarantine, scripts – ikke i TRUTH-replay-sti til runner.)

---

## DEL B — POLICY MUTATION SITES

**POLICY_MUTATION_SITES:**

- **file:** `gx1/execution/oanda_demo_runner.py`  
  **line_range:** 1952–1959  
  **mutation_type:** merge – entry_cfg slås inn i `self.policy`; kun nøkler fra entry_cfg; `entry_models` overskrives bare hvis ikke allerede i policy. **exit_config endres ikke her.**

- **file:** `gx1/execution/oanda_demo_runner.py`  
  **line_range:** 1967–1971, 2119–2120  
  **mutation_type:** overwrite – `self.policy["tick_exit"]`, `self.policy["broker_side_tp_sl"]` settes. **exit_config endres ikke.**

- **file:** `gx1/execution/oanda_demo_runner.py`  
  **line_range:** 1624  
  **mutation_type:** overwrite – `self.policy = load_yaml_config(policy_path)` (første og eneste direkte tildeling av hele policy-dict).

Ingen sted i koden som er søkt, setter eller overskriver `exit_config` programmatisk. Ingen `copy`/`deepcopy` av policy før den sendes til runner; runner får dict rett fra `load_yaml_config`.

---

## DEL C — RUNNER INSTANTIATION

**RUNNER_INSTANTIATION_SITES:**

- **file:** `gx1/execution/replay_chunk.py`  
  **function:** `process_chunk` (linje 196)  
  **policy_source:** Parameter `policy_path: Path`; kalleren sender inn policy_path.  
  Linje 412–413: `runner = GX1DemoRunner(policy_path, replay_mode=True, ...)`.  
  Runner bruker kun `policy_path`; policy bygges inne i runner: `self.policy = load_yaml_config(policy_path)` (oanda_demo_runner.py:1624).  
  **Konklusjon:** Policy-objektet kommer direkte fra `load_yaml_config(policy_path)`; ingen mellomledd som endrer eller velger annen policy.

Kaller til `process_chunk` med policy_path:

- **file:** `gx1/scripts/run_truth_e2e_sanity.py`  
  **function:** `main` → `_run_replay(..., policy_path, ...)` (linje 1298–1301).  
  **policy_path:** Satt linje 1238–1243: `Path(os.environ.get("GX1_CANONICAL_POLICY_PATH", str(ENGINE / "gx1" / "configs" / "policies" / "sniper_snapshot" / "2025_SNIPER_V1" / "GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"))).expanduser().resolve()`.  
  **Konklusjon:** Policy path er enten env `GX1_CANONICAL_POLICY_PATH` eller default canonical YAML-path; deretter sendes samme path til _run_replay → process_chunk → GX1DemoRunner(policy_path).

---

## DEL D — CACHE / SINGLETON

**POLICY_CACHE_OR_SINGLETON:**

- **found:** NO  
- **details:** Grep etter POLICY_CACHE, cached_policy, GLOBAL_POLICY, singleton i gx1 ga ingen treff for policy-cache. Kun replay_eval_collectors.py omtaler "not singleton/global" for _collected_trade_ids. Policy lastes i runner __init__ fra disk ved hvert kall; ingen modul-scope policy-lagring funnet.

---

## DEL E — RUN-ARTEFAKTER

**RUN_ARTEFACT_POLICY_EVIDENCE:**

- **run_dir:** `/home/andre2/GX1_DATA/reports/truth_e2e_sanity/E2E_SANITY_20260219_203609` (LAST_GO).

- **run_header.json** (replay/chunk_0/run_header.json):  
  - **policy_path_in_run:** `/home/andre2/src/GX1_ENGINE/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml`  
  - **exit_config_in_run:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/MASTER_EXIT_V1_A.yaml`

- **REPLAY_SSoT_HEADER.json:**  
  - **policy_config_path:** `/home/andre2/src/GX1_ENGINE/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml`  
  - Ingen exit_config-felt i REPLAY_SSoT_HEADER.

- **matches_expected_policy:**  
  - **Policy path:** YES – samme canonical YAML som forventet.  
  - **exit_config:** NO – run viser `exit_config: MASTER_EXIT_V1_A.yaml`; forventet fra nåværende fil på disk er `EXIT_TRANSFORMER_ONLY_V0.yaml`.

---

## DEL F — SLUTTKONKLUSJON (KUN FAKTA)

**FINAL POLICY TRACE CONCLUSION:**

- **actual_policy_loaded_from:** Filen som ble lest fra disk er den som `policy_path` pekte på: `/home/andre2/src/GX1_ENGINE/gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml` (bevis: run_header.json policy_path og REPLAY_SSoT_HEADER policy_config_path; runner bruker kun denne path i load_yaml_config).

- **policy_object_passed_to_runner_from:** Policy-objektet er ikke «passed inn» som dict; runner bygger det selv i __init__ med `self.policy = load_yaml_config(policy_path)` (oanda_demo_runner.py:1624). Kilde er altså `load_yaml_config(policy_path)` der policy_path kommer fra run_truth_e2e_sanity (env eller default canonical path).

- **exit_config_value_in_policy_object:** I den aktuelle runen var verdien i policy-objektet `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/MASTER_EXIT_V1_A.yaml` (bevis: EXIT_CONFIG_RESOLVE_PROOF.json exit_config_raw og run_header.json exit_config).

- **mismatch_root_cause:** Ukjent. Koden endrer ikke `exit_config` etter load; eneste kilde er innholdet i YAML-filen ved `load_yaml_config(policy_path)`. Nåværende fil på disk (GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml) har `exit_config: .../EXIT_TRANSFORMER_ONLY_V0.yaml` (linje 63). Run-artefakter viser samme policy_path men exit_config MASTER_EXIT_V1_A. Mulige forklaringer: (1) YAML-filen på disk den 19.02.2026 inneholdt exit_config: MASTER_EXIT_V1_A (fil endret senere), eller (2) en annen kode-/lastesti som ikke er funnet. Ingen fil/linje i kodebasen som eksplisitt setter exit_config til MASTER_EXIT_V1_A.
