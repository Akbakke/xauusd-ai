# TRUTH/ONE-UNIVERSE – Exit IOV3 eneste versjon (audit med disk-bevis)

**Dato:** 2026-02-20  
**Premiss:** Kun én IO-versjon for exit-transformer: IOV3. IOV1/IOV2 = legacy, må identifiseres eksplisitt.  
**Metode:** Kun lesing av filer og artefakter på disk; ingen kodeendringer.

---

## DEL 1 – FASTSLÅ HVA SOM FAKTISK KJØRES (DISK > KODE)

### 1.1 Faktisk exit-modell som lastes

**Kilde policy:**  
`gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/EXIT_TRANSFORMER_ONLY_V0.yaml` linje 17:

```yaml
model_path: "models/exit_v0_ctx/EXIT_TRANSFORMER_V0_IOV2"
```

**Resolvert absolutt path (GX1_DATA fra canonical truth):**  
- `canonical_truth_signal_only.json` peker på paths under `/home/andre2/GX1_DATA`.  
- Resolvert path: **`/home/andre2/GX1_DATA/models/exit_v0_ctx/EXIT_TRANSFORMER_V0_IOV2`**

**Sjekk på disk:**  
- Lesing av `/home/andre2/GX1_DATA/models/exit_v0_ctx/EXIT_TRANSFORMER_V0_IOV2/exit_transformer_config.json` → **Fil finnes ikke (File not found).**  
- Glob etter `**/exit_v0_ctx/**` under `/home/andre2/GX1_DATA` → **0 filer.**  
- Mappen som policy peker på, **finnes ikke** på disk.

**Eneste exit_transformer_config.json funnet på disk:**  
- Path: **`/home/andre2/GX1_DATA/models/exit_transformer_v0/f3f5d2941612274c931559b84952ca05cd46ab21c08826b6d52e7227e1910aa3/exit_transformer_config.json`**  
- Denne mappen er **ikke** den policy refererer til (policy: `models/exit_v0_ctx/EXIT_TRANSFORMER_V0_IOV2`; disk: `models/exit_transformer_v0/<sha>`).

**Konklusjon 1.1:**  
Den modellmappen som policy angir, finnes ikke. Ved replay vil lasting feile (FileNotFoundError) med mindre GX1_DATA peker andre steder eller mappen opprettes. Den eneste exit-config som finnes på disk ligger under `exit_transformer_v0/<dataset_sha>` og er fra tidlig trening (IOV1).

---

### 1.2 Rapport fra exit_transformer_config.json

**A) Fra den mappen policy peker på:**  
- **Ikke mulig** – mappen `/home/andre2/GX1_DATA/models/exit_v0_ctx/EXIT_TRANSFORMER_V0_IOV2` finnes ikke.  
- **UKJENT** for input_dim/window_len/feature_names_hash/exit_ml_io_version for den konfigurasjonen som er tenkt brukt.

**B) Fra eneste config funnet på disk**  
Fil: `/home/andre2/GX1_DATA/models/exit_transformer_v0/f3f5d2941612274c931559b84952ca05cd46ab21c08826b6d52e7227e1910aa3/exit_transformer_config.json`

**Innhold (ordrett):**

```json
{"window_len":8,"d_model":64,"n_layers":2,"input_dim":19,"feature_names_hash":"2302f7451a9e9c87"}
```

- **input_dim:** 19  
- **window_len:** 8  
- **feature_names_hash:** 2302f7451a9e9c87  
- **exit_ml_io_version:** Feltet finnes ikke i filen.

**TRAIN_REPORT.json (samme mappe):**  
- `feature_count`: 19  
- `io_version` finnes ikke i rapporten (implisitt IOV1).  
- Bevis: `TRAIN_REPORT.json` linje 11–12 (feature_count, feature_names_hash).

**Konklusjon (obligatorisk):**  
- For **policy-path** (EXIT_TRANSFORMER_V0_IOV2): **UKJENT** – config finnes ikke på disk.  
- For **eneste config på disk** (exit_transformer_v0/&lt;sha&gt;): **AKTIV IO-VERSJON ≠ IOV3**. **RED FLAG: input_dim = 19 (IOV1).**  
- For at «AKTIV IO-VERSJON = IOV3» skal gjelde: det må finnes en mappe (f.eks. under exit_v0_ctx eller tilsvarende) med `exit_transformer_config.json` der **input_dim == 37**. Slik mappe er ikke funnet.

---

## DEL 2 – DEN ENESTE SANNE LISTEN: ALLE 37 FEATURES (NAVN + REKKEFØLGE)

Bygget fra kontraktene i rekkefølge IOV3 (IOV1 → ctx 6/6 → V3-ekstra).  
Kilder: `exit_transformer_io_v1.py`, `exit_transformer_io_v2.py`, `exit_transformer_io_v3.py`, `signal_bridge_v1.py`.

**Semantiske ctx-navn:** ONE UNIVERSE bruker `ORDERED_CTX_CONT_NAMES_EXTENDED` og `ORDERED_CTX_CAT_NAMES_EXTENDED` fra `signal_bridge_v1.py` (ikke ctx_cont_0/ctx_cat_0 i rapporten).

| # | Feature-navn | Type | Opprinnelse | Formel / beskrivelse |
|---|----------------|------|--------------|-----------------------|
| 1 | p_long | signal | XGB | signal_bridge_v1.ORDERED_FIELDS – sannsynlighet long |
| 2 | p_short | signal | XGB | Sannsynlighet short |
| 3 | p_flat | signal | XGB | Sannsynlighet flat |
| 4 | p_hat | signal | XGB | Max av retningsprobs |
| 5 | uncertainty_score | signal | XGB | Normalisert entropi |
| 6 | margin_top1_top2 | signal | XGB | top1 − top2 over 3-klasse probs |
| 7 | entropy | signal | XGB | Rå entropi (nats) |
| 8 | p_long_entry | entry_snapshot | Entry-snapshot | p_long ved entry (frosset) |
| 9 | p_hat_entry | entry_snapshot | Entry-snapshot | p_hat ved entry |
| 10 | uncertainty_entry | entry_snapshot | Entry-snapshot | uncertainty_score ved entry |
| 11 | entropy_entry | entry_snapshot | Entry-snapshot | entropy ved entry |
| 12 | margin_entry | entry_snapshot | Entry-snapshot | margin ved entry |
| 13 | pnl_bps_now | trade_state | Trade-state | Nåværende PnL i bps |
| 14 | mfe_bps | trade_state | Trade-state | Max favorable excursion (bps) |
| 15 | mae_bps | trade_state | Trade-state | Max adverse excursion (bps) |
| 16 | dd_from_mfe_bps | trade_state | Trade-state | Drawdown fra MFE (bps) |
| 17 | bars_held | trade_state | Trade-state | Antall bars i trade |
| 18 | time_since_mfe_bars | trade_state | Trade-state | Bars siden MFE |
| 19 | atr_bps_now | trade_state | Trade-state | ATR i bps (nåværende bar) |
| 20 | atr_bps | ctx_cont | Context | signal_bridge_v1: atr_bps (baseline) |
| 21 | spread_bps | ctx_cont | Context | signal_bridge_v1: spread_bps (baseline) |
| 22 | D1_dist_from_ema200_atr | ctx_cont | Context | signal_bridge_v1: slow core |
| 23 | H1_range_compression_ratio | ctx_cont | Context | signal_bridge_v1: slow core |
| 24 | D1_atr_percentile_252 | ctx_cont | Context | signal_bridge_v1: slow core |
| 25 | M15_range_compression_ratio | ctx_cont | Context | signal_bridge_v1: slow core |
| 26 | session_id | ctx_cat | Context | signal_bridge_v1: 0=ASIA, 1=EU, 2=US, 3=OVERLAP |
| 27 | trend_regime_id | ctx_cat | Context | signal_bridge_v1: 0/1/2 |
| 28 | vol_regime_id | ctx_cat | Context | signal_bridge_v1: 0–3 |
| 29 | atr_bucket | ctx_cat | Context | signal_bridge_v1: 0–3 |
| 30 | spread_bucket | ctx_cat | Context | signal_bridge_v1: 0–2 |
| 31 | H4_trend_sign_cat | ctx_cat | Context | signal_bridge_v1: 0/1/2 |
| 32 | drawdown_from_peak_bps | v3_extra | Derived | exit_transformer_io_v3: pnl_bps_now − mfe_bps |
| 33 | pnl_over_atr | v3_extra | Derived | pnl_bps_now / atr_bps_now |
| 34 | spread_over_atr | v3_extra | Derived | spread_bps_now / atr_bps_now |
| 35 | bars_since_mfe | v3_extra | Derived | int(time_since_mfe_bars) – alias |
| 36 | price_dist_from_entry_bps | v3_extra | Derived | (price_now − entry_price) / entry_price × 10000 |
| 37 | price_dist_from_entry_over_atr | v3_extra | Derived | price_dist_from_entry_bps / atr_bps_now |

**Kildekontrakter:**  
- 1–19: `exit_transformer_io_v1.py` ORDERED_SIGNAL_FIELDS (7), ORDERED_ENTRY_SNAPSHOT_FIELDS (5), ORDERED_TRADE_STATE_FIELDS (7).  
- 20–25: `signal_bridge_v1.py` ORDERED_CTX_CONT_NAMES_EXTENDED (6).  
- 26–31: `signal_bridge_v1.py` ORDERED_CTX_CAT_NAMES_EXTENDED (6).  
- 32–37: `exit_transformer_io_v3.py` ORDERED_EXIT_EXTRA_FIELDS_V3 (6).

---

## DEL 3 – OVERLAPP / DUBLETTER / SEMANTISK KOLLISJON

| Feature A | Feature B | Relasjon | Vurdering |
|-----------|-----------|----------|-----------|
| pnl_bps_now | price_dist_from_entry_bps | Forskjellig: PnL vs. prisavstand i bps | OK |
| time_since_mfe_bars | bars_since_mfe | Samme verdi (bars_since_mfe = int(time_since_mfe_bars)) | REDUNDANT (alias) |
| dd_from_mfe_bps | drawdown_from_peak_bps | Begge “drawdown fra topp”: dd_from_mfe_bps i state, drawdown_from_peak_bps = pnl_bps_now − mfe_bps | REDUNDANT (samme semantikk, to steder) |
| atr_bps_now (trade_state) | atr_bps (ctx_cont) | Begge ATR-relatert; trade_state er nå-bar, ctx er makro/entry-context | OK (forskjellig kontekst) |
| spread_bps_now (brukt i V3) | spread_bps (ctx_cont) | Nå-bar spread vs. makro spread | OK (forskjellig kontekst) |
| pnl_bps_now | pnl_over_atr | Rå vs. ATR-normalisert | OK (derived) |
| price_dist_from_entry_bps | price_dist_from_entry_over_atr | Rå bps vs. ATR-normalisert | OK (derived) |

**Svar på spørsmål:**  
- Samme informasjon flere ganger: **Ja** – time_since_mfe_bars og bars_since_mfe; dd_from_mfe_bps og drawdown_from_peak_bps (to representasjoner av “drawdown fra topp”).  
- Samme signal i entry_snapshot og trade_state: **Nei** – entry_snapshot er entry-tidspunkt, trade_state er løpende; ingen direkte duplikat av samme felt.  
- Samme verdi både rå og ATR-normalisert: **Ja** – pnl_over_atr, spread_over_atr, price_dist_from_entry_over_atr er ATR-normaliserte versjoner av rå felter; det er bevisst (exit sharpeners).

**Konklusjon DEL 3:**  
IOV3 er ikke helt semantisk ren: **REDUNDANT** – (1) time_since_mfe_bars (18) og bars_since_mfe (35); (2) dd_from_mfe_bps (16) og drawdown_from_peak_bps (32). Resten er enten OK eller bevisst raw+derived.

---

## DEL 4 – ENTRY VS EXIT: ER INPUTS KONSISTENTE?

### Får entry-transformer XGB sine 7 signaler?

**Ja.**  
- Navn og kilde: `signal_bridge_v1.py:45-51` ORDERED_FIELDS:  
  **p_long, p_short, p_flat, p_hat, uncertainty_score, margin_top1_top2, entropy.**  
- Runtime: `oanda_demo_runner.py` ~9410–9417 – seq_data/snap_data fylles fra field_to_seq/field_to_snap i ORDERED_FIELDS-rekkefølge; validate_seq_signal/validate_snap_signal brukes.

### Får exit-transformer de samme 7 signalene?

**Ja.**  
- Kontrakt: `exit_transformer_io_v1.py:19-27` ORDERED_SIGNAL_FIELDS identisk med signal_bridge_v1 (samme 7 navn).  
- Mapping exits jsonl → row → vektor: Runner bygger `row["signals"]` med p_long_now, p_short_now, p_hat_now, uncertainty_score, margin_top1_top2, entropy (`oanda_demo_runner.py:16439-16445`). IOV2/IOV3 bruker _row_to_iov1_slice / row_to_feature_vector_v2/v3 som leser row["signals"] i samme rekkefølge som ORDERED_SIGNAL_FIELDS (`exit_transformer_io_v2.py:105-111`).  
- Bekreftelse: De første 7 komponentene i exit-vektoren er de samme 7 signalene som entry.

### Bruker entry og exit samme ctx_cont / ctx_cat-semantikk?

**Ja.**  
- Entry: `entry_context_features.py` + `signal_bridge_v1` ORDERED_CTX_CONT_NAMES_EXTENDED / ORDERED_CTX_CAT_NAMES_EXTENDED (6+6).  
- Exit: row["context"]["ctx_cont"] og row["context"]["ctx_cat"] fylles fra candle_row ctx_cont/ctx_cat (6/6) i runner (`oanda_demo_runner.py:16463-16472`); disse kommer fra samme ONE UNIVERSE-context som entry.  
- Navn: ctx lagres som lister (indeks 0–5); semantisk rekkefølge er signal_bridge_v1 ORDERED_CTX_CONT_NAMES_EXTENDED og ORDERED_CTX_CAT_NAMES_EXTENDED. Exit bruker ikke andre ctx-navn enn det entry bruker.

### Finnes det ctx-features som bare entry ser, eller bare exit?

**Nei.**  
- Begge har 6 ctx_cont + 6 ctx_cat i samme rekkefølge (ONE UNIVERSE). Exit får i tillegg trade_state, entry_snapshot og V3-ekstra; ingen ekstra rene ctx-features som kun entry eller kun exit har.

---

## DEL 5 – LEGACY DØDSFALL: FINNES GAMLE VEIER?

**Søk (fil + status):**

| Referanse | Fil | Status | Bevis |
|-----------|-----|--------|-------|
| row_to_feature_vector_v2 | exit_transformer_io_v2.py | Definert | Definisjon 127. |
| row_to_feature_vector_v2 | exit_transformer_io_v3.py | Brukes av IOV3 | IOV3 kaller row_to_feature_vector_v2 for prefix (249). |
| row_to_feature_vector_v2 | oanda_demo_runner.py:16480-16486 | Brukes i runtime | Hvis expected_dim ≠ 37 → vec = row_to_feature_vector_v2(...). **KAN TRIGGES** når lastet modell har input_dim 31. |
| row_to_feature_vector_v2 | exit_transformer_dataset_v1.py | Brukes ved trening | use_io_v2=True → row_to_feature_vector_v2. **KAN TRIGGES** (trening produserer IOV2). |
| validate_window_v2 | exit_transformer_io_v2.py | Definert | 169. |
| validate_window_v2 | exit_transformer_v0.py:240 | Brukes i decider | Hvis input_dim ≠ 19 og ≠ 37 → validate_window_v2. **KAN TRIGGES**. |
| validate_window_v2 | exit_transformer_dataset_v1.py:159 | Ved trening | Ved use_io_v2. **KAN TRIGGES**. |
| FEATURE_DIM_V2 | – | Ikke som konstant | Grep: ingen eksplisitt konstant FEATURE_DIM_V2; dim 31 kommer fra feature_dim_v2(). **ELIMINERT** (ingen symbol). |
| use_io_v2 | exit_transformer_v0.py:305, 349, 374 | Trenings-API | train_from_exits_jsonl(use_io_v2=...). **KAN TRIGGES** – trening med IOV2. |
| use_io_v2 | exit_transformer_dataset_v1.py:104, 122, 192, 232 | Dataset | load_dataset_from_exits_jsonl(build_sequences_and_labels) use_io_v2. **KAN TRIGGES**. |
| use_io_v2 | run_truth_e2e_sanity.py:1113 | E2E trening | --require-io-v2 → use_io_v2=require_io_v2. **KAN TRIGGES**. |

**IOV1/IOV2 trigges via default/fallback?**  
- **Ja.** Runtime: hvis config.input_dim == 31, brukes row_to_feature_vector_v2 og validate_window_v2 (`oanda_demo_runner.py:16482-16486`, `exit_transformer_v0.py:236-241`). Hvis input_dim == 19, brukes IOV1-validering og (ved trening) _row_to_feature_vector. Ingen kode som tvinger 37; valget kommer fra lastet config.

**Trening produserer fortsatt IOV2-modeller?**  
- **Ja.** `exit_transformer_dataset_v1.py` har kun IOV1 og IOV2 (row_to_feature_vector_v2); ingen row_to_feature_vector_v3 eller 37-dims. `train_from_exits_jsonl` med use_io_v2=True/require_io_v2 produserer 31-dims og skriver det i config. Bevis: dataset_v1 122–125, 151; exit_transformer_v0 305, 349, 374.

**Imports utenfor _quarantine som refererer legacy exit?**  
- **Ja.** `exit_transformer_v0.py` importerer exit_transformer_io_v1 og exit_transformer_io_v2. `oanda_demo_runner.py` importerer row_to_feature_vector_v2 (og v3). `exit_transformer_dataset_v1.py` importerer exit_transformer_io_v1 og exit_transformer_io_v2. Kun `_quarantine/legacy_exit_rules_20260216/exit_master_v1.py` importerer exit_transformer_io_v1 (ml_ctx_to_feature_vector) – den er under quarantine. **KAN TRIGGES:** aktiv runtime og trening bruker fortsatt IOV1/IOV2-kontrakter og -funksjoner.

---

## DEL 6 – TRENING: ER EXIT-TRANSFORMER FAKTISK TRENT FOR 37?

**Treningsscript:**  
- `gx1/policy/exit_transformer_v0.py` – `train_from_exits_jsonl()` (bl.a. 288–362).  
- Den kaller `load_dataset_from_exits_jsonl()` fra `gx1/datasets/exit_transformer_dataset_v1.py`.

**Dataset-builder:**  
- `gx1/datasets/exit_transformer_dataset_v1.py`: `load_dataset_from_exits_jsonl()`, `build_sequences_and_labels()`.  
- Kun IOV1 (use_io_v2=False) og IOV2 (use_io_v2=True): `_row_to_feature_vector` (19) eller `row_to_feature_vector_v2` (31).  
- **Ingen bruk av row_to_feature_vector_v3, FEATURE_DIM_V3, eller 37-dims.** Bevis: grep i `gx1/datasets` etter row_to_feature_vector_v3/IOV3/FEATURE_DIM_V3 → **0 treff.**

**Svar:**  
- **Nei** – det finnes ingen kode i trenings-/dataset-laget som bygger IOV3-vektorer (37) under trening.  
- **Konklusjon:** Runtime støtter IOV3 (oanda_demo_runner + exit_transformer_io_v3), men trening gjør det ikke (ennå).  
- **Manglende kobling:**  
  - Fil: `gx1/datasets/exit_transformer_dataset_v1.py`.  
  - Mangler: kall til `row_to_feature_vector_v3` eller tilsvarende 37-dims bygg; `load_dataset_from_exits_jsonl` (188–241) og `build_sequences_and_labels` (98–164) har ingen use_io_v3/IOV3-branch.  
  - Bevis: use_io_v2-branch finnes (linje 104, 122–125, 151, 158–159, 192, 232); ingen import av exit_transformer_io_v3 eller row_to_feature_vector_v3.

---

## SLUTTRESULTAT (OBLIGATORISK OPPSUMMERING)

```
AKTIV EXIT IO-VERSJON:
- IO: IOV2 / IOV1 (ikke IOV3)
- input_dim: 19 (eneste config på disk); policy-path (37) finnes ikke
- Bevis: /home/andre2/GX1_DATA/models/exit_transformer_v0/<sha>/exit_transformer_config.json (input_dim: 19).
- Bevis policy-path: EXIT_TRANSFORMER_ONLY_V0.yaml:17 model_path: "models/exit_v0_ctx/EXIT_TRANSFORMER_V0_IOV2";
  resolvert path /home/andre2/GX1_DATA/models/exit_v0_ctx/EXIT_TRANSFORMER_V0_IOV2/exit_transformer_config.json – fil finnes ikke.

ANTALL EXIT FEATURES (IOV3-kontrakt):
- Totalt: 37
- Signaler: 7
- Entry snapshot: 5
- Trade state: 7
- Context: 12 (6 cont + 6 cat)
- Exit sharpeners: 6

SEMANTISK STATUS:
- Overlapp: JA – time_since_mfe_bars vs bars_since_mfe (alias); dd_from_mfe_bps vs drawdown_from_peak_bps (samme semantikk).
- Legacy paths aktive: JA – row_to_feature_vector_v2 og validate_window_v2 brukes i runtime når input_dim=31;
  trening bruker kun IOV1/IOV2 (exit_transformer_dataset_v1.py, train_from_exits_jsonl).
- Trening konsistent med runtime: NEI – runtime kan kjøre IOV3 (37), trening bygger bare 19 eller 31.

TRUTH-VERDIKT:
- ONE UNIVERSE HOLDES: NEI – policy krever én IO (IOV3), men (1) policy-peker mappe finnes ikke på disk,
  (2) eneste config på disk er IOV1 (input_dim 19), (3) trening produserer ikke IOV3-modeller.
```

---

*Rapport slutt. Kun filer åpnet og lest; ingen kode endret.*
