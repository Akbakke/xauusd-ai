# Rebuild prebuilt 2025 with ctx_cont+2 (PREBUILT_CTX_CONT_FAIL fix)

TRUTH-style: deterministic, no fallback, clear verification.

## Paths (SSoT)

- **GX1_ENGINE**: `/home/andre2/src/GX1_ENGINE`
- **GX1_DATA**: `/home/andre2/GX1_DATA`

## Gammel parquet path (før rebuild)

```
/home/andre2/GX1_DATA/data/data/prebuilt/V13_REFINED3/2025/xauusd_m5_2025_features_v13_refined3_20260215_150308__CTX2PLUS_20260218_111721.parquet
```

(Canonical truth `canonical_prebuilt_parquet` peker på denne inntil du oppdaterer.)

## Ny parquet path (etter rebuild)

Ny fil får navn med ctx-dim og timestamp, f.eks.:

```
/home/andre2/GX1_DATA/data/data/prebuilt/V13_REFINED3/2025/xauusd_m5_2025_features_v13_refined3_20260215_150308__CTX_CONT4_CAT5_<YYYYMMDD_HHMMSS>.parquet
```

## Kolonner lagt til (ctx_cont+2 for dim 4)

- `D1_dist_from_ema200_atr`
- `H1_range_compression_ratio`

(Kontrakt: `gx1.contracts.signal_bridge_v1.ORDERED_CTX_CONT_NAMES_EXTENDED[2:4]`.)

---

## Kommandoer

### 1) Forventede kolonner og diff (valgfritt)

```bash
cd /home/andre2/src/GX1_ENGINE
GX1_ENGINE=/home/andre2/src/GX1_ENGINE GX1_DATA=/home/andre2/GX1_DATA \
  /home/andre2/venvs/gx1/bin/python -m gx1.scripts.expected_ctx_cont_cols \
  --truth-file gx1/configs/canonical_truth_signal_only.json \
  --output-dir .
```

- Skriver `expected_ctx_cont_cols.txt` og `missing_in_prebuilt.txt`.
- Exit 2 hvis prebuilt mangler kolonner.

### 2) Rebuild prebuilt 2025 (med ctx_cont+2)

**Variant A – wrapper (anbefalt)**

```bash
cd /home/andre2/src/GX1_ENGINE
GX1_ENGINE=/home/andre2/src/GX1_ENGINE GX1_DATA=/home/andre2/GX1_DATA \
  /home/andre2/venvs/gx1/bin/python -m gx1.scripts.rebuild_prebuilt_ctx_cont_2025 \
  --truth-file gx1/configs/canonical_truth_signal_only.json
```

**Variant B – direkte kall på builder (ikke-karantenert)**

```bash
cd /home/andre2/src/GX1_ENGINE
GX1_ENGINE=/home/andre2/src/GX1_ENGINE GX1_DATA=/home/andre2/GX1_DATA \
  /home/andre2/venvs/gx1/bin/python -m gx1.scripts.add_ctx_cont_columns_to_prebuilt \
  --prebuilt_parquet "$GX1_DATA/data/data/prebuilt/V13_REFINED3/2025/xauusd_m5_2025_features_v13_refined3_20260215_150308.parquet" \
  --output_parquet "$GX1_DATA/data/data/prebuilt/V13_REFINED3/2025/xauusd_m5_2025_features_v13_refined3_20260215_150308__CTX_CONT4_CAT5_$(date -u +%Y%m%d_%H%M%S).parquet"
```
(Default raw M5: `$GX1_DATA/data/data/_staging/XAUUSD_M5_2020_2025_bidask__TEMP_CTX2PLUS.parquet`.)

### 3) Oppdater canonical truth

Rediger `gx1/configs/canonical_truth_signal_only.json` og sett `canonical_prebuilt_parquet` til den nye parquet-pathen fra steg 2.

### 4) Sanity-run (kort vindu)

```bash
cd /home/andre2/src/GX1_ENGINE
GX1_ENGINE=/home/andre2/src/GX1_ENGINE GX1_DATA=/home/andre2/GX1_DATA \
  GX1_TRUTH_MODE=1 \
  /home/andre2/venvs/gx1/bin/python -m gx1.scripts.run_truth_e2e_sanity \
  --truth-file gx1/configs/canonical_truth_signal_only.json \
  --start-ts 2025-06-03 --end-ts 2025-06-04 \
  --run-id "PREBUILT_CTX_REBUILD_$(date -u +%Y%m%d_%H%M%S)"
```

### 5) Verifiser prebuilt_used i chunk_footer

```bash
# Erstatt <RUN_ROOT> med faktisk run-dir under GX1_DATA (sanity-run output)
RUN_ROOT="/home/andre2/GX1_DATA/..."   # f.eks. .../runs/truth_e2e_.../...

grep -r "prebuilt_used\|prebuilt_parquet_path" "$RUN_ROOT" --include="chunk_footer.json"
# eller
find "$RUN_ROOT" -name "chunk_footer.json" -exec jq -r '.prebuilt_used, .prebuilt_parquet_path' {} \;
```

Forventet: `prebuilt_used=true` og `prebuilt_parquet_path=<ny parquet path>`.

---

## Filer endret (leveranse)

| Område | Fil |
|--------|-----|
| Fail-early logging | `gx1/execution/oanda_demo_runner.py` – forbedret PREBUILT_CTX_CONT_FAIL med expected_ctx_cont_dim, manglende kolonner (max 50 + full liste i fil), prebuilt path, første 30 kolonner |
| Prebuilt columns ved load | `gx1/execution/chunk_bootstrap.py` – `prebuilt_required_columns` utvides med ctx_cont+2; **hard-fail** hvis expected_ctx_cont_dim ∈ {4,6} og listen ikke slutter med ctx_cont-kolonner i kontraktsrekkefølge (bootstrapping bug) |
| Expected/diff script | `gx1/scripts/expected_ctx_cont_cols.py` (ny) |
| Rebuild wrapper | `gx1/scripts/rebuild_prebuilt_ctx_cont_2025.py` (ny) |
| Truth (etter rebuild) | `gx1/configs/canonical_truth_signal_only.json` – kun `canonical_prebuilt_parquet` oppdateres til ny path |
| Dokumentasjon | `gx1/docs/REBUILD_PREBUILT_CTX_NOTE.md` (denne filen) |

Builder som faktisk legger til kolonnene:  
`gx1/scripts/add_ctx_cont_columns_to_prebuilt.py` (ikke-karantenert; ingen avhengighet av `gx1._quarantine`). Wrapperen `rebuild_prebuilt_ctx_cont_2025.py` importerer og kaller `run_add_ctx_cont_columns()` direkte.
