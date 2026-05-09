"""V12 Phase 1: Run Entry-IQL v2 inference on ALL forward-outcome candidates.

Goal: Produce per-candidate (candidate_uid, action_label, q_values, advantage)
parquet so V12 builder can filter to TAKE_LONG_NOW/TAKE_SHORT_NOW only
(drop SKIP for 1-til-1 entry/exit coupling).

Source: forward-outcome dataset (50,824 candidates, 276 weeks) — has all 165
Entry-IQL state features pre-computed (45 ctx_cont + 6 ctx_cat + 145 chunk_0/
canonical/V10/XGB-derived) so no extra joining required.

Output: ENTRY_IQL_INFERENCE_FOR_V12_<TS>/decisions.parquet
        + per_week/<week>.parquet shards for downstream join
        + symlinked /tmp/v12_entry_iql_decisions.parquet

Run:
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 /tmp/v12_phase1_entry_iql_inference.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/home/andre2/src/GX1_ENGINE")
from gx1.runtime.entry_iql_v2_adapter import EntryIQLV2Adapter  # noqa: E402

ENTRY_IQL_LOCK = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "BUILD_ENTRY_IQL_V2_20260506T195420Z_LOCK"
)
FORWARD_OUTCOME_DIR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "CANDIDATE_FORWARD_OUTCOME_RUN/v1_full/per_week"
)
OUT_DIR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    f"ENTRY_IQL_INFERENCE_FOR_V12_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
)


def main() -> int:
    print(f"V12 PHASE 1 — Entry-IQL v2 inference on forward-outcome candidates")
    print(f"Output dir: {OUT_DIR}")

    # ── Adapter ──────────────────────────────────────────────────────────
    print(f"\n[0/3] Loading Entry-IQL v2 adapter...")
    adapter = EntryIQLV2Adapter.load(
        artifact_root=ENTRY_IQL_LOCK,
        variant="R_NET_REAL",
        fold_id="FOLD_1",
        aggregator="mean",
        beta=1.0,
        prefer_cuda=True,
        min_advantage_bps=0.0,  # raw decisions; V9 Q-adv filter applied downstream
    )
    feat_names = adapter.feature_names
    print(f"  adapter ready: {len(feat_names)} features, device={adapter.model.device}")

    # ── Determine columns to read from forward-outcome ───────────────────
    onehot_features = [f for f in feat_names if "__" in f]
    onehot_base_cols = list({f.partition("__")[0] for f in onehot_features})
    numeric_features = [f for f in feat_names if "__" not in f]
    needed_cols = list({"candidate_uid", "decision_ts_utc", *numeric_features, *onehot_base_cols})

    files = sorted(FORWARD_OUTCOME_DIR.glob("*.parquet"))
    print(f"\n[1/3] Forward-outcome weekly parquets: {len(files)}")

    fw_cols = set(pq.ParquetFile(files[0]).schema.names)
    missing = [c for c in numeric_features if c not in fw_cols]
    missing_oh = [c for c in onehot_base_cols if c not in fw_cols]
    if missing:
        print(f"  ⚠️  {len(missing)} numeric features missing from forward-outcome:")
        for m in missing[:10]:
            print(f"      - {m}")
    if missing_oh:
        print(f"  ⚠️  {len(missing_oh)} categorical sources missing: {missing_oh}")
    cols_to_read = [c for c in needed_cols if c in fw_cols]
    print(f"  reading {len(cols_to_read)} cols / {len(needed_cols)} needed")

    # ── Per-week inference + write ───────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_week_dir = OUT_DIR / "per_week"
    per_week_dir.mkdir(exist_ok=True)
    all_decisions = []
    total_rows = 0
    t0 = time.time()
    print(f"\n[2/3] Running Entry-IQL inference per week...")
    for w_idx, fp in enumerate(files, start=1):
        week_id = fp.stem  # forward_outcomes_TRUTH_MONFRI_WEEK_YYYYMMDD_YYYYMMDD
        cands = pd.read_parquet(fp, columns=cols_to_read)
        if len(cands) == 0:
            continue
        candidate_dicts = cands.to_dict("records")
        recs = adapter.predict(candidate_dicts)

        out = pd.DataFrame({
            "candidate_uid": cands["candidate_uid"].values,
            "decision_ts_utc": cands["decision_ts_utc"].values if "decision_ts_utc" in cands.columns else None,
            "action_label_v1": [r.action_label_v1 for r in recs],
            "action_id_v1": [r.action_id_v1 for r in recs],
            "advantage_over_skip_v1": [r.advantage_over_skip_v1 for r in recs],
            "advantage_over_realized_v1": [r.advantage_over_realized_v1 for r in recs],
            "q_skip_v1": [r.q_per_action_v1[0] for r in recs],
            "q_take_long_v1": [r.q_per_action_v1[1] for r in recs],
            "q_take_short_v1": [r.q_per_action_v1[2] for r in recs],
            "confidence_softmax_max_v1": [float(r.confidence_softmax_v1.max()) for r in recs],
        })
        out.to_parquet(per_week_dir / f"{week_id}.parquet", index=False)
        all_decisions.append(out)
        total_rows += len(out)
        if w_idx % 25 == 0 or w_idx == len(files):
            elapsed = time.time() - t0
            print(f"  [{w_idx}/{len(files)}] {len(out):3d} cands  elapsed={elapsed:.0f}s  total={total_rows:,}")

    # ── Aggregate + summary ──────────────────────────────────────────────
    print(f"\n[3/3] Writing aggregated decisions + summary...")
    full = pd.concat(all_decisions, ignore_index=True)
    aggregated_path = OUT_DIR / "decisions.parquet"
    full.to_parquet(aggregated_path, index=False)
    print(f"  Wrote {len(full):,} decisions → {aggregated_path}")

    print("\n--- action distribution ---")
    print(full["action_label_v1"].value_counts())
    print("\n--- advantage_over_skip distribution ---")
    print(full["advantage_over_skip_v1"].describe(percentiles=[0.1, 0.5, 0.7, 0.9]))

    n_take = int((full["action_label_v1"].isin(["TAKE_LONG_NOW", "TAKE_SHORT_NOW"])).sum())
    p70 = float(full["advantage_over_skip_v1"].quantile(0.70))
    p90 = float(full["advantage_over_skip_v1"].quantile(0.90))
    print(f"\n  TAKE candidates: {n_take:,} / {len(full):,} = {100*n_take/len(full):.2f}%")
    print(f"  Q-adv P70={p70:.2f}  P90={p90:.2f}  (V9 sweet spots)")

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_iql_lock": str(ENTRY_IQL_LOCK),
        "variant": "R_NET_REAL",
        "fold_id": "FOLD_1",
        "n_candidates_total": int(len(full)),
        "n_take": n_take,
        "action_dist": full["action_label_v1"].value_counts().to_dict(),
        "advantage_p70": p70,
        "advantage_p90": p90,
        "missing_numeric_features": missing,
        "missing_categorical_sources": missing_oh,
        "n_features_used": len(feat_names),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    tmp_link = Path("/tmp/v12_entry_iql_decisions.parquet")
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(aggregated_path)
    print(f"  → symlinked /tmp/v12_entry_iql_decisions.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
