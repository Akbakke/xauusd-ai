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

import argparse
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
from gx1.scripts import entry_iql_multi_head_gpu_core_v1 as iql_core  # noqa: E402  ACTION_* labels (one-truth)

# 2026-06-12 (promoted out of _legacy_disabled — it is load-bearing for the
# nightly candidate gate): ALL hardcoded defaults removed. The old
# DEFAULT_ENTRY_IQL_LOCK pointed at a dead 2026-05-14 bundle — running bare
# would have silently inferred with a stale vintage (rule 8). Every input is
# now an explicit required arg; callers resolve bundles via gx1_guards or pass
# an explicit candidate (the gate's sanctioned override).


def main() -> int:
    parser = argparse.ArgumentParser(description="V12 Phase 1 — Entry-IQL v2 inference")
    parser.add_argument("--entry-iql-lock", type=str, required=True,
                        help="Entry-IQL v2 trained-models LOCK dir (from materialize_build_entry_iql_v2.py). "
                             "REQUIRED — no default (rule 8: never a guessed vintage).")
    parser.add_argument("--forward-outcome-dir", type=str, required=True,
                        help="Path to forward-outcome per_week dir (the dir CONTAINING the *.parquet shards).")
    parser.add_argument("--out-root", type=str, required=True,
                        help="Parent dir under which a ENTRY_IQL_INFERENCE_FOR_V12_<TS> dir is created.")
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--fold-id", type=str, default="FOLD_1")
    args = parser.parse_args()

    ENTRY_IQL_LOCK = Path(args.entry_iql_lock)
    FORWARD_OUTCOME_DIR = Path(args.forward_outcome_dir)
    OUT_DIR = Path(args.out_root) / (
        f"ENTRY_IQL_INFERENCE_FOR_V12_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    print(f"V12 PHASE 1 — Entry-IQL v2 inference on forward-outcome candidates")
    print(f"  ENTRY_IQL_LOCK       = {ENTRY_IQL_LOCK}")
    print(f"  FORWARD_OUTCOME_DIR  = {FORWARD_OUTCOME_DIR}")
    print(f"  Output dir           = {OUT_DIR}")

    # ── Adapter ──────────────────────────────────────────────────────────
    print(f"\n[0/3] Loading Entry-IQL v2 adapter...")
    adapter = EntryIQLV2Adapter.load(
        artifact_root=ENTRY_IQL_LOCK,
        variant=args.variant,
        fold_id=args.fold_id,
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
        # 2026-06-08: the candidate lineage (no chunk_0) lacks the 60 group-A/dip-struct ctx cols.
        # The Entry-IQL BUILD attaches them via the one-truth helper; mirror that here so inference
        # state == train state (train==serve). Read needed cols + OHLC + decision_ts so the attach
        # can compute them; the helper is idempotent (no-op if already present, e.g. chunk_0 lineage).
        _read = [c for c in (set(cols_to_read) | {"high", "low", "close", "open", "decision_ts_utc"}) if c in fw_cols]
        cands = pd.read_parquet(fp, columns=_read)
        if len(cands) == 0:
            continue
        if "time" not in cands.columns and "decision_ts_utc" in cands.columns:
            cands["time"] = pd.to_datetime(cands["decision_ts_utc"], utc=True)
        from gx1.scripts.augment_forward_outcome_v2 import attach_group_a_dip_struct_ctx_columns
        cands = attach_group_a_dip_struct_ctx_columns(cands, journal_label="phase1_entry_inference")
        candidate_dicts = cands.to_dict("records")
        recs = adapter.predict(candidate_dicts)

        # DIPFIX serve-time selection overlay (2026-06-15, default-OFF byte-identical). ONE TRUTH with the
        # live serve (gx1.execution.v12_entry_iql_live.apply_dipfix_overlay) so the OOT gate evaluates the
        # SAME selection the paper runner would apply. When GX1_ENTRY_DIPFIX is unset this is a pure no-op
        # (apply_dipfix_overlay returns the action_id unchanged + {}), so the gate baseline arm is identical
        # to the prior raw-IQL inference (the volbal baseline CSV). The marker columns are emitted always
        # (False/"" off-rows) so the per-candidate CSV attributes any fired overlay.
        # MARGIN-FLOOR overlay applied IMMEDIATELY AFTER dipfix (same order as the live serve:
        # conviction-gate → STEP-1 → dipfix → margin-floor). ONE TRUTH with the live serve
        # (gx1.execution.v12_entry_iql_live.apply_margin_floor_overlay) so the OOT gate scores the
        # IDENTICAL selection the paper runner would apply. Default-OFF (GX1_ENTRY_MARGIN_FLOOR unset)
        # ⇒ pure no-op → gate baseline arm byte-identical to the prior inference.
        from gx1.execution.v12_entry_iql_live import apply_dipfix_overlay, apply_margin_floor_overlay
        new_action_ids, dipfix_applied, dipfix_mode_col, dipfix_from_col, dipfix_to_col = [], [], [], [], []
        margin_floor_applied = []
        for r, c in zip(recs, candidate_dicts):
            nid, mk = apply_dipfix_overlay(int(r.action_id_v1), c)
            nid, mf = apply_margin_floor_overlay(nid, c)
            new_action_ids.append(nid)
            dipfix_applied.append(bool(mk.get("dipfix_applied", False)))
            dipfix_mode_col.append(mk.get("dipfix_mode", ""))
            dipfix_from_col.append(mk.get("dipfix_from", ""))
            dipfix_to_col.append(mk.get("dipfix_to", ""))
            margin_floor_applied.append(bool(mf.get("margin_floor_applied", False)))

        out = pd.DataFrame({
            "candidate_uid": cands["candidate_uid"].values,
            "decision_ts_utc": cands["decision_ts_utc"].values if "decision_ts_utc" in cands.columns else None,
            "action_label_v1": [iql_core.ACTION_LABELS_V1[a] for a in new_action_ids],
            "action_id_v1": new_action_ids,
            "dipfix_applied": dipfix_applied,
            "dipfix_mode": dipfix_mode_col,
            "dipfix_from": dipfix_from_col,
            "dipfix_to": dipfix_to_col,
            "margin_floor_applied": margin_floor_applied,
            "raw_action_label_v1": [r.action_label_v1 for r in recs],
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
        "forward_outcome_dir": str(FORWARD_OUTCOME_DIR),
        "variant": args.variant,
        "fold_id": args.fold_id,
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
