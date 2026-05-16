#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MFE_MIN_BPS = 24.0
DEFAULT_MAE_MAX_BPS = 2.5
DEFAULT_PATH_MIN_BPS = 10.0
DEFAULT_BAD_PATH_MAX = 0.0


def relabel_df(
    df: pd.DataFrame,
    *,
    mfe_min_bps: float,
    mae_max_bps: float,
    path_min_bps: float,
    bad_path_max: float,
) -> pd.DataFrame:
    required = [
        "y_direction",
        "y_tradable",
        "y_early_move",
        "y_quality_score",
        "y_bad_path",
        "mae_first_n_bps",
        "mfe_first_n_bps",
        "path_quality_bps",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"[RELABEL_MISSING_COLUMNS] {missing}")

    out = df.copy()
    y_dir = out["y_direction"].astype(np.int32).to_numpy(copy=True)
    y_bad = out["y_bad_path"].astype(np.float32).to_numpy(copy=False)
    mfe = out["mfe_first_n_bps"].astype(np.float32).to_numpy(copy=False)
    mae = out["mae_first_n_bps"].astype(np.float32).to_numpy(copy=False)
    path = out["path_quality_bps"].astype(np.float32).to_numpy(copy=False)

    side_mask = (y_dir == 0) | (y_dir == 1)
    obvious_mask = (
        side_mask
        & (mfe >= float(mfe_min_bps))
        & (np.abs(mae) <= float(mae_max_bps))
        & (path >= float(path_min_bps))
        & (y_bad <= float(bad_path_max))
    )

    flatten_mask = side_mask & (~obvious_mask)
    out.loc[flatten_mask, "y_direction"] = 2
    out.loc[flatten_mask, "y_tradable"] = 0
    out.loc[flatten_mask, "y_early_move"] = 0.0
    out.loc[flatten_mask, "y_quality_score"] = 0.0
    out.loc[flatten_mask, "mfe_first_n_bps"] = 0.0
    out.loc[flatten_mask, "mae_first_n_bps"] = 0.0
    out.loc[flatten_mask, "path_quality_bps"] = 0.0

    # Keep obvious-edge samples explicitly tradable.
    out.loc[obvious_mask, "y_tradable"] = 1
    out.attrs["obvious_edge_relabel"] = {
        "mfe_min_bps": float(mfe_min_bps),
        "mae_max_bps": float(mae_max_bps),
        "path_min_bps": float(path_min_bps),
        "bad_path_max": float(bad_path_max),
        "direction_follows_obvious_edge": True,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel ENTRY_V10_CTX parquet to obvious-edge-only semantics.")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--mfe-min-bps", type=float, default=DEFAULT_MFE_MIN_BPS)
    parser.add_argument("--mae-max-bps", type=float, default=DEFAULT_MAE_MAX_BPS)
    parser.add_argument("--path-min-bps", type=float, default=DEFAULT_PATH_MIN_BPS)
    parser.add_argument("--bad-path-max", type=float, default=DEFAULT_BAD_PATH_MAX)
    args = parser.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp)
    before = {
        "rows": int(len(df)),
        "long_rate": float((df["y_direction"].astype(int) == 0).mean()),
        "short_rate": float((df["y_direction"].astype(int) == 1).mean()),
        "flat_rate": float((df["y_direction"].astype(int) == 2).mean()),
        "tradable_rate": float(df["y_tradable"].astype(float).mean()),
    }
    relabeled = relabel_df(
        df,
        mfe_min_bps=args.mfe_min_bps,
        mae_max_bps=args.mae_max_bps,
        path_min_bps=args.path_min_bps,
        bad_path_max=args.bad_path_max,
    )
    after = {
        "rows": int(len(relabeled)),
        "long_rate": float((relabeled["y_direction"].astype(int) == 0).mean()),
        "short_rate": float((relabeled["y_direction"].astype(int) == 1).mean()),
        "flat_rate": float((relabeled["y_direction"].astype(int) == 2).mean()),
        "tradable_rate": float(relabeled["y_tradable"].astype(float).mean()),
    }
    relabeled.to_parquet(out, index=False)

    proof = {
        "input": str(inp),
        "output": str(out),
        "thresholds": relabeled.attrs["obvious_edge_relabel"],
        "before": before,
        "after": after,
    }
    proof_path = out.with_suffix(".relabel_proof.json")
    proof_path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
