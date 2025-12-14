# gx1/analysis/build_exit_policy_training_dataset_v3.py

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal, Tuple

import pandas as pd


RunPolicyHint = Literal["RULE5_ONLY", "RULE6A_ONLY", "MIXED"]


def infer_run_policy_hint(tag: str) -> RunPolicyHint:
    """Infer run-level policy hint (RULE5-only, RULE6A-only, or mixed) from the tag."""
    tag_upper = (tag or "").upper()
    if "RULE5" in tag_upper:
        return "RULE5_ONLY"
    if "RULE6A" in tag_upper:
        return "RULE6A_ONLY"
    return "MIXED"


def _policy_name_from_exit_profile(exit_profile: Optional[str]) -> Optional[str]:
    """Map an exit profile string to a logical policy label if we recognize it."""
    if not exit_profile or pd.isna(exit_profile):
        return None
    if not isinstance(exit_profile, str):
        return None
    ep_upper = exit_profile.upper()
    if "RULES_ADAPTIVE" in ep_upper or "RULE6A" in ep_upper:
        return "RULE6A"
    if "RULES_A" in ep_upper or "RULE5" in ep_upper:
        return "RULE5"
    return None


def _csv_row_to_trade_dict(row: pd.Series) -> Dict[str, Any]:
    """Konverterer en CSV-rad til trade-dict format som JSON-filer har."""
    trade_dict = {
        "trade_id": row.get("trade_id"),
        "exit_profile": row.get("exit_profile"),
        "entry": {
            "timestamp": row.get("entry_time"),
            "session": row.get("session_entry"),
        },
        "exit": {
            "timestamp": row.get("exit_time"),
        },
        "metrics": {
            "pnl_bps": row.get("pnl_bps"),
        },
        "extra": {},
    }
    
    # Parse extra-kolonnen hvis den finnes
    extra_str = row.get("extra")
    if pd.notna(extra_str) and extra_str:
        try:
            if isinstance(extra_str, str):
                trade_dict["extra"] = json.loads(extra_str)
            else:
                trade_dict["extra"] = extra_str
        except (json.JSONDecodeError, TypeError):
            trade_dict["extra"] = {}
    
    # Ekstraher atr_pct og spread_pct fra exit_hybrid hvis de finnes der
    exit_hybrid = trade_dict["extra"].get("exit_hybrid", {})
    if exit_hybrid:
        if "atr_pct" in exit_hybrid and pd.notna(exit_hybrid.get("atr_pct")):
            trade_dict["extra"]["atr_pct"] = exit_hybrid["atr_pct"]
        if "spread_pct" in exit_hybrid and pd.notna(exit_hybrid.get("spread_pct")):
            trade_dict["extra"]["spread_pct"] = exit_hybrid["spread_pct"]
        if "regime" in exit_hybrid:
            trade_dict["extra"]["router_regime"] = exit_hybrid["regime"]
        if "session" in exit_hybrid:
            trade_dict["extra"]["session"] = exit_hybrid["session"]
    
    # Legg til direkte kolonner i extra hvis de finnes (overstyrer exit_hybrid hvis de finnes)
    if "atr_pct" in row and pd.notna(row.get("atr_pct")):
        trade_dict["extra"]["atr_pct"] = row["atr_pct"]
    if "spread_pct" in row and pd.notna(row.get("spread_pct")):
        trade_dict["extra"]["spread_pct"] = row["spread_pct"]
    if "atr_bucket" in row and pd.notna(row.get("atr_bucket")):
        trade_dict["extra"]["atr_bucket"] = row["atr_bucket"]
    if "farm_entry_session" in row and pd.notna(row.get("farm_entry_session")):
        trade_dict["extra"]["session"] = row["farm_entry_session"]
    
    return trade_dict


def _extract_trade_data(
    t: Dict[str, Any],
    run_dir: Path,
    tag: str,
    run_policy_hint: RunPolicyHint,
) -> Dict[str, Any]:
    """Ekstraherer trade-data fra en trade-dict (fra JSON eller CSV)."""
    # Grunnleggende felt
    trade_id = t.get("trade_id")
    entry = t.get("entry", {})
    exit_ = t.get("exit", {})
    metrics = t.get("metrics", {})
    extra: Dict[str, Any] = t.get("extra", {}) or {}

    exit_profile = t.get("exit_profile")
    router_name = extra.get("router_name") or extra.get("router_version") or extra.get("router_used")

    # PnL
    pnl_bps = metrics.get("pnl_bps")

    # Hypotetiske PnLs per policy (kan være generert av egne analyseruns / sim)
    # Disse er valgfrie – hvis de mangler, bruker vi bare actual.
    pnl_rule5 = extra.get("pnl_rule5_bps")
    pnl_rule6a = extra.get("pnl_rule6a_bps")

    # Features vi vil bruke i V3
    atr_pct = extra.get("atr_pct")
    spread_pct = extra.get("spread_pct")
    atr_bucket = extra.get("atr_bucket")  # f.eks. LOW/MEDIUM/HIGH
    farm_regime = extra.get("farm_regime") or extra.get("router_regime")
    session = extra.get("session") or entry.get("session")

    distance_to_range = extra.get("distance_to_range")
    micro_volatility = extra.get("micro_volatility")
    volume_stability = extra.get("volume_stability")
    range_edge_dist_atr = extra.get("range_edge_dist_atr")

    # ---- V4-B: range_pos feature ----
    # B.1: Foretrukket kilde - sjekk om range_pos finnes direkte
    range_pos = extra.get("range_pos")
    
    # B.2: Hvis ikke direkte, beregn fra distance_to_range + side-indikator
    if range_pos is None:
        range_side = (
            extra.get("range_side") or 
            extra.get("range_edge") or 
            extra.get("pos_in_range") or 
            extra.get("range_location")
        )
        
        if range_side is not None and distance_to_range is not None:
            side_upper = str(range_side).upper()
            try:
                dtr_val = float(distance_to_range)
                dtr_clamped = max(0.0, min(1.0, dtr_val))
                
                if any(x in side_upper for x in ["LOW", "BOTTOM", "SUPPORT"]):
                    range_pos = dtr_clamped  # nær low -> small values
                elif any(x in side_upper for x in ["HIGH", "TOP", "RESISTANCE"]):
                    range_pos = 1.0 - dtr_clamped  # nær high -> large values
                elif any(x in side_upper for x in ["MID", "CENTER"]):
                    range_pos = 0.5
                else:
                    range_pos = 0.5  # ukjent side -> midrange
            except (ValueError, TypeError):
                range_pos = 0.5
        else:
            # B.3: Fallback hvis ingen info
            range_pos = 0.5

    # Quarter / periode – prøv å hente fra extra, og fallback til tag hvis ikke
    quarter = extra.get("quarter") or tag

    # Sett policy-actual basert på hint + exit_profile
    policy_actual = _policy_name_from_exit_profile(exit_profile)
    if run_policy_hint == "RULE5_ONLY":
        policy_actual = "RULE5"
        if pnl_rule5 is None and pnl_bps is not None:
            pnl_rule5 = pnl_bps
    elif run_policy_hint == "RULE6A_ONLY":
        policy_actual = "RULE6A"
        if pnl_rule6a is None and pnl_bps is not None:
            pnl_rule6a = pnl_bps

    if policy_actual is None:
        policy_actual = "UNKNOWN"

    # Best policy-label (foreløpig bare RULE5 vs RULE6A)
    # Hvis vi har hypotetiske PnLs for begge, bruk de til å velge beste.
    # Hvis ikke, fall tilbake til actual.

    policy_best: Optional[str] = None
    if (pnl_rule5 is not None) and (pnl_rule6a is not None):
        # Velg policy med høyest PnL
        if pnl_rule5 >= pnl_rule6a:
            policy_best = "RULE5"
        else:
            policy_best = "RULE6A"
    else:
        # Ingen hypotetiske PnLs – vi kan bare anta at actual var "best"
        policy_best = policy_actual

    return {
        # Meta
        "run_tag": tag,
        "run_dir": str(run_dir),
        "trade_id": trade_id,
        "quarter": quarter,
        "symbol": extra.get("symbol") or extra.get("instrument") or "XAUUSD",

        "entry_time": entry.get("timestamp"),
        "exit_time": exit_.get("timestamp"),

        # Router / profile
        "router_name": router_name,
        "exit_profile_actual": exit_profile,
        "policy_actual": policy_actual,
        "policy_best": policy_best,

        # PnL
        "pnl_bps": pnl_bps,
        "pnl_rule5_bps": pnl_rule5,
        "pnl_rule6a_bps": pnl_rule6a,

        # V3-features
        "atr_pct": atr_pct,
        "spread_pct": spread_pct,
        "atr_bucket": atr_bucket,
        "farm_regime": farm_regime,
        "session": session,
        "distance_to_range": distance_to_range,
        "range_pos": range_pos,
        "range_edge_dist_atr": range_edge_dist_atr,
        "micro_volatility": micro_volatility,
        "volume_stability": volume_stability,
    }


def load_trades_from_run(
    run_dir: Path,
    tag: str,
    run_policy_hint: Optional[RunPolicyHint] = None,
) -> pd.DataFrame:
    """
    Leser trades fra en run-mappe.
    Støtter både:
      - run_dir / "trades" / *.json (JSON-format)
      - run_dir / "trade_log_*.csv" (CSV-format med extra-kolonne)
    
    'tag' kan være f.eks. "FULLYEAR_ADAPTIVE" eller "Q2_ADAPTIVE".
    run_policy_hint lar oss oppgi om run er RULE5-only, RULE6A-only eller MIXED.
    """

    rows: List[Dict[str, Any]] = []
    policy_hint: RunPolicyHint = run_policy_hint if run_policy_hint else "MIXED"

    # Prøv først trades-katalog med JSON-filer
    trades_dir = run_dir / "trades"
    if trades_dir.exists():
        for f in sorted(trades_dir.glob("*.json")):
            with f.open() as fh:
                t = json.load(fh)
            rows.append(_extract_trade_data(t, run_dir, tag, policy_hint))
    else:
        # Fallback til CSV-filer
        csv_files = list(run_dir.glob("trade_log_*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"Fant verken trades-katalog eller trade_log CSV-filer i {run_dir}"
            )
        
        # Les CSV og konverter hver rad til trade-dict
        for csv_file in sorted(csv_files):
            df_csv = pd.read_csv(csv_file)
            for _, row in df_csv.iterrows():
                t = _csv_row_to_trade_dict(row)
                rows.append(_extract_trade_data(t, run_dir, tag, policy_hint))

    df = pd.DataFrame(rows)

    # Konverter tider til datetime hvis de finnes
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")

    return df


def build_dataset_v3(run_dirs: Dict[str, Path]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    run_dirs: mapping tag -> run_dir path
    Returns: (df, metadata_dict) hvor metadata_dict inneholder missing counts
    """
    dfs = []
    for tag, run_dir in run_dirs.items():
        hint = infer_run_policy_hint(tag)
        df_run = load_trades_from_run(run_dir, tag, run_policy_hint=hint)
        dfs.append(df_run)

    df = pd.concat(dfs, ignore_index=True)

    # Hard krav: vi trenger minst atr_pct for å kunne bruke dette i routeren
    before = len(df)
    df = df.dropna(subset=["atr_pct"]).copy()
    after = len(df)
    print(f"[build_exit_policy_v3] Droppet {before - after} trades uten atr_pct (av {before}).")

    # Fyll NaNs i viktige features med "sikre" defaults
    df["spread_pct"] = df["spread_pct"].fillna(1.0)  # antar worst case = høy spread
    for col in ["atr_bucket", "farm_regime", "session"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    # ---- V4-A: distance_to_range som first-class numeric feature ----
    if "distance_to_range" in df.columns:
        missing_dtr_before = int(pd.to_numeric(df["distance_to_range"], errors="coerce").isna().sum())
        df["distance_to_range"] = pd.to_numeric(df["distance_to_range"], errors="coerce")
        df["distance_to_range"] = df["distance_to_range"].fillna(0.5)

        # Logg stats for inspeksjon
        dtr_series = df["distance_to_range"].astype(float)
        print(f"[build_exit_policy_v3] distance_to_range missing før fill: {missing_dtr_before}")
        print(
            "[build_exit_policy_v3] distance_to_range stats etter cleaning: "
            f"min={float(dtr_series.min()):.6f}, "
            f"max={float(dtr_series.max()):.6f}, "
            f"mean={float(dtr_series.mean()):.6f}, "
            f"median={float(dtr_series.median()):.6f}"
        )

    # ---- V4-B: range_pos som first-class numeric feature ----
    missing_rp_before = 0
    if "range_pos" in df.columns:
        missing_rp_before = int(pd.to_numeric(df["range_pos"], errors="coerce").isna().sum())
        df["range_pos"] = pd.to_numeric(df["range_pos"], errors="coerce")
        df["range_pos"] = df["range_pos"].fillna(0.5)
        df["range_pos"] = df["range_pos"].clip(0.0, 1.0)

        # Logg stats for inspeksjon
        rp_series = df["range_pos"].astype(float)
        print(f"[build_exit_policy_v3] range_pos missing før fill: {missing_rp_before}")
        print(
            "[build_exit_policy_v3] range_pos stats etter cleaning: "
            f"min={float(rp_series.min()):.6f}, "
            f"max={float(rp_series.max()):.6f}, "
            f"mean={float(rp_series.mean()):.6f}, "
            f"median={float(rp_series.median()):.6f}"
        )
        print()  # Tom linje for bedre lesbarhet

    # ---- V4-C: range_edge_dist_atr som first-class numeric feature ----
    missing_reda_before = 0
    if "range_edge_dist_atr" in df.columns:
        missing_reda_before = int(pd.to_numeric(df["range_edge_dist_atr"], errors="coerce").isna().sum())
        df["range_edge_dist_atr"] = pd.to_numeric(df["range_edge_dist_atr"], errors="coerce")
        df["range_edge_dist_atr"] = df["range_edge_dist_atr"].fillna(0.0)
        df["range_edge_dist_atr"] = df["range_edge_dist_atr"].clip(0.0, 10.0)

        # Logg stats for inspeksjon
        reda_series = df["range_edge_dist_atr"].astype(float)
        print(f"[build_exit_policy_v3] range_edge_dist_atr missing før fill: {missing_reda_before}")
        print(
            "[build_exit_policy_v3] range_edge_dist_atr stats etter cleaning: "
            f"min={float(reda_series.min()):.6f}, "
            f"max={float(reda_series.max()):.6f}, "
            f"mean={float(reda_series.mean()):.6f}, "
            f"median={float(reda_series.median()):.6f}"
        )
        print()  # Tom linje for bedre lesbarhet

    # Distance / micro-vol / volume: fyll med 0.0 (nøytral) - MEN IKKE distance_to_range
    for col in ["micro_volatility", "volume_stability"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    metadata = {
        "range_pos_missing_before_fill": missing_rp_before,
        "range_edge_dist_atr_missing_before_fill": missing_reda_before,
    }
    return df, metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("TAG", "PATH"),
        help=(
            "Legg til en run med tag og path. Inkluder RULE5 eller RULE6A i taggen for"
            " å hint om single-policy runs. "
            "Eksempel: --run ADAPTIVE_FULL gx1/wf_runs/.../ADAPTIVE_FULLYEAR"
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="gx1/data/exit_policy",
        help="Katalog der datasets skal lagres.",
    )
    args = ap.parse_args()

    if not args.run:
        raise SystemExit(
            "Du må spesifisere minst én --run TAG PATH\n"
            "Eksempel:\n"
            "  poetry run python gx1/analysis/build_exit_policy_training_dataset_v3.py \\\n"
            "    --run FULLYEAR gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_ADAPTIVE_FULLYEAR"
        )

    run_dirs: Dict[str, Path] = {}
    for tag, path_str in args.run:
        run_dirs[tag] = Path(path_str)

    df, build_metadata = build_dataset_v3(run_dirs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "exit_policy_training_dataset_v3.csv"
    parquet_path = out_dir / "exit_policy_training_dataset_v3.parquet"
    meta_path = out_dir / "exit_policy_training_dataset_v3_metadata.json"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    # Beregn range_pos stats for metadata
    range_pos_stats = {}
    if "range_pos" in df.columns:
        rp_series = df["range_pos"].astype(float)
        range_pos_stats = {
            "min": float(rp_series.min()),
            "max": float(rp_series.max()),
            "mean": float(rp_series.mean()),
            "median": float(rp_series.median()),
        }

    # Beregn range_edge_dist_atr stats for metadata
    range_edge_dist_atr_stats = {}
    if "range_edge_dist_atr" in df.columns:
        reda_series = df["range_edge_dist_atr"].astype(float)
        range_edge_dist_atr_stats = {
            "min": float(reda_series.min()),
            "max": float(reda_series.max()),
            "mean": float(reda_series.mean()),
            "median": float(reda_series.median()),
        }

    meta = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "runs": {tag: str(path) for tag, path in run_dirs.items()},
    }
    if range_pos_stats:
        meta["range_pos_stats"] = range_pos_stats
        meta["range_pos_missing_before_fill"] = build_metadata["range_pos_missing_before_fill"]
    if range_edge_dist_atr_stats:
        meta["range_edge_dist_atr_stats"] = range_edge_dist_atr_stats
        meta["range_edge_dist_atr_missing_before_fill"] = build_metadata["range_edge_dist_atr_missing_before_fill"]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[build_exit_policy_v3] Skrev dataset til:")
    print(f"  - {csv_path}")
    print(f"  - {parquet_path}")
    print(f"  - {meta_path}")


if __name__ == "__main__":
    main()
