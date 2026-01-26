#!/usr/bin/env python3
"""
build_exit_dataset_v1.py

Bygger et exit-datasett for SNIPER P4.1 basert på trade-log fra replay.

Mål:
- Bruke eksisterende P4/P4.1-tradelogger (entry_v9 + exit_v3) som grunnlag
- Lage EXIT-labels (EXIT_NOW / SCALP_PROFIT / HOLD) basert på PnL + MFE/MAE
- Ekstrahere robuste features fra det vi allerede logger:
  - trend_regime, vol_regime, session
  - pnl_bps, mfe_bps, mae_bps, bars_held
  - entry_time, entry_hour, entry_dow
  - ATR/spread ved entry (hvis tilgjengelig)

Output:
- Parquet-fil med én rad per trade, klar for trening av ExitCritic V1

Bruk:
  python build_exit_dataset_v1.py \
    --trade-log-glob "runs/replay_shadow/SNIPER_P4_1/trade_log_entry_v9_exit_v3_adaptive_3SEG_merged*.csv" \
    --output-path "data/rl/exit_critic_dataset_p4_1_v1.parquet"
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# -------------------------
# Konfigurasjon / thresholds
# -------------------------

# Definerer hva vi regner som "stygg tail"
TAIL_LOSS_BPS = -60.0

# Minimum MFE (maks gunstig avvik) som indikerer at vi hadde
# en reell sjanse til å komme oss mye bedre ut enn vi gjorde
MIN_MFE_FOR_BETTER_EXIT_BPS = 10.0

# Hvor mye dårligere endelig PnL kan være enn MFE før vi sier
# "her burde vi ha sikret mye tidligere"
MIN_MFE_MINUS_PNL_FOR_SCALP = 8.0

# Vi bryr oss mest om trades som faktisk var "levende" en stund
MIN_BARS_HELD_FOR_LEARNING = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exit-critic dataset from SNIPER trade logs.")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="runs/replay_shadow/SNIPER_P4_1/**/*.csv",
        help="Glob pattern for trade log CSV-filer (default: runs/replay_shadow/SNIPER_P4_1/**/*.csv)",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default="data/rl/exit_critic/exit_critic_dataset_v1.parquet",
        help="Sti til output parquet-fil (default: data/rl/exit_critic/exit_critic_dataset_v1.parquet)",
    )
    parser.add_argument(
        "--min-bars-held",
        type=int,
        default=MIN_BARS_HELD_FOR_LEARNING,
        help=f"Min antall bars held for å inkludere trade (default={MIN_BARS_HELD_FOR_LEARNING}).",
    )
    return parser.parse_args()


def _find_files(glob_pattern: str) -> List[Path]:
    from glob import glob
    # Use glob for recursive patterns (**)
    matches = glob(glob_pattern, recursive=True)
    paths = sorted([Path(p) for p in matches if Path(p).is_file()])
    return paths


def _safe_get(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    """Returner kolonne hvis den finnes, ellers fyll med default."""
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _parse_datetime(series: pd.Series) -> Optional[pd.Series]:
    """Prøver å parse en datetime-serie; returnerer None hvis kolonnen mangler."""
    if series is None:
        return None
    try:
        return pd.to_datetime(series, errors="coerce", utc=True)
    except Exception:
        return None


def _infer_exit_label(row: pd.Series) -> str:
    """
    Heuristisk EXIT-label:

    EXIT_NOW (label=2):
        - trade ender med svært dårlig PnL (pnl_bps <= -80) eller
        - trade endte i SL (exit_reason inneholder SL_)
        -> vi burde ha kommet oss ut langt tidligere enn faktisk SL.

    SCALP_PROFIT (label=1):
        - slutt-PnL er >= 40 bps (current_pnl_bps >= 40)
        -> dette er kandidater hvor vi burde ha sikret mer aggressivt.
        (Note: uten MFE-data, bruker vi kun pnl >= 40 som proxy)

    HOLD (label=0):
        - resten: enten grei utnyttelse, eller ingen reell bedre exit-mulighet.
    """
    pnl = row.get("pnl_bps", 0.0)
    mfe = row.get("mfe_bps", 0.0)
    exit_reason = str(row.get("exit_reason", "") or "")

    # Beskytt mot NaN
    if pd.isna(pnl):
        return "HOLD"

    # 1) EXIT_NOW: Svært dårlig PnL eller SL
    # Hvis vi har MFE-data, sjekk at vi hadde mulighet til bedre exit
    if (pnl <= -80.0) or ("SL" in exit_reason.upper()):
        if mfe >= MIN_MFE_FOR_BETTER_EXIT_BPS or mfe == 0.0:  # mfe==0.0 betyr manglende data, tillat allikevel
            return "EXIT_NOW"

    # 2) SCALP_PROFIT: Høy gevinst
    # Uten MFE-data, bruker vi kun pnl >= 40 som proxy for "burde ha sikret mer"
    if pnl >= 40.0:
        # Hvis vi har MFE-data, sjekk at vi ga tilbake mye
        if mfe > 0.0:
            if (mfe - pnl) >= MIN_MFE_MINUS_PNL_FOR_SCALP:
                return "SCALP_PROFIT"
        else:
            # Uten MFE-data, bruk pnl >= 40 som proxy
            return "SCALP_PROFIT"

    # 3) Resten er nøytrale/ok
    return "HOLD"


def _encode_label(label: str) -> int:
    """Map tekstlabel til heltall for ML."""
    mapping = {
        "HOLD": 0,
        "SCALP_PROFIT": 1,
        "EXIT_NOW": 2,
    }
    return mapping.get(label, 0)


def build_exit_dataset(
    glob_pattern: str,
    output_path: Path,
    min_bars_held: int = MIN_BARS_HELD_FOR_LEARNING,
) -> pd.DataFrame:
    logging.info("Leter etter trade-logger med pattern: %s", glob_pattern)
    files = _find_files(glob_pattern)
    if not files:
        raise FileNotFoundError(f"Ingen trade-log CSV-filer funnet for pattern: {glob_pattern}")

    logging.info("Fant %d filer", len(files))
    dfs = []
    for path in files:
        logging.info("Leser %s", path)
        df_part = pd.read_csv(path)
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)
    logging.info("Total trades (rå): %d", len(df))

    # -------------------------
    # Grunnleggende rens
    # -------------------------

    # Filtrer til trades som faktisk er lukket, hvis vi har slik kolonne
    if "left_open" in df.columns:
        before = len(df)
        df = df[df["left_open"] == 0].copy()
        logging.info("Filtrert left_open==0: %d -> %d", before, len(df))

    # Filtrer på min bars_held
    if "bars_held" in df.columns:
        before = len(df)
        df = df[df["bars_held"] >= min_bars_held].copy()
        logging.info(
            "Filtrert bars_held >= %d: %d -> %d",
            min_bars_held,
            before,
            len(df),
        )
    else:
        logging.warning("bars_held-kolonne mangler; hopper over bars-filter.")

    # Sørg for at sentrale kolonner finnes
    if "pnl_bps" not in df.columns:
        raise KeyError("Forventer kolonne 'pnl_bps' i trade-log for å bygge exit-datasett.")
    
    # MFE/MAE kan mangle - sett defaults hvis de ikke finnes
    if "mfe_bps" not in df.columns:
        logging.warning("mfe_bps kolonne mangler - setter til 0.0 (vil påvirke label-heuristikk)")
        df["mfe_bps"] = 0.0
    if "mae_bps" not in df.columns:
        logging.warning("mae_bps kolonne mangler - setter til 0.0 (vil påvirke label-heuristikk)")
        df["mae_bps"] = 0.0

    # Parse timestamps hvis vi har dem
    entry_time = _parse_datetime(df["entry_time"]) if "entry_time" in df.columns else None
    exit_time = _parse_datetime(df["exit_time"]) if "exit_time" in df.columns else None

    if entry_time is not None:
        df["entry_time_dt"] = entry_time
        df["entry_hour"] = entry_time.dt.hour
        df["entry_dow"] = entry_time.dt.dayofweek  # 0=mandag
    else:
        logging.warning("entry_time-kolonne mangler; lager ikke time-of-day features.")

    if exit_time is not None:
        df["exit_time_dt"] = exit_time
    else:
        logging.warning("exit_time-kolonne mangler; exit_time_dt settes ikke.")

    # -------------------------
    # Feature-engineering
    # -------------------------

    # Defensive: konverter sentrale numeriske kolonner til float
    for col in ["pnl_bps", "mfe_bps", "mae_bps", "bars_held"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ATR/spread ved entry hvis tilgjengelig
    df["atr_bps_entry"] = _safe_get(df, "atr_bps_entry", np.nan)
    df["spread_bps_entry"] = _safe_get(df, "spread_bps_entry", np.nan)

    # RR-type features
    eps = 1e-6
    df["mae_abs"] = df["mae_bps"].abs()
    df["mfe_abs"] = df["mfe_bps"].abs()
    df["rr_mfe_over_mae"] = df["mfe_abs"] / (df["mae_abs"] + eps)

    # Session/Regime som kategoriske features
    if "session" in df.columns:
        df["session_cat"] = df["session"].astype("category")
    else:
        df["session_cat"] = "UNKNOWN"

    if "trend_regime" in df.columns:
        df["trend_regime_cat"] = df["trend_regime"].astype("category")
    else:
        df["trend_regime_cat"] = "UNKNOWN"

    if "vol_regime" in df.columns:
        df["vol_regime_cat"] = df["vol_regime"].astype("category")
    else:
        df["vol_regime_cat"] = "UNKNOWN"

    # -------------------------
    # Labels
    # -------------------------
    logging.info("Genererer exit-labels...")
    df["exit_label"] = df.apply(_infer_exit_label, axis=1)
    df["exit_label_id"] = df["exit_label"].apply(_encode_label)

    # -------------------------
    # Kolonneutvalg for ML
    # -------------------------

    base_cols = [
        "trade_id",
        "pnl_bps",
        "mfe_bps",
        "mae_bps",
        "bars_held",
        "atr_bps_entry",
        "spread_bps_entry",
        "rr_mfe_over_mae",
        "session",
        "trend_regime",
        "vol_regime",
        "session_cat",
        "trend_regime_cat",
        "vol_regime_cat",
        "exit_label",
        "exit_label_id",
    ]

    time_cols = [
        "entry_time_dt",
        "exit_time_dt",
        "entry_hour",
        "entry_dow",
    ]

    # Ta kun de kolonnene som faktisk finnes
    selected_cols: List[str] = []
    for col in base_cols + time_cols:
        if col in df.columns:
            selected_cols.append(col)

    dataset = df[selected_cols].copy()

    # -------------------------
    # Logging av label-distribusjon
    # -------------------------
    label_counts = dataset["exit_label"].value_counts(dropna=False)
    logging.info("Label-distribution:\n%s", label_counts.to_string())

    # -------------------------
    # Lagre
    # -------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Lagrer exit-datasett til %s", output_path)
    dataset.to_parquet(output_path, index=False)

    logging.info("Ferdig. Antall trades i datasett: %d", len(dataset))
    return dataset


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args()
    build_exit_dataset(
        glob_pattern=args.input_glob,
        output_path=Path(args.output_parquet),
        min_bars_held=args.min_bars_held,
    )


if __name__ == "__main__":
    main()
