# gx1/features/runtime_v9.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# Import feature builders (same as training pipeline)
from gx1.features.basic_v1 import build_basic_v1
from gx1.seq.sequence_features import build_sequence_features


# NB: må matche leakage-guarden i entry_v9_dataset.py
LEAKAGE_SUBSTRINGS = [
    "mfe",
    "mae",
    "net_margin",
    "pnl",
    "label",
    "t_mfe",
    "t_mae",
    "first_hit",
]


class V9RuntimeFeatureError(RuntimeError):
    """Feil relatert til V9 runtime feature-pipeline."""


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sørg for at df har en DatetimeIndex.
    Bruker eksisterende index hvis den allerede er datetime,
    ellers prøver den ts/timestamp/time/datetime/date-kolonner.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    for col in ["ts", "timestamp", "time", "datetime", "date"]:
        if col in df.columns:
            idx = pd.to_datetime(df[col], utc=True, errors="coerce")
            if idx.isna().all():
                continue
            df = df.copy()
            df.index = idx
            log.info("[ENTRY_V9] Runtime: satt DatetimeIndex fra kolonne '%s'", col)
            return df

    raise V9RuntimeFeatureError(
        "[ENTRY_V9] Runtime: fant ingen tidskolonne for DatetimeIndex "
        "(forventet en av: ts/timestamp/time/datetime/date)"
    )


def _load_feature_meta(meta_path: Path) -> Tuple[List[str], List[str]]:
    """
    Leser feature metadata for V9 og returnerer (seq_features, snap_features).

    Støtter flere mulige strukturer:
    - {"features": {"seq": [...], "snap": [...]} }
    - {"seq_features": [...], "snap_features": [...]}
    - {"seq": [...], "snap": [...]}
    """
    if not meta_path.exists():
        raise V9RuntimeFeatureError(
            f"[ENTRY_V9] Runtime: feature_meta-fil finnes ikke: {meta_path}"
        )

    with meta_path.open("r", encoding="utf-8") as f:
        import json

        meta = json.load(f)

    seq_features: Optional[List[str]] = None
    snap_features: Optional[List[str]] = None

    # Fleksibel parsing
    if isinstance(meta, dict):
        if "features" in meta and isinstance(meta["features"], dict):
            feat = meta["features"]
            seq_features = feat.get("seq") or feat.get("seq_features")
            snap_features = feat.get("snap") or feat.get("snap_features")

        if seq_features is None:
            seq_features = (
                meta.get("seq_features") or meta.get("seq_feature_names") or meta.get("seq") or []
            )

        if snap_features is None:
            snap_features = (
                meta.get("snap_features") or meta.get("snap_feature_names") or meta.get("snap") or []
            )

    if not seq_features or not snap_features:
        raise V9RuntimeFeatureError(
            f"[ENTRY_V9] Runtime: klarte ikke å lese seq/snap features fra meta: {meta_path}"
        )

    log.debug(
        "[ENTRY_V9] Runtime: feature_meta lastet: %d seq features, %d snap features",
        len(seq_features),
        len(snap_features),
    )
    return list(seq_features), list(snap_features)


def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dropper kolonner som inneholder leakage-substrings.
    Matcher logikken i ENTRY_V9_CLEAN (entry_v9_dataset leakage guard).
    """
    cols = list(df.columns)
    bad_cols = [
        c
        for c in cols
        for sub in LEAKAGE_SUBSTRINGS
        if sub.lower() in str(c).lower()
    ]

    if bad_cols:
        log.warning(
            "[ENTRY_V9] Runtime: fjerner %d potensielle leakage-kolonner: %s",
            len(bad_cols),
            ", ".join(sorted(set(map(str, bad_cols)))),
        )
        df = df.drop(columns=list(set(bad_cols)), errors="ignore")

    return df


def _load_scaler(path: Optional[Path]):
    if path is None:
        return None
    if joblib is None:
        raise V9RuntimeFeatureError(
            "[ENTRY_V9] Runtime: joblib ikke tilgjengelig, men scaler-path ble gitt"
        )
    if not path.exists():
        raise V9RuntimeFeatureError(
            f"[ENTRY_V9] Runtime: scaler-fil finnes ikke: {path}"
        )
    return joblib.load(path)


def _apply_scalers(
    df_features: pd.DataFrame,
    seq_features: List[str],
    snap_features: List[str],
    seq_scaler_path: Optional[Path],
    snap_scaler_path: Optional[Path],
) -> pd.DataFrame:
    """
    Bruker seq/snap scalere på respektive feature-subsets.

    Forutsetter at scalere ble fittet på de samme kolonnene i training.
    """
    df = df_features.copy()

    # Sequence scaler
    if seq_scaler_path is not None:
        scaler = _load_scaler(seq_scaler_path)
        seq_present = [c for c in seq_features if c in df.columns]
        if seq_present:
            arr = df[seq_present].to_numpy(dtype=np.float32)
            arr_scaled = scaler.transform(arr)
            df.loc[:, seq_present] = arr_scaled
            log.debug(
                "[ENTRY_V9] Runtime: seq_scaler brukt på %d features",
                len(seq_present),
            )
        else:
            log.warning(
                "[ENTRY_V9] Runtime: ingen av seq_features finnes i df ved scaling; "
                "seq_scaler_path=%s",
                seq_scaler_path,
            )

    # Snapshot scaler
    if snap_scaler_path is not None:
        scaler = _load_scaler(snap_scaler_path)
        snap_present = [c for c in snap_features if c in df.columns]
        if snap_present:
            arr = df[snap_present].to_numpy(dtype=np.float32)
            arr_scaled = scaler.transform(arr)
            df.loc[:, snap_present] = arr_scaled
            log.debug(
                "[ENTRY_V9] Runtime: snap_scaler brukt på %d features",
                len(snap_present),
            )
        else:
            log.warning(
                "[ENTRY_V9] Runtime: ingen av snap_features finnes i df ved scaling; "
                "snap_scaler_path=%s",
                snap_scaler_path,
            )

    return df


def build_v9_live_base_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Bygger HELE V9-featurestacket på live/replay-data,
    så tett som mulig på treningspipen (ENTRY_V9_CLEAN).
    
    Input: df_raw med OHLCV + spread + session + etc.
    Output: df_features med alle kolonner som treningspipen forventer.
    
    Dette matcher logikken i entry_v9_dataset.prepare_entry_v9_data():
    1. build_sequence_features() - bygger seq features (atr50, atr_z, body_pct, etc.)
    2. build_basic_v1() - bygger _v1_* tabulære features
    3. Alle numeriske kolonner (minus seq features og metadata) blir snapshot features
    """
    if df_raw is None or len(df_raw) == 0:
        raise V9RuntimeFeatureError(
            "[ENTRY_V9] Runtime: df_raw er tom – kan ikke bygge base features"
        )
    
    df = df_raw.copy()
    
    # Ensure DatetimeIndex
    df = _ensure_datetime_index(df)
    df = df.sort_index()
    
    # Ensure required OHLCV columns exist (normalize case)
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in required_cols:
            col_mapping[col] = col_lower
    
    # Rename columns to lowercase if needed
    if col_mapping:
        df = df.rename(columns=col_mapping)
    
    # Add missing columns with defaults
    if 'volume' not in df.columns:
        df['volume'] = 0.0
        log.warning("[ENTRY_V9] Runtime: 'volume' kolonne manglet, satt til 0.0")
    
    # Ensure 'ts' column exists (required by basic_v1)
    if 'ts' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df['ts'] = df.index
        else:
            raise V9RuntimeFeatureError(
                "[ENTRY_V9] Runtime: mangler 'ts' kolonne og index er ikke DatetimeIndex"
            )
    
    # Ensure 'session' column exists (for build_sequence_features)
    if 'session' not in df.columns:
        # Infer from timestamp
        from gx1.execution.live_features import infer_session_tag
        df['session'] = df.index.map(lambda ts: infer_session_tag(ts))
        log.info("[ENTRY_V9] Runtime: infererte 'session' fra timestamp")
    
    log.debug(
        "[ENTRY_V9] Runtime: bygger base features fra %d rader med kolonner: %s",
        len(df), list(df.columns)[:10]
    )
    
    # 1) Build basic_v1 tabular features (_v1_*) FIRST
    # Note: build_basic_v1 expects 'ts' column and returns (df, newcols)
    # This must be done BEFORE build_sequence_features because atr_regime_id in 
    # build_sequence_features depends on _v1_atr_regime_id from build_basic_v1
    df, _ = build_basic_v1(df)
    log.debug(
        "[ENTRY_V9] Runtime: build_basic_v1 ferdig – shape=%s",
        df.shape
    )
    
    # 2) Build sequence features (atr50, atr_z, body_pct, wick_asym, session_id, etc.)
    # This uses _v1_atr_regime_id from basic_v1 to compute atr_regime_id
    df = build_sequence_features(df)
    log.debug(
        "[ENTRY_V9] Runtime: build_sequence_features ferdig – shape=%s",
        df.shape
    )
    
    # 3) Ensure 'atr' column exists (some features expect it)
    if 'atr' not in df.columns:
        # Compute ATR from True Range
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14, min_periods=1).mean().fillna(0.0)
    
    # 4) Ensure 'mid' column exists (some features expect it)
    if 'mid' not in df.columns:
        df['mid'] = (df['high'] + df['low']) / 2.0
    
    # 5) Ensure 'range' column exists (some features expect it)
    if 'range' not in df.columns:
        df['range'] = df['high'] - df['low']
    
    # 6) Ensure 'ret_1', 'ret_5', 'ret_20' exist (some features expect them)
    if 'ret_1' not in df.columns:
        df['ret_1'] = df['close'].pct_change(1).fillna(0.0)
    if 'ret_5' not in df.columns:
        df['ret_5'] = df['close'].pct_change(5).fillna(0.0)
    if 'ret_20' not in df.columns:
        df['ret_20'] = df['close'].pct_change(20).fillna(0.0)
    
    # 7) Ensure 'rvol_20', 'rvol_60', 'vol_ratio' exist (some features expect them)
    if 'rvol_20' not in df.columns:
        df['rvol_20'] = df['ret_1'].rolling(window=20, min_periods=1).std().fillna(0.0)
    if 'rvol_60' not in df.columns:
        df['rvol_60'] = df['ret_1'].rolling(window=60, min_periods=1).std().fillna(0.0)
    if 'vol_ratio' not in df.columns:
        df['vol_ratio'] = (df['rvol_20'] / (df['rvol_60'] + 1e-8)).fillna(1.0)
    
    # 8) Ensure 'is_EU', 'is_US' exist (some features expect them)
    if 'is_EU' not in df.columns:
        from gx1.execution.live_features import infer_session_tag
        df['is_EU'] = df.index.map(lambda ts: 1 if infer_session_tag(ts) == 'EU' else 0).astype(int)
    if 'is_US' not in df.columns:
        from gx1.execution.live_features import infer_session_tag
        df['is_US'] = df.index.map(lambda ts: 1 if infer_session_tag(ts) == 'US' else 0).astype(int)
    
    # 9) Ensure session tag dummies exist (_v1_session_tag_EU, etc.)
    if '_v1_session_tag_EU' not in df.columns:
        df['_v1_session_tag_EU'] = (df['session'] == 'EU').astype(float)
    if '_v1_session_tag_US' not in df.columns:
        df['_v1_session_tag_US'] = (df['session'] == 'US').astype(float)
    if '_v1_session_tag_OVERLAP' not in df.columns:
        df['_v1_session_tag_OVERLAP'] = (df['session'] == 'OVERLAP').astype(float)
    
    # 10) Ensure 'brain_risk_score' exists (some features expect it, default to 0)
    if 'brain_risk_score' not in df.columns:
        df['brain_risk_score'] = 0.0
    
    # 11) Ensure 'prob_long', 'prob_short', 'prob_neutral', 'side' exist (some features expect them, default to 0)
    if 'prob_long' not in df.columns:
        df['prob_long'] = 0.0
    if 'prob_short' not in df.columns:
        df['prob_short'] = 0.0
    if 'prob_neutral' not in df.columns:
        df['prob_neutral'] = 0.0
    if 'side' not in df.columns:
        df['side'] = ''
    
    log.debug(
        "[ENTRY_V9] Runtime: build_v9_live_base_features ferdig – shape=%s, "
        "%d kolonner",
        df.shape, len(df.columns)
    )
    
    return df


def build_v9_runtime_features(
    df_raw: pd.DataFrame,
    feature_meta_path: Path,
    seq_scaler_path: Optional[Path] = None,
    snap_scaler_path: Optional[Path] = None,
    fillna_value: float = 0.0,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Hoved-entrypoint for V9 runtime features.

    Først bygger vi hele V9-featurestacket med build_v9_live_base_features(),
    deretter subsetter vi til de features modellen forventer og skalerer.

    Returnerer:
        df_feats: DataFrame med KUN seq+snap features som modellen forventer
                  (og samme index som df_raw etter standardisering)
        seq_features: liste over sequence-feature-navn i df_feats
        snap_features: liste over snapshot-feature-navn i df_feats
    """
    if df_raw is None or len(df_raw) == 0:
        raise V9RuntimeFeatureError(
            "[ENTRY_V9] Runtime: df_raw er tom – kan ikke bygge features"
        )

    # 1) Bygg hele V9-featurestacket på live-data (samme som treningspipen)
    df_full = build_v9_live_base_features(df_raw)
    
    df = df_full.copy()
    df = df.sort_index()
    # Fjerner duplikate indekser – bedre enn å kollapse vilkårlig
    if not df.index.is_monotonic_increasing:
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        log.info(
            "[ENTRY_V9] Runtime: fjernet duplikate tidsrader – ny shape=%s",
            df.shape,
        )

    # 2) Dropp leakage kolonner
    df = _drop_leakage_columns(df)

    # 2) Last feature-meta
    seq_features, snap_features = _load_feature_meta(feature_meta_path)

    expected_features = set(seq_features) | set(snap_features)
    present_features = set(df.columns)
    
    # Map case-insensitive: CLOSE -> close, etc.
    # Create a mapping from expected (case-insensitive) to actual column names
    col_lower_map = {col.lower(): col for col in df.columns}
    missing = []
    for exp_feat in expected_features:
        exp_lower = exp_feat.lower()
        if exp_feat not in present_features:
            # Try case-insensitive match
            if exp_lower in col_lower_map:
                # Rename column to match expected case
                actual_col = col_lower_map[exp_lower]
                df = df.rename(columns={actual_col: exp_feat})
                log.debug(
                    "[ENTRY_V9] Runtime: mappet kolonne '%s' -> '%s' (case-insensitive)",
                    actual_col, exp_feat
                )
            else:
                missing.append(exp_feat)
    
    # Re-check after case-insensitive mapping
    present_features = set(df.columns)
    missing = sorted(expected_features - present_features)
    extra = sorted(present_features - expected_features)

    if missing:
        log.error(
            "[ENTRY_V9] Runtime: mangler %d forventede features: %s",
            len(missing),
            ", ".join(missing),
        )
        log.error(
            "[ENTRY_V9] Runtime: ekstra features (ikke forventet): %s",
            ", ".join(sorted(extra)) if extra else "ingen",
        )
        # Hard fail – dette er kritisk for entry-paritet
        # No silent fallbacks, no "continue with what we have"
        raise V9RuntimeFeatureError(
            f"[ENTRY_V9] Runtime: V9 feature mismatch detected!\n"
            f"  Missing {len(missing)} expected features: {sorted(missing)}\n"
            f"  Extra {len(extra)} unexpected features: {sorted(extra)}\n"
            f"  This indicates a mismatch between training and runtime feature pipelines.\n"
            f"  Fix: Ensure build_v9_live_base_features() produces all features from entry_v9_feature_meta.json"
        )

    # Note: extra features are logged above in error message if missing features exist
    # If no missing features, extra features are just informational (not critical)
    if extra and not missing:
        log.debug(
            "[ENTRY_V9] Runtime: %d ekstra kolonner som ikke brukes av modellen (ikke kritisk): %s",
            len(extra),
            ", ".join(extra[:20]) + (" ..." if len(extra) > 20 else ""),
        )

    # 3) Subsett til eksakt feature-set modellen forventer (stabil rekkefølge)
    ordered_cols: List[str] = list(seq_features) + list(snap_features)
    df_feats = df[ordered_cols].copy()
    
    # 3.5) Convert all columns to numeric (handle string columns like 'session', 'side', etc.)
    for col in df_feats.columns:
        if df_feats[col].dtype == 'object':
            # Try to convert to numeric, replace non-numeric with fillna_value
            df_feats[col] = pd.to_numeric(df_feats[col], errors='coerce').fillna(fillna_value)
        # Ensure float32
        df_feats[col] = df_feats[col].astype(np.float32)

    # 4) NaN/inf-håndtering – identisk filosofi som i training
    arr = df_feats.to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=fillna_value, posinf=fillna_value, neginf=fillna_value)
    df_feats.loc[:, :] = arr

    # 5) Optional: apply scalers (hvis V9 bruker lagrede scalere)
    if seq_scaler_path is not None or snap_scaler_path is not None:
        df_feats = _apply_scalers(
            df_feats,
            seq_features=seq_features,
            snap_features=snap_features,
            seq_scaler_path=seq_scaler_path,
            snap_scaler_path=snap_scaler_path,
        )

    log.debug(
        "[ENTRY_V9] Runtime: build_v9_runtime_features ferdig – shape=%s, "
        "%d seq features, %d snap features",
        df_feats.shape,
        len(seq_features),
        len(snap_features),
    )

    return df_feats, seq_features, snap_features
