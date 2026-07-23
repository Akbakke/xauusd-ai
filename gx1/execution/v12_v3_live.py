#!/usr/bin/env python3
"""V12 V3 model-native exit-transformer live inference wrapper.

V3 is the per-M1 exit-signal transformer served only under the exact V8
semantic-V2 contract (173 features × 512 observed M1 bars). For
each M1 bar during an open trade it produces 4 outputs that the Exit-IQL V12.1
state vector expects:

    v3_v8_should_exit_prob       sigmoid(main head)         primary exit signal
    v3_v8_profit_protect_prob    sigmoid(profit-protect)
    v3_v8_family_argmax          argmax(4-class family head)
    v3_v8_family_logit_max       max(family logits)

Shared input-prefix layout across the admitted contracts:
    0-6   :  XGB signal_bridge_v1 (7-dim)              — from XGB v5 at M5 bucket
    7-... :  Mix of canonical features + 19 trade-state slots that get OVERLAID
             for in-trade bars. Pre-trade bars (the 511 historical bars before
             this trade opened) have the 19 trade-state slots = 0 by training
             convention.

This module:
  1. Loads V3 v8 from bundle (matches transformer_config.json)
  2. Builds the exact contract-width input matrix per inference:
     - Reads BASE34 prebuilt (M1 cadence) for canonical + ctx columns
     - Runs XGB v5 on the unique M5 buckets in window → signal_bridge per bar
     - Applies trade-state overlay if a TradeState is provided
  3. Forward passes → 4 outputs

Performance target: < 300 ms per inference. The transformer forward on
(1, 512, 91) takes ~50-100ms; XGB on ~103 M5 buckets ~50ms; assembly ~30ms.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gx1.exits.contracts.exit_io_v6_ctx_v3canonical_m1l512 import (
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES as V6_FEATURES,
)
# V8 semantic V2 makes cadence part of the IO identity: M1-native volume and
# historical per-M1 latest-closed-M5 context. Legacy V6/V7/V8-broadcast
# artifacts are audit-only and are never admitted by this live owner.
from gx1.exits.contracts.exit_io_v8_regime_m1l512 import (
    EXIT_IO_V8_REGIME_M1L512_FEATURES as V8_FEATURES,
    EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT as V8_FEATURE_COUNT,
    EXIT_IO_V8_REGIME_M1L512_IO_VERSION as V8_IO_VERSION,
)
from gx1.policy.exit_transformer_v0 import ExitTransformerV0
from gx1.exits.contracts.registry import get_exit_io_contract
from gx1.exits.training.thin_record_dataset import (
    require_reproducible_v3_training_lineage,
)
from gx1.execution.v12_m1_to_m5_downsample import (
    closed_m5_start_for_m1_bar_labels,
)

LOG = logging.getLogger("v12_v3_live")

def _resolve_default_v3_bundle() -> Path:
    """Resolve the registry-bound V3 artifact with no live environment override."""

    from gx1_guards.artifacts import load_decision_artifact

    return Path(load_decision_artifact("v3_exit"))

# One-truth live admission: only the exact cadence-bound V8 semantic contract.
SUPPORTED_V3_CONTRACTS: dict = {
    V8_IO_VERSION: (list(V8_FEATURES), V8_FEATURE_COUNT),
}
WINDOW_LEN = 512   # 512 observed M1 bars (market gaps mean this is not always 8.5 wall-clock hours)

# Indices of the 7 XGB-bridge features (positions 0..6 in the V6 contract)
XGB_BRIDGE_NAMES = V6_FEATURES[:7]   # ['p_long','p_short','p_flat','p_hat','uncertainty_score','margin_top1_top2','entropy']

# 19 features that get overlaid with trade-state values when in-trade.
# These are 0 by default (pre-trade convention from training).
TRADE_STATE_FEATURE_NAMES = [
    "p_long_entry", "p_hat_entry", "uncertainty_entry", "entropy_entry", "margin_entry",
    "pnl_bps_now", "mfe_bps", "mae_bps", "dd_from_mfe_bps",
    "distance_from_peak_mfe_bps", "bars_held", "time_since_mfe_bars",
    "mfe_decay_rate", "pnl_velocity", "pnl_acceleration",
    "rolling_slope_since_entry", "atr_bps_now",
    "giveback_ratio", "giveback_acceleration",
]


def _v3_m1_window(
    end_ts: pd.Timestamp,
    base34_prebuilt: pd.DataFrame,
) -> pd.DataFrame:
    """Select the exact ordered 512-row M1 window used by V3.

    The live caller must not sort, deduplicate, forward-fill, or substitute an
    older endpoint. Any malformed index or missing decision row is evidence
    that no V3 input exists for this cadence step.
    """
    if not isinstance(base34_prebuilt, pd.DataFrame) or base34_prebuilt.empty:
        raise RuntimeError("V3_M1_SOURCE_EMPTY")
    try:
        observed_index = pd.to_datetime(
            base34_prebuilt.index,
            utc=True,
            errors="coerce",
        )
    except Exception as exc:  # noqa: BLE001 - convert schema failure to contract evidence
        raise RuntimeError("V3_M1_INDEX_INVALID") from exc
    if (
        observed_index.hasnans
        or not observed_index.is_monotonic_increasing
        or not observed_index.is_unique
    ):
        raise RuntimeError(
            "V3_M1_INDEX_INVALID: "
            f"has_nat={observed_index.hasnans} "
            f"monotonic={observed_index.is_monotonic_increasing} "
            f"unique={observed_index.is_unique}"
        )

    end_bucket = pd.Timestamp(end_ts)
    end_bucket = (
        end_bucket.tz_localize("UTC")
        if end_bucket.tzinfo is None
        else end_bucket.tz_convert("UTC")
    ).floor("min")
    matches = int(np.count_nonzero(observed_index == end_bucket))
    if matches != 1:
        raise RuntimeError(
            "V3_M1_EXACT_ENDPOINT_MISSING"
            if matches == 0
            else "V3_M1_EXACT_ENDPOINT_DUPLICATE"
        )

    source = base34_prebuilt.copy(deep=False)
    source.index = observed_index
    win = source.loc[:end_bucket].tail(WINDOW_LEN)
    if len(win) != WINDOW_LEN:
        raise RuntimeError(
            f"V3_M1_HISTORY_MISMATCH: observed={len(win)} required={WINDOW_LEN}"
        )
    if win.index[-1] != end_bucket:
        raise RuntimeError(
            f"V3_M1_ENDPOINT_MISMATCH: observed={win.index[-1]} expected={end_bucket}"
        )
    return win


def _closed_m5_key_per_m1(m1_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map M1 bar-start labels to the newest M5 row closed at M1 availability."""
    return closed_m5_start_for_m1_bar_labels(pd.DatetimeIndex(m1_index))


def required_closed_m5_keys_for_v3_window(
    end_ts: pd.Timestamp,
    base34_prebuilt: pd.DataFrame,
) -> pd.DatetimeIndex:
    """Return the exact ordered unique closed-M5 keys needed by a V3 window.

    An M1 bar labelled ``T`` becomes available at ``T+1 minute``. Therefore its
    newest admissible M5 feature row is
    ``floor(T+1 minute, 5 minutes) - 5 minutes``. Mapping with ``floor(T, 5m)``
    reads a forming M5 candle for four out of five M1 phases.
    """
    win = _v3_m1_window(end_ts, base34_prebuilt)
    per_m1 = _closed_m5_key_per_m1(pd.DatetimeIndex(win.index))
    required = pd.DatetimeIndex(per_m1.unique())
    if (
        required.empty
        or required.hasnans
        or not required.is_monotonic_increasing
        or not required.is_unique
    ):
        raise RuntimeError("V3_REQUIRED_CLOSED_M5_KEYS_INVALID")
    return required


def build_v3_base_feature_rows(
    *,
    target_m1: pd.DataFrame,
    volume_history_m1: pd.DataFrame,
    canonical_v3: pd.DataFrame,
    xgb_inferer: Any,
    feature_names: list[str],
) -> np.ndarray:
    """Build exact pre-trade V3 rows for both dataset materialization and live.

    ``target_m1`` owns the output cadence. ``volume_history_m1`` may include the
    causal prefix required by rolling volume features. All other inputs map
    directly to target rows or to their latest fully closed M5 key. The 19
    trade-state columns are explicitly zero by the pre-trade contract and are
    overlaid later by the shared trade-state owner.
    """

    if not isinstance(target_m1, pd.DataFrame) or target_m1.empty:
        raise RuntimeError("V3_BASE_TARGET_M1_EMPTY")
    if not isinstance(volume_history_m1, pd.DataFrame) or volume_history_m1.empty:
        raise RuntimeError("V3_BASE_VOLUME_HISTORY_EMPTY")
    if not isinstance(canonical_v3, pd.DataFrame) or canonical_v3.empty:
        raise RuntimeError("V3_CANONICAL_WINDOW_EMPTY")
    if (
        not feature_names
        or len(feature_names) != len(set(feature_names))
        or list(feature_names[:7]) != list(XGB_BRIDGE_NAMES)
    ):
        raise RuntimeError("V3_FEATURE_CONTRACT_INVALID")

    def normalized_index(frame: pd.DataFrame, name: str) -> pd.DatetimeIndex:
        try:
            index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{name}_INDEX_INVALID") from exc
        if index.hasnans or not index.is_monotonic_increasing or not index.is_unique:
            raise RuntimeError(f"{name}_INDEX_INVALID")
        return pd.DatetimeIndex(index)

    target = target_m1.copy(deep=False)
    target.index = normalized_index(target, "V3_BASE_TARGET_M1")
    volume_history = volume_history_m1.copy(deep=False)
    volume_history.index = normalized_index(
        volume_history,
        "V3_BASE_VOLUME_HISTORY",
    )
    canonical = canonical_v3.copy(deep=False)
    canonical.index = normalized_index(canonical, "V3_CANONICAL")
    positions = volume_history.index.get_indexer(target.index)
    if np.any(positions < 0) or not np.array_equal(
        positions,
        np.arange(positions[0], positions[0] + len(target), dtype=np.int64),
    ):
        raise RuntimeError("V3_BASE_TARGET_NOT_CONTIGUOUS_IN_VOLUME_HISTORY")

    n_rows = len(target)
    per_m1_closed_m5 = _closed_m5_key_per_m1(target.index)
    required_m5 = pd.DatetimeIndex(per_m1_closed_m5.unique())
    missing_m5 = required_m5.difference(canonical.index)
    if len(missing_m5):
        raise RuntimeError(
            "V3_CANONICAL_CLOSED_M5_COVERAGE_MISSING: "
            f"missing={len(missing_m5)} required={len(required_m5)} "
            f"first_missing={missing_m5[0]}"
        )

    from gx1.features.volume_features import (
        VOLUME_FEATURE_NAMES,
        compute_volume_features,
    )

    matrix = np.empty((n_rows, len(feature_names)), dtype=np.float32)
    filled = np.zeros(len(feature_names), dtype=bool)
    trade_feature_names = set(TRADE_STATE_FEATURE_NAMES)
    volume_feature_names = set(VOLUME_FEATURE_NAMES)

    matrix[:, :7] = 0.0
    filled[:7] = True
    for column_index, feature_name in enumerate(feature_names[7:], start=7):
        if feature_name in trade_feature_names:
            matrix[:, column_index] = 0.0
            filled[column_index] = True
            continue
        if feature_name in volume_feature_names:
            continue
        if feature_name in target.columns:
            if list(target.columns).count(feature_name) != 1:
                raise RuntimeError(f"V3_M1_FEATURE_DUPLICATE: {feature_name}")
            raw = target[feature_name]
            source_name = "M1"
        elif feature_name in canonical.columns:
            if list(canonical.columns).count(feature_name) != 1:
                raise RuntimeError(
                    f"V3_CANONICAL_FEATURE_DUPLICATE: {feature_name}"
                )
            raw = canonical.loc[per_m1_closed_m5, feature_name]
            source_name = "closed-M5"
        else:
            raise RuntimeError(f"V3_ACTIVE_FEATURE_MISSING: {feature_name}")
        try:
            values = pd.to_numeric(raw, errors="raise").to_numpy(
                dtype=np.float64
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"V3_ACTIVE_FEATURE_NOT_NUMERIC: {feature_name} "
                f"source={source_name}"
            ) from exc
        if values.shape != (n_rows,) or not np.isfinite(values).all():
            raise RuntimeError(
                f"V3_ACTIVE_FEATURE_INVALID: {feature_name} "
                f"source={source_name} shape={values.shape}"
            )
        matrix[:, column_index] = values.astype(np.float32)
        filled[column_index] = True

    active_volume_features = [
        name for name in VOLUME_FEATURE_NAMES if name in feature_names
    ]
    if active_volume_features:
        from gx1.features.volume_features import VOLUME_FEATURE_PREFIX_ROWS

        if int(positions[0]) < VOLUME_FEATURE_PREFIX_ROWS:
            raise RuntimeError(
                "V3_VOLUME_HISTORY_PREFIX_MISSING: "
                f"observed={int(positions[0])} "
                f"required={VOLUME_FEATURE_PREFIX_ROWS}"
            )
        if (
            "volume" not in volume_history.columns
            or list(volume_history.columns).count("volume") != 1
        ):
            raise RuntimeError("V3_VOLUME_SOURCE_MISSING_OR_DUPLICATE: volume")
        close_name = (
            "close" if "close" in volume_history.columns else "bid_close"
        )
        if (
            close_name not in volume_history.columns
            or list(volume_history.columns).count(close_name) != 1
        ):
            raise RuntimeError(
                "V3_VOLUME_SOURCE_MISSING_OR_DUPLICATE: close|bid_close"
            )
        volume_values = compute_volume_features(
            pd.DataFrame(
                {
                    "volume": volume_history["volume"].to_numpy(),
                    "close": volume_history[close_name].to_numpy(),
                }
            )
        )
        for feature_name in active_volume_features:
            column_index = feature_names.index(feature_name)
            values = np.asarray(
                volume_values[feature_name],
                dtype=np.float32,
            )[positions]
            if values.shape != (n_rows,) or not np.isfinite(values).all():
                raise RuntimeError(
                    f"V3_VOLUME_FEATURE_INVALID: {feature_name}"
                )
            matrix[:, column_index] = values
            filled[column_index] = True

    m5_input = canonical.loc[required_m5]
    xgb_output = xgb_inferer.predict(m5_input)
    if not isinstance(xgb_output, dict) or "signal_bridge_v1" not in xgb_output:
        raise RuntimeError("V3_XGB_BRIDGE_OUTPUT_MISSING")
    try:
        bridge = np.asarray(
            xgb_output["signal_bridge_v1"],
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("V3_XGB_BRIDGE_OUTPUT_NOT_NUMERIC") from exc
    if bridge.shape != (len(required_m5), 7) or not np.isfinite(bridge).all():
        raise RuntimeError(
            "V3_XGB_BRIDGE_OUTPUT_INVALID: "
            f"shape={bridge.shape} expected=({len(required_m5)}, 7)"
        )
    bridge_positions = required_m5.get_indexer(per_m1_closed_m5)
    if np.any(bridge_positions < 0):
        raise RuntimeError("V3_XGB_BRIDGE_MAPPING_INCOMPLETE")
    matrix[:, :7] = bridge[bridge_positions].astype(np.float32)

    if not filled.all():
        missing = [
            feature_names[index]
            for index in np.flatnonzero(~filled)
        ]
        raise RuntimeError(f"V3_BASE_FEATURES_UNFILLED: {missing}")
    if not np.isfinite(matrix).all():
        raise RuntimeError("V3_INPUT_MATRIX_NONFINITE")
    return matrix


@dataclass
class V3LiveInference:
    bundle_dir: Path
    device: str = "cpu"
    _model: ExitTransformerV0 | None = field(default=None)
    # V12.2 multi-TF: detected from transformer_config.json at load time
    _enable_multi_tf: bool = False
    _mtf_seq_dims: dict = field(default_factory=dict)
    _mtf_seq_lens: dict = field(default_factory=dict)
    # Phase 3b q_head: set when bundle was distilled from Exit-IQL teacher.
    _enable_q_head: bool = False
    # 2026-06-02 fix: contract-aware feature list. V7 (155 feats) vs V6 (91)
    # — build_window MUST consult these instead of hardcoding V6 (which silently
    # broke V3 inference for the entire COSTFIX live era when V7 was loaded).
    _features: list = field(default_factory=list)
    _feature_count: int = 0
    _training_lineage: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        bundle_dir: Path | None = None,
        device: str = "cpu",
    ) -> "V3LiveInference":
        bundle_dir = (
            _resolve_default_v3_bundle()
            if bundle_dir is None
            else Path(bundle_dir)
        )
        cfg_path = bundle_dir / "transformer_config.json"
        state_path = bundle_dir / "exit_transformer_v0.pt"
        if not cfg_path.exists():
            raise FileNotFoundError(f"V3 v8 config missing: {cfg_path}")
        if not state_path.exists():
            raise FileNotFoundError(f"V3 v8 weights missing: {state_path}")

        cfg = json.loads(cfg_path.read_text())
        # Cadence semantics are versioned with the feature layout. An old
        # broadcast-context artifact must fail before weights are loaded.
        io_version = cfg.get("exit_ml_io_version")
        if io_version not in SUPPORTED_V3_CONTRACTS:
            raise RuntimeError(
                f"V3 io_version {io_version!r} not in supported: "
                f"{sorted(SUPPORTED_V3_CONTRACTS)}"
            )
        _expected_features, _expected_count = SUPPORTED_V3_CONTRACTS[io_version]
        registry_contract = get_exit_io_contract(io_version)
        if int(cfg["input_dim"]) != _expected_count:
            raise RuntimeError(
                f"V3 input_dim={cfg['input_dim']} != contract {io_version}={_expected_count}"
            )
        if (
            type(cfg.get("feature_count")) is not int
            or cfg["feature_count"] != _expected_count
            or cfg.get("feature_names_hash") != registry_contract["feature_hash"]
        ):
            raise RuntimeError(
                "V3 config feature identity differs from the selected IO contract"
            )
        if int(cfg["window_len"]) != WINDOW_LEN:
            raise RuntimeError(f"V3 window_len={cfg['window_len']} != {WINDOW_LEN}")
        training_lineage = require_reproducible_v3_training_lineage(
            bundle_dir=bundle_dir,
            config=cfg,
            state_path=state_path,
        )
        LOG.info(f"V3 contract={io_version} input_dim={_expected_count}")

        # V12.2: detect multi-TF mode from config (v8 bundles default enabled=False).
        mtf_cfg = cfg.get("multi_tf", {}) or {}
        enable_mtf = bool(mtf_cfg.get("enabled", False))
        # V12.2 hard-fail: live runtime REQUIRES multi-TF V3 bundles.
        # Single-TF V3 bundles (EXIT_V6/V7/V8_DISK_*) are pre-cement and not validated.
        if not enable_mtf:
            raise RuntimeError(
                f"V3 bundle is NOT multi-TF (transformer_config.json missing or "
                f"multi_tf.enabled != true). V12.2 live REQUIRES multi-TF. "
                f"Bundle: {bundle_dir}"
            )
        mtf_kwargs = {}
        if enable_mtf:
            mtf_kwargs = dict(
                enable_multi_tf=True,
                m5_seq_dim=int(mtf_cfg["m5_seq_dim"]),
                m15_seq_dim=int(mtf_cfg["m15_seq_dim"]),
                h1_seq_dim=int(mtf_cfg["h1_seq_dim"]),
                h4_seq_dim=int(mtf_cfg["h4_seq_dim"]),
                d1_seq_dim=int(mtf_cfg["d1_seq_dim"]),
                m5_seq_len=int(mtf_cfg["m5_seq_len"]),
                m15_seq_len=int(mtf_cfg["m15_seq_len"]),
                h1_seq_len=int(mtf_cfg["h1_seq_len"]),
                h4_seq_len=int(mtf_cfg["h4_seq_len"]),
                d1_seq_len=int(mtf_cfg["d1_seq_len"]),
            )
            LOG.info("V3 bundle is multi-TF: M5/M15/H1/H4/D1 active")

        # Phase 3b: bundle was distilled with enable_q_head=True iff the config
        # carries the flag. Falls back to state_dict probe for older bundles.
        state_dict = torch.load(state_path, map_location=device, weights_only=True)
        enable_q_head = bool(cfg.get("enable_q_head", False)) or ("q_head.weight" in state_dict)
        if enable_q_head:
            LOG.info("V3 q_head (distilled) detected — predict() will emit v3_q_per_action")

        model = ExitTransformerV0(
            input_dim=cfg["input_dim"], window_len=cfg["window_len"],
            d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            dropout=cfg.get("dropout", 0.1),
            enable_q_head=enable_q_head,
            enable_pos_enc=bool(cfg.get("enable_pos_enc", False)),
            enable_dip_head=bool(cfg.get("enable_dip_head", False)) or ("dip_head.weight" in state_dict),
            # 2026-05-26 new aux heads — detect from cfg or state_dict so strict load matches.
            enable_timing_head=bool(cfg.get("enable_timing_head", False)) or ("timing_head.weight" in state_dict),
            enable_tail_risk_head=bool(cfg.get("enable_tail_risk_head", False)) or ("tail_risk_head.weight" in state_dict),
            enable_vol_forecast_head=bool(cfg.get("enable_vol_forecast_head", False)) or ("vol_forecast_head.weight" in state_dict),
            enable_forecast_head=bool(cfg.get("enable_forecast_head", False)) or ("forecast_head.weight" in state_dict),
            **mtf_kwargs,
        )
        model.load_state_dict(state_dict)
        model.to(device).eval()
        LOG.info(f"V3 v9 (multi-TF) loaded: {bundle_dir.name}  device={device}  "
                  f"input_dim={cfg['input_dim']}  window_len={cfg['window_len']}  multi_tf={enable_mtf}")
        # 2026-06-02 fix: store contract-aware feature list so build_window uses
        # V7 (155) when V7 bundle, V6 (91) when V6 bundle. Hardcoded V6 paths
        # silently failed for the entire COSTFIX live era.
        return cls(
            bundle_dir=bundle_dir, device=device, _model=model,
            _enable_multi_tf=enable_mtf,
            _enable_q_head=enable_q_head,
            _mtf_seq_dims={k: int(mtf_cfg.get(f"{k.lower()}_seq_dim", 0))
                            for k in ("M5", "M15", "H1", "H4", "D1")} if enable_mtf else {},
            _mtf_seq_lens={k: int(mtf_cfg.get(f"{k.lower()}_seq_len", 96))
                            for k in ("M5", "M15", "H1", "H4", "D1")} if enable_mtf else {},
            _features=list(_expected_features),
            _feature_count=int(_expected_count),
            _training_lineage=training_lineage,
        )

    @classmethod
    def load_default(cls) -> "V3LiveInference":
        return cls.load()

    # ── feature-matrix construction ─────────────────────────────────────

    def build_window(
        self,
        end_ts: pd.Timestamp,
        base34_prebuilt: pd.DataFrame,
        xgb_inferer,                                  # XGBLiveInference
        canonical_v3_window: pd.DataFrame | None = None,
        trade_overlay: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Assemble the exact contract-width input matrix for V3.

        Args:
            end_ts: timestamp of the decision M1 bar (the LAST row in the window).
            base34_prebuilt: M1-cadence prebuilt with 56 columns (BASE34_CTX16CAT6).
                Must cover at least [end_ts - 511 min, end_ts].
            xgb_inferer: XGBLiveInference instance for computing signal_bridge per
                unique M5 bucket in window.
            canonical_v3_window: M5-cadence frame containing every exact,
                closed-M5 key required by the selected M1 window.
            trade_overlay: dict of 19 (window-len,) arrays with trade-state values
                for in-trade portion of the window. If None, those features stay 0.

        Returns: (512, contract feature count) float32 numpy array.
        """
        win = _v3_m1_window(end_ts, base34_prebuilt)

        # 2026-06-02 fix: use the contract's actual feature list (V6=91 or V7=155)
        # instead of hardcoded V6. Hardcoded V6 silently broke V7 inference for
        # the entire COSTFIX live era.
        if not self._features or self._feature_count <= 0:
            raise RuntimeError("V3_FEATURE_CONTRACT_UNINITIALIZED")
        _feat_list = self._features
        _feat_count = self._feature_count
        if len(_feat_list) != _feat_count or len(set(_feat_list)) != _feat_count:
            raise RuntimeError(
                "V3_FEATURE_CONTRACT_INVALID: "
                f"names={len(_feat_list)} unique={len(set(_feat_list))} "
                f"count={_feat_count}"
            )
        if list(_feat_list[:7]) != list(XGB_BRIDGE_NAMES):
            raise RuntimeError("V3_XGB_BRIDGE_PREFIX_INVALID")

        if canonical_v3_window is None or not isinstance(
            canonical_v3_window,
            pd.DataFrame,
        ):
            raise RuntimeError("V3_CANONICAL_WINDOW_REQUIRED")
        from gx1.features.volume_features import (
            VOLUME_FEATURE_NAMES,
            VOLUME_FEATURE_PREFIX_ROWS,
        )

        source = base34_prebuilt.copy(deep=False)
        source.index = pd.to_datetime(source.index, utc=True, errors="coerce")
        volume_source = source.loc[:win.index[-1]].tail(
            WINDOW_LEN + VOLUME_FEATURE_PREFIX_ROWS
        )
        active_volume_features = set(_feat_list).intersection(VOLUME_FEATURE_NAMES)
        if active_volume_features:
            expected_volume_rows = WINDOW_LEN + VOLUME_FEATURE_PREFIX_ROWS
            if len(volume_source) != expected_volume_rows:
                raise RuntimeError(
                    "V3_VOLUME_HISTORY_MISMATCH: "
                    f"observed={len(volume_source)} required={expected_volume_rows}"
                )
        else:
            volume_source = win
        mat = build_v3_base_feature_rows(
            target_m1=win,
            volume_history_m1=volume_source,
            canonical_v3=canonical_v3_window,
            xgb_inferer=xgb_inferer,
            feature_names=_feat_list,
        )

        # Apply the complete contract-owned trade overlay if provided.
        active_trade_features = [
            name for name in TRADE_STATE_FEATURE_NAMES if name in _feat_list
        ]
        if trade_overlay is not None:
            missing_overlay = [
                name for name in active_trade_features if name not in trade_overlay
            ]
            if missing_overlay:
                raise RuntimeError(
                    f"V3_TRADE_OVERLAY_FEATURE_MISSING: {missing_overlay[:5]}"
                )
            for fname in active_trade_features:
                try:
                    overlay_arr = np.asarray(trade_overlay[fname], dtype=np.float64)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(
                        f"V3_TRADE_OVERLAY_NOT_NUMERIC: {fname}"
                    ) from exc
                if (
                    overlay_arr.ndim != 1
                    or not 1 <= len(overlay_arr) <= WINDOW_LEN
                    or not np.isfinite(overlay_arr).all()
                ):
                    raise RuntimeError(
                        f"V3_TRADE_OVERLAY_INVALID: {fname} shape={overlay_arr.shape}"
                    )
                col_idx = _feat_list.index(fname)
                mat[-len(overlay_arr):, col_idx] = overlay_arr.astype(np.float32)

        if not np.isfinite(mat).all():
            raise RuntimeError("V3_INPUT_MATRIX_NONFINITE")

        return mat

    # ── inference ────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_from_matrix(self, window: np.ndarray,
                              multi_tf_windows: dict | None = None) -> dict[str, Any]:
        """Forward V3 on an exact contract-width input matrix."""
        if self._model is None:
            raise RuntimeError("V3 v8 not loaded — call .load() first")
        if not self._features or self._feature_count <= 0:
            raise RuntimeError("V3 feature contract is uninitialized")
        _expected_cnt = self._feature_count
        if window.shape != (WINDOW_LEN, _expected_cnt):
            raise RuntimeError(f"window shape {window.shape} != ({WINDOW_LEN}, {_expected_cnt})")
        if not np.isfinite(window).all():
            raise RuntimeError("V3 input window contains non-finite values")

        x = torch.from_numpy(window).unsqueeze(0).to(self.device)

        # V12.2: build multi-TF kwargs if model needs them
        mtf_kwargs = {}
        if self._enable_multi_tf:
            if multi_tf_windows is None:
                raise RuntimeError(
                    "V3 bundle is multi-TF — caller must pass multi_tf_windows "
                    "(via PrebuiltStateLoader.get_multi_tf_windows())"
                )
            for k in ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1"):
                if k not in multi_tf_windows:
                    raise RuntimeError(f"multi-TF V3 needs {k} but missing from multi_tf_windows")
                arr = multi_tf_windows[k]
                if arr.ndim == 2:
                    arr = arr[np.newaxis, :, :]
                mtf_kwargs[k] = torch.from_numpy(arr.astype(np.float32, copy=False)).to(self.device)

        main_logit = self._model.forward_logits(x, **mtf_kwargs).item()
        pp_logit = self._model.forward_profit_protect_logits(x, **mtf_kwargs).item()
        family_logits = self._model.forward_family_logits(x, **mtf_kwargs).cpu().numpy().flatten()
        result = {
            "v3_v8_should_exit_prob": float(_sigmoid(main_logit)),
            "v3_v8_profit_protect_prob": float(_sigmoid(pp_logit)),
            "v3_v8_family_argmax": int(np.argmax(family_logits)),
            "v3_v8_family_logit_max": float(np.max(family_logits)),
        }
        # Phase 3b q_head: emit Exit-IQL-distilled Q-values when bundle has q_head.
        if self._enable_q_head:
            q = self._model.forward_q_per_action(x, **mtf_kwargs).cpu().numpy()[0].astype(float)
            q_hold, q_exit = float(q[0]), float(q[1])
            result["v3_q_per_action_v1"] = [q_hold, q_exit]
            result["v3_q_hold_v1"] = q_hold
            result["v3_q_exit_v1"] = q_exit
            result["v3_q_advantage_v1"] = q_exit - q_hold
            result["v3_q_action_id_v1"] = 1 if q_exit > q_hold else 0
        return result

    def predict(
        self,
        end_ts: pd.Timestamp,
        base34_prebuilt: pd.DataFrame,
        canonical_v3_window: pd.DataFrame,
        xgb_inferer,
        trade_overlay: dict[str, np.ndarray] | None = None,
        multi_tf_windows: dict | None = None,
    ) -> dict[str, Any]:
        """One-shot: build window + forward V3 v8 (or v9 with multi-TF)."""
        window = self.build_window(
            end_ts, base34_prebuilt, xgb_inferer,
            canonical_v3_window=canonical_v3_window,
            trade_overlay=trade_overlay,
        )
        return self.predict_from_matrix(window, multi_tf_windows=multi_tf_windows)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))
