#!/usr/bin/env python3
"""LIVE model-native seq513 Entry state builder.

Builds the per-M5-bar model-native state — seq (96, 513) + snap (513)
+ ctx_cont (142) + ctx_cat (5) — from the LIVE cv3/BASE28 prebuilts, bit-parity-exact
with the offline dataset builder and the active bundle's model_native_state_contract.

This module is the sole live model-native Entry state builder. Every formula is
imported from the offline one-truth builders; nothing is re-derived here:

  - group-A + dip/struct (60)   -> gx1.scripts.augment_forward_outcome_v2.attach_group_a_dip_struct_ctx_columns
  - volume features (4)         -> gx1.features.volume_features.add_volume_features
  - entry smart ctx (19)        -> gx1.features.entry_smart_context.add_entry_smart_context_features
  - 479 extension signals       -> gx1.scripts.build_entry_v10_ctx_training_dataset_v3._build_inline_seq_structure_extension
  - all other source columns    -> live cv3+BASE28 prebuilts (PrebuiltStateLoader.get_window),
                                   which must come from an admitted immutable
                                   snapshot publisher using the same one-truth
                                   augmenters as the offline chain. No such
                                   live-tail publisher is currently admitted.

TRAIN==SERVE FRAME CONVENTION (critical): every TRAIN/VAL/TEST build and serving
decision starts causal feature construction at the bundle's single immutable
``feature_history_start_utc``.  Validation and test never reset rolling state.
The categorical ATR/spread ranks are digitized for every row against one frozen
TRAIN-only ECDF artifact whose fit ends exactly at ``rank_fit_end_utc``.  That
artifact contains distributions only: no timestamps, per-row categories, test
state, or pinned ATR values.  Group-A, structure, volume, session, price-action
and all specialist layers remain genuine model evidence and are recomputed over
the common causal frame.  Immutable serve parity remains mandatory.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CAT_MIN_MAX,
    MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_CTX_CONT_ENTRY_SMART_DERIVED_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
    MODEL_NATIVE_CTX_CONT_MICRO_FIELDS,
    MODEL_NATIVE_CTX_CONT_SESSION_FIELDS,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_state_v2 import (
    TrainRankReferenceV2,
    apply_train_rank_reference_v2,
    load_train_rank_reference_v2,
    validate_state_contract_metadata_v2,
)
from gx1.features.volume_features import VOLUME_FEATURE_NAMES
from gx1.features.swing_structure_v1 import (
    SWING_ATR_PERIOD_V1,
    SWING_LOOKBACK_V1,
    compute_swing_structure_features,
)

LOG = logging.getLogger("v12_model_native_state_live")

SEQ_LEN_MODEL_NATIVE = MODEL_NATIVE_SEQ_LEN
SIGNAL_DIM_MODEL_NATIVE = MODEL_NATIVE_SIGNAL_DIM
CTX_CONT_DIM_MODEL_NATIVE = MODEL_NATIVE_CTX_CONT_DIM
CTX_CAT_DIM_MODEL_NATIVE = MODEL_NATIVE_CTX_CAT_DIM

if len(MODEL_NATIVE_CTX_CONT_FIELDS) != CTX_CONT_DIM_MODEL_NATIVE:
    raise RuntimeError(
        "MODEL_NATIVE_CTX_CONT_CONTRACT_MISMATCH: "
        f"ordered={len(MODEL_NATIVE_CTX_CONT_FIELDS)} expected={CTX_CONT_DIM_MODEL_NATIVE}; "
        "the full regime surface is unconditional"
    )
if len(MODEL_NATIVE_CTX_CAT_FIELDS) != CTX_CAT_DIM_MODEL_NATIVE:
    raise RuntimeError(
        "MODEL_NATIVE_CTX_CAT_CONTRACT_MISMATCH: "
        f"ordered={len(MODEL_NATIVE_CTX_CAT_FIELDS)} expected={CTX_CAT_DIM_MODEL_NATIVE}; "
        "the full regime surface is unconditional"
    )

# Columns the temp source-parquet must carry for the extension's price/candle
# layers (they read by NAME from the parquet):
#   _build_price_derived_layer  -> time + close + atr
#   _build_candlestick_*        -> time/open/high/low/close
_SOURCE_PARQUET_COLS = ["time", "close", "atr", "open", "high", "low"]

# Semantic domains of the exact five categorical Entry inputs.  These are
# narrower than the transformer's embedding capacity by design: spare
# embedding rows are not permission to serve malformed context.  Keep this
# mapping in the same order as the model-native context contract.
_MODEL_NATIVE_CTX_CAT_DOMAINS = MODEL_NATIVE_CTX_CAT_MIN_MAX
if tuple(_MODEL_NATIVE_CTX_CAT_DOMAINS) != tuple(MODEL_NATIVE_CTX_CAT_FIELDS):
    raise RuntimeError(
        "MODEL_NATIVE_CTX_CAT_DOMAIN_ORDER_MISMATCH: "
        f"domains={list(_MODEL_NATIVE_CTX_CAT_DOMAINS)} "
        f"contract={list(MODEL_NATIVE_CTX_CAT_FIELDS)}"
    )

_ENTRY_SESSION_CONT_DOMAINS: dict[str, tuple[int, int] | None] = {
    "is_ASIA": (0, 1),
    "minutes_since_session_open": None,
    "minutes_to_next_session_boundary": None,
    "session_change_flag": (0, 1),
    "session_tradable": (0, 1),
}


def _require_model_native_entry_context_frame(
    frame: pd.DataFrame,
    *,
    context: str,
) -> None:
    """Fail Entry closed unless categorical/session context is exact.

    The shared session detector intentionally retains historical behavior for
    non-Entry consumers.  Entry does not inherit its unknown-label-to-ASIA
    convenience: timestamps, labels, semantic category domains and derived
    session values are checked here before extension construction or inference.
    """
    required = [
        "time",
        *MODEL_NATIVE_CTX_CAT_FIELDS,
        *_ENTRY_SESSION_CONT_DOMAINS,
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(
            "[MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION] "
            f"{context}: missing categorical/session fields: {missing}"
        )

    times = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True, errors="coerce"))
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise RuntimeError(
            "[MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION] "
            f"{context}: timestamps must be finite, unique and chronological"
        )

    categorical: dict[str, np.ndarray] = {}
    for name, (lower, upper) in _MODEL_NATIVE_CTX_CAT_DOMAINS.items():
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(
                "[MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION] "
                f"{context}: {name} contains missing/non-finite values"
            )
        rounded = np.rint(values)
        if not np.array_equal(values, rounded):
            raise RuntimeError(
                "[MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION] "
                f"{context}: {name} contains non-integral category values"
            )
        exact = rounded.astype(np.int64)
        if ((exact < lower) | (exact > upper)).any():
            observed = sorted(set(exact[(exact < lower) | (exact > upper)].tolist()))
            raise RuntimeError(
                "[MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION] "
                f"{context}: {name} outside semantic domain [{lower}, {upper}]: "
                f"{observed[:10]}"
            )
        categorical[name] = exact

    # Derive the expected label without get_session_id_vectorized's retained
    # fillna(0), so an unknown label can never become ASIA at this boundary.
    from gx1.time.session_detector import (
        SESSION_ID_MAP,
        get_session_minutes_since_open_vectorized,
        get_session_minutes_to_next_boundary_vectorized,
        get_session_vectorized,
    )

    from gx1.time.session_detector import m5_decision_availability

    decision_times = m5_decision_availability(times)
    labels = get_session_vectorized(decision_times)
    if labels.isna().any() or not labels.isin(tuple(SESSION_ID_MAP)).all():
        raise RuntimeError(
            "[MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION] "
            f"{context}: session label unavailable; ASIA fallback forbidden"
        )
    expected_session = labels.map(SESSION_ID_MAP).to_numpy(dtype=np.int64)
    if not np.array_equal(categorical["session_id"], expected_session):
        mismatch = int(np.flatnonzero(categorical["session_id"] != expected_session)[0])
        raise RuntimeError(
            "[MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION] "
            f"{context}: session_id disagrees with UTC timestamp at row {mismatch}; "
            "ASIA fallback forbidden"
        )

    session_values: dict[str, np.ndarray] = {}
    for name, domain in _ENTRY_SESSION_CONT_DOMAINS.items():
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(
                "[MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION] "
                f"{context}: {name} contains missing/non-finite values"
            )
        if domain is not None:
            lower, upper = domain
            rounded = np.rint(values)
            if not np.array_equal(values, rounded):
                raise RuntimeError(
                    "[MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION] "
                    f"{context}: {name} contains non-integral flag values"
                )
            if ((rounded < lower) | (rounded > upper)).any():
                raise RuntimeError(
                    "[MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION] "
                    f"{context}: {name} outside semantic domain [{lower}, {upper}]"
                )
        session_values[name] = values

    expected_session_values = {
        "is_ASIA": (expected_session == SESSION_ID_MAP["ASIA"]).astype(np.float64),
        "minutes_since_session_open": get_session_minutes_since_open_vectorized(
            decision_times
        ).to_numpy(dtype=np.float64),
        "minutes_to_next_session_boundary": (
            get_session_minutes_to_next_boundary_vectorized(decision_times).to_numpy(
                dtype=np.float64
            )
        ),
        "session_change_flag": labels.ne(labels.shift(1)).to_numpy(dtype=np.float64),
        "session_tradable": (expected_session != SESSION_ID_MAP["ASIA"]).astype(
            np.float64
        ),
    }
    for name, expected in expected_session_values.items():
        observed = session_values[name]
        # prepare_frame computes this flag before trimming its causal warmup
        # prefix.  The first retained row can therefore legitimately describe
        # the removed predecessor; every subsequent row remains verifiable.
        if name == "session_change_flag" and len(expected):
            expected = expected.copy()
            expected[0] = observed[0]
        if not np.array_equal(observed, expected):
            raise RuntimeError(
                "[MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION] "
                f"{context}: {name} disagrees with UTC-derived session context"
            )


@dataclass(frozen=True)
class ModelNativeStateContract:
    """Dataset-specific state convention for model-native train==serve parity."""

    feature_history_start_utc: pd.Timestamp
    rank_fit_start_utc: pd.Timestamp
    rank_fit_end_utc: pd.Timestamp
    rank_reference_npz: Path
    rank_reference: TrainRankReferenceV2
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metadata(
        cls,
        raw: Any,
        *,
        require_xau_direction_repair: bool = False,
    ) -> "ModelNativeStateContract":
        del require_xau_direction_repair  # v2 is strict for every model-native bundle.
        data = validate_state_contract_metadata_v2(raw, require_artifact=True)
        rank_ref = Path(str(data["rank_reference_npz"])).expanduser().resolve()
        reference = load_train_rank_reference_v2(
            rank_ref,
            expected_sha256=str(data["rank_reference_npz_sha256"]),
        )
        contract = cls(
            feature_history_start_utc=pd.Timestamp(data["feature_history_start_utc"]),
            rank_fit_start_utc=pd.Timestamp(data["rank_fit_start_utc"]),
            rank_fit_end_utc=pd.Timestamp(data["rank_fit_end_utc"]),
            rank_reference_npz=rank_ref,
            rank_reference=reference,
            raw=data,
        )
        return contract

    def as_report(self) -> dict[str, Any]:
        return {
            "feature_history_start_utc": str(self.feature_history_start_utc),
            "rank_fit_start_utc": str(self.rank_fit_start_utc),
            "rank_fit_end_utc": str(self.rank_fit_end_utc),
            "rank_reference_npz": str(self.rank_reference_npz),
            "rank_reference_npz_sha256": str(
                self.raw.get("rank_reference_npz_sha256") or ""
            ),
            "rank_reference_sidecar_sha256": str(
                self.raw.get("rank_reference_sidecar_sha256") or ""
            ),
            "rank_reference_schema_version": str(
                self.raw.get("rank_reference_schema_version") or ""
            ),
            "rank_reference_fit_scope": str(
                self.raw.get("rank_reference_fit_scope") or ""
            ),
            "rank_transform": str(self.raw.get("rank_transform") or ""),
            "feature_history_mode": str(self.raw.get("feature_history_mode") or ""),
            "split_reset_allowed": self.raw.get("split_reset_allowed"),
            "post_fit_rows_in_rank_reference": self.raw.get(
                "post_fit_rows_in_rank_reference"
            ),
            "runtime_rule_free": self.raw.get("runtime_rule_free"),
            "schema_version": str(self.raw.get("schema_version") or ""),
        }


def _require_state_contract(
    state_contract: ModelNativeStateContract | None,
) -> ModelNativeStateContract:
    if state_contract is None:
        raise RuntimeError(
            "[MODEL_NATIVE_STATE_CONTRACT] explicit model-native state contract required"
        )
    return state_contract


@dataclass
class ModelNativeStateBuilder:
    """Builds model-native seq513 states from the live common-history frame.

    ordered_signal_names: the ACTIVE bundle's 513 ordered signal names
        (bundle_metadata.ordered_signal_names) — [0:34] MUST equal the
        model-native base contract; [34:] are the 479 extension names
        computed inline (manifest order == bundle order, verified at init).
    """

    ordered_signal_names: list[str]
    state_contract: ModelNativeStateContract
    signal_contract: dict[str, Any]
    multi_tf: dict | None = None  # in-memory MTF-v2 bundle for group-A recompute
    _ext_names: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        require_model_native_signal_contract(
            self.signal_contract,
            context="MODEL_NATIVE_STATE_BUILDER",
        )
        if self.ordered_signal_names != list(self.signal_contract["fields"]):
            raise RuntimeError(
                "[MODEL_NATIVE_STATE] ordered_signal_names do not match signal contract"
            )
        if len(self.ordered_signal_names) != SIGNAL_DIM_MODEL_NATIVE:
            raise RuntimeError(
                f"[MODEL_NATIVE_STATE] bundle signal dim {len(self.ordered_signal_names)} "
                f"!= {SIGNAL_DIM_MODEL_NATIVE}"
            )
        base = list(self.ordered_signal_names[: len(MODEL_NATIVE_BASE_FIELDS)])
        if base != list(MODEL_NATIVE_BASE_FIELDS):
            raise RuntimeError(
                "[MODEL_NATIVE_STATE] bundle ordered_signal_names[:34] != MODEL_NATIVE_BASE_FIELDS"
            )
        forbidden = sorted(
            set(self.ordered_signal_names) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS)
        )
        if forbidden:
            raise RuntimeError(
                f"[MODEL_NATIVE_STATE] forbidden legacy bridge fields: {forbidden}"
            )
        self._ext_names = list(
            self.ordered_signal_names[len(MODEL_NATIVE_BASE_FIELDS) :]
        )

    # ── common-history frame preparation ────────────────────────────────────

    def prepare_frame(
        self,
        joined: pd.DataFrame,
        bucket_ctx_cat: pd.DataFrame | None = None,
        multi_tf: dict | None = None,
        context_m5: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Take joined cv3+BASE28 rows [history start .. decision bar] and
        recompute the frame-dependent families
        on it — the offline builder's exact order:
        model-native base -> group-A/dip-struct -> volume -> 'atr' -> entry-smart.
        Returns a NEW frame with a 'time' column (offline builders join on it).

        `multi_tf`: explicit MTF-v2 bundle for the group-A recompute. The LIVE
        async-context path (serving-wave gap 3) passes the snapshot's bundle so
        one decision is internally consistent even if the background refresh
        swaps `self.multi_tf` mid-call; None keeps the legacy self.multi_tf.

        `context_m5`: complete causal M5 prefix through the decision cutoff.
        Live passes it explicitly so D1/H4 liquidity state is not reset at the
        model's feature-history boundary.
        """
        if joined.empty:
            raise RuntimeError("[MODEL_NATIVE_STATE] empty joined frame")
        frame = joined.copy()
        frame.index.name = None  # joined index is named 'time' — avoid label ambiguity
        if "time" not in frame.columns:
            frame.insert(0, "time", frame.index)
        parsed_time = pd.DatetimeIndex(
            pd.to_datetime(frame["time"], utc=True, errors="coerce")
        )
        if parsed_time.hasnans:
            raise RuntimeError(
                "[MODEL_NATIVE_STATE] frame contains missing/invalid timestamps"
            )
        if parsed_time.has_duplicates:
            raise RuntimeError("[MODEL_NATIVE_STATE] frame timestamps are not unique")
        if not parsed_time.is_monotonic_increasing:
            raise RuntimeError(
                "[MODEL_NATIVE_STATE] frame timestamps are not strictly chronological"
            )
        frame["time"] = parsed_time
        frame = frame.reset_index(drop=True)

        forbidden_present = sorted(
            set(frame.columns) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS)
        )
        if forbidden_present:
            raise RuntimeError(
                "[MODEL_NATIVE_STATE] live frame contains retired bridge columns: "
                f"{forbidden_present}"
            )

        # 1b) Exact shared causal ATR/spread plus TRAIN-fit categorical ranks.
        #     This is the same owner used by the dataset builder.  Existing
        #     source values and categories are overwritten, never trusted or
        #     pinned per timestamp.
        frame = apply_train_rank_reference_v2(
            frame,
            self.state_contract.rank_reference,
        )

        # 1c) full-frame override columns — ctx_cat buckets (offline
        #     frame-global-rank convention, compute_bucket_ctx_cat_full_frame)
        #     + the 5 long-lookback HTF ctx cols (compute_htf_ctx_full_frame).
        #     Overrides the daemon/B28 live values by time-join: the B28 M1-lane
        #     rows in the daemon's INCREMENTAL region are stamped one M5 bar
        #     behind the offline M5-row-label convention (parity gate finding
        #     2026-07-08), so B28 must never be the state source for these.
        if bucket_ctx_cat is not None:
            target_index = pd.DatetimeIndex(frame["time"])
            missing_override_times = target_index.difference(bucket_ctx_cat.index)
            if len(missing_override_times):
                raise RuntimeError(
                    "[MODEL_NATIVE_STATE] frame_overrides missing rows for common-history window: "
                    f"{list(missing_override_times[:5])}"
                )
            aligned = bucket_ctx_cat.reindex(target_index)
            for col in aligned.columns:
                if col in (
                    "vol_regime_id",
                    "atr_bucket",
                    "spread_bucket",
                    "H4_trend_sign_cat",
                ):
                    if col in ("vol_regime_id", "atr_bucket", "spread_bucket"):
                        if aligned[col].isna().any():
                            raise RuntimeError(
                                f"[MODEL_NATIVE_STATE] categorical override unavailable: {col}"
                            )
                        observed = frame[col].to_numpy(dtype=np.int64)
                        expected = aligned[col].to_numpy(dtype=np.int64)
                        if not np.array_equal(observed, expected):
                            raise RuntimeError(
                                f"[MODEL_NATIVE_STATE] TRAIN-rank override mismatch: {col}"
                            )
                    # H4 is a causal HTF projection and may be unavailable only
                    # in the contiguous common-history warmup prefix.
                    if col == "H4_trend_sign_cat":
                        frame[col] = pd.to_numeric(
                            aligned[col], errors="coerce"
                        ).to_numpy(dtype=np.float64)
                    else:
                        frame[col] = aligned[col].to_numpy(np.int64)
                else:
                    frame[col] = aligned[col].to_numpy(np.float64)

        # 1d) micro (5) + swing (5) + session (5) ctx. Swing uses the exact
        #     causal confirmation-lag owner used by the dataset builder; a
        #     pivot is never stamped before its confirming bars exist.
        from gx1.execution.v12_ctx_augment_live import (
            _add_micro_features,
            _add_session_features,
        )

        _frame_ts = frame.set_index(pd.DatetimeIndex(frame["time"]))
        _add_micro_features(_frame_ts)
        _add_session_features(_frame_ts)
        for col in (
            list(MODEL_NATIVE_CTX_CONT_MICRO_FIELDS)
            + list(MODEL_NATIVE_CTX_CONT_SESSION_FIELDS)
            + ["session_id"]
        ):
            frame[col] = _frame_ts[col].to_numpy()
        del _frame_ts
        for col, arr in compute_swing_structure_features(
            frame["high"].to_numpy(dtype=np.float64),
            frame["low"].to_numpy(dtype=np.float64),
            frame["close"].to_numpy(dtype=np.float64),
            lookback=SWING_LOOKBACK_V1,
            atr_period=SWING_ATR_PERIOD_V1,
        ).items():
            frame[col] = arr

        # 2) group-A (24) + dip/struct (36) — DROP the loader's full-history
        #    in-memory values and recompute on THIS common-history frame.
        #    attach_... is
        #    idempotent-if-present, so the drop is what forces the recompute.
        ga_cols = list(MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS) + list(
            MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS
        )
        frame = frame.drop(columns=[c for c in ga_cols if c in frame.columns])
        from gx1.scripts.augment_forward_outcome_v2 import (
            attach_group_a_dip_struct_ctx_columns,
            trim_causal_context_warmup_prefix,
        )

        mtf = multi_tf if multi_tf is not None else self.multi_tf
        if mtf is None:
            raise RuntimeError(
                "[MODEL_NATIVE_STATE] multi_tf bundle required for group-A recompute — "
                "pass the in-memory MTF-v2 dict (build_multi_tf_from_cv3)"
            )
        frame = attach_group_a_dip_struct_ctx_columns(
            frame,
            multi_tf=mtf,
            journal_label="model_native_live",
            context_m5=context_m5,
        )
        from gx1.features.regime_v4_features import (
            REGIME_V4_DERIVED_COLS,
            REGIME_V4_SOURCE_COLS,
        )

        causal_required = list(
            dict.fromkeys(
                ga_cols + list(REGIME_V4_SOURCE_COLS) + list(REGIME_V4_DERIVED_COLS)
            )
        )
        frame = trim_causal_context_warmup_prefix(frame, causal_required).reset_index(
            drop=True
        )

        # 3) volume features (4) — common-history causal recompute.
        from gx1.features.volume_features import add_volume_features

        if "volume" not in frame.columns:
            raise RuntimeError(
                "[MODEL_NATIVE_STATE] 'volume' column missing from live frame"
            )
        frame = frame.drop(
            columns=[c for c in VOLUME_FEATURE_NAMES if c in frame.columns]
        )
        add_volume_features(frame)

        # 4) bare 'atr' for the extension price layer: computed in step 1b.

        # 5) entry smart ctx (19) — common-history recompute AFTER group-A (consumes
        #    dist_to_* / dip_* from step 2), mirroring the offline order
        #    (build_entry_v10_ctx_training_dataset_v3.py:2280-2282).
        from gx1.features.entry_smart_context import add_entry_smart_context_features

        frame = frame.drop(
            columns=[
                c
                for c in MODEL_NATIVE_CTX_CONT_ENTRY_SMART_DERIVED_FIELDS
                if c in frame.columns
            ]
        )
        add_entry_smart_context_features(frame)

        # Contract completeness — fail loud (never zero-fill for decisioning).
        missing_sig = [c for c in MODEL_NATIVE_BASE_FIELDS if c not in frame.columns]
        missing_ctx = [
            c for c in MODEL_NATIVE_CTX_CONT_FIELDS if c not in frame.columns
        ]
        missing_cat = [c for c in MODEL_NATIVE_CTX_CAT_FIELDS if c not in frame.columns]
        if missing_sig or missing_ctx or missing_cat:
            raise RuntimeError(
                f"[MODEL_NATIVE_STATE] contract columns missing from live frame — "
                f"sig={missing_sig[:10]} ctx={missing_ctx[:10]} cat={missing_cat}"
            )
        _require_model_native_entry_context_frame(
            frame,
            context="prepare_frame",
        )
        return frame

    # ── state assembly ──────────────────────────────────────────────────────

    def build_states(
        self,
        frame: pd.DataFrame,
        target_times: Sequence[pd.Timestamp],
    ) -> dict[str, Any]:
        """Compute states for `target_times` (must exist in frame['time']).

        Returns dict with:
          seq      (n, 96, 513) float32
          snap     (n, 513)     float32
          ctx_cont (n, 142)     float32
          ctx_cat  (n, 5)       int64
          entry_trend_regime_id (n,) int64
          times    list[pd.Timestamp]
        Mirrors build_dataset_canonical's emission exactly:
        seq = sig_mat[i-95:i+1]; snap = sig_mat[i] (builder line 3024-3026).
        """
        _require_model_native_entry_context_frame(
            frame,
            context="build_states",
        )
        times = pd.DatetimeIndex(pd.to_datetime(list(target_times), utc=True))
        pos_by_time = pd.Index(frame["time"])
        idxs: list[int] = []
        for ts in times:
            loc = pos_by_time.get_indexer([ts])
            if loc[0] < 0:
                raise RuntimeError(
                    f"[MODEL_NATIVE_STATE] target bar {ts} not in live frame"
                )
            if loc[0] < SEQ_LEN_MODEL_NATIVE - 1:
                raise RuntimeError(
                    f"[MODEL_NATIVE_STATE] target bar {ts} has only {loc[0]} prior frame bars "
                    f"(needs >= {SEQ_LEN_MODEL_NATIVE - 1}; common-history frame too short)"
                )
            idxs.append(int(loc[0]))

        # 479 extension signals — offline one-truth inline computation. The
        # price/candle layers read the SOURCE PARQUET by name, so hand them the
        # common-history frame's own price columns via a temp parquet so offline
        # and serve share the exact EMA/candlestick history boundary.
        from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
            _build_inline_seq_structure_extension,
        )

        with tempfile.NamedTemporaryFile(
            suffix="_model_native_src.parquet", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            frame[[c for c in _SOURCE_PARQUET_COLS if c in frame.columns]].to_parquet(
                tmp_path, index=False
            )
            ext_mat, ext_names, _meta = _build_inline_seq_structure_extension(
                frame,
                requested_features=self._ext_names,
                ctx_cont_names=list(MODEL_NATIVE_CTX_CONT_FIELDS),
                ctx_cat_names=list(MODEL_NATIVE_CTX_CAT_FIELDS),
                source_parquet=tmp_path,
                base_signal_fields=list(MODEL_NATIVE_BASE_FIELDS),
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        if list(ext_names) != self._ext_names:
            raise RuntimeError(
                "[MODEL_NATIVE_STATE] extension name order mismatch vs bundle"
            )

        base_mat = frame[list(MODEL_NATIVE_BASE_FIELDS)].astype(np.float32).to_numpy()
        sig_mat = np.concatenate([base_mat, ext_mat], axis=1).astype(
            np.float32, copy=False
        )
        if sig_mat.shape[1] != SIGNAL_DIM_MODEL_NATIVE:
            raise RuntimeError(
                f"[MODEL_NATIVE_STATE] signal width {sig_mat.shape[1]} != {SIGNAL_DIM_MODEL_NATIVE}"
            )
        ctx_cont_mat = (
            frame[list(MODEL_NATIVE_CTX_CONT_FIELDS)].astype(np.float32).to_numpy()
        )
        # The exact-integral/domain validation above precedes this conversion;
        # pandas/numpy truncation can therefore never turn a malformed category
        # into a valid embedding index.
        ctx_cat_mat = (
            frame[list(MODEL_NATIVE_CTX_CAT_FIELDS)].astype(np.int64).to_numpy()
        )
        if "trend_regime_id" not in frame.columns:
            raise RuntimeError(
                "[MODEL_NATIVE_STATE] exact entry trend_regime_id is missing"
            )
        trend_regime_numeric = pd.to_numeric(
            frame["trend_regime_id"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if (
            not np.isfinite(trend_regime_numeric).all()
            or not np.array_equal(
                trend_regime_numeric,
                trend_regime_numeric.astype(np.int64),
            )
            or not np.isin(
                trend_regime_numeric.astype(np.int64),
                (0, 1, 2),
            ).all()
        ):
            raise RuntimeError(
                "[MODEL_NATIVE_STATE] entry trend_regime_id is outside 0,1,2"
            )
        trend_regime_mat = trend_regime_numeric.astype(np.int64)

        n = len(idxs)
        seq = np.empty(
            (n, SEQ_LEN_MODEL_NATIVE, SIGNAL_DIM_MODEL_NATIVE), dtype=np.float32
        )
        snap = np.empty((n, SIGNAL_DIM_MODEL_NATIVE), dtype=np.float32)
        ctx_cont = np.empty((n, CTX_CONT_DIM_MODEL_NATIVE), dtype=np.float32)
        ctx_cat = np.empty((n, CTX_CAT_DIM_MODEL_NATIVE), dtype=np.int64)
        entry_trend_regime_id = np.empty(n, dtype=np.int64)
        for k, i in enumerate(idxs):
            seq[k] = sig_mat[i - (SEQ_LEN_MODEL_NATIVE - 1) : i + 1]
            snap[k] = sig_mat[i]
            ctx_cont[k] = ctx_cont_mat[i]
            ctx_cat[k] = ctx_cat_mat[i]
            entry_trend_regime_id[k] = trend_regime_mat[i]
        if (
            not np.isfinite(seq).all()
            or not np.isfinite(snap).all()
            or not np.isfinite(ctx_cont).all()
        ):
            raise RuntimeError(
                "[MODEL_NATIVE_STATE] non-finite state values — refusing to serve"
            )
        return {
            "seq": seq,
            "snap": snap,
            "ctx_cont": ctx_cont,
            "ctx_cat": ctx_cat,
            "entry_trend_regime_id": entry_trend_regime_id,
            "times": [pd.Timestamp(t) for t in times],
        }


def compute_bucket_ctx_cat_full_frame(
    cv3: pd.DataFrame,
    state_contract: ModelNativeStateContract | None = None,
) -> pd.DataFrame:
    """Derive absolute buckets plus a distinct causal local-volatility regime.

    Raw ATR/spread inputs are causal over the common history frame. Absolute
    ATR/spread buckets use frozen TRAIN distributions; vol_regime_id uses the
    shared trailing-288-bar rank. No row-level category table exists.
    """
    if not isinstance(cv3.index, pd.DatetimeIndex):
        raise RuntimeError(
            "[MODEL_NATIVE_STATE] cv3 must have a DatetimeIndex for bucket recompute"
        )
    contract = _require_state_contract(state_contract)
    required = ["high", "low", "close", "bid_close", "ask_close"]
    missing = [name for name in required if name not in cv3.columns]
    if missing:
        raise RuntimeError(
            f"[MODEL_NATIVE_STATE] rank source columns missing: {missing}"
        )
    sub = cv3.loc[
        cv3.index >= contract.feature_history_start_utc,
        required,
    ].copy()
    ranked = apply_train_rank_reference_v2(sub, contract.rank_reference)
    return pd.DataFrame(
        {
            "vol_regime_id": ranked["vol_regime_id"].to_numpy(dtype=np.int64),
            "atr_bucket": ranked["atr_bucket"].to_numpy(dtype=np.int64),
            "spread_bucket": ranked["spread_bucket"].to_numpy(dtype=np.int64),
        },
        index=sub.index,
    )


def compute_htf_ctx_full_frame(
    cv3: pd.DataFrame,
    state_contract: ModelNativeStateContract | None = None,
) -> pd.DataFrame:
    """The 5 long-lookback HTF ctx columns (D1_dist_from_ema200_atr,
    D1_atr_percentile_252, H1_range_compression_ratio,
    M15_range_compression_ratio, H4_trend_sign_cat) recomputed FRESH over the
    common history frame [state_contract.feature_history_start_utc, now] via the
    ONE-TRUTH mirror v12_ctx_augment_live._add_htf_features (== the offline
    add_ctx_cont HTF block). NEVER taken from B28: the daemon's incremental
    M1-lane rows stamp these one M5 bar behind the offline convention (parity
    gate finding 2026-07-08). A bare work-frame is passed so the function's
    preserve-guard cannot short-circuit onto stale values.
    """
    from gx1.execution.v12_ctx_augment_live import _add_htf_features

    if not isinstance(cv3.index, pd.DatetimeIndex):
        raise RuntimeError(
            "[MODEL_NATIVE_STATE] cv3 must have a DatetimeIndex for HTF recompute"
        )
    contract = _require_state_contract(state_contract)
    sub_idx = cv3.index[cv3.index >= contract.feature_history_start_utc]
    # Compute on the complete causal cv3 prefix and only then slice to the
    # model history window.  Starting at feature_history_start discarded prior
    # D1 transitions and made bars_since_d1_regime_change frame-dependent.
    full_idx = cv3.index
    m5 = cv3.loc[full_idx, ["open", "high", "low", "close"]].copy()
    work = pd.DataFrame(index=full_idx)
    _add_htf_features(work, m5)
    cols = [
        "D1_dist_from_ema200_atr",
        "D1_atr_percentile_252",
        "H1_range_compression_ratio",
        "M15_range_compression_ratio",
        "H4_trend_sign_cat",
    ]
    missing = [c for c in cols if c not in work.columns]
    if missing:
        raise RuntimeError(
            f"[MODEL_NATIVE_STATE] HTF recompute missing cols: {missing}"
        )

    # REGIME_V4 DERIVED ctx (8) — one-truth add_regime_v4_features on the full
    # model-range frame, fed the RECOMPUTED D1_dist series (the daemon's cv3
    # carries d1_dist_roc_288_v3 / d1_dist_to_boundary_v3 built from ITS OWN
    # D1_dist history, which skews on weekend-open bars — parity gate finding
    # 2026-07-08). Per-TF sources ({tf}_regime_class_id_v2 / _ema_stack_aligned_v2 /
    # _trend_age_bars_norm_v2) come from cv3 (parity-verified equal offline).
    from gx1.features.regime_v4_features import (
        REGIME_V4_SOURCE_COLS,
        add_regime_v4_features,
    )

    src_cols = [c for c in REGIME_V4_SOURCE_COLS if c != "D1_dist_from_ema200_atr"]
    missing_src = [c for c in src_cols if c not in cv3.columns]
    if missing_src:
        raise RuntimeError(
            f"[MODEL_NATIVE_STATE] REGIME_V4 source cols missing from cv3: {missing_src}"
        )
    rv_work = cv3.loc[full_idx, src_cols].copy()
    rv_work["D1_dist_from_ema200_atr"] = work["D1_dist_from_ema200_atr"].to_numpy()
    add_regime_v4_features(rv_work)
    derived = [
        "regime_tf_agreement_v3",
        "regime_stack_sum_v3",
        "regime_divergence_flag_v3",
        "d1_dist_roc_288_v3",
        "d1_dist_to_boundary_v3",
        "d1_regime_changed_flag_v3",
        "bars_since_d1_regime_change_v3",
        "d1_trend_age_mature_flag_v3",
    ]
    out = work[cols].copy()
    for c in derived:
        out[c] = rv_work[c].to_numpy()
    return out.loc[sub_idx].copy()


def build_multi_tf_from_cv3(
    cv3: pd.DataFrame,
    *,
    matrix_contract: str,
    feature_names: list[str],
) -> dict:
    """Build the exact bundle-declared in-memory MTF surface from live OHLCV.

    The active Entry architecture requires the all-eight-family V4 matrix.
    The caller must repeat the immutable bundle identity; this owner neither
    guesses a version nor falls back to the older generic V2 surface.

    Uses float32-cast OHLCV,
    the EXACT dtype convention of both the offline disk cache
    (`gx1.scripts.prebuild_multi_tf_cache_v4`) and the trainer/eval dataset
    (entry_v10_ctx_train_v3.py:1634-1651). NOTE: PrebuiltStateLoader.build_multi_tf_features
    exists but feeds the EXIT chain with float64 OHLC — the entry parity target is the
    float32 cache convention, hence this thin one-truth wrapper (build fn is shared).
    """
    from gx1.features.htf_features import (
        HTF_V4_MATRIX_CONTRACT,
        MULTI_TF_PER_BAR_FEATURES_V4,
        build_multi_tf_per_bar_features_v4,
    )

    if (
        matrix_contract != HTF_V4_MATRIX_CONTRACT
        or tuple(feature_names) != MULTI_TF_PER_BAR_FEATURES_V4
    ):
        raise RuntimeError(
            "[MODEL_NATIVE_STATE] exact bundle-declared V4 MTF contract "
            "required"
        )

    cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in cols if c not in cv3.columns]
    if missing:
        raise RuntimeError(
            f"[MODEL_NATIVE_STATE] cv3 missing OHLCV for MTF build: {missing}"
        )
    m5 = cv3[cols].copy()
    for c in cols:
        m5[c] = m5[c].astype(np.float32)
    if not isinstance(m5.index, pd.DatetimeIndex):
        raise RuntimeError(
            "[MODEL_NATIVE_STATE] cv3 must have a DatetimeIndex for MTF build"
        )
    from gx1.execution.v12_state_from_prebuilt import (
        _require_v29_registry_constants_from_bound_cache,
    )

    built = build_multi_tf_per_bar_features_v4(
        m5,
        v29_registry_constants=_require_v29_registry_constants_from_bound_cache(),
    )
    for timeframe, frame in built.items():
        if (
            frame.attrs.get("htf_feature_contract") != matrix_contract
            or tuple(frame.columns) != tuple(feature_names)
        ):
            raise RuntimeError(
                "[MODEL_NATIVE_STATE] live MTF contract mismatch "
                f"timeframe={timeframe}"
            )
    return built
