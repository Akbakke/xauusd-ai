#!/usr/bin/env python3
"""One-model XAU live pipeline.

The contract-admitted model bundle is the only decision authority.  Its shared
feature encoder must emit both:

* ``direction_logits`` ordered LONG/SHORT/FLAT for Entry; and
* ``exit_action_logits`` ordered HOLD/EXIT_NOW for positions opened from that
  exact Entry snapshot.

There is no auxiliary decision model, compatibility bridge, rule overlay, or
synthetic HOLD/FLAT path.  Until a bundle proves the unified Entry/Exit
contract, startup and Exit inference fail closed with structured evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC,
    MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS,
    ModelNativeRuntimeEvidenceError,
    require_model_native_runtime_evidence,
)
from gx1.execution.v12_m1_to_m5_downsample import latest_closed_m5_start_at
from gx1.execution.v12_model_native_state_live import (
    SEQ_LEN_MODEL_NATIVE as ENTRY_SEQ_LEN,
)
from gx1.execution.v12_state_from_prebuilt import PrebuiltStateLoader
from gx1.models.entry_v10.direction_decision_contract import (
    require_unified_entry_exit_contract,
    require_unified_exit_output,
)

LOG = logging.getLogger("v12_pipeline")

ENTRY_DECISION_AVAILABILITY_LAG = pd.Timedelta(minutes=5)
ENTRY_MAX_DECISION_LATENCY_SEC = float(
    MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC
)
ENTRY_MAX_CANONICAL_CUTOFF_AGE_SEC = (
    ENTRY_DECISION_AVAILABILITY_LAG.total_seconds()
    + ENTRY_MAX_DECISION_LATENCY_SEC
)


class EntryDecisionUnavailable(RuntimeError):
    """No model direction exists for this poll; never synthesize FLAT/SKIP."""

    def __init__(self, reason: str, **evidence: Any):
        self.reason = str(reason)
        self.evidence = dict(evidence)
        super().__init__(f"{self.reason}: {self.evidence}")


class ExitDecisionUnavailable(RuntimeError):
    """No unified-model Exit decision exists; never synthesize HOLD/EXIT."""

    def __init__(self, reason: str, **evidence: Any):
        self.reason = str(reason)
        self.evidence = dict(evidence)
        super().__init__(f"{self.reason}: {self.evidence}")


def _utc_ts(ts: pd.Timestamp | Any) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _latest_closed_m5_start(now_minute: pd.Timestamp) -> pd.Timestamp:
    return latest_closed_m5_start_at(_utc_ts(now_minute))


def _entry_decision_latency_fields(
    now_minute: pd.Timestamp,
    decision_m5: pd.Timestamp,
) -> dict[str, Any]:
    now_ts = _utc_ts(now_minute)
    decision_ts = _utc_ts(decision_m5)
    available_ts = decision_ts + ENTRY_DECISION_AVAILABILITY_LAG
    latency_sec = (now_ts - available_ts).total_seconds()
    return {
        "decision_ts": str(decision_ts),
        "decision_available_ts": str(available_ts),
        "entry_signal_latency_sec": float(latency_sec),
        "entry_signal_latency_min": float(latency_sec / 60.0),
        "entry_signal_latency_cap_sec": ENTRY_MAX_DECISION_LATENCY_SEC,
        "entry_signal_stale": bool(
            latency_sec > ENTRY_MAX_DECISION_LATENCY_SEC
        ),
    }


def _require_unified_model(model_adapter: object, *, context: str) -> None:
    metadata = getattr(model_adapter, "_meta", None)
    if not isinstance(metadata, Mapping):
        raise RuntimeError(
            f"{context}: UNIFIED_MODEL_METADATA_UNAVAILABLE"
        )
    require_unified_entry_exit_contract(metadata, context=context)
    model = getattr(model_adapter, "_model", None)
    if model is None or not hasattr(model, "state_dict"):
        raise RuntimeError(f"{context}: UNIFIED_MODEL_STATE_UNAVAILABLE")
    state_keys = set(model.state_dict())
    required_exit_keys = {
        "head_exit_action.weight",
        "head_exit_action.bias",
    }
    missing = sorted(required_exit_keys - state_keys)
    if missing:
        raise RuntimeError(
            f"{context}: UNIFIED_MODEL_EXIT_HEAD_MISSING keys={missing}"
        )
    if not callable(getattr(model_adapter, "decide_exit", None)):
        raise RuntimeError(
            f"{context}: UNIFIED_MODEL_EXIT_ADAPTER_MISSING"
        )


def _bind_admitted_m1_surface_if_declared(
    model_adapter: object,
    *,
    context: str,
) -> None:
    """Bind the exact M1 surface declared by the frozen model metadata.

    Missing metadata leaves Exit unavailable until a complete lifecycle/model
    admission exists. A declared but malformed binding is a startup failure;
    it is never ignored or replaced with a live feature rebuild.
    """

    metadata = getattr(model_adapter, "_meta", None)
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"{context}: M1_FEATURE_BINDING_METADATA_UNAVAILABLE")
    binding = metadata.get("m1_feature_surface_binding")
    if binding is None:
        raise RuntimeError(f"{context}: M1_FEATURE_BINDING_MISSING")
    if not isinstance(binding, Mapping):
        raise RuntimeError(f"{context}: M1_FEATURE_BINDING_SCHEMA_INVALID")
    expected = {
        "parquet_path",
        "manifest_path",
        "dataset_run_id",
        "pair_generation_id",
        "parquet_sha256",
        "manifest_sha256",
        "feature_field_order_sha256",
    }
    if set(binding) != expected:
        raise RuntimeError(f"{context}: M1_FEATURE_BINDING_KEYS_INVALID")
    binder = getattr(model_adapter, "bind_admitted_m1_feature_surface", None)
    if not callable(binder):
        raise RuntimeError(f"{context}: M1_FEATURE_BINDING_ADAPTER_MISSING")
    binder(
        parquet_path=Path(str(binding["parquet_path"])),
        manifest_path=Path(str(binding["manifest_path"])),
        dataset_run_id=str(binding["dataset_run_id"]),
        pair_generation_id=str(binding["pair_generation_id"]),
        parquet_sha256=str(binding["parquet_sha256"]),
        manifest_sha256=str(binding["manifest_sha256"]),
        feature_field_order_sha256=str(
            binding["feature_field_order_sha256"]
        ),
    )


@dataclass
class V12Pipeline:
    """Runtime holder for one contract-admitted model and its shared state."""

    prebuilt_loader: PrebuiltStateLoader
    smart_entry: object | None = None
    _last_smart_bucket: pd.Timestamp | None = None
    _last_augmented_bucket: pd.Timestamp | None = None
    _last_augmented: pd.DataFrame | None = None
    _last_entry_prebuilt_snapshot: object | None = None

    @classmethod
    def load_exit_recovery(cls, trade: object) -> "V12Pipeline":
        """Load the exact frozen bundle for an already-open trade only."""

        binding = getattr(trade, "model_bundle_binding", None)
        if not isinstance(binding, Mapping):
            raise RuntimeError(
                "EXIT_RECOVERY_MODEL_BUNDLE_BINDING_MISSING"
            )
        require_snapshot = getattr(trade, "require_entry_snapshot", None)
        if not callable(require_snapshot):
            raise RuntimeError("EXIT_RECOVERY_ENTRY_SNAPSHOT_MISSING")
        require_snapshot()
        # Direction runtime evidence no longer carries the sizing authority:
        # sizing may never sit inside the direction surface. The authority is
        # owned by this trade's learned sizing application.
        sizing_execution = getattr(trade, "sizing_execution_evidence", None)
        sizing_application = (
            sizing_execution.get("sizing_application")
            if isinstance(sizing_execution, Mapping)
            else None
        )
        sizing_authority = (
            sizing_application.get("sizing_authority_contract")
            if isinstance(sizing_application, Mapping)
            else None
        )
        if not isinstance(sizing_authority, Mapping):
            raise RuntimeError(
                "EXIT_RECOVERY_SIZING_AUTHORITY_MISSING"
            )

        loader = PrebuiltStateLoader()
        loader.load()
        from gx1.execution.v12_smart_entry_live import (
            SmartEntryLiveInference,
        )

        model_adapter = (
            SmartEntryLiveInference.load_immutable_exit_recovery(
                bundle_dir=Path(str(binding["bundle_dir"])),
                expected_bundle_sha256=str(binding["bundle_sha256"]),
                operating_point=binding["operating_point"],
                sizing_authority=sizing_authority,
                device="cpu",
            )
        )
        _require_unified_model(
            model_adapter,
            context="V12_PIPELINE_EXIT_RECOVERY",
        )
        _bind_admitted_m1_surface_if_declared(
            model_adapter,
            context="V12_PIPELINE_EXIT_RECOVERY",
        )
        return cls(
            prebuilt_loader=loader,
            smart_entry=model_adapter,
        )

    def _refresh_entry_canonical(self, now_minute: pd.Timestamp) -> None:
        """Load the exact latest closed M5 window under the live contract."""

        now_ts = _utc_ts(now_minute)
        expected_m5 = _latest_closed_m5_start(now_ts)
        try:
            changed = bool(self.prebuilt_loader.refresh_if_changed())
        except Exception as exc:
            raise EntryDecisionUnavailable(
                "entry_canonical_refresh_failed",
                now_minute=str(now_ts),
                expected_m5=str(expected_m5),
                error_type=type(exc).__name__,
            ) from exc
        if changed:
            self._last_augmented_bucket = None
            self._last_augmented = None
            self._last_entry_prebuilt_snapshot = None

        try:
            snapshot = self.prebuilt_loader.acquire_serving_snapshot()
            raw_cutoff = snapshot.cutoff_ts
        except Exception as exc:
            raise EntryDecisionUnavailable(
                "entry_canonical_cutoff_unavailable",
                now_minute=str(now_ts),
                expected_m5=str(expected_m5),
                error_type=type(exc).__name__,
            ) from exc
        if raw_cutoff is None:
            raise EntryDecisionUnavailable(
                "entry_canonical_cutoff_missing",
                now_minute=str(now_ts),
                expected_m5=str(expected_m5),
            )

        prior_snapshot = self._last_entry_prebuilt_snapshot
        if (
            prior_snapshot is not None
            and prior_snapshot.pair_generation_id
            != snapshot.pair_generation_id
        ):
            self._last_augmented_bucket = None
            self._last_augmented = None
        try:
            cutoff = _utc_ts(raw_cutoff)
        except Exception as exc:
            raise EntryDecisionUnavailable(
                "entry_canonical_cutoff_invalid",
                cutoff=repr(raw_cutoff),
                error_type=type(exc).__name__,
            ) from exc
        if pd.isna(cutoff):
            raise EntryDecisionUnavailable(
                "entry_canonical_cutoff_invalid",
                cutoff=repr(raw_cutoff),
            )

        cutoff_age_sec = float((now_ts - cutoff).total_seconds())
        evidence = {
            "now_minute": str(now_ts),
            "expected_m5": str(expected_m5),
            "canonical_cutoff": str(cutoff),
            "canonical_cutoff_age_sec": cutoff_age_sec,
            "canonical_cutoff_age_cap_sec": (
                ENTRY_MAX_CANONICAL_CUTOFF_AGE_SEC
            ),
        }
        if cutoff_age_sec > ENTRY_MAX_CANONICAL_CUTOFF_AGE_SEC:
            raise EntryDecisionUnavailable(
                "entry_canonical_stale",
                **evidence,
            )
        if cutoff < expected_m5:
            raise EntryDecisionUnavailable(
                "entry_latest_closed_m5_unavailable",
                **evidence,
            )

        bucket = expected_m5.floor("5min")
        augmented = self._last_augmented
        if self._last_augmented_bucket != bucket or augmented is None:
            try:
                augmented = self.prebuilt_loader.get_window(
                    expected_m5,
                    n_bars=ENTRY_SEQ_LEN,
                    snapshot=snapshot,
                )
            except Exception as exc:
                raise EntryDecisionUnavailable(
                    "entry_canonical_window_read_failed",
                    error_type=type(exc).__name__,
                    **evidence,
                ) from exc

        if augmented is None or augmented.empty:
            raise EntryDecisionUnavailable(
                "entry_canonical_window_empty",
                **evidence,
            )
        if len(augmented) != ENTRY_SEQ_LEN:
            raise EntryDecisionUnavailable(
                "entry_canonical_history_mismatch",
                observed_bars=int(len(augmented)),
                required_bars=int(ENTRY_SEQ_LEN),
                **evidence,
            )
        observed_index = pd.to_datetime(
            augmented.index,
            utc=True,
            errors="coerce",
        )
        if (
            observed_index.hasnans
            or not observed_index.is_monotonic_increasing
            or not observed_index.is_unique
        ):
            raise EntryDecisionUnavailable(
                "entry_canonical_index_invalid",
                has_nat=bool(observed_index.hasnans),
                monotonic=bool(observed_index.is_monotonic_increasing),
                unique=bool(observed_index.is_unique),
                **evidence,
            )
        if observed_index[-1] != expected_m5:
            raise EntryDecisionUnavailable(
                "entry_canonical_exact_m5_missing",
                observed_latest_m5=str(observed_index[-1]),
                **evidence,
            )

        self._last_augmented_bucket = bucket
        self._last_augmented = augmented
        self._last_entry_prebuilt_snapshot = snapshot

    def make_entry_decision(
        self,
        now_minute: pd.Timestamp,
        bid: float,
        ask: float,
    ) -> dict[str, Any]:
        """Return one exact model LONG/SHORT/FLAT decision for a fresh M5 row.

        Operational no-data/stale/cadence states raise; none becomes FLAT.
        """

        del bid, ask
        if self.smart_entry is None:
            raise EntryDecisionUnavailable("unified_model_not_loaded")
        _require_unified_model(
            self.smart_entry,
            context="V12_PIPELINE_ENTRY",
        )
        self._refresh_entry_canonical(now_minute)
        augmented = self._last_augmented
        if augmented is None or len(augmented) != ENTRY_SEQ_LEN:
            raise EntryDecisionUnavailable(
                "canonical_history_mismatch",
                observed_bars=0 if augmented is None else int(len(augmented)),
                required_bars=int(ENTRY_SEQ_LEN),
            )

        decision_m5 = augmented.index[-1]
        latency_fields = _entry_decision_latency_fields(
            now_minute,
            decision_m5,
        )
        if (
            self._last_smart_bucket is not None
            and decision_m5 <= self._last_smart_bucket
        ):
            raise EntryDecisionUnavailable(
                "awaiting_new_m5_bar",
                **latency_fields,
            )
        if latency_fields["entry_signal_stale"]:
            self._last_smart_bucket = decision_m5
            raise EntryDecisionUnavailable(
                "entry_signal_stale",
                **latency_fields,
            )

        from gx1.execution.v12_smart_entry_live import SmartContextStaleError

        try:
            snapshot = self._last_entry_prebuilt_snapshot
            if snapshot is None:
                raise RuntimeError(
                    "ENTRY_PREBUILT_SERVING_SNAPSHOT_UNAVAILABLE"
                )
            head = self.smart_entry.predict_live_bar(
                self.prebuilt_loader,
                decision_m5,
                prebuilt_snapshot=snapshot,
            )
        except SmartContextStaleError as exc:
            raise EntryDecisionUnavailable(
                "smart_ctx_stale_refresh_pending",
                context_age_m5_bars=exc.age,
                context_cutoff_ts=str(exc.ctx_cutoff),
                **latency_fields,
            ) from exc
        except Exception as exc:
            raise EntryDecisionUnavailable(
                "smart_entry_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                **latency_fields,
            ) from exc

        if "atr_bps" not in augmented.columns:
            raise EntryDecisionUnavailable(
                "model_native_atr_missing",
                decision_m5=str(decision_m5),
                **latency_fields,
            )
        atr_bps = float(
            pd.to_numeric(
                augmented.iloc[-1]["atr_bps"],
                errors="coerce",
            )
        )
        if not np.isfinite(atr_bps) or atr_bps <= 0.0:
            raise EntryDecisionUnavailable(
                "model_native_atr_invalid",
                decision_m5=str(decision_m5),
                atr_bps=atr_bps,
                **latency_fields,
            )
        try:
            decision = self.smart_entry.decide(head, atr_bps=atr_bps)
        except Exception as exc:
            raise EntryDecisionUnavailable(
                "model_native_direction_decision_invalid",
                decision_m5=str(decision_m5),
                error_type=type(exc).__name__,
                error=str(exc),
                **latency_fields,
            ) from exc
        if not isinstance(decision, dict) or not decision:
            raise EntryDecisionUnavailable(
                "model_native_direction_decision_invalid",
                decision_m5=str(decision_m5),
                **latency_fields,
            )

        decision.update(latency_fields)
        decision["entry_source_pair_generation_id"] = (
            snapshot.pair_generation_id
        )
        decision["entry_source_pair_manifest_sha256"] = (
            snapshot.pair_manifest_sha256
        )
        raw_snapshot = decision.get("_v10_snapshot")
        if not isinstance(raw_snapshot, dict) or not raw_snapshot:
            raise EntryDecisionUnavailable(
                "model_native_runtime_evidence_missing",
                decision_m5=str(decision_m5),
            )
        timing_evidence = {
            key: decision[key]
            for key in MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
            if key in decision
        }
        if set(timing_evidence) != set(
            MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
        ):
            raise EntryDecisionUnavailable(
                "model_native_timing_evidence_incomplete",
                missing_fields=sorted(
                    MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
                    - set(timing_evidence)
                ),
                decision_m5=str(decision_m5),
            )
        executable_snapshot = dict(raw_snapshot)
        executable_snapshot.update(timing_evidence)
        try:
            executable_snapshot = require_model_native_runtime_evidence(
                executable_snapshot,
                context="V12_PIPELINE_ENTRY",
            )
        except ModelNativeRuntimeEvidenceError as exc:
            raise EntryDecisionUnavailable(
                "model_native_runtime_evidence_invalid",
                decision_m5=str(decision_m5),
                error=str(exc),
            ) from exc
        if decision.get("policy") != executable_snapshot["model_policy"]:
            raise EntryDecisionUnavailable(
                "model_native_policy_mismatch",
                decision_policy=decision.get("policy"),
                snapshot_policy=executable_snapshot["model_policy"],
            )
        decision["_v10_snapshot"] = executable_snapshot
        self._last_smart_bucket = decision_m5
        return decision

    def make_exit_decision(
        self,
        trade: object,
        now_minute: pd.Timestamp,
        bid: float,
        ask: float,
        *,
        on_bar_committed: Callable[[object], None],
    ) -> dict[str, Any]:
        """Return Exit after durably handing off every complete M1 decision."""

        if self.smart_entry is None:
            raise ExitDecisionUnavailable("unified_model_not_loaded")
        if not callable(on_bar_committed):
            raise ExitDecisionUnavailable(
                "exit_bar_commit_handler_missing",
            )
        try:
            _require_unified_model(
                self.smart_entry,
                context="V12_PIPELINE_EXIT",
            )
        except RuntimeError as exc:
            raise ExitDecisionUnavailable(
                "unified_model_exit_not_admitted",
                error=str(exc),
            ) from exc
        entry_snapshot = getattr(trade, "v10_snapshot", None)
        if not isinstance(entry_snapshot, Mapping) or not entry_snapshot:
            raise ExitDecisionUnavailable(
                "entry_snapshot_missing_for_exit",
            )
        now_value = pd.Timestamp(now_minute)
        if (
            pd.isna(now_value)
            or now_value.tzinfo is None
            or now_value.utcoffset() != pd.Timedelta(0)
            or now_value != now_value.floor("min")
        ):
            raise ExitDecisionUnavailable(
                "exit_clock_not_exact_utc_minute",
                now_minute=str(now_minute),
            )
        now_value = now_value.tz_convert("UTC")
        del bid, ask
        entry_ts = getattr(trade, "entry_ts", None)
        last_processed = getattr(trade, "last_processed_m1_ts", None)
        try:
            entry_ts = pd.Timestamp(entry_ts)
        except Exception as exc:
            raise ExitDecisionUnavailable(
                "trade_entry_fill_time_invalid",
            ) from exc
        if (
            pd.isna(entry_ts)
            or entry_ts.tzinfo is None
            or entry_ts.utcoffset() != pd.Timedelta(0)
        ):
            raise ExitDecisionUnavailable(
                "trade_entry_fill_time_invalid",
            )
        if last_processed is None:
            next_bar_floor = entry_ts.tz_convert("UTC").ceil("min")
            next_bar_side = "left"
        else:
            try:
                last_processed_ts = pd.Timestamp(last_processed)
            except Exception as exc:
                raise ExitDecisionUnavailable(
                    "last_processed_m1_time_invalid",
                ) from exc
            if (
                pd.isna(last_processed_ts)
                or last_processed_ts.tzinfo is None
                or last_processed_ts.utcoffset() != pd.Timedelta(0)
                or last_processed_ts != last_processed_ts.floor("min")
            ):
                raise ExitDecisionUnavailable(
                    "last_processed_m1_time_invalid",
                    last_processed_m1_ts=str(last_processed),
                )
            next_bar_floor = last_processed_ts.tz_convert("UTC")
            next_bar_side = "right"
        wall_latest_closed_bar = now_value - pd.Timedelta(minutes=1)
        try:
            self.prebuilt_loader.refresh_if_changed()
            source_snapshot = (
                self.prebuilt_loader.acquire_serving_snapshot()
            )
            source_index = source_snapshot.base28.index
        except Exception as exc:
            raise ExitDecisionUnavailable(
                "exit_source_refresh_or_snapshot_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            ) from exc
        if (
            not isinstance(source_index, pd.DatetimeIndex)
            or source_index.empty
            or source_index.hasnans
            or not source_index.is_unique
            or not source_index.is_monotonic_increasing
        ):
            raise ExitDecisionUnavailable(
                "exit_source_m1_index_invalid",
            )
        source_latest_closed_bar = pd.Timestamp(source_index[-1])
        if (
            source_latest_closed_bar.tzinfo is None
            or source_latest_closed_bar.utcoffset() != pd.Timedelta(0)
            or source_latest_closed_bar
            != source_latest_closed_bar.floor("min")
        ):
            raise ExitDecisionUnavailable(
                "exit_source_m1_tail_invalid",
                source_latest_m1_bar=str(source_latest_closed_bar),
            )
        source_latest_closed_bar = (
            source_latest_closed_bar.tz_convert("UTC")
        )
        if source_latest_closed_bar > wall_latest_closed_bar:
            raise ExitDecisionUnavailable(
                "exit_source_contains_unclosed_m1_bar",
                source_latest_m1_bar=source_latest_closed_bar.isoformat(),
                wall_latest_closed_m1_bar=(
                    wall_latest_closed_bar.isoformat()
                ),
            )
        latest_closed_bar = source_latest_closed_bar
        source_times_ns = source_index.as_unit("ns").asi8
        next_position = int(
            np.searchsorted(
                source_times_ns,
                next_bar_floor.value,
                side=next_bar_side,
            )
        )
        if next_position >= len(source_index):
            raise ExitDecisionUnavailable(
                "awaiting_authoritative_closed_m1_bar",
                next_required_m1_bar_after=next_bar_floor.isoformat(),
                source_latest_closed_m1_bar=(
                    latest_closed_bar.isoformat()
                ),
                wall_latest_closed_m1_bar=(
                    wall_latest_closed_bar.isoformat()
                ),
            )

        last_decision: dict[str, Any] | None = None
        for raw_next_bar in source_index[next_position:]:
            next_bar = pd.Timestamp(raw_next_bar).tz_convert("UTC")
            if next_bar > latest_closed_bar:
                break
            try:
                closed_bar = self.prebuilt_loader.get_closed_m1_bar(
                    next_bar,
                    snapshot=source_snapshot,
                )
                staged = trade.clone_for_exit_decision()
                staged.update_bar(**closed_bar)
                staged_snapshot = staged.require_entry_snapshot()
                path_envelope = staged.build_closed_m1_path_evidence()
                exit_feature_surface = (
                    self.smart_entry.build_exit_feature_surface(
                        decision_time=next_bar,
                        prebuilt_snapshot=source_snapshot,
                    )
                )
                decision = self.smart_entry.decide_exit(
                    entry_snapshot=staged_snapshot,
                    exit_path_envelope=path_envelope,
                    exit_feature_surface=exit_feature_surface,
                    entry_bid=staged.entry_bid,
                    entry_ask=staged.entry_ask,
                    side=staged.side,
                )
                bundle_sha256 = getattr(
                    self.smart_entry,
                    "_bundle_sha256",
                    None,
                )
                decision = require_unified_exit_output(
                    decision,
                    context="V12_PIPELINE_EXIT",
                    expected_bundle_sha256=bundle_sha256,
                    entry_snapshot=staged_snapshot,
                    exit_path_envelope=path_envelope,
                )
                staged.bind_unified_exit_decision(
                    decision,
                    expected_bundle_sha256=bundle_sha256,
                )
                trade.commit_complete_exit_bar(staged)
                # The runtime owner must persist this exact atomic state and
                # journal its complete model proof before catch-up advances to
                # another M1 bar or exposes EXIT_NOW to broker execution.
                on_bar_committed(trade)
            except Exception as exc:
                raise ExitDecisionUnavailable(
                    "unified_exit_decision_failed",
                    m1_bar=next_bar.isoformat(),
                    error_type=type(exc).__name__,
                    error=str(exc),
                ) from exc
            last_decision = decision
            if decision["action"] == "EXIT_NOW":
                return decision

        if last_decision is None:
            raise ExitDecisionUnavailable(
                "unified_exit_decision_missing",
                latest_closed_m1_bar=latest_closed_bar.isoformat(),
            )
        return last_decision
