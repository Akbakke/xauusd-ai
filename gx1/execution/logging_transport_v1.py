from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional


LOGGING_TRANSPORT_LAYER_ID = "LOGGING_TRANSPORT_INFO_THROTTLE_V1"
EVENT_TRACE_FILE = "OBSERVABILITY_EVENT_TRACE_V1.jsonl"
SUMMARY_FILE = "OBSERVABILITY_TRANSPORT_SUMMARY_V1.json"
MANIFEST_FILE = "OBSERVABILITY_TRANSPORT_MANIFEST_V1.json"
STATUS_FILE = "OBSERVABILITY_TRANSPORT_STATUS_V1.json"
RINGBUFFER_FILE = "OBSERVABILITY_TRANSPORT_RINGBUFFER_V1.json"

INFO_POLICIES = {
    "KEEP_INFO",
    "FIRST_ONLY",
    "FIRST_AND_FINAL",
    "FIRST_EVERY_N_FINAL",
    "SUMMARY_ONLY",
    "DEBUG_ONLY",
}

POLICY_ACTIONS = {
    "KEEP_INFO",
    "RATE_LIMIT_INFO",
    "MOVE_TO_TRACE",
    "MOVE_TO_SUMMARY",
    "DEBUG_ONLY",
    "REMOVE_REDUNDANT",
}

MID_EDGE_EXPECTED_SKIP_REASONS = {
    "SKIP_NOT_LONG",
    "SKIP_OTHER",
    "SKIP_SESSION_FILTER",
    "SKIP_NO_MEANINGFUL_MFE_FIELD",
    "SKIP_MFE_BELOW_10",
    "SKIP_MFE_AT_OR_ABOVE_50",
    "SKIP_PROFIT_PROTECT_SCORE_TOO_LOW",
    "SKIP_PNL_BELOW_FLOOR",
    "SKIP_ALREADY_IN_OTHER_STATE",
    "SKIP_THRESHOLD_PATH_TAKES_PRECEDENCE",
}

HOT_PATH_LOGGING_POLICIES_V1: Dict[str, Dict[str, Any]] = {
    "REPLAY_PROGRESS_HEARTBEAT": {
        "file": "gx1/execution/oanda_demo_runner.py",
        "function": "_run_replay_impl",
        "event_name": "REPLAY_PROGRESS",
        "trigger_type": "progress_heartbeat",
        "frequency_type": "per-run",
        "observability_purpose": "heartbeat",
        "policy_action": "KEEP_INFO",
        "default_info_policy": "KEEP_INFO",
        "every_n": 0,
        "trace_mode": "NEVER",
        "existing_artifact_truth": "REPLAY_SUMMARY.json, RUN_COMPLETED.json",
        "same_truth_already_exists": True,
        "risk_if_muted": "Operator loses live progress heartbeat.",
    },
    "REPLAY_PROGRESS_REDUNDANT": {
        "file": "gx1/execution/oanda_demo_runner.py",
        "function": "_run_replay_impl",
        "event_name": "REPLAY_PROGRESS_REDUNDANT",
        "trigger_type": "duplicate_progress_heartbeat",
        "frequency_type": "per-run",
        "observability_purpose": "heartbeat",
        "policy_action": "REMOVE_REDUNDANT",
        "default_info_policy": "DEBUG_ONLY",
        "every_n": 0,
        "trace_mode": "NEVER",
        "existing_artifact_truth": "Primary [REPLAY PROGRESS] heartbeat already exists.",
        "same_truth_already_exists": True,
        "risk_if_muted": "None; duplicate of the main replay heartbeat.",
    },
    "ENTRY_NO_TRADE_CLOSED_WINDOW_PROOF": {
        "file": "gx1/execution/entry_manager.py",
        "function": "EntryManager.evaluate_entry",
        "event_name": "ENTRY_NO_TRADE_CLOSED_WINDOW_PROOF",
        "trigger_type": "closed_market_window",
        "frequency_type": "per-bar",
        "observability_purpose": "proof",
        "policy_action": "RATE_LIMIT_INFO",
        "default_info_policy": "FIRST_ONLY",
        "every_n": 0,
        "trace_mode": "EMIT_OR_ABNORMAL",
        "existing_artifact_truth": "Summary counters only; INFO is not canonical.",
        "same_truth_already_exists": False,
        "risk_if_muted": "Lose minute-by-minute stdout proof; preserved by counters and final summary.",
    },
    "ENTRY_GAP_GUARD": {
        "file": "gx1/execution/entry_manager.py",
        "function": "EntryManager.evaluate_entry",
        "event_name": "ENTRY_GAP_GUARD",
        "trigger_type": "gap_detected_or_gap_cooldown",
        "frequency_type": "per-bar",
        "observability_purpose": "abnormality",
        "policy_action": "RATE_LIMIT_INFO",
        "default_info_policy": "FIRST_ONLY",
        "every_n": 0,
        "trace_mode": "EMIT_OR_ABNORMAL",
        "existing_artifact_truth": "Summary counters only; INFO is not canonical.",
        "same_truth_already_exists": False,
        "risk_if_muted": "Lose repeated cooldown spam, but keep first-signal and count of blocked bars.",
    },
    "MID_EDGE_10_50_PROBE": {
        "file": "gx1/execution/exit_manager.py",
        "function": "ExitManager._protected_profit_state",
        "event_name": "MID_EDGE_10_50_PROBE",
        "trigger_type": "mid_edge_state_transition",
        "frequency_type": "per-trade",
        "observability_purpose": "proof",
        "policy_action": "MOVE_TO_SUMMARY",
        "default_info_policy": "SUMMARY_ONLY",
        "every_n": 0,
        "trace_mode": "EMIT_OR_ABNORMAL",
        "existing_artifact_truth": "EXIT_EVAL_TRACE.csv carries exit-state scalars; summary carries reason counts.",
        "same_truth_already_exists": True,
        "risk_if_muted": "Repeated skip reasons vanish from stdout; ARMED and abnormal transitions remain visible.",
    },
    "EXIT_CATA_GUARD_EVENT": {
        "file": "gx1/execution/exit_manager.py",
        "function": "ExitManager.evaluate_and_close_trades",
        "event_name": "EXIT_CATA_GUARD_EVENT",
        "trigger_type": "catastrophic_guard_trigger",
        "frequency_type": "per-event",
        "observability_purpose": "abnormality",
        "policy_action": "RATE_LIMIT_INFO",
        "default_info_policy": "KEEP_INFO",
        "every_n": 0,
        "trace_mode": "EMIT_OR_ABNORMAL",
        "existing_artifact_truth": "EXIT_EVAL_TRACE.csv, trade_log.csv, trade_outcomes parquet.",
        "same_truth_already_exists": True,
        "risk_if_muted": "Operator loses the immediate guard-close signal; kept as one structured INFO event.",
    },
    "EXIT_INPUT_PREP_BREAKDOWN": {
        "file": "gx1/execution/exit_manager.py",
        "function": "ExitManager._build_exit_window_array",
        "event_name": "EXIT_INPUT_PREP_BREAKDOWN",
        "trigger_type": "profiling_snapshot",
        "frequency_type": "per-event",
        "observability_purpose": "profiling",
        "policy_action": "MOVE_TO_SUMMARY",
        "default_info_policy": "FIRST_EVERY_N_FINAL",
        "every_n": 5,
        "trace_mode": "SUMMARY_SAMPLE_ONLY",
        "existing_artifact_truth": "Summary snapshot in observability transport only.",
        "same_truth_already_exists": False,
        "risk_if_muted": "Lose live profiling details unless they are checkpointed to summary.",
    },
    "EXIT_DECISION_RESULT": {
        "file": "gx1/execution/exit_manager.py",
        "function": "ExitManager.evaluate_and_close_trades",
        "event_name": "EXIT_DECISION_RESULT",
        "trigger_type": "actual_close_decision_result",
        "frequency_type": "per-event",
        "observability_purpose": "proof",
        "policy_action": "KEEP_INFO",
        "default_info_policy": "KEEP_INFO",
        "every_n": 0,
        "trace_mode": "NEVER",
        "existing_artifact_truth": "trade_log.csv, trade_journal parquet, EXIT_EVAL_TRACE.csv",
        "same_truth_already_exists": True,
        "risk_if_muted": "Lose accepted/rejected close visibility.",
    },
    "EXIT_REPLAY_THRESHOLD_REJECT_HOLD_FASTPATH": {
        "file": "gx1/execution/exit_manager.py",
        "function": "ExitManager.evaluate_and_close_trades",
        "event_name": "EXIT_REPLAY_THRESHOLD_REJECT_HOLD_FASTPATH",
        "trigger_type": "replay_hold_fastpath",
        "frequency_type": "per-event",
        "observability_purpose": "abnormality",
        "policy_action": "RATE_LIMIT_INFO",
        "default_info_policy": "FIRST_EVERY_N_FINAL",
        "every_n": 5000,
        "trace_mode": "EMIT_OR_ABNORMAL",
        "existing_artifact_truth": "EXIT_EVAL_TRACE.csv",
        "same_truth_already_exists": True,
        "risk_if_muted": "Low; existing rate-limited contract already protects stdout.",
    },
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, deque):
        return [_json_ready(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_ready(v) for v in value)
    if isinstance(value, Counter):
        return {str(k): int(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def build_logging_surface_inventory_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for family, payload in HOT_PATH_LOGGING_POLICIES_V1.items():
        row = dict(payload)
        row["family"] = family
        rows.append(row)
    return rows


def is_mid_edge_unexpected(reason: str) -> bool:
    cleaned = str(reason or "").strip().upper()
    return bool(cleaned) and cleaned not in MID_EDGE_EXPECTED_SKIP_REASONS and cleaned != "ARMED"


@dataclass
class _FamilyState:
    total_count: int = 0
    emitted_count: int = 0
    suppressed_count: int = 0
    reason_counts: Counter = field(default_factory=Counter)
    key_counts: Counter = field(default_factory=Counter)
    unique_trade_ids: set[str] = field(default_factory=set)
    first_seen_utc: Optional[str] = None
    last_seen_utc: Optional[str] = None
    sample_events: List[Dict[str, Any]] = field(default_factory=list)
    worst_events: List[Dict[str, Any]] = field(default_factory=list)


class ReplayObservabilityTransport:
    def __init__(
        self,
        *,
        output_dir: Path,
        run_id: str,
        chunk_id: str,
        debug_mode: bool = False,
        heavy_week_mode: bool = False,
        canary_mode: bool = False,
        flush_interval_sec: float = 30.0,
        ringbuffer_max: int = 200,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = self.output_dir / "observability_transport_v1"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.event_trace_path = self.artifact_dir / EVENT_TRACE_FILE
        self.summary_path = self.artifact_dir / SUMMARY_FILE
        self.manifest_path = self.artifact_dir / MANIFEST_FILE
        self.status_path = self.artifact_dir / STATUS_FILE
        self.ringbuffer_path = self.artifact_dir / RINGBUFFER_FILE
        self.run_id = str(run_id)
        self.chunk_id = str(chunk_id)
        self.debug_mode = bool(debug_mode or canary_mode)
        self.heavy_week_mode = bool(heavy_week_mode)
        self.canary_mode = bool(canary_mode)
        self.flush_interval_sec = max(float(flush_interval_sec), 1.0)
        self.ringbuffer_max = max(int(ringbuffer_max), 10)
        self._family_state: Dict[str, _FamilyState] = {}
        self._ringbuffer: Deque[Dict[str, Any]] = deque(maxlen=self.ringbuffer_max)
        self._last_flush_monotonic = 0.0
        self._checkpoint_count = 0
        self._failure_mode = False
        self._failure_reason = None
        self._event_trace_rows = 0
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "layer_id": LOGGING_TRANSPORT_LAYER_ID,
            "created_at_utc": _now_utc(),
            "run_id": self.run_id,
            "chunk_id": self.chunk_id,
            "artifact_dir": str(self.artifact_dir),
            "event_trace_path": str(self.event_trace_path),
            "summary_path": str(self.summary_path),
            "status_path": str(self.status_path),
            "ringbuffer_path": str(self.ringbuffer_path),
            "debug_mode": self.debug_mode,
            "heavy_week_mode": self.heavy_week_mode,
            "canary_mode": self.canary_mode,
            "flush_interval_sec": self.flush_interval_sec,
            "ringbuffer_max": self.ringbuffer_max,
            "design_contract_v1": {
                "info_is_not_proof_canonical": True,
                "truth_lives_in_trace_or_artifacts": True,
                "summary_gets_richer_when_info_gets_quieter": True,
                "debug_mode_is_opt_in": True,
                "normal_replay_should_be_heartbeat_dominated": True,
            },
        }
        self.manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")

    def _family(self, family: str) -> _FamilyState:
        if family not in self._family_state:
            self._family_state[family] = _FamilyState()
        return self._family_state[family]

    def _should_emit(
        self,
        *,
        policy: str,
        occurrence: int,
        every_n: int,
        final: bool,
    ) -> bool:
        if policy not in INFO_POLICIES:
            policy = "SUMMARY_ONLY"
        if policy == "KEEP_INFO":
            return True
        if policy == "FIRST_ONLY":
            return occurrence == 1
        if policy == "FIRST_AND_FINAL":
            return occurrence == 1 or final
        if policy == "FIRST_EVERY_N_FINAL":
            if occurrence == 1 or final:
                return True
            if every_n > 0 and (occurrence % every_n) == 0:
                return True
            return False
        if policy == "DEBUG_ONLY":
            return bool(self.debug_mode)
        return False

    def _append_event_trace(self, payload: Dict[str, Any]) -> None:
        with self.event_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_ready(payload), sort_keys=True) + "\n")
        self._event_trace_rows += 1

    def _push_ringbuffer(self, payload: Dict[str, Any]) -> None:
        self._ringbuffer.append(_json_ready(payload))

    def _update_samples(
        self,
        *,
        state: _FamilyState,
        payload: Optional[Dict[str, Any]],
        severity: Optional[float],
    ) -> None:
        if not payload:
            return
        normalized = _json_ready(payload)
        if len(state.sample_events) < 5:
            state.sample_events.append(normalized)
        if severity is None:
            return
        record = dict(normalized)
        record["_severity"] = float(severity)
        state.worst_events.append(record)
        state.worst_events.sort(key=lambda item: float(item.get("_severity", 0.0)), reverse=True)
        del state.worst_events[5:]

    def record_event(
        self,
        family: str,
        *,
        reason: Optional[str] = None,
        ts: Optional[Any] = None,
        trade_id: Optional[str] = None,
        key: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        info_policy: Optional[str] = None,
        every_n: Optional[int] = None,
        final: bool = False,
        abnormal: bool = False,
        severity: Optional[float] = None,
    ) -> bool:
        cfg = HOT_PATH_LOGGING_POLICIES_V1.get(family, {})
        policy = str(info_policy or cfg.get("default_info_policy", "SUMMARY_ONLY"))
        cadence = int(cfg.get("every_n", 0) if every_n is None else every_n)
        capture_mode = str(cfg.get("trace_mode", "EMIT_OR_ABNORMAL"))
        normalized_reason = str(reason or "UNSPECIFIED")
        normalized_key = str(key or normalized_reason or family)
        ts_text = None
        if ts is not None:
            try:
                ts_text = ts.isoformat()
            except Exception:
                ts_text = str(ts)

        state = self._family(family)
        state.total_count += 1
        state.reason_counts[normalized_reason] += 1
        state.key_counts[normalized_key] += 1
        if trade_id:
            state.unique_trade_ids.add(str(trade_id))
        if state.first_seen_utc is None:
            state.first_seen_utc = ts_text or _now_utc()
        state.last_seen_utc = ts_text or _now_utc()

        occurrence = int(state.key_counts[normalized_key])
        emit_info = self._should_emit(policy=policy, occurrence=occurrence, every_n=cadence, final=final)
        if self.heavy_week_mode and policy == "KEEP_INFO":
            emit_info = occurrence == 1
        if self.debug_mode and policy != "SUMMARY_ONLY":
            emit_info = True

        if emit_info:
            state.emitted_count += 1
        else:
            state.suppressed_count += 1

        event_payload = {
            "family": family,
            "reason": normalized_reason,
            "key": normalized_key,
            "run_id": self.run_id,
            "chunk_id": self.chunk_id,
            "timestamp_utc": ts_text or _now_utc(),
            "trade_id": trade_id,
            "abnormal": bool(abnormal),
            "info_policy": policy,
            "occurrence": occurrence,
            "payload": payload or {},
        }
        if emit_info or abnormal or self.debug_mode:
            self._push_ringbuffer(event_payload)
        if capture_mode == "EMIT_OR_ABNORMAL" and (emit_info or abnormal or self.debug_mode):
            self._append_event_trace(event_payload)
        elif capture_mode == "ALWAYS_DEBUG" and self.debug_mode:
            self._append_event_trace(event_payload)
        self._update_samples(state=state, payload=event_payload, severity=severity)
        return emit_info

    def mark_failure(self, reason: str) -> None:
        self._failure_mode = True
        self._failure_reason = str(reason)
        self.flush(reason="failure", force=True)

    def build_summary(self, *, checkpoint_reason: str) -> Dict[str, Any]:
        families = {}
        for family, state in sorted(self._family_state.items()):
            families[family] = {
                "policy": HOT_PATH_LOGGING_POLICIES_V1.get(family, {}).get("default_info_policy"),
                "policy_action": HOT_PATH_LOGGING_POLICIES_V1.get(family, {}).get("policy_action"),
                "total_count": int(state.total_count),
                "emitted_count": int(state.emitted_count),
                "suppressed_count": int(state.suppressed_count),
                "unique_trade_count": int(len(state.unique_trade_ids)),
                "first_seen_utc": state.first_seen_utc,
                "last_seen_utc": state.last_seen_utc,
                "reason_counts": _json_ready(state.reason_counts),
                "top_keys": [
                    {"key": str(key), "count": int(count)}
                    for key, count in state.key_counts.most_common(5)
                ],
                "sample_events": _json_ready(state.sample_events),
                "worst_events": _json_ready(state.worst_events),
            }
        return {
            "layer_id": LOGGING_TRANSPORT_LAYER_ID,
            "checkpoint_reason": checkpoint_reason,
            "updated_at_utc": _now_utc(),
            "run_id": self.run_id,
            "chunk_id": self.chunk_id,
            "debug_mode": self.debug_mode,
            "heavy_week_mode": self.heavy_week_mode,
            "canary_mode": self.canary_mode,
            "event_trace_rows": int(self._event_trace_rows),
            "checkpoint_count": int(self._checkpoint_count),
            "failure_mode": self._failure_mode,
            "failure_reason": self._failure_reason,
            "families": families,
        }

    def flush(self, *, reason: str, force: bool = False) -> Optional[Path]:
        now_mono = time.monotonic()
        if not force and (now_mono - self._last_flush_monotonic) < self.flush_interval_sec:
            return None
        self._checkpoint_count += 1
        self._last_flush_monotonic = now_mono
        summary = self.build_summary(checkpoint_reason=reason)
        self.summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
        self.status_path.write_text(
            json.dumps(
                _json_ready(
                    {
                        "layer_id": LOGGING_TRANSPORT_LAYER_ID,
                        "updated_at_utc": _now_utc(),
                        "checkpoint_reason": reason,
                        "summary_path": str(self.summary_path),
                        "event_trace_rows": int(self._event_trace_rows),
                        "failure_mode": self._failure_mode,
                        "failure_reason": self._failure_reason,
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        self.ringbuffer_path.write_text(
            json.dumps(
                _json_ready(
                    {
                        "updated_at_utc": _now_utc(),
                        "run_id": self.run_id,
                        "chunk_id": self.chunk_id,
                        "events": list(self._ringbuffer),
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        return self.summary_path


def default_transport_output_dir(
    *,
    explicit_output_dir: Optional[Path],
    output_dir: Optional[Path],
    log_dir: Optional[Path],
) -> Optional[Path]:
    if explicit_output_dir is not None:
        return Path(explicit_output_dir)
    if output_dir is not None:
        return Path(output_dir)
    if log_dir is not None:
        return Path(log_dir).parent
    return None


def transport_debug_mode_from_env(*, canary_mode: bool) -> bool:
    return bool(canary_mode or os.environ.get("GX1_LOG_TRANSPORT_DEBUG", "0") == "1")


def transport_heavy_week_mode_from_env() -> bool:
    return bool(os.environ.get("GX1_LOG_TRANSPORT_HEAVY_WEEK", "0") == "1")


def transport_ringbuffer_max_from_env() -> int:
    try:
        return max(int(os.environ.get("GX1_LOG_TRANSPORT_RINGBUFFER_MAX", "200")), 10)
    except Exception:
        return 200


def transport_flush_interval_from_env() -> float:
    try:
        return max(float(os.environ.get("GX1_LOG_TRANSPORT_FLUSH_SEC", "30")), 1.0)
    except Exception:
        return 30.0


def ensure_policy_rows_have_valid_contract(rows: Iterable[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    seen: set[str] = set()
    for row in rows:
        family = str(row.get("family", ""))
        if not family:
            errors.append("missing family")
            continue
        if family in seen:
            errors.append(f"duplicate family={family}")
        seen.add(family)
        if str(row.get("default_info_policy", "")) not in INFO_POLICIES:
            errors.append(f"family={family} invalid info policy={row.get('default_info_policy')}")
        if str(row.get("policy_action", "")) not in POLICY_ACTIONS:
            errors.append(f"family={family} invalid policy action={row.get('policy_action')}")
    return errors
