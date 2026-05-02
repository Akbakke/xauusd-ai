from __future__ import annotations

import json
from pathlib import Path
from typing import Any


NOT_READY_FOR_IQL_TRAINING = "NOT_READY_FOR_IQL_TRAINING"
READY_FOR_IQL_TRAINING_STUB_ONLY = "READY_FOR_IQL_TRAINING_STUB_ONLY"

REQUIRED_DATASET_FIELDS_V1 = [
    "episode_id",
    "transition_id",
    "state_vector",
    "state_feature_names",
    "action",
    "action_id",
    "reward",
    "next_state_vector",
    "done",
    "discount",
    "decision_ts",
    "candidate_uid_exact",
    "source_policy_version",
    "behavior_policy_status",
    "support_status",
    "as_of_schema_version",
    "reward_version",
    "outcome_backfill_version",
]

ACTION_ID_MAP_V1 = {"HOLD": 0, "EXIT_NOW": 1}


def _check(name: str, passed: bool, observed: Any, expected: Any, note: str) -> dict[str, Any]:
    return {
        "check_name_v1": name,
        "status_v1": "PASS" if passed else "FAIL",
        "observed_value_v1": observed,
        "expected_value_v1": expected,
        "note_v1": note,
    }


def _field_status(schema: dict[str, Any], field_name: str) -> str:
    fields = schema.get("fields_v1", [])
    if isinstance(fields, list):
        for row in fields:
            if isinstance(row, dict) and row.get("field_name_v1") == field_name:
                return str(row.get("status_v1", "MISSING"))
    return "MISSING"


def _state_features(schema: dict[str, Any]) -> list[str]:
    features = schema.get("state_feature_names_v1", [])
    return [str(item) for item in features] if isinstance(features, list) else []


def validate_iql_dataset_schema_v1(schema: dict[str, Any]) -> dict[str, Any]:
    """Validate an IQL dataset schema/contract without training any model.

    The harness is intentionally conservative. It only returns a ready status if
    the upstream scaffold has already proven MDP readiness, a locked reward, and
    acceptable support. This module is a gate/stub, not an IQL trainer.
    """

    state_features = _state_features(schema)
    readiness = schema.get("readiness_gates_v1", {}) if isinstance(schema.get("readiness_gates_v1"), dict) else {}
    transition_counts = schema.get("transition_counts_v1", {}) if isinstance(schema.get("transition_counts_v1"), dict) else {}
    reward_contract = schema.get("reward_contract_v1", {}) if isinstance(schema.get("reward_contract_v1"), dict) else {}
    support = schema.get("support_v1", {}) if isinstance(schema.get("support_v1"), dict) else {}
    baseline = schema.get("baseline_comparator_presence_v1", {})
    if not isinstance(baseline, dict):
        baseline = {}

    missing_fields = [field for field in REQUIRED_DATASET_FIELDS_V1 if _field_status(schema, field) == "MISSING"]
    unavailable_fields = [
        field
        for field in REQUIRED_DATASET_FIELDS_V1
        if _field_status(schema, field) in {"MISSING", "NOT_ESTABLISHED", "PENDING_REPLAY"}
    ]
    forbidden_state_tokens = [
        field
        for field in state_features
        if any(token in field.lower() for token in ("hindsight", "terminal", "reward", "outcome"))
    ]
    action_id_map = schema.get("action_id_map_v1", {})

    mdp_verdict = str(readiness.get("management_mdp_verdict_v1", "NOT_READY"))
    full_transitions = int(transition_counts.get("full_sequence_ready_transition_count_v1", 0) or 0)
    hold_next = int(transition_counts.get("hold_to_next_state_transition_count_v1", 0) or 0)
    exit_done = int(transition_counts.get("exit_now_to_done_transition_count_v1", 0) or 0)
    locked_reward = bool(reward_contract.get("locked_scalar_reward_v1", False))
    support_verdict = str(support.get("overall_support_verdict_v1", "NOT_ESTABLISHED"))
    baseline_required = {
        "no_rl_baseline_v1",
        "r6_frozen_shadow_fallback_v1",
        "r5_2_frozen_reference_v1",
        "management_harvest_candidate_v1",
        "supervised_exit_local_tree_baseline_v1",
    }
    registered_baseline_statuses = {"READY", "REFERENCE_REGISTERED", "CALIBRATION_PENDING", "SANITY_ONLY_READY", "SANITY_ONLY_REGISTERED"}
    baseline_missing = sorted(
        name
        for name in baseline_required
        if str(baseline.get(name, {}).get("status_v1", "MISSING")) not in registered_baseline_statuses
    )

    checks = [
        _check(
            "REQUIRED_SCHEMA_FIELDS_PRESENT",
            not missing_fields,
            missing_fields,
            "[]",
            "All IQL dataset fields must be declared, even if some are not ready yet.",
        ),
        _check(
            "NO_UNAVAILABLE_REQUIRED_FIELD_FOR_TRAINING",
            not unavailable_fields,
            unavailable_fields,
            "[]",
            "Training cannot start while required fields are missing, pending replay, or not established.",
        ),
        _check(
            "AS_OF_STATE_HAS_NO_HINDSIGHT_OR_REWARD_FIELDS",
            not forbidden_state_tokens,
            forbidden_state_tokens,
            "[]",
            "State features must remain AS_OF only; HINDSIGHT and reward channels are terminal/outcome only.",
        ),
        _check(
            "ACTION_ID_CONTRACT_LOCKED",
            action_id_map == ACTION_ID_MAP_V1,
            action_id_map,
            ACTION_ID_MAP_V1,
            "Management IQL scaffold only supports HOLD=0 and EXIT_NOW=1.",
        ),
        _check(
            "MDP_READY_GATE",
            mdp_verdict == "MDP_READY",
            mdp_verdict,
            "MDP_READY",
            "Sequence IQL requires a real MDP verdict, not bandit-only or partial transitions.",
        ),
        _check(
            "NEXT_STATE_DONE_GATE",
            full_transitions > 0 and hold_next > 0 and exit_done > 0,
            {
                "full_sequence_ready_transition_count_v1": full_transitions,
                "hold_to_next_state_transition_count_v1": hold_next,
                "exit_now_to_done_transition_count_v1": exit_done,
            },
            "positive counts for full, HOLD->next, and EXIT_NOW->done",
            "Terminal-only transitions are not enough for sequence IQL when HOLD has no next state.",
        ),
        _check(
            "LOCKED_REWARD_GATE",
            locked_reward,
            locked_reward,
            True,
            "Reward candidates may exist, but training requires one locked scalar reward_version.",
        ),
        _check(
            "SUPPORT_GATE",
            support_verdict == "SUPPORT_ACCEPTABLE_FOR_RESEARCH",
            support_verdict,
            "SUPPORT_ACCEPTABLE_FOR_RESEARCH",
            "IQL training should not start if support is too thin or only weak/bandit-usable.",
        ),
        _check(
            "BASELINE_COMPARATOR_REGISTRY_PRESENT",
            not baseline_missing,
            baseline_missing,
            "[]",
            "Comparator slots must be registered; calibration/performance analysis is outside this foundation harness.",
        ),
    ]

    failed = [row for row in checks if row["status_v1"] != "PASS"]
    status = READY_FOR_IQL_TRAINING_STUB_ONLY if not failed else NOT_READY_FOR_IQL_TRAINING
    return {
        "harness_id_v1": "IQL_TRAINING_HARNESS_STUB_V1",
        "training_executed_v1": False,
        "status_v1": status,
        "failed_check_count_v1": int(len(failed)),
        "checks_v1": checks,
        "stop_reason_v1": None if not failed else NOT_READY_FOR_IQL_TRAINING,
        "not_live_gate_v1": True,
        "not_controller_v1": True,
        "not_policy_v1": True,
    }


def validate_iql_dataset_schema_file_v1(path: str | Path) -> dict[str, Any]:
    schema_path = Path(path).expanduser().resolve()
    return validate_iql_dataset_schema_v1(json.loads(schema_path.read_text(encoding="utf-8")))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate an IQL dataset schema without training.")
    parser.add_argument("schema_json", type=str)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    result = validate_iql_dataset_schema_file_v1(args.schema_json)
    payload = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).expanduser().resolve().write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
