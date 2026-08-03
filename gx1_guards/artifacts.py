"""
GX1 artifact loading — fail-closed.

Enforces guardrails that are too important to leave to AGENTS.md text:
  - No implicit latest/glob selection for decisioning.
  - Only artifacts marked ACTIVE in the selection contract are decision-valid.
  - No cross-project artifacts in an XAUUSD run (project isolation).
  - No old/invalidated artifacts for decisioning.

A rule in a markdown file is a hope. A function that raises is a guarantee.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
    require_newest_immutable_event,
    select_latest_immutable_event,
)
from gx1.contracts.live_tail_publication_v1 import (
    LiveTailAuthorityError,
    require_live_tail_launch_authority,
    require_newest_live_tail_runtime_authority,
)
from gx1_guards import REPO_ROOT
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_FAMILY_COUNT,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    MODEL_NATIVE_SIZING_MODE_LEARNED,
    ModelNativeSizingUnavailable,
    prepare_model_native_sizing_authority,
    require_model_native_sizing_authority_contract,
)
from gx1.contracts.entry_model_native_sizing_execution_v1 import (
    ModelNativeSizingExecutionContractError,
    load_bound_runtime_sizing_parity,
    require_canonical_unified_replay_launch_authority,
    require_joint_exit_portfolio_capacity,
)
from gx1.contracts.entry_model_native_adaptation_lifecycle_v1 import (
    ModelNativeAdaptationLifecycleError,
    require_launch_adaptation_authority,
)
from gx1.contracts.entry_model_native_launch_approval_v1 import (
    require_entry_launch_approval,
)
from gx1.contracts.entry_model_native_launch_transaction_v1 import (
    EntryLaunchTransactionError,
    require_entry_launch_transaction,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_CLASS_ORDER,
    UNIFIED_ENTRY_EXIT_CONTRACT_SCHEMA_VERSION,
    UNIFIED_EXIT_ACTION_ORDER,
    require_model_direction_operating_point,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    RUNTIME_PREDICTION_EVIDENCE_SCHEMA_VERSION,
)
from gx1.contracts.model_native_serve_gate_v1 import (
    MODEL_NATIVE_DIRECTION_POCKET_SCHEMA_VERSION,
    MODEL_NATIVE_SERVE_PARITY_SCHEMA_VERSION,
    cross_gate_contract_failures,
    serve_gate_event_contract_failures,
)

# The selection contract is the ONE source of truth for what is decision-valid.
# Repo-root-relative so it resolves regardless of the caller's CWD.
SELECTION_CONTRACT = REPO_ROOT / "PROJECT_STATE_artifacts.json"
XAU_DIRECTION_LAUNCH_CONTRACT = REPO_ROOT / "PROJECT_STATE_xau_direction_launch.json"

THIS_PROJECT = "XAUUSD"
SERVE_GATE_EVIDENCE_CONTRACT = {
    "model_native_serve_parity": (
        "MODEL_NATIVE_SERVE_PARITY",
        MODEL_NATIVE_SERVE_PARITY_SCHEMA_VERSION,
    ),
    "model_native_direction_pocket_audit": (
        "MODEL_NATIVE_DIRECTION_POCKET_AUDIT",
        MODEL_NATIVE_DIRECTION_POCKET_SCHEMA_VERSION,
    ),
}
SERVE_GATE_PREDICTION_PATH_FIELDS = {
    "model_native_serve_parity": "pinned_predictions",
    "model_native_direction_pocket_audit": "predictions_parquet",
}
REQUIRED_XAU_CTX_CONT_DIM = 142
REQUIRED_XAU_CTX_CAT_DIM = 5


class ArtifactGuardError(Exception):
    """Raised when a forbidden / unsafe artifact selection is attempted."""


def _load_contract() -> dict:
    if not SELECTION_CONTRACT.exists():
        raise ArtifactGuardError(
            f"No selection contract at {SELECTION_CONTRACT}. "
            "Decisioning requires an explicit ACTIVE contract — refusing to guess."
        )
    return json.loads(SELECTION_CONTRACT.read_text())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_bound_file(raw: object, *, label: str) -> Path:
    path = Path(str(raw or "")).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ArtifactGuardError(f"{label} is missing or not an absolute file: {path}")
    return path.resolve()


def _exact_sha256(raw: object, *, label: str) -> str:
    value = str(raw or "").strip().lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ArtifactGuardError(f"{label} is not an exact SHA-256")
    return value


def _validate_gate_prediction_lineage(
    payload: dict,
    *,
    evidence_name: str,
    accepted_bundle: Path,
) -> None:
    """Bind each launch-authoritative gate to immutable prediction evidence."""

    prediction = payload.get("prediction_evidence")
    if not isinstance(prediction, dict):
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} lacks prediction_evidence"
        )
    if (
        prediction.get("schema_version")
        != RUNTIME_PREDICTION_EVIDENCE_SCHEMA_VERSION
    ):
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction evidence schema mismatch"
        )
    if prediction.get("authoritative") is not True:
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction evidence is not authoritative"
        )
    if prediction.get("runtime_head_evidence_authoritative") is not True:
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} runtime-head evidence "
            "is not authoritative"
        )
    prediction_path = _absolute_bound_file(
        prediction.get("path"),
        label=f"XAU direction launch {evidence_name} prediction parquet",
    )
    if prediction_path.name == "selective_edge_predictions.parquet":
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} names the mutable prediction mirror"
        )
    expected_prediction_sha = _exact_sha256(
        prediction.get("sha256"),
        label=f"XAU direction launch {evidence_name} prediction parquet hash",
    )
    actual_prediction_sha = _sha256_file(prediction_path)
    if actual_prediction_sha != expected_prediction_sha:
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction parquet hash mismatch: "
            f"declared={expected_prediction_sha} actual={actual_prediction_sha}"
        )
    report_prediction_field = SERVE_GATE_PREDICTION_PATH_FIELDS[evidence_name]
    reported_prediction_path = Path(
        str(payload.get(report_prediction_field) or "")
    ).expanduser()
    if (
        not reported_prediction_path.is_absolute()
        or reported_prediction_path.resolve() != prediction_path
    ):
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} {report_prediction_field} does not "
            "equal its authoritative prediction evidence path"
        )

    report_binding = payload.get("prediction_report_evidence")
    if not isinstance(report_binding, dict) or set(report_binding) != {
        "json_path",
        "sha256",
    }:
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction_report_evidence must "
            "contain exact json_path/sha256"
        )
    report_path = _absolute_bound_file(
        report_binding.get("json_path"),
        label=f"XAU direction launch {evidence_name} prediction report",
    )
    if report_path.name.endswith("_latest.json"):
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} binds a mutable latest prediction report"
        )
    try:
        require_newest_immutable_event(
            report_path,
            "ENTRY_CANDIDATE_SELECTIVE_EDGE",
        )
    except ImmutableEventAuthorityError as exc:
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction report authority invalid: {exc}"
        ) from exc
    expected_report_sha = _exact_sha256(
        report_binding.get("sha256"),
        label=f"XAU direction launch {evidence_name} prediction report hash",
    )
    actual_report_sha = _sha256_file(report_path)
    if actual_report_sha != expected_report_sha:
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction report hash mismatch: "
            f"declared={expected_report_sha} actual={actual_report_sha}"
        )
    try:
        prediction_report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction report unreadable: {exc}"
        ) from exc
    if not isinstance(prediction_report, dict):
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction report is not an object"
        )
    if prediction_report.get("schema_version") != "entry_candidate_selective_edge_v1":
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction report schema mismatch"
        )
    if prediction_report.get("decision") != "PASS" or prediction_report.get("failures"):
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction report is not zero-failure PASS"
        )
    report_self_path = Path(
        str(prediction_report.get("json_path") or "")
    ).expanduser()
    if not report_self_path.is_absolute() or report_self_path.resolve() != report_path:
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction report json_path mismatch"
        )
    report_parquet = Path(
        str(prediction_report.get("predictions_path") or "")
    ).expanduser()
    if not report_parquet.is_absolute() or report_parquet.resolve() != prediction_path:
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction report parquet mismatch"
        )
    if prediction_report.get("prediction_evidence") != prediction:
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction declaration mismatch"
        )
    report_bundle = Path(
        str(prediction_report.get("bundle_dir") or "")
    ).expanduser()
    if not report_bundle.is_absolute() or report_bundle.resolve() != accepted_bundle.resolve():
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction bundle mismatch"
        )
    event_dataset = Path(str(payload.get("dataset_dir") or "")).expanduser()
    report_dataset = Path(
        str(prediction_report.get("dataset_dir") or "")
    ).expanduser()
    if (
        not event_dataset.is_absolute()
        or not report_dataset.is_absolute()
        or event_dataset.resolve() != report_dataset.resolve()
    ):
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction dataset mismatch"
        )
    metadata_path = _absolute_bound_file(
        prediction.get("bundle_metadata_path"),
        label=f"XAU direction launch {evidence_name} prediction bundle metadata",
    )
    if metadata_path != (accepted_bundle / "bundle_metadata.json").resolve():
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction metadata path mismatch"
        )
    expected_metadata_sha = _exact_sha256(
        prediction.get("bundle_metadata_sha256"),
        label=f"XAU direction launch {evidence_name} prediction metadata hash",
    )
    if _sha256_file(metadata_path) != expected_metadata_sha:
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction metadata hash mismatch"
        )
    if str(prediction_report.get("bundle_metadata_sha256") or "").lower() != expected_metadata_sha:
        raise ArtifactGuardError(
            f"XAU direction launch {evidence_name} prediction report metadata hash mismatch"
        )


def _validate_serve_gate_evidence(
    state: dict,
    *,
    accepted_bundle: Path,
) -> dict[str, dict[str, str]]:
    raw = state.get("serve_gate_evidence")
    if not isinstance(raw, dict) or set(raw) != set(SERVE_GATE_EVIDENCE_CONTRACT):
        raise ArtifactGuardError(
            "XAU direction launch contract serve_gate_evidence must have exact keys "
            f"{sorted(SERVE_GATE_EVIDENCE_CONTRACT)}"
        )
    validated: dict[str, dict[str, str]] = {}
    gate_payloads: dict[str, dict] = {}
    for name, (event_prefix, schema_version) in SERVE_GATE_EVIDENCE_CONTRACT.items():
        declaration = raw.get(name)
        if not isinstance(declaration, dict) or set(declaration) != {"json_path", "sha256"}:
            raise ArtifactGuardError(
                f"XAU direction launch {name} evidence must contain exact json_path/sha256"
            )
        event_path = Path(str(declaration.get("json_path") or "")).expanduser()
        if not event_path.is_absolute() or not event_path.is_file():
            raise ArtifactGuardError(
                f"XAU direction launch {name} event is missing or not absolute: {event_path}"
            )
        expected_sha = str(declaration.get("sha256") or "").strip().lower()
        if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
            raise ArtifactGuardError(f"XAU direction launch {name} lacks an exact SHA-256")
        actual_sha = _sha256_file(event_path)
        if actual_sha != expected_sha:
            raise ArtifactGuardError(
                f"XAU direction launch {name} hash mismatch: "
                f"contract={expected_sha} actual={actual_sha} path={event_path}"
            )
        try:
            newest = select_latest_immutable_event(event_path.parent, event_prefix)
        except ImmutableEventAuthorityError as exc:
            raise ArtifactGuardError(
                f"XAU direction launch {name} immutable event authority invalid: {exc}"
            ) from exc
        if newest is None or newest.resolve() != event_path.resolve():
            raise ArtifactGuardError(
                f"XAU direction launch {name} is not the newest immutable event: "
                f"declared={event_path.resolve()} newest={newest}"
            )
        try:
            payload = json.loads(event_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactGuardError(
                f"XAU direction launch {name} event unreadable: {event_path}: {exc}"
            ) from exc
        if payload.get("schema_version") != schema_version:
            raise ArtifactGuardError(
                f"XAU direction launch {name} schema mismatch: "
                f"{payload.get('schema_version')!r}"
            )
        if payload.get("decision") != "PASS":
            raise ArtifactGuardError(
                f"XAU direction launch newest {name} decision={payload.get('decision')!r}, not PASS"
            )
        semantic_failures = serve_gate_event_contract_failures(
            payload,
            evidence_name=name,
        )
        if semantic_failures:
            raise ArtifactGuardError(
                f"XAU direction launch {name} semantic contract invalid: "
                + " | ".join(semantic_failures)
            )
        if name == "model_native_serve_parity":
            fusion = payload.get("direction_evidence_fusion_influence")
            if not isinstance(fusion, dict):
                raise ArtifactGuardError(
                    "XAU direction launch serve parity lacks fusion influence binding"
                )
            expected_metadata_path = (accepted_bundle / "bundle_metadata.json").resolve()
            expected_lock_path = (
                accepted_bundle / "MASTER_TRANSFORMER_LOCK.json"
            ).resolve()
            for field, hash_field, expected_path in (
                (
                    "bundle_metadata_path",
                    "bundle_metadata_sha256",
                    expected_metadata_path,
                ),
                (
                    "master_transformer_lock_path",
                    "master_transformer_lock_sha256",
                    expected_lock_path,
                ),
            ):
                observed_path = Path(str(fusion.get(field) or "")).expanduser()
                if (
                    not observed_path.is_absolute()
                    or observed_path.resolve() != expected_path
                    or not expected_path.is_file()
                ):
                    raise ArtifactGuardError(
                        f"XAU direction launch fusion influence {field} mismatch"
                    )
                expected_hash = _exact_sha256(
                    fusion.get(hash_field),
                    label=f"XAU direction launch fusion influence {hash_field}",
                )
                if _sha256_file(expected_path) != expected_hash:
                    raise ArtifactGuardError(
                        f"XAU direction launch fusion influence {hash_field} mismatch"
                    )
        report_bundle = Path(str(payload.get("bundle_dir") or "")).expanduser()
        if not report_bundle.is_absolute() or report_bundle.resolve() != accepted_bundle.resolve():
            raise ArtifactGuardError(
                f"XAU direction launch {name} bundle mismatch: "
                f"report={report_bundle} accepted={accepted_bundle}"
            )
        _validate_gate_prediction_lineage(
            payload,
            evidence_name=name,
            accepted_bundle=accepted_bundle,
        )
        validated[name] = {
            "json_path": str(event_path.resolve()),
            "sha256": actual_sha,
        }
        gate_payloads[name] = payload
    cross_failures = cross_gate_contract_failures(
        gate_payloads["model_native_serve_parity"],
        gate_payloads["model_native_direction_pocket_audit"],
    )
    if cross_failures:
        raise ArtifactGuardError(
            "XAU direction launch serve-gate cross-event contract invalid: "
            + " | ".join(cross_failures)
        )
    return validated


def _check_v10_entry_launch_contract(
    path: Path,
    *,
    launch_contract_path: Path | None = None,
    selection_contract_path: Path | None = None,
    target_launch_contract_path: Path | None = None,
    target_selection_contract_path: Path | None = None,
    expected_selection_contract: dict | None = None,
) -> dict:
    """Reject historical/stale Entry bundles even if still marked ACTIVE.

    ``PROJECT_STATE_artifacts.json`` remains the explicit artifact selector,
    while this second contract is the current XAU direction admission state.
    Both must agree before a decisioning caller may resolve ``v10_entry``.
    This prevents an old green ACTIVE pointer from surviving a newer hard-red
    model event.
    """

    launch_path = (
        XAU_DIRECTION_LAUNCH_CONTRACT
        if launch_contract_path is None
        else Path(launch_contract_path)
    )
    selection_path = (
        SELECTION_CONTRACT
        if selection_contract_path is None
        else Path(selection_contract_path)
    )
    target_launch_path = (
        launch_path
        if target_launch_contract_path is None
        else Path(target_launch_contract_path)
    )
    target_selection_path = (
        selection_path
        if target_selection_contract_path is None
        else Path(target_selection_contract_path)
    )
    if not launch_path.is_file():
        raise ArtifactGuardError(
            f"XAU direction launch contract missing: {launch_path}. "
            "Refusing historical ACTIVE v10_entry."
        )
    try:
        state = json.loads(launch_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactGuardError(
            f"XAU direction launch contract unreadable: {launch_path}: {exc}"
        ) from exc
    if state.get("schema_version") != "gx1_xau_direction_launch_state_v1":
        raise ArtifactGuardError(
            "XAU direction launch contract schema mismatch: "
            f"{state.get('schema_version')!r}"
        )
    if state.get("project") != THIS_PROJECT:
        raise ArtifactGuardError(
            f"XAU direction launch contract project={state.get('project')!r}, "
            f"expected {THIS_PROJECT!r}"
        )
    decision = state.get("decision")
    if decision != "ALLOW":
        event_id = str(state.get("latest_terminal_event_id") or "UNKNOWN")
        blockers = [str(item) for item in (state.get("blockers") or [])]
        raise ArtifactGuardError(
            "XAU v10_entry launch is fail-closed: "
            f"decision={decision or 'MISSING'} latest_terminal_event_id={event_id} "
            f"blockers={blockers[:5]}"
        )
    required_exact = {
        "decision_surface": "model_direction_argmax",
        "public_trade_flat_surface": "public_trade_flat_decision_logits",
        "latest_terminal_event_decision": "PASS",
        "required_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "required_unified_entry_exit_contract": (
            UNIFIED_ENTRY_EXIT_CONTRACT_SCHEMA_VERSION
        ),
        "required_entry_action_order": list(MODEL_DIRECTION_CLASS_ORDER),
        "required_exit_action_order": list(UNIFIED_EXIT_ACTION_ORDER),
        "required_same_bundle_shared_encoder": True,
        "required_exact_closed_m1_exit_path_envelope": True,
        "external_decision_models_allowed": False,
        "required_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "required_base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "required_selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "required_mandatory_causal_layer_feature_count": (
            MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
        ),
        "required_train_ranked_remainder_feature_count": (
            MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
        ),
        "required_mandatory_causal_layer_count": MODEL_NATIVE_MANDATORY_FAMILY_COUNT,
        "required_ctx_cont_dim": REQUIRED_XAU_CTX_CONT_DIM,
        "required_ctx_cat_dim": REQUIRED_XAU_CTX_CAT_DIM,
        "sizing_adoption_mode": MODEL_NATIVE_SIZING_MODE_LEARNED,
    }
    mismatches = [
        f"{key}={state.get(key)!r} expected={expected!r}"
        for key, expected in required_exact.items()
        if state.get(key) != expected
    ]
    if mismatches:
        raise ArtifactGuardError(
            "XAU direction launch contract is not model-native/current: " + " | ".join(mismatches)
        )
    if not str(state.get("dataset_event_id") or "").strip():
        raise ArtifactGuardError("XAU direction launch contract lacks dataset_event_id")
    for evidence_name in (
        "joint_exit_execution_proof_evidence",
        "sizing_runtime_parity_evidence",
        "adaptation_lifecycle_evidence",
    ):
        if not isinstance(state.get(evidence_name), dict):
            raise ArtifactGuardError(
                f"XAU direction launch contract lacks {evidence_name}"
            )
    try:
        require_live_tail_launch_authority(
            state.get("new_entry_live_tail_authority")
        )
    except LiveTailAuthorityError as exc:
        raise ArtifactGuardError(
            f"XAU direction launch static live-tail authority invalid: {exc}"
        ) from exc
    try:
        authority = require_model_native_sizing_authority_contract(
            state.get("sizing_authority_contract"),
            context="XAU_DIRECTION_LAUNCH_SIZING_AUTHORITY",
            required_mode=MODEL_NATIVE_SIZING_MODE_LEARNED,
        )
        sizing_snapshot = prepare_model_native_sizing_authority(
            authority,
            context="XAU_DIRECTION_LAUNCH_SIZING_AUTHORITY",
        )
    except ModelNativeSizingUnavailable as exc:
        raise ArtifactGuardError(
            f"XAU direction launch sizing authority invalid: {exc}"
        ) from exc
    adoption = sizing_snapshot.adoption
    if Path(adoption["bundle_dir"]).resolve() != path.resolve():
        raise ArtifactGuardError(
            "XAU direction launch sizing adoption bundle differs from accepted bundle"
        )
    candidate_bundle_path = Path(
        str(sizing_snapshot.candidate_bundle_authority.get("bundle_dir") or "")
    ).expanduser()
    if (
        not candidate_bundle_path.is_absolute()
        or candidate_bundle_path.resolve() != path.resolve()
    ):
        raise ArtifactGuardError(
            "XAU direction launch sizing proof binds a different candidate "
            "bundle than the launch target"
        )
    if (
        state.get("joint_exit_execution_proof_evidence")
        != adoption["joint_exit_sizing_proof_artifact"]
    ):
        raise ArtifactGuardError(
            "XAU direction launch joint Exit evidence differs from sizing adoption"
        )
    try:
        operating_point = require_model_direction_operating_point(
            state.get("operating_point"),
            context="XAU direction launch state",
        )
        require_canonical_unified_replay_launch_authority(
            sizing_snapshot.joint_proof,
            context="XAU_DIRECTION_LAUNCH_UNIFIED_REPLAY_PRODUCER",
        )
        require_joint_exit_portfolio_capacity(
            sizing_snapshot.joint_proof,
            max_trades=int(operating_point["max_trades"]),
            context="XAU_DIRECTION_LAUNCH_PORTFOLIO_CAPACITY",
        )
    except RuntimeError as exc:
        raise ArtifactGuardError(
            f"XAU direction launch portfolio capacity invalid: {exc}"
        ) from exc
    accepted_bundle = Path(str(state.get("accepted_bundle_dir") or "")).expanduser()
    if (
        not accepted_bundle.is_absolute()
        or accepted_bundle.resolve() != path.resolve()
    ):
        raise ArtifactGuardError(
            "XAU direction launch accepted_bundle_dir differs from selected artifact"
        )
    try:
        require_entry_launch_approval(
            state,
            accepted_bundle=accepted_bundle.resolve(),
        )
    except RuntimeError as exc:
        raise ArtifactGuardError(
            f"XAU direction launch approval invalid: {exc}"
        ) from exc
    metadata_path = accepted_bundle.resolve() / "bundle_metadata.json"
    if not metadata_path.is_file() or _sha256_file(metadata_path) != str(
        state.get("bundle_metadata_sha256") or ""
    ).lower():
        raise ArtifactGuardError(
            "XAU direction launch bundle_metadata_sha256 mismatch"
        )
    _validate_serve_gate_evidence(state, accepted_bundle=accepted_bundle.resolve())
    if not isinstance(state.get("sizing_runtime_parity_evidence"), dict):
        raise ArtifactGuardError(
            "XAU direction launch lacks sizing_runtime_parity_evidence"
        )
    try:
        runtime_parity, runtime_binding = load_bound_runtime_sizing_parity(
            state["sizing_runtime_parity_evidence"],
            adoption=adoption,
            calibration=sizing_snapshot.calibration,
            adoption_artifact=authority["adoption_artifact"],
            context="XAU_DIRECTION_LAUNCH_SIZING_RUNTIME_PARITY",
            verify_source_files=True,
        )
    except ModelNativeSizingExecutionContractError as exc:
        raise ArtifactGuardError(
            f"XAU direction launch sizing runtime parity invalid: {exc}"
        ) from exc
    if runtime_binding != state["sizing_runtime_parity_evidence"]:
        raise ArtifactGuardError(
            "XAU direction launch sizing runtime parity binding is noncanonical"
        )
    if runtime_parity["bundle_identity"]["bundle_dir"] != str(path.resolve()):
        raise ArtifactGuardError(
            "XAU direction launch sizing runtime parity bundle mismatch"
        )
    try:
        _, adaptation_binding = require_launch_adaptation_authority(
            state["adaptation_lifecycle_evidence"],
            accepted_bundle=path.resolve(),
            serve_gate_evidence=state["serve_gate_evidence"],
            joint_exit_execution_proof_evidence=state[
                "joint_exit_execution_proof_evidence"
            ],
            sizing_runtime_parity_evidence=state[
                "sizing_runtime_parity_evidence"
            ],
            context="XAU_DIRECTION_LAUNCH_ADAPTATION_LIFECYCLE",
        )
    except ModelNativeAdaptationLifecycleError as exc:
        raise ArtifactGuardError(
            f"XAU direction launch adaptation lifecycle invalid: {exc}"
        ) from exc
    if adaptation_binding != state["adaptation_lifecycle_evidence"]:
        raise ArtifactGuardError(
            "XAU direction launch adaptation lifecycle binding is noncanonical"
        )
    try:
        require_entry_launch_transaction(
            state,
            launch_state_bytes_path=launch_path,
            registry_bytes_path=selection_path,
            target_launch_state_path=target_launch_path,
            target_registry_path=target_selection_path,
            accepted_bundle=path.resolve(),
            expected_registry=expected_selection_contract,
        )
    except EntryLaunchTransactionError as exc:
        raise ArtifactGuardError(
            f"XAU direction launch transaction invalid: {exc}"
        ) from exc
    return state


def require_new_entry_live_tail_admission(
    launch_state: dict,
    *,
    expected_pair_generation_id: str | None = None,
    expected_generation_manifest_sha256: str | None = None,
    now_utc: object | None = None,
) -> dict:
    """Require newest fresh pair authority solely for opening new exposure."""

    if not isinstance(launch_state, dict):
        raise ArtifactGuardError(
            "new-Entry live-tail launch state is unavailable"
        )
    try:
        authority = require_live_tail_launch_authority(
            launch_state.get("new_entry_live_tail_authority")
        )
        return require_newest_live_tail_runtime_authority(
            authority,
            expected_pair_generation_id=expected_pair_generation_id,
            expected_generation_manifest_sha256=(
                expected_generation_manifest_sha256
            ),
            now_utc=now_utc,
        )
    except LiveTailAuthorityError as exc:
        raise ArtifactGuardError(
            f"new-Entry live-tail admission unavailable: {exc}"
        ) from exc


def load_decision_entry(role: str) -> dict:
    """
    Return the full ACTIVE entry dict for a role — path resolved + active_variant,
    active_fold(s), active_aggregator and any other fields the contract carries.

    Same guard semantics as ``load_decision_artifact`` (status ACTIVE, no
    in_sample_only, project isolation, path exists). Use this when live-wiring
    needs the per-role config (variant/fold/aggregator), not just the path.
    """
    contract = _load_contract()
    if contract.get("project") != THIS_PROJECT:
        raise ArtifactGuardError(
            f"Contract project is {contract.get('project')!r}, "
            f"expected {THIS_PROJECT!r}. Refusing cross-project load."
        )
    admission = contract.get("production_admission")
    if (
        not isinstance(admission, dict)
        or admission.get("status") != "ALLOW"
        or admission.get("selection_registry_is_launch_authority") is not True
    ):
        raise ArtifactGuardError(
            "Production admission is not ALLOW and authoritative; refusing "
            f"decision-artifact role {role!r}."
        )
    entry = contract.get("active", {}).get(role)
    if entry is None:
        raise ArtifactGuardError(
            f"No ACTIVE artifact declared for role '{role}'. "
            "Add an explicit entry to the selection contract first (vedtak required)."
        )
    if entry.get("status") != "ACTIVE":
        raise ArtifactGuardError(
            f"Artifact for '{role}' has status {entry.get('status')!r}, not ACTIVE."
        )
    if entry.get("in_sample_only"):
        raise ArtifactGuardError(
            f"Artifact for '{role}' is flagged in_sample_only — never decision-valid."
        )
    path_str = entry["path"]
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise ArtifactGuardError(
            f"ACTIVE artifact for '{role}' not found on disk: {path}."
        )
    launch_state: dict | None = None
    if role == "v10_entry":
        launch_state = _check_v10_entry_launch_contract(
            path,
            expected_selection_contract=contract,
        )
        try:
            require_model_direction_operating_point(
                entry.get("operating_point"),
                context="XAU direction selection contract",
            )
        except Exception as exc:
            raise ArtifactGuardError(
                f"XAU direction operating point invalid: {exc}"
            ) from exc
    out = dict(entry)
    out["path"] = path  # absolute resolved Path overrides string
    if launch_state is not None:
        out["xau_direction_launch_state"] = launch_state
    return out


def load_decision_artifact(role: str) -> Path:
    """
    Return the ACTIVE artifact path for a declared decision role.

    Thin wrapper over load_decision_entry — kept for callers that only need
    the path. There is deliberately NO glob, NO 'latest', NO mtime sorting.
    """
    return load_decision_entry(role)["path"]


def forbid_synthetic(data_source: str, *, allow_synthetic: bool = False) -> None:
    """
    Call at the top of any decisioning path that ingests data.
    Refuses dummy/synthetic/degraded fallbacks unless explicitly allowed
    (allow_synthetic should ONLY ever be True in unit tests, never in decisioning).
    """
    bad = ("dummy", "synthetic", "fallback", "degraded", "placeholder", "mock")
    if not allow_synthetic and any(b in data_source.lower() for b in bad):
        raise ArtifactGuardError(
            f"Refusing synthetic/degraded input for decisioning: {data_source!r}. "
            "Decisioning must use real artifacts only."
        )
