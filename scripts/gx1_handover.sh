#!/usr/bin/env bash
# Read-only, fail-closed takeover status for the GX1 gold/XAUUSD project.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)
HANDOVER="$REPO/HANDOVER_XAU_DIRECTION_REPAIR_20260714.md"
LAUNCH_STATE="$REPO/PROJECT_STATE_xau_direction_launch.json"
PY="$REPO/.venv/bin/python"

# Keep the authority fingerprint path-ordered. Historical prose is reference
# only; GX1_RULES.md defines the active scope.
sources=(
  "$REPO/AGENTS.md"
  "$REPO/CLAUDE.md"
  "$REPO/GX1_RULES.md"
  "$REPO/README.md"
  "$REPO/SYSTEM_MAP.md"
  "$HANDOVER"
  "$REPO/docs/CURRENT_AUDIT_STATUS_20260828.md"
  "$REPO/docs/OFFLINE_CHAMPION_CHALLENGER_V1.md"
  "$REPO/docs/DATA_CONTRACT.md"
  "$REPO/docs/ATTENDED_STAGED_PREFLIGHT_DESIGN_20260823.md"
  "$REPO/docs/CANONICAL_HOST_GPU_TELEMETRY_BRIDGE_CONTRACT.md"
  "$REPO/docs/CANDIDATE_THROUGHPUT_DECISION_20260830.md"
  "$REPO/docs/V8_CANDIDATE_HOST_HANG_INCIDENT_20260901.md"
  "$REPO/docs/FEATURE_VALUE_REVIEW_20260813.md"
  "$REPO/docs/INDICATOR_FIDELITY_AUDIT_20260813.md"
  "$REPO/docs/GIT_WORKTREE_POLICY.md"
  "$REPO/docs/POST_BUILD_INTEGRITY_GATE_20260825.md"
  "$REPO/docs/PREREGISTERED_DIRECTION_TEST_20260820.md"
  "$REPO/docs/RECIPE_DECISION_DRAFT_20260808.md"
  "$REPO/docs/V29_EVENT_SURFACE_DESIGN_20260811.md"
  "$REPO/docs/TRAIN_WINDOW_WIDENING_20260819.md"
  "$LAUNCH_STATE"
)

usage() {
  cat <<'EOF'
Usage: scripts/gx1_handover.sh [--check|--verbose]

Default prints compact status. --check prints only deterministic authority and
worktree identity. --verbose appends the exact handover document.
EOF
}

mode=compact
case "${1:-}" in
  "") ;;
  --check) mode=check ;;
  --verbose) mode=verbose ;;
  -h|--help) usage; exit 0 ;;
  *) printf 'FATAL: unsupported argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "FATAL: expected at most one argument" >&2; exit 2; }

for source in "${sources[@]}"; do
  [[ -f "$source" ]] || { echo "FATAL: authority input missing: $source" >&2; exit 2; }
done
CURRENT_PAIR_MANIFEST=$("$PY" - "$LAUNCH_STATE" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
path = state.get("current_pair_manifest")
if not isinstance(path, str) or not path.startswith("/"):
    raise SystemExit("FATAL: launch authority current_pair_manifest is invalid")
print(path)
PY
)
[[ -f "$CURRENT_PAIR_MANIFEST" && ! -L "$CURRENT_PAIR_MANIFEST" ]] || {
  echo "FATAL: current pair manifest missing/non-regular: $CURRENT_PAIR_MANIFEST" >&2
  exit 2
}
[[ -x "$PY" ]] || { echo "FATAL: repository Python is not executable: $PY" >&2; exit 2; }
cd "$REPO"

readarray -t identity < <("$PY" - "$REPO" "$LAUNCH_STATE" \
  "$CURRENT_PAIR_MANIFEST" "${sources[@]}" "$CURRENT_PAIR_MANIFEST" <<'PY'
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_MANDATORY_FULL_STACK_SCHEMA_VERSION,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
)
from gx1.contracts.current_audited_dataset_evidence_v1 import (
    require_blocked_launch_state_with_current_audited_dataset,
)
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    MANIFEST_NAME as BUNDLE_COMMIT_MANIFEST_NAME,
    require_bundle_commit_manifest,
)
from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
    canonical_json_sha256,
    require_pretest_technical_recipe_metadata,
)
from gx1.contracts.entry_model_native_train_launch_v1 import (
    PRETEST_TECHNICAL_TRAIN_WRAPPER_RELATIVE_PATH,
    recipe_source_bindings,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
)
from gx1.features.htf_features import (
    HTF_V4_CACHE_SCHEMA_VERSION,
    HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION,
    HTF_V4_MATRIX_CONTRACT,
    MULTI_TF_FEATURE_COUNT_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
)

repo = Path(sys.argv[1])
launch_path = Path(sys.argv[2])
pair_path = Path(sys.argv[3])
authority_paths = tuple(Path(raw) for raw in sys.argv[4:])

state = json.loads(launch_path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_json(path: Path, *, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"FATAL: candidate {label} path is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"FATAL: candidate {label} JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"FATAL: candidate {label} JSON is not an object: {path}")
    return value


def _canonical_session_json_sha256(value: dict) -> str:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _active_candidate_session_status(launch_state: dict) -> tuple[str, ...]:
    """Verify the declared session, recipe and live two-slot pointer.

    This is intentionally read-only.  The trainer independently verifies the
    recipe's source-file bindings immediately before it can resume; this
    handover check proves that the selected state, recipe identity and session
    lineage have not drifted into a prose-only checkpoint claim.
    """

    reference = launch_state.get("active_candidate_training_session")
    expected_reference_keys = {
        "schema_version", "session_dir", "recipe_audit_path", "recipe_audit_sha256",
        "source_commit", "source_bindings_sha256", "run_id", "dataset_run_id",
    }
    if not isinstance(reference, dict) or set(reference) != expected_reference_keys:
        raise SystemExit("FATAL: active candidate session reference is invalid")
    if reference.get("schema_version") != "gx1_active_candidate_training_session_reference_v1":
        raise SystemExit("FATAL: active candidate session reference schema is invalid")
    for key in ("session_dir", "recipe_audit_path"):
        if not isinstance(reference.get(key), str) or not reference[key].startswith("/"):
            raise SystemExit(f"FATAL: active candidate session {key} is invalid")
    for key in ("recipe_audit_sha256", "source_bindings_sha256"):
        if not isinstance(reference.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", reference[key]) is None:
            raise SystemExit(f"FATAL: active candidate session {key} is invalid")
    if not isinstance(reference.get("source_commit"), str) or re.fullmatch(r"[0-9a-f]{40}", reference["source_commit"]) is None:
        raise SystemExit("FATAL: active candidate session source_commit is invalid")
    if not all(isinstance(reference.get(key), str) and reference[key] for key in ("run_id", "dataset_run_id")):
        raise SystemExit("FATAL: active candidate session run identity is invalid")

    session_dir = Path(reference["session_dir"])
    if session_dir.is_symlink() or not session_dir.is_dir():
        raise SystemExit("FATAL: active candidate session directory is invalid")
    recipe_path = Path(reference["recipe_audit_path"])
    recipe = _read_regular_json(recipe_path, label="recipe")
    if _sha256_file(recipe_path) != reference["recipe_audit_sha256"]:
        raise SystemExit("FATAL: active candidate recipe SHA-256 mismatch")
    if (
        recipe.get("decision") != "PASS"
        or recipe.get("run_id") != reference["run_id"]
        or recipe.get("dataset_run_id") != reference["dataset_run_id"]
        or recipe.get("source_commit") != reference["source_commit"]
        or recipe.get("source_bindings_sha256") != reference["source_bindings_sha256"]
    ):
        raise SystemExit("FATAL: active candidate recipe identity mismatch")

    bindings = recipe.get("source_bindings")
    if (
        not isinstance(bindings, dict)
        or not bindings
        or hashlib.sha256(
            json.dumps(bindings, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        != reference["source_bindings_sha256"]
    ):
        raise SystemExit("FATAL: active candidate source bindings are invalid")
    for binding_name, binding in bindings.items():
        if not isinstance(binding_name, str) or not isinstance(binding, dict):
            raise SystemExit("FATAL: active candidate source binding shape is invalid")
        binding_path = binding.get("path")
        binding_sha256 = binding.get("sha256")
        if (
            not isinstance(binding_path, str)
            or not isinstance(binding_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", binding_sha256) is None
        ):
            raise SystemExit("FATAL: active candidate source binding value is invalid")
        try:
            relative_path = Path(binding_path).relative_to(repo).as_posix()
        except ValueError as exc:
            raise SystemExit(
                "FATAL: active candidate source binding escapes repository"
            ) from exc
        frozen = subprocess.run(
            [
                "git", "-C", str(repo), "show",
                f"{reference['source_commit']}:{relative_path}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if (
            frozen.returncode != 0
            or hashlib.sha256(frozen.stdout).hexdigest() != binding_sha256
        ):
            raise SystemExit(
                "FATAL: active candidate frozen source closure mismatch: "
                f"{binding_name}"
            )

    contract = _read_regular_json(
        session_dir / "CANDIDATE_TRAINING_SESSION_CONTRACT.json", label="contract"
    )
    expected_authority = {
        "candidate_training": True, "bundle": False, "validation": False,
        "test": False, "promotion": False, "paper": False, "live": False,
    }
    if (
        contract.get("schema_version") != "gx1_candidate_training_session_v1"
        or contract.get("authority") != expected_authority
        or contract.get("run_id") != reference["run_id"]
        or contract.get("dataset_run_id") != reference["dataset_run_id"]
        or contract.get("source_commit") != reference["source_commit"]
        or contract.get("profile") != "candidate"
        or contract.get("execution_tier") != "canonical"
    ):
        raise SystemExit("FATAL: active candidate contract identity mismatch")
    contract_sha256 = _canonical_session_json_sha256(contract)

    pointer = _read_regular_json(
        session_dir / "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json", label="pointer"
    )
    expected_pointer_keys = {
        "schema_version", "session_contract_sha256", "slot", "checkpoint_index",
        "state_sha256", "phase", "epoch_index", "next_batch_offset",
        "global_optimizer_steps", "complete",
    }
    if set(pointer) != expected_pointer_keys:
        raise SystemExit("FATAL: active candidate pointer keys are invalid")
    if (
        pointer.get("schema_version") != "gx1_candidate_training_session_v1"
        or pointer.get("session_contract_sha256") != contract_sha256
        or pointer.get("slot") not in (0, 1)
        or not isinstance(pointer.get("checkpoint_index"), int)
        or int(pointer["checkpoint_index"]) < 1
        or pointer.get("phase") not in ("train", "validation")
        or not isinstance(pointer.get("epoch_index"), int)
        or int(pointer["epoch_index"]) < 0
        or not isinstance(pointer.get("next_batch_offset"), int)
        or int(pointer["next_batch_offset"]) < 0
        or not isinstance(pointer.get("global_optimizer_steps"), int)
        or int(pointer["global_optimizer_steps"]) < 0
        or not isinstance(pointer.get("complete"), bool)
        or not isinstance(pointer.get("state_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", pointer["state_sha256"]) is None
    ):
        raise SystemExit("FATAL: active candidate pointer is invalid")
    state_path = session_dir / f"candidate_training_state_slot_{pointer['slot']}.pt"
    if state_path.is_symlink() or not state_path.is_file():
        raise SystemExit("FATAL: active candidate state path is invalid")
    if _sha256_file(state_path) != pointer["state_sha256"]:
        raise SystemExit("FATAL: active candidate state SHA-256 mismatch")

    validation = "NOT_REACHED" if pointer["phase"] == "train" else "REQUIRES_AUDIT"
    return (
        "SESSION_INTACT"
        f"__checkpoint={pointer['checkpoint_index']}"
        f"__phase={pointer['phase']}"
        f"__epoch={pointer['epoch_index']}"
        f"__next_batch={pointer['next_batch_offset']}"
        f"__optimizer_steps={pointer['global_optimizer_steps']}"
        f"__complete={int(pointer['complete'])}",
        validation,
        contract_sha256,
        str(pointer["state_sha256"]),
        str(reference["recipe_audit_sha256"]),
        str(reference["source_bindings_sha256"]),
        "FROZEN_COMMIT_BYTES_MATCH_RECIPE",
    )


def _current_source_technical_recipe_status(
    launch_state: dict,
) -> tuple[str, str, str, str]:
    """Verify the separate current-source technical smoke recipe.

    The retained candidate session is immutable historical evidence. This
    reference proves the next technical smoke is bound to *live* executable
    source bytes without granting CUDA, candidate, TEST or execution authority.
    """
    reference = launch_state.get("current_source_technical_recipe")
    base_reference_keys = {
        "schema_version", "status", "recipe_path", "recipe_sha256",
        "source_commit", "source_bindings_sha256", "run_id", "dataset_run_id",
        "out_bundle_dir",
    }
    executed_status = (
        "EXECUTED_TECHNICAL_SMOKE__POSTRUN_AUDIT_PENDING__"
        "NO_CANDIDATE_AUTHORITY"
    )
    audited_status = (
        "EXECUTED_TECHNICAL_SMOKE__POSTRUN_AUDIT_FAIL__"
        "CANDIDATE_READINESS_READY__NO_PROMOTION_AUTHORITY"
    )
    gated_status = (
        "EXECUTED_TECHNICAL_SMOKE__POSTRUN_AUDIT_FAIL__"
        "CANDIDATE_READINESS_READY__CANDIDATE_GATE_READY__"
        "NO_PROMOTION_AUTHORITY"
    )
    expected_reference_keys = set(base_reference_keys)
    if isinstance(reference, dict) and reference.get("status") in {
        executed_status, audited_status, gated_status,
    }:
        expected_reference_keys.update({
            "bundle_commit_manifest_sha256", "bundle_commit_sha256",
            "bundle_metadata_sha256",
        })
    if isinstance(reference, dict) and reference.get("status") in {
        audited_status, gated_status,
    }:
        expected_reference_keys.update({
            "postrun_bundle_audit_path", "postrun_bundle_audit_sha256",
            "postrun_bundle_audit_decision", "candidate_readiness_path",
            "candidate_readiness_sha256", "candidate_readiness_decision",
        })
    if isinstance(reference, dict) and reference.get("status") == gated_status:
        expected_reference_keys.update({
            "candidate_launch_gate_path", "candidate_launch_gate_sha256",
            "candidate_launch_gate_decision",
        })
    if not isinstance(reference, dict) or set(reference) != expected_reference_keys:
        raise SystemExit("FATAL: current-source technical recipe reference is invalid")
    if reference.get("schema_version") != "gx1_current_source_technical_recipe_reference_v1":
        raise SystemExit("FATAL: current-source technical recipe status is invalid")
    status = str(reference.get("status") or "")
    if status not in {
        "MATERIALIZED_CPU_LAUNCH_DRY_RUN_PASS__CUDA_NOT_EXECUTED",
        executed_status,
        audited_status,
        gated_status,
    }:
        raise SystemExit("FATAL: current-source technical recipe status is invalid")
    for key in ("recipe_path", "out_bundle_dir"):
        if not isinstance(reference.get(key), str) or not reference[key].startswith("/"):
            raise SystemExit(f"FATAL: current-source technical recipe {key} is invalid")
    for key in ("recipe_sha256", "source_bindings_sha256"):
        if (
            not isinstance(reference.get(key), str)
            or re.fullmatch(r"[0-9a-f]{64}", reference[key]) is None
        ):
            raise SystemExit(f"FATAL: current-source technical recipe {key} is invalid")
    if (
        not isinstance(reference.get("source_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", reference["source_commit"]) is None
        or not all(
            isinstance(reference.get(key), str) and reference[key]
            for key in ("run_id", "dataset_run_id")
        )
    ):
        raise SystemExit("FATAL: current-source technical recipe identity is invalid")

    recipe_path = Path(reference["recipe_path"])
    recipe = _read_regular_json(recipe_path, label="current-source technical recipe")
    if _sha256_file(recipe_path) != reference["recipe_sha256"]:
        raise SystemExit("FATAL: current-source technical recipe SHA-256 mismatch")
    try:
        validated = require_pretest_technical_recipe_metadata(
            recipe,
            expected_profile="smoke",
            expected_run_id=reference["run_id"],
            expected_dataset_run_id=reference["dataset_run_id"],
            expected_out_bundle_dir=reference["out_bundle_dir"],
        )
    except RuntimeError as exc:
        raise SystemExit(
            "FATAL: current-source technical recipe metadata is invalid"
        ) from exc
    if (
        validated.get("source_commit") != reference["source_commit"]
        or validated.get("source_bindings_sha256")
        != reference["source_bindings_sha256"]
        or validated["trainer_cli"].get("execution_tier") != "canonical"
        or validated["trainer_cli"].get("device") != "cuda"
        or validated["trainer_cli"].get("subsample_rows") != 32
    ):
        raise SystemExit("FATAL: current-source technical recipe contract mismatch")
    live_bindings = recipe_source_bindings(
        repo=repo,
        wrapper_path=(repo / PRETEST_TECHNICAL_TRAIN_WRAPPER_RELATIVE_PATH),
    )
    if (
        validated.get("source_bindings") != live_bindings
        or canonical_json_sha256(live_bindings)
        != reference["source_bindings_sha256"]
    ):
        raise SystemExit("FATAL: current-source technical recipe live source mismatch")
    out_bundle_dir = Path(reference["out_bundle_dir"])
    if status == "MATERIALIZED_CPU_LAUNCH_DRY_RUN_PASS__CUDA_NOT_EXECUTED":
        if out_bundle_dir.exists() or out_bundle_dir.is_symlink():
            raise SystemExit("FATAL: current-source technical recipe has executed CUDA")
        closure = "LIVE_SOURCE_BYTES_MATCH_RECIPE__CUDA_NOT_EXECUTED"
    else:
        try:
            bundle_commit = require_bundle_commit_manifest(out_bundle_dir)
        except RuntimeError as exc:
            raise SystemExit(
                "FATAL: current-source technical smoke bundle commit is invalid"
            ) from exc
        bundle_commit_path = out_bundle_dir / BUNDLE_COMMIT_MANIFEST_NAME
        metadata_path = out_bundle_dir / "bundle_metadata.json"
        if (
            _sha256_file(bundle_commit_path)
            != reference["bundle_commit_manifest_sha256"]
            or bundle_commit.get("commit_sha256")
            != reference["bundle_commit_sha256"]
            or bundle_commit.get("bundle_kind") != "trained"
            or bundle_commit["artifacts"]["bundle_metadata.json"].get("sha256")
            != reference["bundle_metadata_sha256"]
            or _sha256_file(metadata_path) != reference["bundle_metadata_sha256"]
        ):
            raise SystemExit("FATAL: current-source technical smoke bundle mismatch")
        bundle_metadata = _read_regular_json(
            metadata_path,
            label="current-source technical smoke bundle metadata",
        )
        provenance = bundle_metadata.get("recipe_source_provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("recipe_audit_path") != str(recipe_path)
            or provenance.get("recipe_audit_sha256") != reference["recipe_sha256"]
            or provenance.get("source_commit") != reference["source_commit"]
            or provenance.get("source_bindings_sha256")
            != reference["source_bindings_sha256"]
            or bundle_metadata.get("execution_tier") != "canonical"
        ):
            raise SystemExit(
                "FATAL: current-source technical smoke bundle provenance mismatch"
            )
        closure = (
            "LIVE_SOURCE_BYTES_MATCH_RECIPE__BUNDLE_COMMIT_VALID__"
            "POSTRUN_AUDIT_PENDING"
        )
        if status in {audited_status, gated_status}:
            for prefix, expected_decision in (
                ("postrun_bundle_audit", "FAIL"),
                ("candidate_readiness", "READY_FOR_CANDIDATE_TRAINING"),
            ):
                event_path = Path(reference[f"{prefix}_path"])
                if (
                    _sha256_file(event_path) != reference[f"{prefix}_sha256"]
                    or _read_regular_json(event_path, label=prefix).get("decision") != expected_decision
                    or reference[f"{prefix}_decision"] != expected_decision
                ):
                    raise SystemExit(f"FATAL: current-source technical {prefix} mismatch")
            closure = (
                "LIVE_SOURCE_BYTES_MATCH_RECIPE__BUNDLE_COMMIT_VALID__"
                "POSTRUN_AUDIT_FAIL__CANDIDATE_READINESS_READY"
            )
        if status == gated_status:
            gate_path = Path(reference["candidate_launch_gate_path"])
            gate = _read_regular_json(gate_path, label="candidate launch gate")
            expected_authority = {
                "candidate_training": True,
                "live": False,
                "paper": False,
                "promotion": False,
                "shadow": False,
                "test": False,
            }
            if (
                _sha256_file(gate_path) != reference["candidate_launch_gate_sha256"]
                or gate.get("schema_version")
                != "entry_pretest_candidate_launch_gate_v1"
                or gate.get("decision")
                != "READY_FOR_PRETEST_CANDIDATE_TRAINING"
                or gate.get("failures") != []
                or gate.get("activation_authority") is not False
                or gate.get("authority") != expected_authority
                or gate.get("json_path") != str(gate_path)
                or reference["candidate_launch_gate_decision"]
                != "READY_FOR_PRETEST_CANDIDATE_TRAINING"
            ):
                raise SystemExit("FATAL: current-source candidate launch gate mismatch")
            expected_smoke_audit = {
                "path": reference["postrun_bundle_audit_path"],
                "sha256": reference["postrun_bundle_audit_sha256"],
            }
            expected_readiness = {
                "path": reference["candidate_readiness_path"],
                "sha256": reference["candidate_readiness_sha256"],
            }
            if (
                gate.get("smoke_bundle_audit") != expected_smoke_audit
                or gate.get("candidate_readiness") != expected_readiness
            ):
                raise SystemExit("FATAL: current-source candidate gate inputs mismatch")
            candidate_recipe_binding = gate.get("recipe")
            if not isinstance(candidate_recipe_binding, dict):
                raise SystemExit("FATAL: current-source candidate gate recipe is invalid")
            candidate_recipe_path = Path(str(candidate_recipe_binding.get("path") or ""))
            candidate_recipe = _read_regular_json(
                candidate_recipe_path,
                label="current-source candidate recipe",
            )
            if _sha256_file(candidate_recipe_path) != candidate_recipe_binding.get("sha256"):
                raise SystemExit("FATAL: current-source candidate recipe SHA-256 mismatch")
            try:
                require_pretest_technical_recipe_metadata(
                    candidate_recipe,
                    expected_profile="candidate",
                    expected_run_id=str(gate["run_id"]),
                    expected_dataset_run_id=str(gate["dataset_run_id"]),
                    expected_out_bundle_dir=str(gate["out_bundle_dir"]),
                )
            except (KeyError, RuntimeError) as exc:
                raise SystemExit(
                    "FATAL: current-source candidate gate recipe metadata is invalid"
                ) from exc
            closure = (
                "LIVE_SOURCE_BYTES_MATCH_RECIPE__BUNDLE_COMMIT_VALID__"
                "POSTRUN_AUDIT_FAIL__CANDIDATE_READINESS_READY__"
                "CANDIDATE_GATE_READY"
            )
    return (
        status,
        str(reference["recipe_sha256"]),
        str(reference["source_bindings_sha256"]),
        closure,
    )


candidate_session = _active_candidate_session_status(state)
current_source_technical_recipe = _current_source_technical_recipe_status(state)
try:
    audited_dataset = require_blocked_launch_state_with_current_audited_dataset(
        state
    )
except RuntimeError as exc:
    raise SystemExit(f"FATAL: current audited dataset evidence invalid: {exc}") from exc
expected_state = {
    "required_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
    "required_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
    "required_base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
    "required_selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    "required_mandatory_causal_layer_feature_count": (
        MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
    ),
    "required_available_candidate_feature_count": (
        MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT
    ),
    "required_mandatory_causal_layer_count": len(
        MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
    ),
    "required_ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
    "required_ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
}
for key, expected in expected_state.items():
    if state.get(key) != expected:
        raise SystemExit(
            f"FATAL: launch authority {key}={state.get(key)!r} "
            f"does not match source owner {expected!r}"
        )
if len(MULTI_TF_PER_BAR_FEATURES_V4) != MULTI_TF_FEATURE_COUNT_V4:
    raise SystemExit("FATAL: MTF tuple/count owner mismatch")
pair = json.loads(pair_path.read_text(encoding="utf-8"))
pair_id = str(pair.get("pair_generation_id") or "")
artifacts = pair.get("artifacts")
lineage = pair.get("lineage")
native = lineage.get("native_sources") if isinstance(lineage, dict) else None
if (
    len(pair_id) != 64
    or not isinstance(artifacts, dict)
    or not isinstance(native, dict)
):
    raise SystemExit("FATAL: current pair authority is invalid")

authority = hashlib.sha256()
authority.update(b"gx1-takeover-authority-v3\0")
for index, path in enumerate(authority_paths):
    path_bytes = str(path).encode("utf-8")
    payload = path.read_bytes()
    authority.update(index.to_bytes(4, "big"))
    authority.update(len(path_bytes).to_bytes(8, "big"))
    authority.update(path_bytes)
    authority.update(len(payload).to_bytes(8, "big"))
    authority.update(payload)

def git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE
    ).stdout

worktree = hashlib.sha256()
worktree.update(b"gx1-worktree-identity-v1\0")
for label, payload in (
    (b"head", git_bytes("rev-parse", "HEAD")),
    (b"tracked-diff", git_bytes("diff", "--binary", "--no-ext-diff", "HEAD", "--")),
):
    worktree.update(len(label).to_bytes(4, "big"))
    worktree.update(label)
    worktree.update(len(payload).to_bytes(8, "big"))
    worktree.update(payload)
for raw in filter(None, git_bytes("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")):
    path = repo / os.fsdecode(raw)
    if path.is_symlink():
        kind = b"symlink"
        payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
    elif path.is_file():
        kind = b"file"
        payload = path.read_bytes()
    else:
        raise SystemExit(f"FATAL: unsupported untracked entry: {path}")
    for value in (raw, kind, payload):
        worktree.update(len(value).to_bytes(8, "big"))
        worktree.update(value)

status = git_bytes("status", "--porcelain=v1", "-z")
changed = len(tuple(filter(None, status.split(b"\0"))))
ignored_status = git_bytes("status", "--ignored", "--porcelain=v1", "-z")
ignored_paths = tuple(
    os.fsdecode(entry[3:])
    for entry in filter(None, ignored_status.split(b"\0"))
    if entry.startswith(b"!! ")
)
ignored = len(ignored_paths)
worktree_porcelain = git_bytes("worktree", "list", "--porcelain").decode("utf-8")
prunable_worktrees = sum(
    1 for line in worktree_porcelain.splitlines() if line.startswith("prunable")
)
reviewed_exclusions = state.get("reviewed_local_runtime_exclusions")
if (
    not isinstance(reviewed_exclusions, dict)
    or set(reviewed_exclusions) != {"schema_version", "paths"}
    or reviewed_exclusions.get("schema_version")
    != "gx1_reviewed_local_runtime_exclusions_v1"
    or not isinstance(reviewed_exclusions.get("paths"), list)
    or any(
        not isinstance(path, str) for path in reviewed_exclusions["paths"]
    )
    or set(reviewed_exclusions["paths"])
    != {".claude/worktrees/", ".env", ".venv/"}
    or len(reviewed_exclusions["paths"]) != 3
):
    raise SystemExit("FATAL: reviewed local runtime exclusions are invalid")
environment_file = repo / ".env"
if environment_file.is_symlink() or (
    environment_file.exists()
    and (
        not environment_file.is_file()
        or os.stat(environment_file, follow_symlinks=False).st_mode & 0o077
    )
):
    raise SystemExit("FATAL: reviewed local .env exclusion is unsafe")
venv = repo / ".venv"
if venv.is_symlink() or (
    venv.exists()
    and (
        not venv.is_dir()
        or (venv / "pyvenv.cfg").is_symlink()
        or not (venv / "pyvenv.cfg").is_file()
    )
):
    raise SystemExit("FATAL: reviewed local virtual environment exclusion is invalid")
worktree_root = repo / ".claude" / "worktrees"
registered_worktree_paths = {
    Path(line.removeprefix("worktree ")).resolve()
    for line in worktree_porcelain.splitlines()
    if line.startswith("worktree ")
}
if worktree_root.is_symlink() or (
    worktree_root.exists()
    and (
        not worktree_root.is_dir()
        or not any(
            str(path).startswith(str(worktree_root.resolve()) + os.sep)
            for path in registered_worktree_paths
        )
    )
):
    raise SystemExit("FATAL: reviewed local Claude worktree exclusion is invalid")
declared_exclusion_paths = set(reviewed_exclusions["paths"])


def reviewed_ignored_path(path: str) -> bool:
    """Return whether an ignored path is declared local state or regenerable cache."""
    return (
        path in declared_exclusion_paths
        or path in {".pytest_cache/", ".ruff_cache/"}
        or path.endswith("/__pycache__/")
    )


reviewed_ignored = sum(
    reviewed_ignored_path(path) for path in ignored_paths
)
unexpected_ignored = sorted(
    path for path in ignored_paths if not reviewed_ignored_path(path)
)
print(authority.hexdigest())
print(worktree.hexdigest())
print(changed)
print(ignored)
print(prunable_worktrees)
print(reviewed_ignored)
print(len(unexpected_ignored))
print(state.get("required_contract_mode", "MISSING"))
print(state.get("dataset_event_id") or "NONE")
print(state.get("dataset_admission_stage") or "NONE")
print(audited_dataset["status"])
print(audited_dataset["dataset_run_id"])
print(audited_dataset["report_count"])
print(audited_dataset["blocker"])
print(pair_id)
print(artifacts["canonical_v3"]["parquet_path"])
print(artifacts["base28"]["parquet_path"])
print(native["m1"]["root"])
print(native["m5"]["root"])
print(lineage["coverage"]["base28_time_max_utc"])
print(lineage["coverage"]["canonical_time_max_utc"])
print(
    "local=M5 sequence=96 "
    f"signal={MODEL_NATIVE_SIGNAL_DIM} "
    f"ctx_cont={MODEL_NATIVE_CTX_CONT_DIM} "
    f"ctx_cat={MODEL_NATIVE_CTX_CAT_DIM} "
    f"mtf_per_tf={MULTI_TF_FEATURE_COUNT_V4} mtf=M15,H1,H4,D1"
)
print(
    f"signal={MODEL_NATIVE_SIGNAL_SCHEMA_VERSION} "
    f"split={MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION} "
    f"mandatory={MODEL_NATIVE_MANDATORY_FULL_STACK_SCHEMA_VERSION} "
    f"matrix={HTF_V4_MATRIX_CONTRACT} "
    f"cache={HTF_V4_CACHE_SCHEMA_VERSION} "
    f"liveness={HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION}"
)
for value in candidate_session:
    print(value)
for value in current_source_technical_recipe:
    print(value)
PY
)

authority_sha256=${identity[0]}
worktree_sha256=${identity[1]}
changed_path_count=${identity[2]}
ignored_path_count=${identity[3]}
prunable_worktree_count=${identity[4]}
reviewed_ignored_path_count=${identity[5]}
unexpected_ignored_path_count=${identity[6]}
required_contract_mode=${identity[7]}
dataset_event_id=${identity[8]}
dataset_admission_stage=${identity[9]}
audited_dataset_status=${identity[10]}
audited_dataset_run_id=${identity[11]}
audited_dataset_report_count=${identity[12]}
audited_dataset_blocker=${identity[13]}
pair_generation_id=${identity[14]}
canonical_v3_path=${identity[15]}
base28_path=${identity[16]}
native_m1_root=${identity[17]}
native_m5_root=${identity[18]}
m1_time_max=${identity[19]}
m5_time_max=${identity[20]}
entry_contract_summary=${identity[21]}
feature_contract_summary=${identity[22]}
candidate_session_status=${identity[23]}
candidate_validation_status=${identity[24]}
candidate_session_contract_sha256=${identity[25]}
candidate_session_state_sha256=${identity[26]}
candidate_recipe_sha256=${identity[27]}
candidate_source_bindings_sha256=${identity[28]}
candidate_source_closure=${identity[29]}
current_source_technical_recipe_status=${identity[30]}
current_source_technical_recipe_sha256=${identity[31]}
current_source_technical_recipe_bindings_sha256=${identity[32]}
current_source_technical_recipe_closure=${identity[33]}
head_commit=$(git rev-parse HEAD)
if (( prunable_worktree_count > 0 )); then
  source_identity_gate=BLOCK_PRUNABLE_WORKTREE_REGISTRATION
elif (( changed_path_count > 0 )); then
  source_identity_gate=BLOCK_DIRTY_WORKTREE
elif (( unexpected_ignored_path_count > 0 )); then
  source_identity_gate=BLOCK_UNEXPECTED_IGNORED_CONTENT
else
  source_identity_gate=READY_CLEAN_WORKTREE__REVIEWED_LOCAL_EXCLUSIONS
fi

if [[ "$mode" == check ]]; then
  echo "mode: check"
  echo "authority_fingerprint: $authority_sha256"
  echo "decision: BLOCK"
  echo "head_commit: $head_commit"
  echo "changed_path_count: $changed_path_count"
  echo "ignored_path_count: $ignored_path_count"
  echo "reviewed_ignored_path_count: $reviewed_ignored_path_count"
  echo "unexpected_ignored_path_count: $unexpected_ignored_path_count"
  echo "prunable_worktree_count: $prunable_worktree_count"
  echo "worktree_fingerprint: $worktree_sha256"
  echo "candidate_session: $candidate_session_status"
  echo "candidate_recipe_sha256: $candidate_recipe_sha256"
  echo "candidate_source_closure: $candidate_source_closure"
  echo "current_source_technical_recipe: $current_source_technical_recipe_status"
  echo "current_source_technical_recipe_closure: $current_source_technical_recipe_closure"
  exit 0
fi

echo "# GX1 XAU Direction Repair Takeover (compact)"
echo "mode: $mode"
echo "takeover_entrypoint: scripts/entry_next_edge_control.sh handover"
echo "handover_owner: scripts/gx1_handover.sh"
echo
echo "## Goal"
echo "Build the GX1 trading bot for gold/XAUUSD as one learned Entry/Exit bundle."
echo "Entry selects LONG/SHORT/FLAT direction; Exit selects HOLD/EXIT_NOW."
echo "The active path has no competing direction route, fallback or soft pass-through."
echo "Near-perfect practical precision remains a target, not a proven result."
echo
echo "## Current verdict"
echo "decision: BLOCK"
echo "required_contract_mode: $required_contract_mode"
echo "dataset_event_id: $dataset_event_id"
echo "dataset_admission_stage: $dataset_admission_stage"
echo "accepted_bundle_dir: NONE"
echo "current_audited_dataset_status: $audited_dataset_status"
echo "current_audited_dataset_run_id: $audited_dataset_run_id"
echo "current_audited_dataset_report_count: $audited_dataset_report_count"
echo "dataset_contract: HASH_BOUND_AUDITED_REPORT_ONLY_PRODUCTION_ECONOMICS_BLOCKED"
echo "train_recipe: HISTORICAL_V6_V8_BLOCKED__CURRENT_SOURCE_V9_CANDIDATE_SESSION_ACTIVE_OR_RESUMABLE__NO_PROMOTION_AUTHORITY"
echo "model_contract: NO_ADMITTED_UNIFIED_BUNDLE"
echo "historical_pnl_winrate: UNPROVEN"
echo "strict_preflight: PASS_V4_TECHNICAL_PIPELINE_ONLY_NO_EXTERNAL_TRAIN_AUTHORITY"
echo "strict_preflight_test_accessed: NO"
echo "technical_checkpoint_bundle_parity: PASS_TECHNICAL_ONLY_NOT_CANDIDATE"
echo "technical_checkpoint_bundle_parity_method: CLEAN_CPU_TO_CLEAN_CPU_EXACT__CUDA_HASH_BOUND_NOT_BITWISE_CLAIMED"
echo "val_decision_journal: PASS_VAL_ONLY_PLUMBING_NOT_EDGE_OR_BACKTEST"
echo "candidate_static_gate_source_policy: EXIT_ONLY_PROVISIONAL_POSITIVE_OPEN__HASH_BOUND_DIRECT_EXIT_INPUT_REQUIRED__ENTRY_STRICT"
echo "candidate_static_gate_runtime_evidence: COMPLETE_31004_BATCH_TRAIN_AND_FULL_VAL__TECHNICAL_BUNDLE_ONLY__NO_CANDIDATE_ACCEPTANCE"
echo "candidate_session: $candidate_session_status"
echo "candidate_validation: $candidate_validation_status"
echo "candidate_session_contract_sha256: $candidate_session_contract_sha256"
echo "candidate_session_state_sha256: $candidate_session_state_sha256"
echo "candidate_recipe_sha256: $candidate_recipe_sha256"
echo "candidate_source_bindings_sha256: $candidate_source_bindings_sha256"
echo "candidate_source_closure: $candidate_source_closure"
echo "current_source_technical_recipe: $current_source_technical_recipe_status"
echo "current_source_technical_recipe_sha256: $current_source_technical_recipe_sha256"
echo "current_source_technical_recipe_bindings_sha256: $current_source_technical_recipe_bindings_sha256"
echo "current_source_technical_recipe_closure: $current_source_technical_recipe_closure"
echo "external_full_training: NO_GO_PENDING_EXPLICIT_COST_REVIEW_FROZEN_COMMIT_RECIPE_AND_FULL_CANDIDATE_PLAN"
echo "fresh_31004_train: BLOCKED_PENDING_HOST_RESTART_AND_EXPLICIT_REAUTHORIZATION"
echo "exit_contract: LOCAL_M1_PLUS_CAUSAL_M5_M15_H1_H4_D1_REQUIRED"
# A restated test count goes stale the moment anyone adds a test — and
# every restated number in this repository has (rule 13/25). State the
# standing requirement, which cannot rot, and date the last verification.
echo "source_regression: RELEVANT_CONTRACT_TESTS_MUST_PASS_BEFORE_EACH_SOURCE_CHANGE"
echo "source_regression_last_verified: focused source-binding/gate/parity regressions must pass on the frozen repair commit; no whole-repository green claim is made here"
echo "pair_generation_id: $pair_generation_id"
echo "native_m1_root: $native_m1_root"
echo "native_m5_root: $native_m5_root"
echo "canonical_v3_path: $canonical_v3_path"
echo "base28_path: $base28_path"
echo "source_time_max: M1=$m1_time_max M5=$m5_time_max"
echo
echo "## Fixed architecture"
echo "feature_owners: SAME_8_IMPLEMENTATIONS_NATIVE_M5_AND_M1_NO_VALUE_COPY"
# Current counts and identities are imported above from the code-owned signal
# and feature owners. The local surface exposes the entire candidate pool in
# owner order; no shell-restated top-k, quota or score cutoff is authoritative.
echo "entry: $entry_contract_summary"
echo "feature_contracts: $feature_contract_summary"
echo "entry_feature_surface: HASH_BOUND_NATIVE_M5_LOADED_ONCE_EXACT_ZERO_COPY_SPLIT_WINDOWS"
echo "exit: local=M1 sequence=480 mtf=M5,M15,H1,H4,D1 same_contract_plus_causal_path shared_encoder=true"
echo "mtf_construction: CLOSED_OHLCV_BEFORE_FEATURES_NO_COMPUTED_M1_RESAMPLING"
echo "direction_authority: UNIQUE_RAW_BPS_ENTRY_Q_ARGMAX_OR_FAIL_CLOSED"
echo "exit_authority: UNIQUE_SAME_BUNDLE_MODEL_ARGMAX_OR_FAIL_CLOSED"
echo "execution_path: DETERMINISTIC_FP32 feature_workers=1 dataloader_workers=0"
echo
echo "## Resume boundary"
echo "scope: OFFLINE_SHARED_FEATUREBASE_ONLY"
echo "source_identity_gate: $source_identity_gate"
echo "resume_stage: RETAIN_V6_EPOCH_ONE_TECHNICAL_RESULT__V7_SMOKE_PREFLIGHT_MATERIALIZED"
echo "dataset_rebuild: NOT_REQUIRED_FOR_OFFLINE_RESEARCH; PRODUCTION_ECONOMICS_REVIEW_MAY_REQUIRE_A_SUCCESSOR"
echo "production_economics_blocker: $audited_dataset_blocker"
echo "capacity: audits=4G training_max=20G swap=512M cpu=0-1 dataloader_workers=0 one_job_at_a_time"
echo "local_cuda: V5_TECHNICAL_SMOKE_COMPLETED__63C_211_56W_9447MIB__POSTRUN_CPU_AUDIT_FAIL__CANDIDATE_READINESS_READY"
echo "cuda_speed: CUDA_ACTIVATION_RETENTION_0_45_ALLOCATOR_FENCE_FP32_ONLY__64_BATCHES_101_889S_TO_86_863S__FULL_TRAIN_EPOCH_APPROX_11_7H"
echo "current_cuda_authority: PARTIAL_SESSION_MECHANICS_PROVED__NO_AUTOMATIC_FULL_EPOCH_OR_VAL_TEST_EXECUTION"
echo "remote_compute: PREPARE_ONLY_UNTIL_EXPLICIT_COST_APPROVAL_FROZEN_COMMIT_AND_V46_HASHES_REQUIRED"
echo "environment: CPYTHON_3.10.12 PINNED_DIRECT_REQUIREMENTS"
echo "ordered_control_routes:"
echo "  1. run this handover and confirm clean source, no competing job, retained checkpoint-640 evidence and the V5 bundle commit/source identity"
echo "  2. run the CPU-only V5 smoke-bundle audit, then a fresh candidate-readiness recheck; do not resume V4 under current source"
echo "  3. reach first complete TRAIN epoch and full VAL before interpreting learning; checkpoint selection and early-stop policy stay frozen"
echo "  4. repeat candidate audit only after a completed full candidate; no partial-session metric is an edge claim"
echo "  5. run preregistered untouched-TEST evaluation only after the candidate/OOS gates, never as a troubleshooting input"
echo "  6. bind immutable broker costs, financing, gap/terminal treatment and portfolio capital before demo, paper, live or production-net claims"
echo "forbidden_routes: live, paper, broker, daemon, promotion, drift-adaptation"
echo
echo "## Source worktree"
echo "head_commit: $head_commit"
echo "changed_path_count: $changed_path_count"
echo "ignored_path_count: $ignored_path_count"
echo "reviewed_ignored_path_count: $reviewed_ignored_path_count"
echo "unexpected_ignored_path_count: $unexpected_ignored_path_count"
echo "ignored_content_scope: DECLARED_LOCAL_RUNTIME_EXCLUSIONS_PLUS_REGENERABLE_CACHE_ONLY"
echo "worktree_fingerprint: $worktree_sha256"
echo "authority_fingerprint: $authority_sha256"
echo "registered_worktrees: $(git worktree list --porcelain | awk '$1 == "worktree" {count++} END {print count+0}')"
echo "prunable_worktree_count: $prunable_worktree_count"
echo "active_training_processes: $(pgrep -fc 'gx1.models.entry_v10.entry_v10_ctx_train_v3 --train' || true)"

if [[ "$mode" == verbose ]]; then
  echo
  echo "## Full Handover (--verbose)"
  cat "$HANDOVER"
fi
