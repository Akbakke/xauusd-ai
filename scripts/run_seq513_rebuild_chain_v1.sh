#!/usr/bin/env bash
# One fail-closed chain for the current shared Entry/Exit feature architecture:
#   canonical-pair TRAIN rank -> native M5/M1 owner surfaces (one worker)
#   -> M5 model source + M5/M15/H1/H4/D1 cache -> TRAIN ranking
#   -> one signal manifest -> contract-derived M1/M5 feature surfaces
#   -> preflight -> combined Entry/Exit dataset rebuild.
#
# The driver never discovers artifacts, waits for an external producer, resumes
# from inferred debris, or treats existing split manifests as completion. It
# owns the ranker so a second heavyweight path cannot overlap the dataset job,
# and stops before post-rebuild audits/readiness. Every producer receives the
# same explicit run_id, ranking identity, split window, and event-local
# immutable paths.
set -Eeuo pipefail

ENG=/home/andre2/src/GX1_ENGINE
PY=$ENG/.venv/bin/python

RUN_ID=
EVENT=
RANKING=
MANIFEST=
PRE_OUT=
HISTORY_START=
TRAIN_START=
TRAIN_END=
VAL_START=
VAL_END=
TEST_START=
TEST_END=
M1_LIFECYCLE_PAIR_MANIFEST=
M1_LIFECYCLE_PAIR_GENERATION_ROOT=
REGISTRY_FIT_TRAIN_END=
REGISTRY_FIT_INNER_END=
VOLATILITY_SQUEEZE_MANIFEST=
VOLATILITY_SQUEEZE_MANIFEST_SHA256=

usage() {
  printf '%s\n' \
    "Usage: $0 --run-id ID --event-root /absolute/event/root" \
    "  --feature-ranking-json /absolute/event/root/ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_<UTC>.json" \
    "  --preflight-out-dir /absolute/event/root/fresh-preflight-dir" \
    "  --m1-lifecycle-pair-manifest-json /absolute/immutable/generation/PAIR_MANIFEST.json" \
    "  --m1-lifecycle-pair-generation-root /absolute/immutable/generations" \
    "  Exit target policy is fit and hash-bound from TRAIN native M1" \
    "  Entry direction/early-move policy is fit on TRAIN native M5" \
    "  --registry-fit-inner-end UTC (required chronological inner-TRAIN boundary)" \
    "  [--registry-fit-train-end UTC (defaults to --train-end: same value, one origin)]" \
    "  --volatility-squeeze-manifest /absolute/immutable/six-clock/manifest.json" \
    "  --volatility-squeeze-manifest-sha256 SHA256" \
    "  --history-start UTC --train-start UTC --train-end UTC" \
    "  --val-start UTC --val-end UTC --test-start UTC --test-end UTC" \
    "The ranking and preflight targets must be fresh. The chain allocates the" \
    "signal-manifest path at its producer boundary so its immutable timestamp" \
    "is both newer than ranking and inside the producer's freshness window."
}

die_args() {
  printf '[ABORT] %s\n' "$*" >&2
  usage >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --run-id)
      (($# >= 2)) || die_args "--run-id requires a value"
      [[ -z $RUN_ID ]] || die_args "duplicate --run-id"
      RUN_ID=$2
      shift 2
      ;;
    --event-root)
      (($# >= 2)) || die_args "--event-root requires a value"
      [[ -z $EVENT ]] || die_args "duplicate --event-root"
      EVENT=$2
      shift 2
      ;;
    --feature-ranking-json)
      (($# >= 2)) || die_args "--feature-ranking-json requires a value"
      [[ -z $RANKING ]] || die_args "duplicate --feature-ranking-json"
      RANKING=$2
      shift 2
      ;;
    --preflight-out-dir)
      (($# >= 2)) || die_args "--preflight-out-dir requires a value"
      [[ -z $PRE_OUT ]] || die_args "duplicate --preflight-out-dir"
      PRE_OUT=$2
      shift 2
      ;;
    --history-start)
      (($# >= 2)) || die_args "--history-start requires a value"
      [[ -z $HISTORY_START ]] || die_args "duplicate --history-start"
      HISTORY_START=$2
      shift 2
      ;;
    --train-start)
      (($# >= 2)) || die_args "--train-start requires a value"
      [[ -z $TRAIN_START ]] || die_args "duplicate --train-start"
      TRAIN_START=$2
      shift 2
      ;;
    --train-end)
      (($# >= 2)) || die_args "--train-end requires a value"
      [[ -z $TRAIN_END ]] || die_args "duplicate --train-end"
      TRAIN_END=$2
      shift 2
      ;;
    --val-start)
      (($# >= 2)) || die_args "--val-start requires a value"
      [[ -z $VAL_START ]] || die_args "duplicate --val-start"
      VAL_START=$2
      shift 2
      ;;
    --val-end)
      (($# >= 2)) || die_args "--val-end requires a value"
      [[ -z $VAL_END ]] || die_args "duplicate --val-end"
      VAL_END=$2
      shift 2
      ;;
    --test-start)
      (($# >= 2)) || die_args "--test-start requires a value"
      [[ -z $TEST_START ]] || die_args "duplicate --test-start"
      TEST_START=$2
      shift 2
      ;;
    --test-end)
      (($# >= 2)) || die_args "--test-end requires a value"
      [[ -z $TEST_END ]] || die_args "duplicate --test-end"
      TEST_END=$2
      shift 2
      ;;
    --m1-lifecycle-pair-manifest-json)
      (($# >= 2)) || die_args "--m1-lifecycle-pair-manifest-json requires a value"
      [[ -z $M1_LIFECYCLE_PAIR_MANIFEST ]] || die_args "duplicate --m1-lifecycle-pair-manifest-json"
      M1_LIFECYCLE_PAIR_MANIFEST=$2
      shift 2
      ;;
    --m1-lifecycle-pair-generation-root)
      (($# >= 2)) || die_args "--m1-lifecycle-pair-generation-root requires a value"
      [[ -z $M1_LIFECYCLE_PAIR_GENERATION_ROOT ]] || die_args "duplicate --m1-lifecycle-pair-generation-root"
      M1_LIFECYCLE_PAIR_GENERATION_ROOT=$2
      shift 2
      ;;
    --registry-fit-inner-end)
      (($# >= 2)) || die_args "--registry-fit-inner-end requires a value"
      [[ -z $REGISTRY_FIT_INNER_END ]] || die_args "duplicate --registry-fit-inner-end"
      REGISTRY_FIT_INNER_END=$2
      shift 2
      ;;
    --registry-fit-train-end)
      (($# >= 2)) || die_args "--registry-fit-train-end requires a value"
      [[ -z $REGISTRY_FIT_TRAIN_END ]] || die_args "duplicate --registry-fit-train-end"
      REGISTRY_FIT_TRAIN_END=$2
      shift 2
      ;;
    --volatility-squeeze-manifest)
      (($# >= 2)) || die_args "--volatility-squeeze-manifest requires a value"
      [[ -z $VOLATILITY_SQUEEZE_MANIFEST ]] || die_args "duplicate --volatility-squeeze-manifest"
      VOLATILITY_SQUEEZE_MANIFEST=$2
      shift 2
      ;;
    --volatility-squeeze-manifest-sha256)
      (($# >= 2)) || die_args "--volatility-squeeze-manifest-sha256 requires a value"
      [[ -z $VOLATILITY_SQUEEZE_MANIFEST_SHA256 ]] || die_args "duplicate --volatility-squeeze-manifest-sha256"
      VOLATILITY_SQUEEZE_MANIFEST_SHA256=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die_args "unknown argument: $1"
      ;;
  esac
done

for name in \
  RUN_ID EVENT RANKING PRE_OUT HISTORY_START TRAIN_START TRAIN_END \
  VAL_START VAL_END TEST_START TEST_END M1_LIFECYCLE_PAIR_MANIFEST \
  M1_LIFECYCLE_PAIR_GENERATION_ROOT REGISTRY_FIT_INNER_END \
  VOLATILITY_SQUEEZE_MANIFEST \
  VOLATILITY_SQUEEZE_MANIFEST_SHA256; do
  [[ -n ${!name} ]] || die_args "required argument missing: $name"
done
# The V29 registry TRAIN-fit window end defaults to the chain's one declared
# --train-end: same value, one origin (the chain's split authority), so the
# registry constants are fitted on exactly the TRAIN population the rebuild
# declares (rule 2g).  An explicit --registry-fit-train-end overrides only by
# operator decision.
if [[ -z $REGISTRY_FIT_TRAIN_END ]]; then
  REGISTRY_FIT_TRAIN_END=$TRAIN_END
fi
[[ -x $PY ]] || die_args "repository Python is not executable: $PY"
if [[ ! $RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$ ]]; then
  die_args "--run-id has invalid format"
fi
[[ $EVENT == /* ]] || die_args "--event-root must be absolute"
[[ -d $EVENT && ! -L $EVENT ]] || die_args "--event-root must be an existing regular directory"

SRC="$EVENT/FULL_PLUS_CTX_v3src.parquet"
MTF="$EVENT/MULTI_TF_V4_CACHE"
M1_ENRICHED="$EVENT/m1_enriched.parquet"
M5_ENRICHED="$EVENT/m5_enriched.parquet"
M1_FEATURE_BASE="$EVENT/m1_feature_base.parquet"
M5_FEATURE_BASE="$EVENT/m5_feature_base.parquet"
M1_CHECKPOINT="$EVENT/m1_enriched_checkpoint"
M5_CHECKPOINT="$EVENT/m5_enriched_checkpoint"
OUTPUT="$EVENT/dataset/v10_seq513_dataset__DIR_TRAIN_FIT.parquet"
AUDIT="$EVENT/audit"
EXIT_LIFECYCLE="$EVENT/exit_lifecycle"
TRAIN_GROUP_A_CHECKPOINT_MANIFEST="$EVENT/dataset/_v10_seq513_dataset__DIR_TRAIN_FIT_train_GROUP_A_CHECKPOINT/CHECKPOINT_MANIFEST.json"
STAMP=$("$PY" -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))')
DATASET_REBUILD_TERMINAL="$EVENT/rebuild_authority/ENTRY_MODEL_NATIVE_SEQ513_DATASET_REBUILD_TERMINAL_${STAMP}.json"
PREFREEZE_TEST_SEAL="$EVENT/rebuild_authority/ENTRY_MODEL_NATIVE_SEQ513_UNTOUCHED_TEST_SEAL_${STAMP}.json"
STARTED_UTC=$("$PY" -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())')
IFS= read -r BOOT_ID </proc/sys/kernel/random/boot_id
CHAIN_PID=$$
LOG="$EVENT/CHAIN_LOG_${STAMP}.txt"
STATUS="$EVENT/CHAIN_STATUS.json"

CURRENT_STEP=bootstrap
TERMINAL_WRITTEN=0
STATUS_INITIALIZED=1
GIT_HEAD=
WORKTREE_STATUS=
RANKING_SHA256=
MANIFEST_SHA256=
PREFLIGHT_JSON=
PREFLIGHT_SHA256=
SOURCE_CASCADE="$EVENT/SEQ513_SOURCE_CASCADE_PROOF_${STAMP}.json"
SOURCE_CASCADE_SHA256=
PAIR_MANIFEST_SHA256=
PAIR_GENERATION_ID=
DATASET_REBUILD_TERMINAL_SHA256=
PREFREEZE_TEST_SEAL_SHA256=
CV2=
BASE28=
NATIVE_M1_ROOT=
NATIVE_M5_ROOT=
TAPE=

tg() {
  "$PY" - "$1" <<'PYEOF' || true
import importlib.util
import sys

spec = importlib.util.spec_from_file_location(
    "gx1_tg", "/home/andre2/src/GX1_ENGINE/scripts/gx1_telegram_notifier.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.send(sys.argv[1])
PYEOF
}

write_status() {
  local step=$1
  local state=$2
  local reason=${3:-}
  local exit_code=${4:-}
  "$PY" - \
    "$step" "$state" "$reason" "$exit_code" "$STATUS" "$RUN_ID" "$EVENT" \
    "$LOG" "$GIT_HEAD" "$RANKING" "$RANKING_SHA256" "$MANIFEST" \
    "$MANIFEST_SHA256" "$PRE_OUT" "$PREFLIGHT_JSON" "$PREFLIGHT_SHA256" \
    "$STARTED_UTC" "$BOOT_ID" "$CHAIN_PID" \
    "$SOURCE_CASCADE" "$SOURCE_CASCADE_SHA256" "$OUTPUT" "$AUDIT" \
    "$EXIT_LIFECYCLE" "$M1_FEATURE_BASE" "$M5_FEATURE_BASE" \
    "$M1_LIFECYCLE_PAIR_MANIFEST" "$PAIR_MANIFEST_SHA256" \
    "$PAIR_GENERATION_ID" "$VOLATILITY_SQUEEZE_MANIFEST" \
    "$VOLATILITY_SQUEEZE_MANIFEST_SHA256" "$DATASET_REBUILD_TERMINAL" \
    "$DATASET_REBUILD_TERMINAL_SHA256" "$PREFREEZE_TEST_SEAL" \
    "$PREFREEZE_TEST_SEAL_SHA256" <<'PYEOF'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    step,
    state,
    reason,
    exit_code,
    raw_path,
    run_id,
    event_root,
    log_path,
    git_head,
    ranking_path,
    ranking_sha256,
    manifest_path,
    manifest_sha256,
    preflight_out_dir,
    preflight_json,
    preflight_sha256,
    started_utc,
    boot_id,
    chain_pid,
    source_cascade_path,
    source_cascade_sha256,
    dataset_output_path,
    audit_output_dir,
    exit_lifecycle_dir,
    m1_feature_base_path,
    m5_feature_base_path,
    pair_manifest_path,
    pair_manifest_sha256,
    pair_generation_id,
    volatility_squeeze_manifest_path,
    volatility_squeeze_manifest_sha256,
    dataset_rebuild_terminal_path,
    dataset_rebuild_terminal_sha256,
    prefreeze_test_seal_path,
    prefreeze_test_seal_sha256,
) = sys.argv[1:]
path = Path(raw_path)
now = datetime.now(timezone.utc)
terminal_states = {"GREEN", "RED", "ABORTED"}
terminal_path = None
if state in terminal_states:
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    terminal_path = path.with_name(f"CHAIN_TERMINAL_{stamp}_{state}.json")
payload = {
    "schema_version": "seq513_rebuild_chain_status_v9",
    "entry_run_id": run_id,
    "event_root": event_root,
    "step": step,
    "state": state,
    "started_utc": started_utc,
    "updated_utc": now.isoformat(),
    "boot_id": boot_id,
    "chain_pid": int(chain_pid),
    "log_path": log_path,
    "git_head": git_head or None,
    "feature_ranking": {
        "path": ranking_path,
        "sha256": ranking_sha256 or None,
    },
    "source_cascade": {
        "path": source_cascade_path,
        "sha256": source_cascade_sha256 or None,
    },
    "pair_authority": {
        "manifest_path": pair_manifest_path,
        "manifest_sha256": pair_manifest_sha256 or None,
        "pair_generation_id": pair_generation_id or None,
    },
    "volatility_squeeze_artifact": {
        "manifest_path": volatility_squeeze_manifest_path,
        "manifest_sha256": volatility_squeeze_manifest_sha256,
    },
    "dataset_rebuild_terminal": {
        "path": dataset_rebuild_terminal_path,
        "sha256": dataset_rebuild_terminal_sha256 or None,
    },
    "prefreeze_test_seal": {
        "path": prefreeze_test_seal_path,
        "sha256": prefreeze_test_seal_sha256 or None,
    },
    "signal_manifest": {
        "path": manifest_path,
        "sha256": manifest_sha256 or None,
    },
    "preflight": {
        "out_dir": preflight_out_dir,
        "json_path": preflight_json or None,
        "sha256": preflight_sha256 or None,
    },
    "outputs": {
        "dataset_output_stem": dataset_output_path,
        "audit_output_dir": audit_output_dir,
        "unified_exit_lifecycle_dir": exit_lifecycle_dir,
        "m1_exit_feature_base": m1_feature_base_path,
        "m5_entry_feature_base": m5_feature_base_path,
    },
    "next_boundary": (
        "POST_REBUILD_AUDITS_AND_READINESS"
        if state == "GREEN"
        else "REPAIR_CURRENT_FAILED_STEP_WITHOUT_REUSING_PARTIAL_OUTPUT"
        if state in terminal_states
        else "CURRENT_STEP_IN_PROGRESS"
    ),
    "terminal_event_path": str(terminal_path) if terminal_path else None,
}
if reason:
    payload["reason"] = reason
if exit_code:
    payload["exit_code"] = int(exit_code)
encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(temporary, flags, 0o644)
try:
    view = memoryview(encoded)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError(f"short status write: {temporary}")
        view = view[written:]
    os.fsync(descriptor)
finally:
    os.close(descriptor)
os.replace(temporary, path)
if terminal_path is not None:
    terminal_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    terminal_descriptor = os.open(terminal_path, terminal_flags, 0o644)
    try:
        terminal_view = memoryview(encoded)
        while terminal_view:
            written = os.write(terminal_descriptor, terminal_view)
            if written <= 0:
                raise OSError(f"short terminal status write: {terminal_path}")
            terminal_view = terminal_view[written:]
        os.fsync(terminal_descriptor)
    finally:
        os.close(terminal_descriptor)
PYEOF
}

terminal_status() {
  local state=$1
  local reason=$2
  local exit_code=$3
  if ((TERMINAL_WRITTEN == 0)); then
    if write_status "$CURRENT_STEP" "$state" "$reason" "$exit_code"; then
      TERMINAL_WRITTEN=1
    else
      printf '[chain] could not persist terminal status: state=%s step=%s\n' \
        "$state" "$CURRENT_STEP" >&2
    fi
  fi
}

fail() {
  local reason=${1:-"step failed"}
  local exit_code=${2:-2}
  trap - ERR
  terminal_status RED "$reason" "$exit_code"
  tg "GX1 seq513-kjede STOPPET RED: steg=$CURRENT_STEP run_id=$RUN_ID logg=$LOG"
  printf '[chain] RED at %s: %s — see %s\n' "$CURRENT_STEP" "$reason" "$LOG" >&2
  exit "$exit_code"
}

on_err() {
  local exit_code=$1
  local line=$2
  trap - ERR
  terminal_status RED "unexpected ERR at line $line" "$exit_code"
  tg "GX1 seq513-kjede STOPPET RED: steg=$CURRENT_STEP run_id=$RUN_ID logg=$LOG"
  exit "$exit_code"
}

on_signal() {
  local signal_name=$1
  local exit_code=$2
  trap - ERR TERM INT HUP
  terminal_status ABORTED "received $signal_name" "$exit_code"
  tg "GX1 seq513-kjede AVBRUTT: signal=$signal_name steg=$CURRENT_STEP run_id=$RUN_ID logg=$LOG"
  exit "$exit_code"
}

on_exit() {
  local exit_code=$?
  trap - ERR
  if ((STATUS_INITIALIZED == 1 && TERMINAL_WRITTEN == 0)); then
    terminal_status RED "non-terminal exit" "$exit_code"
  fi
}

trap 'on_err "$?" "$LINENO"' ERR
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT
trap 'on_signal HUP 129' HUP
trap on_exit EXIT

if ! (set -o noclobber; : >"$LOG") 2>/dev/null; then
  fail "immutable chain log path already exists"
fi
printf '[chain] run_id=%s event=%s log=%s\n' "$RUN_ID" "$EVENT" "$LOG"

# Source authority is one exact clean revision. Ignored files do not take part
# in the gate, while tracked changes and every non-ignored untracked file do.
CURRENT_STEP=source-revision
write_status "$CURRENT_STEP" RUNNING
if ! GIT_HEAD=$(git -C "$ENG" rev-parse --verify HEAD 2>>"$LOG"); then
  fail "cannot resolve repository HEAD"
fi
if ! WORKTREE_STATUS=$(git -C "$ENG" status --porcelain --untracked-files=all 2>>"$LOG"); then
  fail "cannot verify repository worktree status"
fi
if [[ -n $WORKTREE_STATUS ]]; then
  fail "repository worktree is not clean"
fi
write_status "$CURRENT_STEP" RUNNING

require_source_identity() {
  local observed_head observed_status
  if ! observed_head=$(git -C "$ENG" rev-parse --verify HEAD 2>>"$LOG"); then
    fail "cannot revalidate repository HEAD"
  fi
  [[ $observed_head == "$GIT_HEAD" ]] || fail "repository HEAD changed after binding"
  if ! observed_status=$(git -C "$ENG" status --porcelain --untracked-files=all 2>>"$LOG"); then
    fail "cannot revalidate repository worktree status"
  fi
  [[ -z $observed_status ]] || fail "repository worktree changed after binding"
}

# One pair manifest owns canonical M5, aligned M1, and both native roots.  The
# chain never accepts event-local copies or separately selected source trees.
CURRENT_STEP=pair-authority
write_status "$CURRENT_STEP" RUNNING
if ! PAIR_BINDING=$("$PY" - \
  "$M1_LIFECYCLE_PAIR_MANIFEST" \
  "$M1_LIFECYCLE_PAIR_GENERATION_ROOT" <<'PYEOF'
import hashlib
import json
import re
import sys
from pathlib import Path


def regular_file(raw: str, *, label: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise RuntimeError(f"{label} must be an exact absolute path: {path}")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing/non-regular: {path}")
    return path


def regular_dir(raw: str, *, label: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise RuntimeError(f"{label} must be an exact absolute path: {path}")
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} is missing/non-regular: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = regular_file(sys.argv[1], label="pair manifest")
generation_root = regular_dir(sys.argv[2], label="pair generation root")
payload = json.loads(manifest.read_text(encoding="utf-8"))
pair_id = str(payload.get("pair_generation_id") or "")
if not re.fullmatch(r"[0-9a-f]{64}", pair_id):
    raise RuntimeError("pair generation id is not an exact SHA-256")
expected_manifest = generation_root / pair_id / "PAIR_MANIFEST.json"
if manifest != expected_manifest:
    raise RuntimeError(
        "pair manifest must be the exact generation-local PAIR_MANIFEST.json"
    )
artifacts = payload.get("artifacts")
lineage = payload.get("lineage")
native = lineage.get("native_sources") if isinstance(lineage, dict) else None
if not isinstance(artifacts, dict) or not isinstance(native, dict):
    raise RuntimeError("pair manifest artifact/native lineage is missing")
canonical_meta = artifacts.get("canonical_v3")
base28_meta = artifacts.get("base28")
if not isinstance(canonical_meta, dict) or not isinstance(base28_meta, dict):
    raise RuntimeError("pair manifest canonical/base28 binding is missing")
canonical = regular_file(
    str(canonical_meta.get("parquet_path") or ""), label="pair canonical parquet"
)
base28 = regular_file(
    str(base28_meta.get("parquet_path") or ""), label="pair base28 parquet"
)
if canonical.parent != generation_root / pair_id or base28.parent != canonical.parent:
    raise RuntimeError("pair artifacts are outside the exact generation directory")
if sha256(canonical) != canonical_meta.get("parquet_sha256"):
    raise RuntimeError("pair canonical parquet hash mismatch")
if sha256(base28) != base28_meta.get("parquet_sha256"):
    raise RuntimeError("pair base28 parquet hash mismatch")

roots = []
tape_manifest = None
tape_manifest_sha256 = None
for timeframe in ("m1", "m5"):
    meta = native.get(timeframe)
    if not isinstance(meta, dict):
        raise RuntimeError(f"pair native {timeframe} binding is missing")
    root = regular_dir(str(meta.get("root") or ""), label=f"native {timeframe} root")
    native_manifest = regular_file(
        str(meta.get("manifest_path") or ""), label=f"native {timeframe} manifest"
    )
    if native_manifest.parent != root or sha256(native_manifest) != meta.get("manifest_sha256"):
        raise RuntimeError(f"pair native {timeframe} manifest mismatch")
    roots.append(root)
    if timeframe == "m5":
        tape_manifest = native_manifest
        tape_manifest_sha256 = str(meta.get("manifest_sha256") or "")

if tape_manifest is None or not re.fullmatch(r"[0-9a-f]{64}", tape_manifest_sha256 or ""):
    raise RuntimeError("native M5 tape manifest binding is missing")

print(
    "\t".join(
        (
            sha256(manifest),
            pair_id,
            str(canonical),
            str(base28),
            str(roots[0]),
            str(roots[1]),
            str(tape_manifest),
            str(tape_manifest_sha256),
        )
    )
)
PYEOF
); then
  fail "current pair authority validation failed"
fi
IFS=$'\t' read -r PAIR_MANIFEST_SHA256 PAIR_GENERATION_ID CV2 BASE28 \
  NATIVE_M1_ROOT NATIVE_M5_ROOT TAPE_MANIFEST TAPE_MANIFEST_SHA256 <<<"$PAIR_BINDING"
TAPE=$NATIVE_M5_ROOT
[[ -n $PAIR_MANIFEST_SHA256 && -n $PAIR_GENERATION_ID && -n $CV2 \
   && -n $BASE28 && -n $NATIVE_M1_ROOT && -n $NATIVE_M5_ROOT \
   && -n $TAPE_MANIFEST && -n $TAPE_MANIFEST_SHA256 ]] \
  || fail "current pair authority emitted an incomplete binding"
require_source_identity
write_status "$CURRENT_STEP" RUNNING

# Validate all operator-supplied identities before any producer starts. The
# ranking and fresh preflight namespace must be exact
# canonical absolute paths below this event root; symlink indirection, mutable
# latest names, pre-existing outputs, and implicit resume debris all fail.
CURRENT_STEP=contract-validation
write_status "$CURRENT_STEP" RUNNING
if ! "$PY" - \
  "$VOLATILITY_SQUEEZE_MANIFEST" \
  "$VOLATILITY_SQUEEZE_MANIFEST_SHA256" \
  "$M1_LIFECYCLE_PAIR_MANIFEST" "$PAIR_MANIFEST_SHA256" \
  "$PAIR_GENERATION_ID" "$TRAIN_START" "$TRAIN_END" >>"$LOG" 2>&1 <<'PYEOF'
import json
import sys
from pathlib import Path

import pandas as pd

from gx1.features.volatility_squeeze_state_v1 import (
    load_volatility_squeeze_artifact_manifest,
)

manifest_path = Path(sys.argv[1]).expanduser()
loaded = load_volatility_squeeze_artifact_manifest(
    manifest_path,
    expected_sha256=sys.argv[2],
)
payload = json.loads(loaded.manifest_path.read_text(encoding="utf-8"))
lineage = payload["common_train_lineage"]
if (
    lineage["pair_manifest_artifact"] != str(Path(sys.argv[3]).resolve(strict=True))
    or lineage["pair_manifest_sha256"] != sys.argv[4]
    or lineage["pair_generation_id"] != sys.argv[5]
    or pd.Timestamp(lineage["declared_train_window_start"]) != pd.Timestamp(sys.argv[6])
    or pd.Timestamp(lineage["declared_train_window_end"]) != pd.Timestamp(sys.argv[7])
):
    raise RuntimeError(
        "VOLATILITY_SQUEEZE_CHAIN_TRAIN_OR_PAIR_LINEAGE_MISMATCH"
    )
PYEOF
then
  fail "volatility-squeeze six-clock artifact identity validation failed"
fi
if ! "$PY" - "$TRAIN_START" "$REGISTRY_FIT_INNER_END" "$REGISTRY_FIT_TRAIN_END" <<'PYEOF'
import sys
import pandas as pd

start, inner, end = (pd.Timestamp(value) for value in sys.argv[1:])
if any(value.tzinfo is None or value.utcoffset() != pd.Timedelta(0) for value in (start, inner, end)):
    raise RuntimeError("registry fit split must use timezone-aware UTC timestamps")
if not start < inner < end:
    raise RuntimeError("registry fit inner boundary must lie strictly inside TRAIN")
PYEOF
then
  fail "registry chronological inner-TRAIN boundary is invalid"
fi
if ! "$PY" - \
  "$EVENT" "$RANKING" "$PRE_OUT" "$OUTPUT" "$AUDIT" \
  "$SRC" "$CV2" "$MTF" "$TAPE" "$M1_LIFECYCLE_PAIR_MANIFEST" \
  "$M1_LIFECYCLE_PAIR_GENERATION_ROOT" "$EXIT_LIFECYCLE" \
  "$BASE28" "$NATIVE_M1_ROOT" "$NATIVE_M5_ROOT" \
  "$M1_ENRICHED" "$M5_ENRICHED" "$M1_FEATURE_BASE" "$M5_FEATURE_BASE" \
  "$M1_CHECKPOINT" "$M5_CHECKPOINT" "$SOURCE_CASCADE" \
  "$DATASET_REBUILD_TERMINAL" "$PREFREEZE_TEST_SEAL" \
  "$HISTORY_START" "$TRAIN_START" "$TRAIN_END" "$VAL_START" "$VAL_END" \
  "$TEST_START" "$TEST_END" >>"$LOG" 2>&1 <<'PYEOF'
import re
import sys
from pathlib import Path

import pandas as pd

(
    raw_event,
    raw_ranking,
    raw_preflight,
    raw_output,
    raw_audit,
    raw_source,
    raw_canonical,
    raw_mtf,
    raw_tape,
    raw_m1_lifecycle_pair_manifest,
    raw_m1_lifecycle_pair_generation_root,
    raw_exit_lifecycle,
    raw_base28,
    raw_native_m1,
    raw_native_m5,
    raw_m1_enriched,
    raw_m5_enriched,
    raw_m1_feature_base,
    raw_m5_feature_base,
    raw_m1_checkpoint,
    raw_m5_checkpoint,
    raw_source_cascade,
    raw_dataset_rebuild_terminal,
    raw_prefreeze_test_seal,
    raw_history_start,
    raw_train_start,
    raw_train_end,
    raw_val_start,
    raw_val_end,
    raw_test_start,
    raw_test_end,
) = sys.argv[1:]


def exact_path(raw: str, *, label: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{label} must be absolute: {path}")
    resolved = path.resolve(strict=False)
    if path != resolved:
        raise RuntimeError(f"{label} must be canonical and symlink-free: {path}")
    if any(part == "latest" or "_latest" in part for part in path.parts):
        raise RuntimeError(f"{label} contains mutable latest identity: {path}")
    return path


event = exact_path(raw_event, label="event root")
if not event.is_dir() or event.is_symlink():
    raise RuntimeError(f"event root is not a regular directory: {event}")

ranking = exact_path(raw_ranking, label="feature ranking")
preflight = exact_path(raw_preflight, label="preflight output directory")
output = exact_path(raw_output, label="dataset output")
audit = exact_path(raw_audit, label="audit output directory")
source = exact_path(raw_source, label="source parquet")
canonical = exact_path(raw_canonical, label="canonical-v2 parquet")
mtf = exact_path(raw_mtf, label="MTF cache")
tape = exact_path(raw_tape, label="tape root")
m1_lifecycle_pair_manifest = exact_path(
    raw_m1_lifecycle_pair_manifest,
    label="M1 lifecycle pair manifest",
)
m1_lifecycle_pair_generation_root = exact_path(
    raw_m1_lifecycle_pair_generation_root,
    label="M1 lifecycle pair generation root",
)
exit_lifecycle = exact_path(
    raw_exit_lifecycle,
    label="Exit lifecycle output directory",
)
base28 = exact_path(raw_base28, label="pair base28 parquet")
native_m1 = exact_path(raw_native_m1, label="native M1 root")
native_m5 = exact_path(raw_native_m5, label="native M5 root")
m1_enriched = exact_path(raw_m1_enriched, label="M1 enriched output")
m5_enriched = exact_path(raw_m5_enriched, label="M5 enriched output")
m1_feature_base = exact_path(raw_m1_feature_base, label="M1 feature-base output")
m5_feature_base = exact_path(raw_m5_feature_base, label="M5 feature-base output")
m1_checkpoint = exact_path(raw_m1_checkpoint, label="M1 checkpoint")
m5_checkpoint = exact_path(raw_m5_checkpoint, label="M5 checkpoint")
source_cascade = exact_path(raw_source_cascade, label="source cascade output")
dataset_rebuild_terminal = exact_path(
    raw_dataset_rebuild_terminal,
    label="dataset rebuild terminal",
)
prefreeze_test_seal = exact_path(
    raw_prefreeze_test_seal,
    label="pre-freeze TEST seal",
)
authority_dir = event / "rebuild_authority"
if (
    dataset_rebuild_terminal.parent != authority_dir
    or prefreeze_test_seal.parent != authority_dir
    or not dataset_rebuild_terminal.name.startswith(
        "ENTRY_MODEL_NATIVE_SEQ513_DATASET_REBUILD_TERMINAL_"
    )
    or not dataset_rebuild_terminal.name.endswith(".json")
    or not prefreeze_test_seal.name.startswith(
        "ENTRY_MODEL_NATIVE_SEQ513_UNTOUCHED_TEST_SEAL_"
    )
    or not prefreeze_test_seal.name.endswith(".json")
):
    raise RuntimeError("TEST authority paths are not exact event-owned events")

for label, path in (
    ("feature ranking", ranking),
    ("preflight output directory", preflight),
    ("dataset output", output),
    ("audit output directory", audit),
    ("source parquet", source),
    ("MTF cache", mtf),
    ("M1 enriched output", m1_enriched),
    ("M5 enriched output", m5_enriched),
    ("M1 feature-base output", m1_feature_base),
    ("M5 feature-base output", m5_feature_base),
    ("M1 checkpoint", m1_checkpoint),
    ("M5 checkpoint", m5_checkpoint),
    ("source cascade output", source_cascade),
    ("dataset rebuild terminal", dataset_rebuild_terminal),
    ("pre-freeze TEST seal", prefreeze_test_seal),
    ("Exit lifecycle output directory", exit_lifecycle),
):
    try:
        path.relative_to(event)
    except ValueError as exc:
        raise RuntimeError(f"{label} must be below event root: {path}") from exc

ranking_pattern = re.compile(
    r"ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_"
    r"(?P<stamp>\d{8}T\d{6}(?:\d{6})?Z)\.json"
)
ranking_match = ranking_pattern.fullmatch(ranking.name)
if ranking.exists() or ranking.is_symlink() or ranking_match is None:
    raise RuntimeError(f"feature ranking output must be a fresh timestamped JSON: {ranking}")
ranking_stamp_raw = ranking_match.group("stamp")
ranking_stamp_format = (
    "%Y%m%dT%H%M%SZ" if len(ranking_stamp_raw) == 16 else "%Y%m%dT%H%M%S%fZ"
)
ranking_stamp = pd.Timestamp(
    pd.to_datetime(ranking_stamp_raw, format=ranking_stamp_format, utc=True, errors="raise")
)
validation_now = pd.Timestamp.now(tz="UTC")
if ranking_stamp > validation_now:
    raise RuntimeError(
        "feature ranking timestamp cannot be in the future: "
        f"ranking={ranking_stamp.isoformat()} now={validation_now.isoformat()}"
    )
if preflight.exists() or preflight.is_symlink():
    raise RuntimeError(f"preflight output directory must be fresh: {preflight}")
if not canonical.is_file() or canonical.is_symlink():
    raise RuntimeError(f"canonical-v2 parquet is missing/non-regular: {canonical}")
if not base28.is_file() or base28.is_symlink():
    raise RuntimeError(f"pair base28 parquet is missing/non-regular: {base28}")
if not native_m1.is_dir() or native_m1.is_symlink():
    raise RuntimeError(f"native M1 root is missing/non-regular: {native_m1}")
if not native_m5.is_dir() or native_m5.is_symlink():
    raise RuntimeError(f"native M5 root is missing/non-regular: {native_m5}")
if not tape.is_dir() or tape.is_symlink():
    raise RuntimeError(f"tape root is missing/non-regular: {tape}")
if tape != native_m5:
    raise RuntimeError("dataset tape must be the pair-bound native M5 root")
if (
    not m1_lifecycle_pair_manifest.is_file()
    or m1_lifecycle_pair_manifest.is_symlink()
):
    raise RuntimeError(
        "M1 lifecycle pair manifest is missing/non-regular: "
        f"{m1_lifecycle_pair_manifest}"
    )
if (
    not m1_lifecycle_pair_generation_root.is_dir()
    or m1_lifecycle_pair_generation_root.is_symlink()
):
    raise RuntimeError(
        "M1 lifecycle pair generation root is missing/non-regular: "
        f"{m1_lifecycle_pair_generation_root}"
    )
if exit_lifecycle.exists() or exit_lifecycle.is_symlink():
    raise RuntimeError(
        f"Exit lifecycle output directory must be fresh: {exit_lifecycle}"
    )

labels = (
    "history_start",
    "train_start",
    "train_end",
    "val_start",
    "val_end",
    "test_start",
    "test_end",
)
raw_times = (
    raw_history_start,
    raw_train_start,
    raw_train_end,
    raw_val_start,
    raw_val_end,
    raw_test_start,
    raw_test_end,
)
times = []
for label, raw in zip(labels, raw_times):
    try:
        parsed = pd.to_datetime(raw, utc=True, errors="raise")
    except Exception as exc:
        raise RuntimeError(f"invalid UTC split timestamp: {label}={raw!r}") from exc
    if pd.isna(parsed):
        raise RuntimeError(f"invalid UTC split timestamp: {label}={raw!r}")
    times.append(pd.Timestamp(parsed))
if not all(left < right for left, right in zip(times, times[1:])):
    raise RuntimeError(
        "split timestamps must be strictly ordered: "
        + ", ".join(f"{label}={value.isoformat()}" for label, value in zip(labels, times))
    )

output_dir = output.parent
output_stem = output.stem
fresh_paths = [
    ranking,
    event / "_ranker_checkpoint.npz",
    event / "_ranker_group_a_checkpoint",
    rank_npz,
    Path(f"{rank_npz}.json"),
    source,
    Path(f"{source}.manifest.json"),
    mtf,
    source_cascade,
    dataset_rebuild_terminal,
    prefreeze_test_seal,
    m1_enriched,
    Path(f"{m1_enriched}.manifest.json"),
    m5_enriched,
    Path(f"{m5_enriched}.manifest.json"),
    m1_feature_base,
    Path(f"{m1_feature_base}.manifest.json"),
    m5_feature_base,
    Path(f"{m5_feature_base}.manifest.json"),
    m1_checkpoint,
    m5_checkpoint,
    output,
    output_dir / "DATASET_BUILD_PROOF.json",
    *(output_dir / f"{output_stem}_{split}.parquet" for split in ("train", "val", "test")),
    *(output_dir / f"{output_stem}_{split}.manifest.json" for split in ("train", "val", "test")),
]
existing = [str(path) for path in fresh_paths if path.exists() or path.is_symlink()]
if audit.exists() or audit.is_symlink():
    existing.append(str(audit))
if existing:
    raise RuntimeError(f"fresh event outputs required; existing={existing}")
PYEOF
then
  fail "explicit immutable path/freshness contract failed"
fi

hash_file() {
  "$PY" - "$1" <<'PYEOF'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PYEOF
}

require_unchanged() {
  local label=$1
  local path=$2
  local expected=$3
  local observed
  observed=$(hash_file "$path")
  [[ $observed == "$expected" ]] || fail "$label changed after identity binding"
}

require_pair_unchanged() {
  require_unchanged "pair manifest" "$M1_LIFECYCLE_PAIR_MANIFEST" \
    "$PAIR_MANIFEST_SHA256"
}

# Build Entry locally on native M5.  The producer itself rejects workers != 1
# and constructs M15/H1/H4/D1 from closed OHLCV before feature ownership.
CURRENT_STEP=m5-enriched-feature-lane
write_status "$CURRENT_STEP" RUNNING
require_source_identity
require_pair_unchanged
if ! (cd "$ENG" && bash scripts/entry_next_edge_control.sh \
  model-native-m5-enriched-frame \
  --native-m5-root "$NATIVE_M5_ROOT" \
  --pair-manifest "$M1_LIFECYCLE_PAIR_MANIFEST" \
  --multi-tf-cache-dir "$MTF" \
  --output-parquet "$M5_ENRICHED" \
  --manifest-path "${M5_ENRICHED}.manifest.json" \
  --checkpoint-dir "$M5_CHECKPOINT" \
  --dataset-run-id "$RUN_ID" \
  --pair-generation-id "$PAIR_GENERATION_ID" \
  --registry-fit-train-start "$TRAIN_START" \
  --registry-fit-train-end "$REGISTRY_FIT_TRAIN_END" \
  --registry-fit-inner-end "$REGISTRY_FIT_INNER_END" \
  --registry-fit-tape-manifest "$TAPE_MANIFEST" \
  --expected-registry-fit-tape-manifest-sha256 "$TAPE_MANIFEST_SHA256" \
  --volatility-squeeze-manifest "$VOLATILITY_SQUEEZE_MANIFEST" \
  --expected-volatility-squeeze-manifest-sha256 "$VOLATILITY_SQUEEZE_MANIFEST_SHA256" \
  --workers 1 --checkpoint-chunk-rows 4096) >>"$LOG" 2>&1; then
  fail "native M5 enriched feature lane failed"
fi
[[ -f $M5_ENRICHED && ! -L $M5_ENRICHED \
   && -f ${M5_ENRICHED}.manifest.json && ! -L ${M5_ENRICHED}.manifest.json \
   && -d $MTF && ! -L $MTF && -f $MTF/manifest.json ]] \
  || fail "native M5 enriched output or its one V4 cache is missing/non-regular"

CURRENT_STEP=m5-model-source
write_status "$CURRENT_STEP" RUNNING
require_source_identity
require_pair_unchanged
if ! (cd "$ENG" && bash scripts/entry_next_edge_control.sh \
  model-native-m5-source-frame \
  --enriched-parquet "$M5_ENRICHED" \
  --multi-tf-cache-dir "$MTF" \
  --native-m5-root "$NATIVE_M5_ROOT" \
  --pair-manifest "$M1_LIFECYCLE_PAIR_MANIFEST" \
  --output-parquet "$SRC" \
  --dataset-run-id "$RUN_ID" \
  --pair-generation-id "$PAIR_GENERATION_ID") >>"$LOG" 2>&1; then
  fail "M5 model-source materialization failed"
fi
[[ -f $SRC && ! -L $SRC && -f ${SRC}.manifest.json && ! -L ${SRC}.manifest.json ]] \
  || fail "M5 model source/manifest is missing or non-regular"
SOURCE_SHA256=$(hash_file "$SRC")

# Prove final source coverage and exact canonical↔model market identity before
# spending hours on ranking or the M1 lane.
CURRENT_STEP=model-source-identity
write_status "$CURRENT_STEP" RUNNING
if ! (cd "$ENG" && bash scripts/gx1_capped_run.sh --class audit --mem 4G --swap 512M -- \
  "$PY" - "$CV2" "$SRC" "$HISTORY_START" "$TRAIN_END" \
  "$TRAIN_START" "$TEST_END") >>"$LOG" 2>&1 <<'PYEOF'; then
import sys
from pathlib import Path

import numpy as np
import pandas as pd

canonical, source = map(Path, sys.argv[1:3])
history_start = pd.Timestamp(sys.argv[3])
train_end = pd.Timestamp(sys.argv[4])
train_start = pd.Timestamp(sys.argv[5])
test_end = pd.Timestamp(sys.argv[6])
times = pd.DatetimeIndex(
    pd.to_datetime(
        pd.read_parquet(source, columns=["time"])["time"],
        utc=True,
        errors="raise",
    )
)
if (
    len(times) == 0
    or times.hasnans
    or not times.is_unique
    or not times.is_monotonic_increasing
):
    raise RuntimeError("final M5 model source time geometry is invalid")
pre_train_rows = int(((times >= history_start) & (times < train_start)).sum())
if times[0] > history_start or times[-1] != test_end or pre_train_rows < 96:
    raise RuntimeError(
        "source does not cover the declared common history/test window: "
        f"first={times[0]} last={times[-1]} test_end={test_end} "
        f"pre_train_rows={pre_train_rows}"
    )
identity_columns = ["time", "high", "low", "close", "bid_close", "ask_close"]
canonical_frame = pd.read_parquet(canonical, columns=identity_columns)
source_frame = pd.read_parquet(source, columns=identity_columns)
for frame in (canonical_frame, source_frame):
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    frame.drop(
        frame.index[
            (frame["time"] < history_start) | (frame["time"] > train_end)
        ],
        inplace=True,
    )
    frame.reset_index(drop=True, inplace=True)
if not canonical_frame["time"].equals(source_frame["time"]):
    raise RuntimeError("canonical/model source timestamp identity mismatch")
if not np.array_equal(
    canonical_frame[identity_columns[1:]].to_numpy(dtype=np.float64),
    source_frame[identity_columns[1:]].to_numpy(dtype=np.float64),
):
    raise RuntimeError("canonical/model source market identity mismatch")
PYEOF
  fail "final M5 model-source identity failed"
fi

# Bind the current pair, final model source, and exact five-timeframe cache.
CURRENT_STEP=source-cascade
write_status "$CURRENT_STEP" RUNNING
require_source_identity
require_pair_unchanged
if ! (cd "$ENG" && bash scripts/gx1_capped_run.sh --class audit --mem 4G --swap 512M -- \
  "$PY" -m gx1.scripts.materialize_current_pair_source_cascade_proof_v1 \
  --run-id "$RUN_ID" \
  --source-parquet "$SRC" \
  --canonical-v2-parquet "$CV2" \
  --mtf-cache-dir "$MTF" \
  --pair-manifest "$M1_LIFECYCLE_PAIR_MANIFEST" \
  --required-history-start "$HISTORY_START" \
  --out "$SOURCE_CASCADE") >>"$LOG" 2>&1; then
  fail "current pair source-cascade proof failed"
fi
[[ -f $SOURCE_CASCADE && ! -L $SOURCE_CASCADE ]] \
  || fail "source cascade proof missing or non-regular"
SOURCE_CASCADE_SHA256=$(hash_file "$SOURCE_CASCADE")
printf '[chain] source-cascade=%s sha256=%s\n' \
  "$SOURCE_CASCADE" "$SOURCE_CASCADE_SHA256" >>"$LOG"

require_source_cascade_unchanged() {
  require_unchanged "source cascade proof" "$SOURCE_CASCADE" \
    "$SOURCE_CASCADE_SHA256"
}

# Ranking is part of this one chain, not a separately launched heavyweight
# producer. The capped runner's global lock makes overlapping V3/V4-style
# ranker/builder execution impossible. Its exact checkpoint namespace is bound
# to run/source/cache/window identity by the producer.
CURRENT_STEP=feature-ranking
write_status "$CURRENT_STEP" RUNNING
require_source_identity
require_source_cascade_unchanged
require_pair_unchanged
run_feature_ranker() {
  # Heavy producer, not an audit: it materializes the full candidate matrix over
  # every TRAIN row. The pre-wave chain ran it at 10G and the last GREEN run (V5)
  # verified memory.max=10737418240. Reclassifying it as an audit capped it at 4G
  # and the cgroup killed it.
  (cd "$ENG" && bash scripts/gx1_capped_run.sh --class producer --mem 10G --swap 512M -- \
    "$PY" -m gx1.scripts.materialize_entry_model_native_train_feature_ranker_v1 \
    --run-id "$RUN_ID" \
    --source-parquet "$SRC" \
    --canonical-v2-parquet "$CV2" \
    --mtf-cache-dir "$MTF" \
    --source-cascade-proof "$SOURCE_CASCADE" \
    --tape-root "$TAPE" \
    --expected-source-time-max "$TEST_END" \
    --history-start "$HISTORY_START" \
    --train-start "$TRAIN_START" \
    --train-end "$TRAIN_END" \
    --out "$RANKING")
}
if ! run_feature_ranker >>"$LOG" 2>&1; then
  if [[ -f $EVENT/_ranker_checkpoint.npz || -f $EVENT/_ranker_group_a_checkpoint/CHECKPOINT_MANIFEST.json ]]; then
    CURRENT_STEP=feature-ranking-exact-checkpoint-resume
    write_status "$CURRENT_STEP" RUNNING "first capped attempt failed; exact checkpoint retry" 0
    require_source_identity
    require_source_cascade_unchanged
    require_pair_unchanged
    if ! run_feature_ranker >>"$LOG" 2>&1; then
      fail "feature ranking exact-checkpoint retry failed"
    fi
  else
    fail "feature ranking failed before an exact resumable checkpoint existed"
  fi
fi
[[ -f $RANKING && ! -L $RANKING ]] || fail "feature ranking output missing/non-regular"
RANKING_SHA256=$(hash_file "$RANKING")
require_source_identity
require_source_cascade_unchanged
require_pair_unchanged
write_status "$CURRENT_STEP" RUNNING
printf '[chain] ranking=%s sha256=%s\n' "$RANKING" "$RANKING_SHA256" >>"$LOG"

# Allocate one exact event-local output at the producer boundary.  Preallocating
# its timestamp before the long ranker run is unsatisfiable: the manifest must
# be newer than ranking and no more than five minutes old.  This is allocation,
# never discovery; no glob, mtime, latest alias, or existing manifest is read.
CURRENT_STEP=signal-manifest
MANIFEST_STAMP=$("$PY" -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))')
MANIFEST="$EVENT/ENTRY_MODEL_NATIVE_SEQ513_SIGNAL_MANIFEST_${MANIFEST_STAMP}.json"
[[ ! -e $MANIFEST && ! -L $MANIFEST ]] || fail "fresh signal manifest allocation collided"
write_status "$CURRENT_STEP" RUNNING
require_source_identity
require_source_cascade_unchanged
require_pair_unchanged
if ! (cd "$ENG" && "$PY" -m gx1.scripts.materialize_entry_model_native_seq513_signal_manifest_v1 \
  --feature-ranking-json "$RANKING" \
  --out "$MANIFEST" \
  --run-id "$RUN_ID") >>"$LOG" 2>&1; then
  fail "signal manifest materialization failed"
fi
MANIFEST_SHA256=$(hash_file "$MANIFEST")
require_unchanged "feature ranking" "$RANKING" "$RANKING_SHA256"
require_source_cascade_unchanged
require_pair_unchanged
write_status "$CURRENT_STEP" RUNNING
printf '[chain] manifest=%s sha256=%s\n' "$MANIFEST" "$MANIFEST_SHA256" >>"$LOG"

# Build Exit independently on native M1.  It uses local M1 plus closed
# M5/M15/H1/H4/D1 OHLCV through the same owners and the same TRAIN-rank bytes.
CURRENT_STEP=m1-enriched-feature-lane
write_status "$CURRENT_STEP" RUNNING
require_source_identity
require_pair_unchanged
require_source_cascade_unchanged
if ! (cd "$ENG" && bash scripts/entry_next_edge_control.sh \
  model-native-m1-enriched-frame \
  --native-m1-root "$NATIVE_M1_ROOT" \
  --pair-manifest "$M1_LIFECYCLE_PAIR_MANIFEST" \
  --multi-tf-cache-dir "$MTF" \
  --output-parquet "$M1_ENRICHED" \
  --manifest-path "${M1_ENRICHED}.manifest.json" \
  --checkpoint-dir "$M1_CHECKPOINT" \
  --dataset-run-id "$RUN_ID" \
  --pair-generation-id "$PAIR_GENERATION_ID" \
  --registry-fit-train-start "$TRAIN_START" \
  --registry-fit-train-end "$REGISTRY_FIT_TRAIN_END" \
  --registry-fit-inner-end "$REGISTRY_FIT_INNER_END" \
  --registry-fit-tape-manifest "$TAPE_MANIFEST" \
  --expected-registry-fit-tape-manifest-sha256 "$TAPE_MANIFEST_SHA256" \
  --volatility-squeeze-manifest "$VOLATILITY_SQUEEZE_MANIFEST" \
  --expected-volatility-squeeze-manifest-sha256 "$VOLATILITY_SQUEEZE_MANIFEST_SHA256" \
  --workers 1 --checkpoint-chunk-rows 4096) >>"$LOG" 2>&1; then
  fail "native M1 enriched feature lane failed"
fi
[[ -f $M1_ENRICHED && ! -L $M1_ENRICHED \
   && -f ${M1_ENRICHED}.manifest.json && ! -L ${M1_ENRICHED}.manifest.json ]] \
  || fail "native M1 enriched feature output is missing/non-regular"

# Project the same ordered current-contract fields into each native clock. M1 is aligned
# to the pair-bound closed-M1 lifecycle rows; M5 retains its native timeline.
CURRENT_STEP=entry-exit-feature-surfaces
write_status "$CURRENT_STEP" RUNNING
require_source_identity
require_pair_unchanged
require_source_cascade_unchanged
if ! (cd "$ENG" && bash scripts/entry_next_edge_control.sh \
  model-native-m1-feature-base \
  --source-parquet "$M1_ENRICHED" \
  --alignment-parquet "$BASE28" \
  --seq-structure-manifest "$MANIFEST" \
  --output-parquet "$M1_FEATURE_BASE" \
  --dataset-run-id "$RUN_ID" \
  --pair-generation-id "$PAIR_GENERATION_ID" \
  --v29-registry-constants-json "${M1_ENRICHED}.manifest.json" \
  --volatility-squeeze-manifest "$VOLATILITY_SQUEEZE_MANIFEST" \
  --expected-volatility-squeeze-manifest-sha256 "$VOLATILITY_SQUEEZE_MANIFEST_SHA256") >>"$LOG" 2>&1; then
  fail "M1 Exit feature-surface materialization failed"
fi
if ! (cd "$ENG" && bash scripts/entry_next_edge_control.sh \
  model-native-m5-feature-base \
  --source-parquet "$M5_ENRICHED" \
  --seq-structure-manifest "$MANIFEST" \
  --output-parquet "$M5_FEATURE_BASE" \
  --dataset-run-id "$RUN_ID" \
  --pair-generation-id "$PAIR_GENERATION_ID" \
  --v29-registry-constants-json "$MTF/manifest.json" \
  --volatility-squeeze-manifest "$VOLATILITY_SQUEEZE_MANIFEST" \
  --expected-volatility-squeeze-manifest-sha256 "$VOLATILITY_SQUEEZE_MANIFEST_SHA256") >>"$LOG" 2>&1; then
  fail "M5 Entry feature-surface materialization failed"
fi
for path in "$M1_FEATURE_BASE" "$M5_FEATURE_BASE"; do
  [[ -f $path && ! -L $path && -f ${path}.manifest.json \
     && ! -L ${path}.manifest.json ]] \
    || fail "Entry/Exit feature surface or manifest is missing/non-regular: $path"
done

# Preflight publishes into one explicit fresh namespace. Its unique output is
# then self-reference checked and hash-bound before the rebuild boundary.
CURRENT_STEP=rebuild-preflight
write_status "$CURRENT_STEP" RUNNING
require_source_identity
require_source_cascade_unchanged
require_pair_unchanged
if ! (cd "$ENG" && bash scripts/entry_next_edge_control.sh model-native-rebuild-preflight \
  --run-id "$RUN_ID" \
  --feature-ranking-json "$RANKING" \
  --source-parquet "$SRC" --canonical-v2-parquet "$CV2" \
  --signal-manifest "$MANIFEST" \
  --mtf-cache-dir "$MTF" --tape-root "$TAPE" \
  --m1-lifecycle-pair-manifest-json "$M1_LIFECYCLE_PAIR_MANIFEST" \
  --m1-lifecycle-pair-generation-root "$M1_LIFECYCLE_PAIR_GENERATION_ROOT" \
  --m1-feature-base-parquet "$M1_FEATURE_BASE" \
  --m5-feature-base-parquet "$M5_FEATURE_BASE" \
  --exit-lifecycle-dir "$EXIT_LIFECYCLE" \
  --output "$OUTPUT" --audit-out-dir "$AUDIT" \
  --history-start "$HISTORY_START" \
  --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
  --val-start "$VAL_START" --val-end "$VAL_END" \
  --test-start "$TEST_START" --test-end "$TEST_END" \
  --out-dir "$PRE_OUT" --quiet) >>"$LOG" 2>&1; then
  fail "rebuild preflight failed"
fi

if ! PREFLIGHT_ID=$("$PY" - "$PRE_OUT" "$RUN_ID" <<'PYEOF'
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_id = sys.argv[2]
pattern = re.compile(
    r"ENTRY_MODEL_NATIVE_SEQ513_REBUILD_PREFLIGHT_\d{8}T\d{6}(?:\d{6})?Z\.json"
)
entries = list(root.iterdir()) if root.is_dir() and not root.is_symlink() else []
if len(entries) != 1:
    raise RuntimeError(f"preflight namespace must contain exactly one artifact: {entries}")
path = entries[0]
if not path.is_file() or path.is_symlink() or not pattern.fullmatch(path.name):
    raise RuntimeError(f"unexpected preflight artifact: {path}")
raw = path.read_bytes()
payload = json.loads(raw)
if payload.get("json_path") != str(path.resolve()):
    raise RuntimeError("preflight json_path is not an exact self-reference")
if payload.get("entry_run_id") != run_id:
    raise RuntimeError("preflight run_id does not match chain run_id")
if payload.get("decision") != "READY_FOR_MODEL_NATIVE_SEQ513_REBUILD":
    raise RuntimeError("preflight is not READY")
print(f"{path.resolve()}\t{hashlib.sha256(raw).hexdigest()}")
PYEOF
); then
  fail "preflight output identity validation failed"
fi
IFS=$'\t' read -r PREFLIGHT_JSON PREFLIGHT_SHA256 <<<"$PREFLIGHT_ID"
[[ -n $PREFLIGHT_JSON && -n $PREFLIGHT_SHA256 ]] || fail "preflight identity is empty"
require_unchanged "feature ranking" "$RANKING" "$RANKING_SHA256"
require_source_cascade_unchanged
require_pair_unchanged
require_unchanged "signal manifest" "$MANIFEST" "$MANIFEST_SHA256"
require_unchanged "preflight artifact" "$PREFLIGHT_JSON" "$PREFLIGHT_SHA256"
write_status "$CURRENT_STEP" RUNNING
printf '[chain] preflight=%s sha256=%s\n' "$PREFLIGHT_JSON" "$PREFLIGHT_SHA256" >>"$LOG"

# The event was proven fresh above, so this invocation always performs exactly
# one rebuild. File presence can never substitute for terminal validation.
CURRENT_STEP=dataset-rebuild
write_status "$CURRENT_STEP" RUNNING
tg "GX1 seq513: preflight GREEN; dataset rebuild startet. run_id=$RUN_ID"
require_source_identity
require_source_cascade_unchanged
require_pair_unchanged
require_unchanged "feature ranking" "$RANKING" "$RANKING_SHA256"
require_unchanged "signal manifest" "$MANIFEST" "$MANIFEST_SHA256"
require_unchanged "preflight artifact" "$PREFLIGHT_JSON" "$PREFLIGHT_SHA256"
run_dataset_rebuild() {
  local resume_mode=$1
  local -a resume_args=()
  if [[ $resume_mode == exact-checkpoint ]]; then
    resume_args+=(--resume-exact-checkpoints)
  fi
  (cd "$ENG" && bash scripts/rebuild_entry_model_native_seq513_dataset.sh \
    --run-id "$RUN_ID" \
    --feature-ranking-json "$RANKING" \
    --source-parquet "$SRC" --canonical-v2-parquet "$CV2" \
    --signal-manifest "$MANIFEST" \
    --mtf-cache-dir "$MTF" --tape-root "$TAPE" \
    --m1-lifecycle-pair-manifest-json "$M1_LIFECYCLE_PAIR_MANIFEST" \
    --m1-lifecycle-pair-generation-root "$M1_LIFECYCLE_PAIR_GENERATION_ROOT" \
    --m1-feature-base-parquet "$M1_FEATURE_BASE" \
    --m5-feature-base-parquet "$M5_FEATURE_BASE" \
    --exit-lifecycle-dir "$EXIT_LIFECYCLE" \
    --output "$OUTPUT" --audit-out-dir "$AUDIT" \
    --rebuild-terminal-json "$DATASET_REBUILD_TERMINAL" \
    --prefreeze-test-seal-json "$PREFREEZE_TEST_SEAL" \
    --history-start "$HISTORY_START" \
    --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
    --val-start "$VAL_START" --val-end "$VAL_END" \
    --test-start "$TEST_START" --test-end "$TEST_END" \
    "${resume_args[@]}")
}

if ! run_dataset_rebuild fresh >>"$LOG" 2>&1; then
  # Exact checkpoint reuse is safe only before any split/audit output has been
  # materialized.  A failure after output emission is either a genuine
  # fail-closed audit result or an interrupted non-atomic split write; neither
  # may be overwritten inside the same immutable lineage.
  DATASET_OUTPUT_STARTED=0
  if [[ -e $AUDIT || -L $AUDIT || -e $EXIT_LIFECYCLE || -L $EXIT_LIFECYCLE ]]; then
    DATASET_OUTPUT_STARTED=1
  fi
  for split in train val test; do
    if [[ -e "$EVENT/dataset/v10_seq513_dataset__DIR_TRAIN_FIT_${split}.parquet" \
       || -e "$EVENT/dataset/v10_seq513_dataset__DIR_TRAIN_FIT_${split}.manifest.json" ]]; then
      DATASET_OUTPUT_STARTED=1
    fi
  done
  if [[ $DATASET_OUTPUT_STARTED -eq 1 ]]; then
    fail "dataset rebuild or post-build audit failed after immutable output materialization; fresh lineage required"
  elif [[ -f $TRAIN_GROUP_A_CHECKPOINT_MANIFEST && -f $EVENT/dataset/DATASET_BUILD_PROOF.json ]]; then
    CURRENT_STEP=dataset-rebuild-exact-checkpoint-resume
    write_status "$CURRENT_STEP" RUNNING "first capped attempt failed; exact checkpoint retry" 0
    require_source_identity
    require_source_cascade_unchanged
    require_pair_unchanged
    require_unchanged "feature ranking" "$RANKING" "$RANKING_SHA256"
    require_unchanged "signal manifest" "$MANIFEST" "$MANIFEST_SHA256"
    require_unchanged "preflight artifact" "$PREFLIGHT_JSON" "$PREFLIGHT_SHA256"
    if ! run_dataset_rebuild exact-checkpoint >>"$LOG" 2>&1; then
      fail "dataset rebuild exact-checkpoint retry failed"
    fi
  else
    fail "dataset rebuild failed before an exact resumable checkpoint existed"
  fi
fi
if ! DATASET_AUTHORITY_BINDING=$("$PY" - \
  "$DATASET_REBUILD_TERMINAL" "$PREFREEZE_TEST_SEAL" "$RUN_ID" \
  "$EVENT/dataset" "$OUTPUT" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


terminal_path = Path(sys.argv[1])
seal_path = Path(sys.argv[2])
run_id = sys.argv[3]
dataset_dir = Path(sys.argv[4])
output_stem = Path(sys.argv[5]).stem
if terminal_path.is_symlink() or not terminal_path.is_file():
    raise RuntimeError("dataset rebuild terminal is missing/non-regular")
if seal_path.is_symlink() or not seal_path.is_file():
    raise RuntimeError("pre-freeze TEST seal is missing/non-regular")
terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
seal = json.loads(seal_path.read_text(encoding="utf-8"))
terminal_bound = dict(terminal)
terminal_content_sha = terminal_bound.pop("content_binding_sha256", None)
seal_bound = dict(seal)
seal_content_sha = seal_bound.pop("content_binding_sha256", None)
test_row = terminal.get("split_artifacts", {}).get("test", {})
if (
    terminal.get("schema_version")
    != "entry_model_native_seq513_dataset_rebuild_terminal_v1"
    or terminal.get("decision")
    != "COMPLETED_MODEL_NATIVE_SEQ513_DATASET_REBUILD"
    or terminal.get("entry_run_id") != run_id
    or terminal.get("dataset_dir") != str(dataset_dir)
    or terminal.get("terminal_event_path") != str(terminal_path)
    or terminal_content_sha != canonical_sha256(terminal_bound)
    or seal.get("schema_version")
    != "entry_model_native_seq513_untouched_test_seal_v2"
    or seal.get("decision") != "SEALED_UNTOUCHED_TEST"
    or seal.get("entry_run_id") != run_id
    or seal.get("dataset_dir") != str(dataset_dir)
    or seal.get("seal_path") != str(seal_path)
    or seal.get("rebuild_terminal", {}).get("path") != str(terminal_path)
    or seal.get("rebuild_terminal", {}).get("sha256") != sha256(terminal_path)
    or seal.get("pair_lineage") != terminal.get("pair_lineage")
    or seal.get("source_lineage") != terminal.get("source_lineage")
    or seal.get("manifest", {}).get("path")
    != str(dataset_dir / f"{output_stem}_test.manifest.json")
    or seal.get("parquet", {}).get("path")
    != str(dataset_dir / f"{output_stem}_test.parquet")
    or seal.get("manifest", {}).get("sha256")
    != test_row.get("manifest_sha256")
    or seal.get("parquet", {}).get("sha256")
    != test_row.get("parquet_sha256")
    or seal_content_sha != canonical_sha256(seal_bound)
):
    raise RuntimeError("dataset rebuild terminal/TEST seal binding mismatch")
print(f"{sha256(terminal_path)}\t{sha256(seal_path)}")
PYEOF
); then
  fail "dataset rebuild did not publish one exact pre-freeze TEST authority"
fi
IFS=$'\t' read -r DATASET_REBUILD_TERMINAL_SHA256 \
  PREFREEZE_TEST_SEAL_SHA256 <<<"$DATASET_AUTHORITY_BINDING"
[[ -n $DATASET_REBUILD_TERMINAL_SHA256 && -n $PREFREEZE_TEST_SEAL_SHA256 ]] \
  || fail "dataset rebuild TEST authority hashes are empty"
require_source_identity
require_source_cascade_unchanged
require_pair_unchanged
require_unchanged "feature ranking" "$RANKING" "$RANKING_SHA256"
require_unchanged "signal manifest" "$MANIFEST" "$MANIFEST_SHA256"
require_unchanged "preflight artifact" "$PREFLIGHT_JSON" "$PREFLIGHT_SHA256"
require_unchanged "dataset rebuild terminal" "$DATASET_REBUILD_TERMINAL" \
  "$DATASET_REBUILD_TERMINAL_SHA256"
require_unchanged "pre-freeze TEST seal" "$PREFREEZE_TEST_SEAL" \
  "$PREFREEZE_TEST_SEAL_SHA256"

CURRENT_STEP=chain-complete
write_status "$CURRENT_STEP" GREEN "stopped before post-rebuild audits/readiness" 0
TERMINAL_WRITTEN=1
tg "GX1 seq513-kjeden er GREEN gjennom kombinert Entry/lifecycle-rebuild. Neste steg er post-rebuild audit/readiness."
printf '[chain] GREEN — combined dataset complete; next boundary is post-rebuild audits/readiness\n'
