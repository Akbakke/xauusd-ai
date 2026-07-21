#!/usr/bin/env bash
# One-shot fail-closed chain driver for one fresh seq513 dataset event:
#   fresh immutable TRAIN rank reference -> fresh explicit TRAIN ranking
#   -> fresh explicit signal manifest -> fresh preflight -> fresh dataset rebuild
#
# The driver never discovers artifacts, waits for an external producer, resumes
# from inferred debris, or treats existing split manifests as completion. It
# owns the ranker so a second heavyweight path cannot overlap the dataset job,
# and stops before the smoke gate. Every producer receives the same explicit
# run_id, ranking identity, split window, and event-local immutable paths.
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

usage() {
  printf '%s\n' \
    "Usage: $0 --run-id ID --event-root /absolute/event/root" \
    "  --feature-ranking-json /absolute/event/root/ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_<UTC>.json" \
    "  --preflight-out-dir /absolute/event/root/fresh-preflight-dir" \
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
  VAL_START VAL_END TEST_START TEST_END; do
  [[ -n ${!name} ]] || die_args "required argument missing: $name"
done
[[ -x $PY ]] || die_args "repository Python is not executable: $PY"
if [[ ! $RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$ ]]; then
  die_args "--run-id has invalid format"
fi
[[ $EVENT == /* ]] || die_args "--event-root must be absolute"
[[ -d $EVENT && ! -L $EVENT ]] || die_args "--event-root must be an existing regular directory"

SRC="$EVENT/FULL_PLUS_CTX_v3src.parquet"
CV2="$EVENT/canonical_features_v2.parquet"
MTF="$EVENT/MULTI_TF_V2_CACHE"
TAPE="$EVENT/m5_tape_repaired_dec2024"
RANK_NPZ="$EVENT/model_native_train_rank_reference_v4.npz"
OUTPUT="$EVENT/dataset/v10_seq513_dataset__HOLD_03B.parquet"
AUDIT="$EVENT/audit"
TRAIN_GROUP_A_CHECKPOINT_MANIFEST="$EVENT/dataset/_v10_seq513_dataset__HOLD_03B_train_GROUP_A_CHECKPOINT/CHECKPOINT_MANIFEST.json"
STAMP=$("$PY" -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))')
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
RANK_REFERENCE_SHA256=
RANK_REFERENCE_SIDECAR_SHA256=
MANIFEST_SHA256=
PREFLIGHT_JSON=
PREFLIGHT_SHA256=

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
    "$STARTED_UTC" "$BOOT_ID" "$CHAIN_PID" "$RANK_NPZ" \
    "$RANK_REFERENCE_SHA256" "$RANK_REFERENCE_SIDECAR_SHA256" <<'PYEOF'
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
    rank_reference_path,
    rank_reference_sha256,
    rank_reference_sidecar_sha256,
) = sys.argv[1:]
path = Path(raw_path)
now = datetime.now(timezone.utc)
terminal_states = {"GREEN", "RED", "ABORTED"}
terminal_path = None
if state in terminal_states:
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    terminal_path = path.with_name(f"CHAIN_TERMINAL_{stamp}_{state}.json")
payload = {
    "schema_version": "seq513_rebuild_chain_status_v4",
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
    "rank_reference": {
        "path": rank_reference_path,
        "sha256": rank_reference_sha256 or None,
        "sidecar_path": f"{rank_reference_path}.json",
        "sidecar_sha256": rank_reference_sidecar_sha256 or None,
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

# Validate all operator-supplied identities before any producer starts. The
# ranking and fresh preflight namespace must be exact
# canonical absolute paths below this event root; symlink indirection, mutable
# latest names, pre-existing outputs, and implicit resume debris all fail.
CURRENT_STEP=contract-validation
write_status "$CURRENT_STEP" RUNNING
if ! "$PY" - \
  "$EVENT" "$RANKING" "$PRE_OUT" "$RANK_NPZ" "$OUTPUT" "$AUDIT" \
  "$SRC" "$CV2" "$MTF" "$TAPE" \
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
    raw_rank_npz,
    raw_output,
    raw_audit,
    raw_source,
    raw_canonical,
    raw_mtf,
    raw_tape,
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
rank_npz = exact_path(raw_rank_npz, label="rank reference")
output = exact_path(raw_output, label="dataset output")
audit = exact_path(raw_audit, label="audit output directory")
source = exact_path(raw_source, label="source parquet")
canonical = exact_path(raw_canonical, label="canonical-v2 parquet")
mtf = exact_path(raw_mtf, label="MTF cache")
tape = exact_path(raw_tape, label="tape root")

for label, path in (
    ("feature ranking", ranking),
    ("preflight output directory", preflight),
    ("rank reference", rank_npz),
    ("dataset output", output),
    ("audit output directory", audit),
    ("source parquet", source),
    ("canonical-v2 parquet", canonical),
    ("MTF cache", mtf),
    ("tape root", tape),
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
if not source.is_file() or source.is_symlink():
    raise RuntimeError(f"source parquet is missing/non-regular: {source}")
if not canonical.is_file() or canonical.is_symlink():
    raise RuntimeError(f"canonical-v2 parquet is missing/non-regular: {canonical}")
if not mtf.is_dir() or mtf.is_symlink():
    raise RuntimeError(f"MTF cache is missing/non-regular: {mtf}")
if not tape.is_dir() or tape.is_symlink():
    raise RuntimeError(f"tape root is missing/non-regular: {tape}")

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

# Reject a context-trimmed source before spending time on TRAIN ranking.  The
# final finite surface must already cover the declared common-history boundary;
# pre-TRAIN row count alone is insufficient when its first timestamp is later
# than --history-start.
source_times = pd.to_datetime(
    pd.read_parquet(source, columns=["time"])["time"], utc=True, errors="coerce"
)
if (
    source_times.empty
    or source_times.isna().any()
    or source_times.duplicated().any()
    or not source_times.is_monotonic_increasing
):
    raise RuntimeError("source time column must be nonempty, finite, unique, and ordered")
source_first = pd.Timestamp(source_times.iloc[0])
source_last = pd.Timestamp(source_times.iloc[-1])
history_start, train_start, _, _, _, _, test_end = times
pre_train_rows = int(((source_times >= history_start) & (source_times < train_start)).sum())
if source_first > history_start or source_last < test_end or pre_train_rows < 96:
    raise RuntimeError(
        "source does not cover the declared common history/test window: "
        f"first={source_first.isoformat()} history_start={history_start.isoformat()} "
        f"last={source_last.isoformat()} test_end={test_end.isoformat()} "
        f"pre_train_rows={pre_train_rows}"
    )

output_dir = output.parent
output_stem = output.stem
fresh_paths = [
    ranking,
    event / "_ranker_checkpoint.npz",
    event / "_ranker_group_a_checkpoint",
    rank_npz,
    Path(f"{rank_npz}.json"),
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

require_rank_reference_unchanged() {
  require_unchanged "rank reference" "$RANK_NPZ" "$RANK_REFERENCE_SHA256"
  require_unchanged "rank reference sidecar" "${RANK_NPZ}.json" \
    "$RANK_REFERENCE_SIDECAR_SHA256"
}

# The frozen TRAIN-only ECDF/ATR state is upstream of feature computation.
# Ranking, manifest, preflight, dataset, and live must therefore bind these
# exact bytes rather than letting the dataset wrapper create a later artifact.
CURRENT_STEP=train-rank-reference
write_status "$CURRENT_STEP" RUNNING
require_source_identity
if ! (cd "$ENG" && bash scripts/gx1_capped_run.sh --mem 30G --swap 2G -- \
  "$PY" -m gx1.scripts.materialize_model_native_train_rank_reference_v2 \
  --run-id "$RUN_ID" \
  --source-parquet "$SRC" \
  --out "$RANK_NPZ" \
  --history-start "$HISTORY_START" \
  --fit-start "$TRAIN_START" \
  --fit-end "$TRAIN_END") >>"$LOG" 2>&1; then
  fail "TRAIN rank-reference materialization failed"
fi
[[ -f $RANK_NPZ && ! -L $RANK_NPZ && -f ${RANK_NPZ}.json && ! -L ${RANK_NPZ}.json ]] \
  || fail "TRAIN rank-reference output/sidecar missing or non-regular"
RANK_REFERENCE_SHA256=$(hash_file "$RANK_NPZ")
RANK_REFERENCE_SIDECAR_SHA256=$(hash_file "${RANK_NPZ}.json")
require_source_identity
write_status "$CURRENT_STEP" RUNNING
printf '[chain] rank-reference=%s sha256=%s sidecar_sha256=%s\n' \
  "$RANK_NPZ" "$RANK_REFERENCE_SHA256" "$RANK_REFERENCE_SIDECAR_SHA256" >>"$LOG"

# Ranking is part of this one chain, not a separately launched heavyweight
# producer. The capped runner's global lock makes overlapping V3/V4-style
# ranker/builder execution impossible. Its exact checkpoint namespace is bound
# to run/source/cache/window identity by the producer.
CURRENT_STEP=feature-ranking
write_status "$CURRENT_STEP" RUNNING
require_source_identity
run_feature_ranker() {
  (cd "$ENG" && bash scripts/gx1_capped_run.sh --mem 30G --swap 2G -- \
    "$PY" -m gx1.scripts.materialize_entry_model_native_train_feature_ranker_v1 \
    --run-id "$RUN_ID" \
    --source-parquet "$SRC" \
    --mtf-cache-dir "$MTF" \
    --rank-reference-npz "$RANK_NPZ" \
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
require_rank_reference_unchanged
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
require_rank_reference_unchanged
if ! (cd "$ENG" && "$PY" -m gx1.scripts.materialize_entry_model_native_seq513_signal_manifest_v1 \
  --feature-ranking-json "$RANKING" \
  --out "$MANIFEST" \
  --run-id "$RUN_ID") >>"$LOG" 2>&1; then
  fail "signal manifest materialization failed"
fi
MANIFEST_SHA256=$(hash_file "$MANIFEST")
require_unchanged "feature ranking" "$RANKING" "$RANKING_SHA256"
write_status "$CURRENT_STEP" RUNNING
printf '[chain] manifest=%s sha256=%s\n' "$MANIFEST" "$MANIFEST_SHA256" >>"$LOG"

# Preflight publishes into one explicit fresh namespace. Its unique output is
# then self-reference checked and hash-bound before the rebuild boundary.
CURRENT_STEP=rebuild-preflight
write_status "$CURRENT_STEP" RUNNING
require_source_identity
require_rank_reference_unchanged
if ! (cd "$ENG" && bash scripts/entry_next_edge_control.sh model-native-rebuild-preflight \
  --run-id "$RUN_ID" \
  --feature-ranking-json "$RANKING" \
  --source-parquet "$SRC" --canonical-v2-parquet "$CV2" \
  --signal-manifest "$MANIFEST" --rank-reference-npz "$RANK_NPZ" \
  --mtf-cache-dir "$MTF" --tape-root "$TAPE" \
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
require_rank_reference_unchanged
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
require_rank_reference_unchanged
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
    --existing-rank-reference \
    --run-id "$RUN_ID" \
    --feature-ranking-json "$RANKING" \
    --source-parquet "$SRC" --canonical-v2-parquet "$CV2" \
    --signal-manifest "$MANIFEST" --rank-reference-npz "$RANK_NPZ" \
    --mtf-cache-dir "$MTF" --tape-root "$TAPE" \
    --output "$OUTPUT" --audit-out-dir "$AUDIT" \
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
  if [[ -e $AUDIT || -L $AUDIT ]]; then
    DATASET_OUTPUT_STARTED=1
  fi
  for split in train val test; do
    if [[ -e "$EVENT/dataset/v10_seq513_dataset__HOLD_03B_${split}.parquet" \
       || -e "$EVENT/dataset/v10_seq513_dataset__HOLD_03B_${split}.manifest.json" ]]; then
      DATASET_OUTPUT_STARTED=1
    fi
  done
  if [[ $DATASET_OUTPUT_STARTED -eq 1 ]]; then
    fail "dataset rebuild or post-build audit failed after immutable output materialization; fresh lineage required"
  elif [[ -f $TRAIN_GROUP_A_CHECKPOINT_MANIFEST && -f $RANK_NPZ && -f ${RANK_NPZ}.json && -f $EVENT/dataset/DATASET_BUILD_PROOF.json ]]; then
    CURRENT_STEP=dataset-rebuild-exact-checkpoint-resume
    write_status "$CURRENT_STEP" RUNNING "first capped attempt failed; exact checkpoint retry" 0
    require_source_identity
    require_rank_reference_unchanged
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
require_source_identity
require_rank_reference_unchanged
require_unchanged "feature ranking" "$RANKING" "$RANKING_SHA256"
require_unchanged "signal manifest" "$MANIFEST" "$MANIFEST_SHA256"
require_unchanged "preflight artifact" "$PREFLIGHT_JSON" "$PREFLIGHT_SHA256"

CURRENT_STEP=chain-complete
write_status "$CURRENT_STEP" GREEN "stopped at smoke gate" 0
TERMINAL_WRITTEN=1
tg "GX1 seq513-kjeden er GREEN gjennom dataset-rebuild. Neste steg er manuell smoke-gate."
printf '[chain] GREEN — stopped at the smoke gate as designed\n'
