#!/usr/bin/env bash
# One-shot fail-closed chain driver for one fresh seq513 dataset event:
#   explicit TRAIN ranking -> fresh explicit signal manifest -> fresh preflight
#   -> fresh dataset rebuild
#
# The driver never discovers artifacts, waits for an external producer, resumes
# from inferred debris, or treats existing split manifests as completion.  It
# stops before the smoke gate.  Every producer receives the same explicit
# vedtak, ranking identity, split window, and event-local immutable paths.
set -Eeuo pipefail

ENG=/home/andre2/src/GX1_ENGINE
PY=$ENG/.venv/bin/python

VEDTAK=
EVENT=
RANKING=
MANIFEST=
PRE_OUT=

usage() {
  printf '%s\n' \
    "Usage: $0 --vedtak ID --event-root /absolute/event/root" \
    "  --feature-ranking-json /absolute/event/root/ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_<UTC>.json" \
    "  --signal-manifest /absolute/event/root/ENTRY_MODEL_NATIVE_SEQ513_SIGNAL_MANIFEST_<UTC>.json" \
    "  --preflight-out-dir /absolute/event/root/fresh-preflight-dir" \
    "The ranking must already exist. The signal manifest path and preflight directory must be fresh."
}

die_args() {
  printf '[ABORT] %s\n' "$*" >&2
  usage >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --vedtak)
      (($# >= 2)) || die_args "--vedtak requires a value"
      [[ -z $VEDTAK ]] || die_args "duplicate --vedtak"
      VEDTAK=$2
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
    --signal-manifest)
      (($# >= 2)) || die_args "--signal-manifest requires a value"
      [[ -z $MANIFEST ]] || die_args "duplicate --signal-manifest"
      MANIFEST=$2
      shift 2
      ;;
    --preflight-out-dir)
      (($# >= 2)) || die_args "--preflight-out-dir requires a value"
      [[ -z $PRE_OUT ]] || die_args "duplicate --preflight-out-dir"
      PRE_OUT=$2
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

for name in VEDTAK EVENT RANKING MANIFEST PRE_OUT; do
  [[ -n ${!name} ]] || die_args "required argument missing: $name"
done
[[ -x $PY ]] || die_args "repository Python is not executable: $PY"
if [[ ! $VEDTAK =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$ ]]; then
  die_args "--vedtak has invalid format"
fi
[[ $EVENT == /* ]] || die_args "--event-root must be absolute"
[[ -d $EVENT && ! -L $EVENT ]] || die_args "--event-root must be an existing regular directory"

SRC="$EVENT/FULL_PLUS_CTX_v3src.parquet"
CV2="$EVENT/canonical_features_v2.parquet"
MTF="$EVENT/MULTI_TF_V2_CACHE"
TAPE="$EVENT/m5_tape_repaired_dec2024"
RANK_NPZ="$EVENT/model_native_train_rank_reference_v3.npz"
OUTPUT="$EVENT/dataset/v10_seq513_dataset__HOLD_03B.parquet"
AUDIT="$EVENT/audit"
# history < train strictly. Source (FULL_PLUS) first row is 2021-01-04T23:55.
HISTORY_START=2021-01-05T00:00:00Z
TRAIN_START=2021-03-16T00:00:00Z
TRAIN_END=2026-03-31T23:59:59Z
VAL_START=2026-04-01T00:00:00Z
VAL_END=2026-04-30T23:59:59Z
TEST_START=2026-05-01T00:00:00Z
TEST_END=2026-06-14T23:55:00Z

STAMP=$("$PY" -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))')
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
    "$step" "$state" "$reason" "$exit_code" "$STATUS" "$VEDTAK" "$EVENT" \
    "$LOG" "$GIT_HEAD" "$RANKING" "$RANKING_SHA256" "$MANIFEST" \
    "$MANIFEST_SHA256" "$PRE_OUT" "$PREFLIGHT_JSON" "$PREFLIGHT_SHA256" <<'PYEOF'
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
    vedtak,
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
) = sys.argv[1:]
path = Path(raw_path)
payload = {
    "schema_version": "seq513_rebuild_chain_status_v2",
    "explicit_vedtak_id": vedtak,
    "event_root": event_root,
    "step": step,
    "state": state,
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "log_path": log_path,
    "git_head": git_head or None,
    "feature_ranking": {
        "path": ranking_path,
        "sha256": ranking_sha256 or None,
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
  tg "GX1 seq513-kjede STOPPET RED: steg=$CURRENT_STEP vedtak=$VEDTAK logg=$LOG"
  printf '[chain] RED at %s: %s — see %s\n' "$CURRENT_STEP" "$reason" "$LOG" >&2
  exit "$exit_code"
}

on_err() {
  local exit_code=$1
  local line=$2
  trap - ERR
  terminal_status RED "unexpected ERR at line $line" "$exit_code"
  tg "GX1 seq513-kjede STOPPET RED: steg=$CURRENT_STEP vedtak=$VEDTAK logg=$LOG"
  exit "$exit_code"
}

on_signal() {
  local signal_name=$1
  local exit_code=$2
  trap - ERR TERM INT HUP
  terminal_status ABORTED "received $signal_name" "$exit_code"
  tg "GX1 seq513-kjede AVBRUTT: signal=$signal_name steg=$CURRENT_STEP vedtak=$VEDTAK logg=$LOG"
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
printf '[chain] vedtak=%s event=%s log=%s\n' "$VEDTAK" "$EVENT" "$LOG"

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
# ranking, fresh manifest target, and fresh preflight namespace must be exact
# canonical absolute paths below this event root; symlink indirection, mutable
# latest names, pre-existing outputs, and implicit resume debris all fail.
CURRENT_STEP=contract-validation
write_status "$CURRENT_STEP" RUNNING
if ! "$PY" - \
  "$EVENT" "$RANKING" "$MANIFEST" "$PRE_OUT" "$RANK_NPZ" "$OUTPUT" "$AUDIT" \
  "$SRC" "$CV2" "$MTF" "$TAPE" >>"$LOG" 2>&1 <<'PYEOF'
import re
import sys
from pathlib import Path

(
    raw_event,
    raw_ranking,
    raw_manifest,
    raw_preflight,
    raw_rank_npz,
    raw_output,
    raw_audit,
    raw_source,
    raw_canonical,
    raw_mtf,
    raw_tape,
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
manifest = exact_path(raw_manifest, label="signal manifest")
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
    ("signal manifest", manifest),
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
    r"ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_\d{8}T\d{6}(?:\d{6})?Z\.json"
)
manifest_pattern = re.compile(
    r"ENTRY_MODEL_NATIVE_SEQ513_SIGNAL_MANIFEST_\d{8}T\d{6}(?:\d{6})?Z\.json"
)
if not ranking.is_file() or ranking.is_symlink() or not ranking_pattern.fullmatch(ranking.name):
    raise RuntimeError(f"feature ranking is not an immutable timestamped regular JSON: {ranking}")
if manifest.exists() or manifest.is_symlink() or not manifest_pattern.fullmatch(manifest.name):
    raise RuntimeError(f"signal manifest output must be a fresh timestamped JSON: {manifest}")
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

output_dir = output.parent
output_stem = output.stem
fresh_paths = [
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

RANKING_SHA256=$(hash_file "$RANKING")
write_status "$CURRENT_STEP" RUNNING
printf '[chain] ranking=%s sha256=%s\n' "$RANKING" "$RANKING_SHA256" >>"$LOG"

# Materialize exactly the caller-selected fresh signal-manifest output. Existing
# manifests are never reused, even if one happens to be present under EVENT.
CURRENT_STEP=signal-manifest
write_status "$CURRENT_STEP" RUNNING
require_source_identity
if ! (cd "$ENG" && "$PY" -m gx1.scripts.materialize_entry_model_native_seq513_signal_manifest_v1 \
  --feature-ranking-json "$RANKING" \
  --out "$MANIFEST" \
  --vedtak "$VEDTAK") >>"$LOG" 2>&1; then
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
if ! (cd "$ENG" && bash scripts/entry_next_edge_control.sh model-native-rebuild-preflight \
  --vedtak "$VEDTAK" \
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

if ! PREFLIGHT_ID=$("$PY" - "$PRE_OUT" "$VEDTAK" <<'PYEOF'
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
vedtak = sys.argv[2]
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
if payload.get("explicit_vedtak_id") != vedtak:
    raise RuntimeError("preflight vedtak does not match chain vedtak")
if payload.get("decision") != "READY_FOR_MODEL_NATIVE_SEQ513_REBUILD_VEDTAK_REVIEW":
    raise RuntimeError("preflight is not READY")
print(f"{path.resolve()}\t{hashlib.sha256(raw).hexdigest()}")
PYEOF
); then
  fail "preflight output identity validation failed"
fi
IFS=$'\t' read -r PREFLIGHT_JSON PREFLIGHT_SHA256 <<<"$PREFLIGHT_ID"
[[ -n $PREFLIGHT_JSON && -n $PREFLIGHT_SHA256 ]] || fail "preflight identity is empty"
require_unchanged "feature ranking" "$RANKING" "$RANKING_SHA256"
require_unchanged "signal manifest" "$MANIFEST" "$MANIFEST_SHA256"
require_unchanged "preflight artifact" "$PREFLIGHT_JSON" "$PREFLIGHT_SHA256"
write_status "$CURRENT_STEP" RUNNING
printf '[chain] preflight=%s sha256=%s\n' "$PREFLIGHT_JSON" "$PREFLIGHT_SHA256" >>"$LOG"

# The event was proven fresh above, so this invocation always performs exactly
# one rebuild. File presence can never substitute for terminal validation.
CURRENT_STEP=dataset-rebuild
write_status "$CURRENT_STEP" RUNNING
tg "GX1 seq513: preflight GREEN; dataset rebuild startet. vedtak=$VEDTAK"
require_source_identity
require_unchanged "feature ranking" "$RANKING" "$RANKING_SHA256"
require_unchanged "signal manifest" "$MANIFEST" "$MANIFEST_SHA256"
require_unchanged "preflight artifact" "$PREFLIGHT_JSON" "$PREFLIGHT_SHA256"
if ! (cd "$ENG" && bash scripts/rebuild_entry_model_native_seq513_dataset.sh \
  --vedtak "$VEDTAK" \
  --feature-ranking-json "$RANKING" \
  --source-parquet "$SRC" --canonical-v2-parquet "$CV2" \
  --signal-manifest "$MANIFEST" --rank-reference-npz "$RANK_NPZ" \
  --mtf-cache-dir "$MTF" --tape-root "$TAPE" \
  --output "$OUTPUT" --audit-out-dir "$AUDIT" \
  --history-start "$HISTORY_START" \
  --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
  --val-start "$VAL_START" --val-end "$VAL_END" \
  --test-start "$TEST_START" --test-end "$TEST_END") >>"$LOG" 2>&1; then
  fail "dataset rebuild failed"
fi
require_source_identity

CURRENT_STEP=chain-complete
write_status "$CURRENT_STEP" GREEN "stopped at smoke gate" 0
TERMINAL_WRITTEN=1
tg "GX1 seq513-kjeden er GREEN gjennom dataset-rebuild. Neste steg er manuell smoke-gate."
printf '[chain] GREEN — stopped at the smoke gate as designed\n'
