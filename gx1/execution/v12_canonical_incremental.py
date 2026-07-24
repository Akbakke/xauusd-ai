#!/usr/bin/env python3
"""V12 canonical/BASE28 pair updater with full-history feature equivalence.

Strategy
--------
For each new native, completed canonical M5 bar:
  1. Verify the one-owner canonical M5 manifest and every partition hash.
  2. Run the canonical feature owners on complete causal M5 history.
  3. Append only exact native M1 market identity to raw BASE28.
  4. Write both complete candidates into an unpublished staging directory.
  5. Publish one immutable pair generation and atomically replace the single
     canonical-v3/BASE28 pair pointer. No individual artifact is ever activated.

BASE28 is deliberately cheap: it is a raw M1 lane, not another derived feature
store.  Context, multi-timeframe and TRAIN-fit rank state have their own owners.

This correctness-first implementation is intentionally not claimed to be
cheap. A future recursive-state accelerator must prove bit-equivalence against
this complete-history owner before it can replace the computation.

This owner is snapshot-driven. It does not poll mutable collector directories
or run a daemon loop; a new immutable native M1+M5 snapshot and explicit
decision are required for every publication.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.contracts.xau_tape_provenance_v1 import (  # noqa: E402
    CANONICAL_NATIVE_REQUIRED_COLUMNS,
    canonical_xau_source_descriptor_v1,
)
from gx1.execution.v12_state_from_prebuilt import (  # noqa: E402
    PREBUILT_PAIR_MANIFEST_PATH,
    PREBUILT_PAIR_ROOT,
    PREBUILT_PAIR_SCHEMA_VERSION,
    inspect_prebuilt_artifact,
    pair_generation_id_for_artifacts,
    read_prebuilt_pair_manifest,
    verify_prebuilt_pair,
)
from gx1.execution.v12_m1_to_m5_downsample import (  # noqa: E402
    closed_m5_start_for_m1_bar_labels,
)
from gx1.features.basic_v1 import (  # noqa: E402
    PLUS5_FEATURES,
    compute_plus5_features,
)
from gx1.scripts.materialize_build_canonical_features_v2 import (  # noqa: E402
    build_canonical_v2,
)
from gx1.scripts.materialize_canonical_v3_augment import (  # noqa: E402
    DROP_COLUMNS,
    add_cyclic_time_features,
    add_smc_premium_state_interaction,
    add_cross_tf_momentum,
)

LOG = logging.getLogger("v12_incr")

PAIR_CANONICAL_FILENAME = "canonical_v3.parquet"
PAIR_BASE28_FILENAME = "base28.parquet"
PAIR_PUBLISH_LOCK_FILENAME = ".canonical_v3_base28_pair_publish.lock"
_PAIR_STAGING_NAME = re.compile(r"\.staging-[0-9a-f]{32}\Z")

# PLUS5: 5 features re-added on 2026-05-21 because the PLUS5 Entry-IQL ensemble
# was trained on real values.  This function is the retained computation source.
M1_MARKET_IDENTITY_COLUMNS = CANONICAL_NATIVE_REQUIRED_COLUMNS[1:]
# BASE28 is the M1-cadence lane. Every field is owned by native M1. Derived
# context and M5 broadcasts are forbidden because they created duplicate,
# precedence-dependent feature authorities in Entry and Exit.
RAW_BASE28_COLUMNS = M1_MARKET_IDENTITY_COLUMNS
PAIR_PRODUCER_OWNER = "gx1.execution.v12_canonical_incremental"
_PAIR_PRODUCER_SOURCE_PATHS = (
    "gx1/contracts/entry_model_native_state_v2.py",
    "gx1/contracts/xau_tape_provenance_v1.py",
    "gx1/execution/v12_canonical_incremental.py",
    "gx1/execution/v12_ctx_augment_live.py",
    "gx1/execution/v12_m1_to_m5_downsample.py",
    "gx1/features/basic_v1.py",
    "gx1/features/htf_features.py",
    "gx1/scripts/augment_forward_outcome_v2.py",
    "gx1/scripts/materialize_build_canonical_features_v1.py",
    "gx1/scripts/materialize_build_canonical_features_v2.py",
    "gx1/scripts/materialize_canonical_v3_augment.py",
    "gx1/time/session_detector.py",
)


def _build_raw_base28_owned_frame(m1: pd.DataFrame) -> pd.DataFrame:
    """Return exact native M1 identity or fail on any derived/stale schema."""

    if (
        not isinstance(m1, pd.DataFrame)
        or m1.empty
        or not isinstance(m1.index, pd.DatetimeIndex)
        or m1.index.hasnans
        or not m1.index.is_unique
        or not m1.index.is_monotonic_increasing
        or m1.index.tz is None
    ):
        raise RuntimeError("RAW_BASE28_M1_INDEX_INVALID")
    missing = [name for name in RAW_BASE28_COLUMNS if name not in m1.columns]
    if missing:
        raise RuntimeError(f"RAW_BASE28_M1_FIELDS_MISSING: {missing}")
    out = m1.loc[:, list(RAW_BASE28_COLUMNS)].copy()
    for name in RAW_BASE28_COLUMNS:
        if pd.api.types.is_bool_dtype(out[name].dtype):
            raise RuntimeError(f"RAW_BASE28_M1_FIELD_BOOL: {name}")
        try:
            values = pd.to_numeric(out[name], errors="raise").to_numpy(
                dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"RAW_BASE28_M1_FIELD_INVALID: {name}") from exc
        if not np.isfinite(values).all():
            raise RuntimeError(f"RAW_BASE28_M1_FIELD_NONFINITE: {name}")
    out.index = out.index.tz_convert("UTC")
    out.index.name = "time"
    return out


def _compute_plus5_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility call-site delegating to the basic_v1 PLUS5 owner."""
    return compute_plus5_features(df)


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _candidate_staging_path(generation_root: Path) -> Path:
    """Reserve a unique unpublished path without creating an empty cycle artifact."""
    generation_root = Path(generation_root)
    generation_root.mkdir(parents=True, exist_ok=True)
    if generation_root.is_symlink() or not generation_root.is_dir():
        raise RuntimeError(f"pair generation root is invalid: {generation_root}")
    generation_root = generation_root.resolve(strict=True)
    return generation_root / f".staging-{uuid.uuid4().hex}"


def _discard_pair_staging_dir(staging_dir: Path, *, generation_root: Path) -> None:
    """Delete only this process's exact unpublished staging directory."""
    generation_root = Path(generation_root).resolve(strict=True)
    staging_dir = Path(staging_dir)
    candidate = staging_dir.absolute()
    if (
        candidate.parent != generation_root
        or _PAIR_STAGING_NAME.fullmatch(candidate.name) is None
    ):
        raise RuntimeError(f"refusing unsafe pair staging cleanup: {staging_dir}")
    if not candidate.exists():
        if candidate.is_symlink():
            raise RuntimeError(f"refusing symlink pair staging cleanup: {candidate}")
        return
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(f"refusing non-directory pair staging cleanup: {candidate}")
    allowed = {PAIR_CANONICAL_FILENAME, PAIR_BASE28_FILENAME}
    entries = list(candidate.iterdir())
    if any(
        item.name not in allowed or item.is_symlink() or not item.is_file()
        for item in entries
    ):
        raise RuntimeError(
            f"refusing pair staging cleanup with unexpected contents: {candidate}"
        )
    shutil.rmtree(candidate)
    _fsync_directory(generation_root)


def _discard_unpublished_generation_dir(
    generation_dir: Path,
    *,
    generation_root: Path,
    pair_manifest_path: Path,
    pair_generation_id: str,
    lineage_contract: dict[str, object],
) -> None:
    """Delete only a generation created by this failed, pre-pointer publication."""
    generation_root = Path(generation_root).resolve(strict=True)
    generation_dir = Path(generation_dir).absolute()
    if (
        generation_dir.parent != generation_root
        or generation_dir.name != pair_generation_id
        or len(pair_generation_id) != 64
        or any(char not in "0123456789abcdef" for char in pair_generation_id)
    ):
        raise RuntimeError(
            f"refusing unsafe unpublished generation cleanup: {generation_dir}"
        )
    if generation_dir.is_symlink() or not generation_dir.is_dir():
        raise RuntimeError(
            f"refusing invalid unpublished generation cleanup: {generation_dir}"
        )
    if set(item.name for item in generation_dir.iterdir()) != {
        PAIR_CANONICAL_FILENAME,
        PAIR_BASE28_FILENAME,
    }:
        raise RuntimeError(
            f"refusing unpublished generation cleanup with unexpected contents: "
            f"{generation_dir}"
        )
    artifacts = {
        "canonical_v3": inspect_prebuilt_artifact(
            generation_dir / PAIR_CANONICAL_FILENAME,
            label="canonical_v3",
        ),
        "base28": inspect_prebuilt_artifact(
            generation_dir / PAIR_BASE28_FILENAME,
            label="base28",
        ),
    }
    if (
        pair_generation_id_for_artifacts(
            artifacts,
            lineage=lineage_contract,
        )
        != pair_generation_id
    ):
        raise RuntimeError(
            f"refusing unpublished generation cleanup after identity mismatch: "
            f"{generation_dir}"
        )
    pair_manifest_path = Path(pair_manifest_path)
    if pair_manifest_path.exists() or pair_manifest_path.is_symlink():
        current = read_prebuilt_pair_manifest(
            pair_manifest_path,
            generation_root=generation_root,
        )
        if current.pair_generation_id == pair_generation_id:
            raise RuntimeError(
                f"refusing cleanup of published pair generation: {generation_dir}"
            )
    shutil.rmtree(generation_dir)
    _fsync_directory(generation_root)


def _write_candidate_parquet(
    frame: pd.DataFrame,
    path: Path,
    *,
    index: bool,
) -> None:
    """Write one unpublished candidate; partial staging is never served."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite pair candidate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=index)
    _fsync_file(path)
    _fsync_directory(path.parent)


def _publish_prebuilt_pair_generation(
    staging_dir: Path,
    *,
    pair_manifest_path: Path,
    generation_root: Path,
    expected_pair_generation_id: str | None,
    expected_manifest_sha256: str | None,
    lineage_contract: dict[str, object],
    created_utc: str | None = None,
) -> str:
    """Publish a complete immutable pair through one atomic pointer replacement.

    Artifact computation is complete before this function begins. Under the
    publisher lock it verifies the previously admitted pointer, renames the
    complete staging directory to its content-derived generation id, fsyncs it,
    and only then replaces the one serving pointer. A failure before the pointer
    replacement leaves the previous generation active.
    """
    staging_dir = Path(staging_dir)
    pair_manifest_path = Path(pair_manifest_path)
    generation_root = Path(generation_root)
    if generation_root.is_symlink() or not generation_root.is_dir():
        raise RuntimeError(f"pair generation root is invalid: {generation_root}")
    generation_root = generation_root.resolve(strict=True)
    try:
        staging_resolved = staging_dir.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"pair staging directory is missing: {staging_dir}") from exc
    if (
        staging_dir.is_symlink()
        or not staging_resolved.is_dir()
        or staging_resolved.parent != generation_root
        or _PAIR_STAGING_NAME.fullmatch(staging_resolved.name) is None
    ):
        raise RuntimeError(f"pair staging directory is not exact: {staging_dir}")

    canonical_candidate = staging_resolved / PAIR_CANONICAL_FILENAME
    base28_candidate = staging_resolved / PAIR_BASE28_FILENAME
    if set(item.name for item in staging_resolved.iterdir()) != {
        PAIR_CANONICAL_FILENAME,
        PAIR_BASE28_FILENAME,
    }:
        raise RuntimeError("pair staging directory must contain exactly two artifacts")
    artifacts = {
        "canonical_v3": inspect_prebuilt_artifact(
            canonical_candidate,
            label="canonical_v3",
        ),
        "base28": inspect_prebuilt_artifact(base28_candidate, label="base28"),
    }
    pair_generation_id = pair_generation_id_for_artifacts(
        artifacts,
        lineage=lineage_contract,
    )
    final_dir = generation_root / pair_generation_id

    pair_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_parent = pair_manifest_path.parent.resolve(strict=True)
    if pair_manifest_path.is_symlink() or manifest_parent != pair_manifest_path.parent:
        raise RuntimeError(f"pair manifest path is not exact: {pair_manifest_path}")
    lock_path = manifest_parent / PAIR_PUBLISH_LOCK_FILENAME
    if lock_path.is_symlink():
        raise RuntimeError(f"pair publish lock path is not exact: {lock_path}")
    with lock_path.open("a+b") as lock_handle:
        if lock_path.is_symlink() or not lock_path.is_file():
            raise RuntimeError(f"pair publish lock path is not exact: {lock_path}")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if expected_pair_generation_id is None or expected_manifest_sha256 is None:
            if pair_manifest_path.exists() or pair_manifest_path.is_symlink():
                raise RuntimeError(
                    "pair bootstrap refused because an active pointer already exists"
                )
        else:
            current = read_prebuilt_pair_manifest(
                pair_manifest_path,
                generation_root=generation_root,
            )
            if (
                current.pair_generation_id != expected_pair_generation_id
                or current.manifest_sha256 != expected_manifest_sha256
            ):
                raise RuntimeError(
                    "active pair changed during candidate computation; "
                    "refusing stale publication"
                )

        created_final_dir = False
        if final_dir.exists() or final_dir.is_symlink():
            if final_dir.is_symlink() or not final_dir.is_dir():
                raise RuntimeError(
                    f"immutable pair generation path is invalid: {final_dir}"
                )
            if set(item.name for item in final_dir.iterdir()) != {
                PAIR_CANONICAL_FILENAME,
                PAIR_BASE28_FILENAME,
            }:
                raise RuntimeError(
                    f"immutable pair generation contents are invalid: {final_dir}"
                )
            existing_artifacts = {
                "canonical_v3": inspect_prebuilt_artifact(
                    final_dir / PAIR_CANONICAL_FILENAME,
                    label="canonical_v3",
                ),
                "base28": inspect_prebuilt_artifact(
                    final_dir / PAIR_BASE28_FILENAME,
                    label="base28",
                ),
            }
            if (
                pair_generation_id_for_artifacts(
                    existing_artifacts,
                    lineage=lineage_contract,
                )
                != pair_generation_id
            ):
                raise RuntimeError(
                    f"immutable pair generation identity collision: {final_dir}"
                )
            _discard_pair_staging_dir(
                staging_resolved,
                generation_root=generation_root,
            )
        else:
            os.rename(staging_resolved, final_dir)
            created_final_dir = True
            _fsync_directory(generation_root)

        pointer_replaced = False
        try:
            published_artifacts: dict[str, dict[str, object]] = {}
            for label, filename in (
                ("canonical_v3", PAIR_CANONICAL_FILENAME),
                ("base28", PAIR_BASE28_FILENAME),
            ):
                contract = dict(artifacts[label])
                contract["parquet_path"] = str(
                    (final_dir / filename).resolve(strict=True)
                )
                published_artifacts[label] = contract
            manifest = {
                "schema_version": PREBUILT_PAIR_SCHEMA_VERSION,
                "pair_generation_id": pair_generation_id,
                "created_utc": created_utc
                or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "lineage": lineage_contract,
                "artifacts": published_artifacts,
            }
            pointer_tmp = manifest_parent / (
                f".{pair_manifest_path.name}.{uuid.uuid4().hex}.tmp"
            )
            encoded = (
                json.dumps(manifest, sort_keys=True, indent=2) + "\n"
            ).encode("utf-8")
            try:
                with pointer_tmp.open("xb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(pointer_tmp, pair_manifest_path)
                pointer_replaced = True
            finally:
                if pointer_tmp.exists():
                    if pointer_tmp.is_symlink() or not pointer_tmp.is_file():
                        raise RuntimeError(
                            "refusing unsafe pair pointer temp cleanup: "
                            f"{pointer_tmp}"
                        )
                    pointer_tmp.unlink()
                    _fsync_directory(manifest_parent)
            _fsync_directory(manifest_parent)

            admitted = read_prebuilt_pair_manifest(
                pair_manifest_path,
                generation_root=generation_root,
            )
            if admitted.pair_generation_id != pair_generation_id:
                raise RuntimeError(
                    "published pair pointer failed identity re-admission"
                )
            verify_prebuilt_pair(admitted)
        except Exception:
            if created_final_dir and not pointer_replaced:
                _discard_unpublished_generation_dir(
                    final_dir,
                    generation_root=generation_root,
                    pair_manifest_path=pair_manifest_path,
                    pair_generation_id=pair_generation_id,
                    lineage_contract=lineage_contract,
                )
            raise
    return pair_generation_id


def _clean_repository_commit(repo_root: Path) -> str:
    root = Path(repo_root)
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve() != root
    ):
        raise RuntimeError("PAIR_BOOTSTRAP_REPOSITORY_ROOT_INVALID")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("PAIR_BOOTSTRAP_REPOSITORY_DIRTY")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise RuntimeError("PAIR_BOOTSTRAP_GIT_COMMIT_INVALID")
    return commit


def _pair_producer_source_inventory(repo_root: Path) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    for relative in _PAIR_PRODUCER_SOURCE_PATHS:
        path = repo_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"PAIR_BOOTSTRAP_PRODUCER_SOURCE_INVALID: {relative}"
            )
        inventory.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return inventory


def _load_native_source_frame(
    descriptor: dict[str, object],
    *,
    timeframe: str,
) -> pd.DataFrame:
    root = Path(str(descriptor["root"]))
    parts = sorted(root.glob("year=*/part-000.parquet"))
    if not parts:
        raise RuntimeError(f"PAIR_BOOTSTRAP_NATIVE_{timeframe}_EMPTY")
    required = list(CANONICAL_NATIVE_REQUIRED_COLUMNS)
    frames = [pd.read_parquet(path, columns=required) for path in parts]
    frame = pd.concat(frames, ignore_index=True)
    if list(frame.columns) != required:
        raise RuntimeError(
            f"PAIR_BOOTSTRAP_NATIVE_{timeframe}_SCHEMA_INVALID"
        )
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if (
        frame["time"].isna().any()
        or frame["time"].duplicated().any()
        or not frame["time"].is_monotonic_increasing
        or len(frame) != int(descriptor["row_count"])
    ):
        raise RuntimeError(
            f"PAIR_BOOTSTRAP_NATIVE_{timeframe}_TIME_OR_ROWS_INVALID"
        )
    numeric = frame[required[1:]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError(
            f"PAIR_BOOTSTRAP_NATIVE_{timeframe}_NONFINITE"
        )
    # Revalidate the complete immutable source after reading every partition.
    observed = canonical_xau_source_descriptor_v1(root, timeframe=timeframe)
    if observed != descriptor:
        raise RuntimeError(
            f"PAIR_BOOTSTRAP_NATIVE_{timeframe}_CHANGED_DURING_LOAD"
        )
    return frame


def _apply_canonical_v3_augment(v2: pd.DataFrame) -> pd.DataFrame:
    """Apply the retained model-agnostic canonical-v3 feature transform."""

    v3 = v2.copy()
    if "time" in v3.columns and not isinstance(v3.index, pd.DatetimeIndex):
        v3["time"] = pd.to_datetime(v3["time"], utc=True)
        v3 = v3.set_index("time")
    v3 = v3.drop(columns=[name for name in DROP_COLUMNS if name in v3.columns])
    v3 = add_cyclic_time_features(v3)
    v3 = add_smc_premium_state_interaction(v3)
    return add_cross_tf_momentum(v3)


def _build_model_agnostic_canonical(
    native_m5: pd.DataFrame,
    *,
    checkpoint_dir: Path,
    checkpoint_key: str,
    workers: int,
) -> pd.DataFrame:
    """Build the complete shared M5 surface once, without TRAIN-fit state."""

    v2 = build_canonical_v2(native_m5)
    canonical = _apply_canonical_v3_augment(v2)
    indexed_m5 = native_m5.set_index("time").sort_index()
    plus5 = _compute_plus5_features(
        indexed_m5[["open", "high", "low", "close", "volume"]]
    )
    for name in PLUS5_FEATURES:
        canonical[name] = plus5[name].reindex(canonical.index)
    # Preserve exact native MBA market identity under the canonical M5 clock.
    for name in CANONICAL_NATIVE_REQUIRED_COLUMNS[1:]:
        canonical[name] = indexed_m5[name].reindex(canonical.index)

    from gx1.features.htf_features import (
        attach_default_regime_v4_v2_scalars,
        build_multi_tf_per_bar_features_v2,
    )
    from gx1.execution.v12_ctx_augment_live import (
        augment_canonical_v3_model_agnostic,
    )
    from gx1.scripts.augment_forward_outcome_v2 import (
        attach_group_a_dip_struct_ctx_columns_parallel,
        trim_causal_context_warmup_prefix,
    )

    attach_default_regime_v4_v2_scalars(canonical)
    canonical = augment_canonical_v3_model_agnostic(
        canonical,
        indexed_m5.loc[
            canonical.index,
            list(CANONICAL_NATIVE_REQUIRED_COLUMNS[1:]),
        ],
    )
    multi_tf = build_multi_tf_per_bar_features_v2(canonical)
    # Group-A long-memory state (60-D1 liquidity, trailing-1yr ATR
    # percentiles, pivots) must see the full causal native prehistory, not the
    # warmup-trimmed decision slice — resetting it at the trim boundary was
    # the exact V11 failure mode.
    pre_attach_columns = set(canonical.columns)
    canonical = attach_group_a_dip_struct_ctx_columns_parallel(
        canonical,
        multi_tf=multi_tf,
        journal_label="native_pair_bootstrap",
        workers=workers,
        checkpoint_dir=checkpoint_dir,
        checkpoint_key=checkpoint_key,
        context_m5=indexed_m5[["open", "high", "low", "close"]],
    )
    # The per-candidate emission has its own causal warmup at the context
    # boundary (per-TF multi-TF snapshot convergence raises
    # CausalContextWarmupError as whole-row NaN). That warmup must be one
    # contiguous prefix; the shared trim owner removes it and fails closed on
    # any interior gap before the immutable all-column finiteness gate.
    attached_columns = [
        name for name in canonical.columns if name not in pre_attach_columns
    ]
    canonical = trim_causal_context_warmup_prefix(canonical, attached_columns)
    canonical = canonical.drop(
        columns=[
            name
            for name in (
                "atr_bucket",
                "spread_bucket",
                "is_model_bar",
                "m5_phase_0",
                "m5_phase_1",
                "m5_phase_2",
                "m5_phase_3",
                "m5_phase_4",
            )
            if name in canonical.columns
        ]
    )
    if (
        canonical.empty
        or not canonical.index.is_unique
        or not canonical.index.is_monotonic_increasing
        or canonical.index.tz is None
        or len(canonical.columns) != len(set(canonical.columns))
    ):
        raise RuntimeError("PAIR_BOOTSTRAP_CANONICAL_INDEX_OR_SCHEMA_INVALID")
    numeric = canonical.apply(pd.to_numeric, errors="coerce")
    invalid = [
        name
        for name in numeric.columns
        if not np.isfinite(numeric[name].to_numpy(dtype=np.float64)).all()
    ]
    if invalid:
        raise RuntimeError(
            f"PAIR_BOOTSTRAP_CANONICAL_NONFINITE: {invalid}"
        )
    canonical.index.name = "time"
    return canonical


def _native_pair_lineage_descriptor(
    descriptor: dict[str, object],
) -> dict[str, object]:
    fields = (
        "root",
        "manifest_path",
        "manifest_sha256",
        "instrument",
        "timeframe",
        "explicit_vedtak_id",
        "source_environment",
        "source_base_url",
        "requested_start_utc",
        "requested_end_utc_exclusive",
        "row_count",
        "time_min_utc",
        "time_max_utc",
        "canonical_rows_sha256",
        "producer_git_commit",
        "producer_source_inventory_sha256",
        "manifest_payload_sha256",
        "year_sha256",
        "year_rows",
    )
    missing = [name for name in fields if name not in descriptor]
    if missing:
        raise RuntimeError(
            f"PAIR_BOOTSTRAP_NATIVE_DESCRIPTOR_FIELDS_MISSING: {missing}"
        )
    return {name: descriptor[name] for name in fields}


def bootstrap_prebuilt_pair(
    *,
    native_m1_root: Path,
    native_m5_root: Path,
    vedtak_id: str,
    checkpoint_dir: Path,
    pair_manifest_path: Path = PREBUILT_PAIR_MANIFEST_PATH,
    generation_root: Path = PREBUILT_PAIR_ROOT,
    repo_root: Path = REPO_ROOT,
    workers: int = 12,
) -> str:
    """Derive and publish the first raw pair from two immutable native sources."""

    from gx1_guards.gates import require_retrain_vedtak

    vedtak = require_retrain_vedtak(vedtak_id)
    commit = _clean_repository_commit(Path(repo_root))
    m1_descriptor = canonical_xau_source_descriptor_v1(
        native_m1_root,
        timeframe="M1",
    )
    m5_descriptor = canonical_xau_source_descriptor_v1(
        native_m5_root,
        timeframe="M5",
    )
    for field in (
        "instrument",
        "explicit_vedtak_id",
        "source_environment",
        "source_base_url",
        "requested_start_utc",
        "requested_end_utc_exclusive",
    ):
        if m1_descriptor.get(field) != m5_descriptor.get(field):
            raise RuntimeError(
                f"PAIR_BOOTSTRAP_NATIVE_{field.upper()}_MISMATCH"
            )
    if m1_descriptor.get("explicit_vedtak_id") != vedtak:
        raise RuntimeError("PAIR_BOOTSTRAP_VEDTAK_SOURCE_MISMATCH")
    checkpoint = Path(checkpoint_dir)
    if not checkpoint.is_absolute() or checkpoint.is_symlink():
        raise RuntimeError("PAIR_BOOTSTRAP_CHECKPOINT_DIR_INVALID")
    checkpoint.mkdir(parents=True, exist_ok=True)
    if checkpoint.resolve() != checkpoint:
        raise RuntimeError("PAIR_BOOTSTRAP_CHECKPOINT_DIR_NOT_CANONICAL")

    native_m1 = _load_native_source_frame(m1_descriptor, timeframe="M1")
    native_m5 = _load_native_source_frame(m5_descriptor, timeframe="M5")
    source_key = hashlib.sha256(
        json.dumps(
            {
                "m1": m1_descriptor["manifest_sha256"],
                "m5": m5_descriptor["manifest_sha256"],
                "commit": commit,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    canonical = _build_model_agnostic_canonical(
        native_m5,
        checkpoint_dir=checkpoint,
        checkpoint_key=source_key,
        workers=workers,
    )

    m1 = native_m1.set_index("time").sort_index()
    closed_keys = closed_m5_start_for_m1_bar_labels(m1.index)
    keep = closed_keys.isin(canonical.index)
    base28 = _build_raw_base28_owned_frame(m1.loc[keep])
    if base28.empty:
        raise RuntimeError("PAIR_BOOTSTRAP_RAW_BASE28_EMPTY")
    if closed_m5_start_for_m1_bar_labels(base28.index)[-1] != canonical.index[-1]:
        raise RuntimeError("PAIR_BOOTSTRAP_RAW_BASE28_TAIL_MISMATCH")

    source_inventory = _pair_producer_source_inventory(Path(repo_root))
    lineage: dict[str, object] = {
        "schema_version": "gx1_native_pair_lineage_v1",
        "explicit_vedtak_id": vedtak,
        "producer_owner": PAIR_PRODUCER_OWNER,
        "producer_git_commit": commit,
        "producer_repository_clean": True,
        "producer_source_files": source_inventory,
        "producer_source_inventory_sha256": hashlib.sha256(
            json.dumps(
                source_inventory,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        "native_sources": {
            "m1": _native_pair_lineage_descriptor(m1_descriptor),
            "m5": _native_pair_lineage_descriptor(m5_descriptor),
        },
        "derivation_contract": {
            "canonical_builder": "canonical_v2_plus_v3_plus5_model_agnostic_ctx_group_a_v1",
            "canonical_ordered_columns_sha256": hashlib.sha256(
                json.dumps(
                    list(canonical.columns),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "raw_base28_columns": list(RAW_BASE28_COLUMNS),
            "raw_base28_columns_sha256": hashlib.sha256(
                json.dumps(
                    list(RAW_BASE28_COLUMNS),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "rank_fit_fields_absent": True,
            "m5_phase_owned_by_m1_time": True,
            "formula_contract": {
                "rank_inputs": "shared_model_native_simple_tr14_hl_mid_bid_ask_v1",
                "train_rank_application": "consumer_boundary_only",
                "group_a": "shared_parallel_exact_checkpointed_v1",
            },
            "timing_contract": {
                "m1_label": "bar_start_utc",
                "m1_availability": "bar_start_plus_1min",
                "m5_label": "bar_start_utc",
                "m5_availability": "bar_start_plus_5min",
                "htf_join_left_key": "m5_bar_start_plus_5min",
                "session_and_cycles": "m5_bar_start_plus_5min",
            },
        },
        "coverage": {
            "native_m1_rows": len(native_m1),
            "native_m5_rows": len(native_m5),
            "canonical_rows": len(canonical),
            "base28_rows": len(base28),
            "canonical_time_min_utc": canonical.index[0].isoformat(),
            "canonical_time_max_utc": canonical.index[-1].isoformat(),
            "base28_time_min_utc": base28.index[0].isoformat(),
            "base28_time_max_utc": base28.index[-1].isoformat(),
            "canonical_warmup_prefix_rows_trimmed": len(native_m5) - len(canonical),
        },
        "parent_pair_generation_id": None,
    }

    staging_dir = _candidate_staging_path(generation_root)
    try:
        _write_candidate_parquet(
            canonical.reset_index(),
            staging_dir / PAIR_CANONICAL_FILENAME,
            index=False,
        )
        _write_candidate_parquet(
            base28.reset_index(),
            staging_dir / PAIR_BASE28_FILENAME,
            index=False,
        )
        return _publish_prebuilt_pair_generation(
            staging_dir,
            pair_manifest_path=pair_manifest_path,
            generation_root=generation_root,
            expected_pair_generation_id=None,
            expected_manifest_sha256=None,
            lineage_contract=lineage,
        )
    finally:
        _discard_pair_staging_dir(
            staging_dir,
            generation_root=generation_root,
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build one immutable canonical-v3/raw-BASE28 pair"
    )
    p.add_argument("--native-m1-root", type=Path, required=True)
    p.add_argument("--native-m5-root", type=Path, required=True)
    p.add_argument("--vedtak", required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument(
        "--pair-manifest",
        type=Path,
        default=PREBUILT_PAIR_MANIFEST_PATH,
    )
    p.add_argument(
        "--generation-root",
        type=Path,
        default=PREBUILT_PAIR_ROOT,
    )
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")

    generation_id = bootstrap_prebuilt_pair(
        native_m1_root=args.native_m1_root,
        native_m5_root=args.native_m5_root,
        vedtak_id=args.vedtak,
        checkpoint_dir=args.checkpoint_dir,
        pair_manifest_path=args.pair_manifest,
        generation_root=args.generation_root,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "pair_generation_id": generation_id,
                "pair_manifest": str(args.pair_manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
