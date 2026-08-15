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
    CANONICAL_NATIVE_SUCCESSOR_MODE,
    CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
    canonical_xau_source_descriptor_v1,
)
from gx1.contracts.gx1_scope_v1 import require_offline_scope  # noqa: E402
from gx1.execution.v12_state_from_prebuilt import (  # noqa: E402
    PREBUILT_PAIR_MANIFEST_PATH,
    PREBUILT_PAIR_ROOT,
    PREBUILT_CANONICAL_BUILDER_CONTRACT,
    PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME,
    PREBUILT_PAIR_FORMULA_CONTRACT,
    PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION,
    PAIR_PUBLISH_LOCK_FILENAME,
    PREBUILT_PAIR_PRODUCER_OWNER,
    PREBUILT_PAIR_SCHEMA_VERSION,
    PREBUILT_PAIR_TIMING_CONTRACT,
    _load_verified_prebuilt,
    inspect_prebuilt_artifact,
    pair_generation_id_for_artifacts,
    read_prebuilt_pair_manifest,
    require_prebuilt_pair_parent,
    require_prebuilt_successor_frame,
    verify_prebuilt_pair,
)
from gx1.execution.v12_m1_to_m5_downsample import (  # noqa: E402
    closed_m5_start_for_m1_bar_labels,
)
from gx1.features.basic_v1 import (  # noqa: E402
    BASIC_V1_FEATURES,
    BASIC_V1_FEATURES_SHA256,
    BASIC_V1_FORMULA_SHA256,
    BASIC_V1_SCHEMA_VERSION,
    PLUS5_FEATURES,
    compute_plus5_features,
)
from gx1.scripts.materialize_build_canonical_features_v2 import (  # noqa: E402
    build_canonical_v2,
)
from gx1.scripts.materialize_canonical_v3_augment import (  # noqa: E402
    DROP_COLUMNS,
    add_cyclic_time_features,
    add_cross_tf_momentum,
)

LOG = logging.getLogger("v12_incr")

PAIR_CANONICAL_FILENAME = "canonical_v3.parquet"
PAIR_BASE28_FILENAME = "base28.parquet"
MODEL_AGNOSTIC_CACHE_SCHEMA = "gx1_model_agnostic_canonical_cache_v6"
_PAIR_STAGING_NAME = re.compile(r"\.staging-[0-9a-f]{32}\Z")

# Historical PLUS5 lane now retains four genuinely local auxiliary fields.
# The misnamed rolling-288 `_v1h1_vwap_drift` was a second H1 owner and is gone.
M1_MARKET_IDENTITY_COLUMNS = CANONICAL_NATIVE_REQUIRED_COLUMNS[1:]
# BASE28 is the M1-cadence lane. Every field is owned by native M1. Derived
# context and M5 broadcasts are forbidden because they created duplicate,
# precedence-dependent feature authorities in Entry and Exit.
RAW_BASE28_COLUMNS = M1_MARKET_IDENTITY_COLUMNS
PAIR_PRODUCER_OWNER = PREBUILT_PAIR_PRODUCER_OWNER
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
    allowed = {
        PAIR_CANONICAL_FILENAME,
        PAIR_BASE28_FILENAME,
        PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME,
    }
    entries = list(candidate.iterdir())
    if any(
        item.name not in allowed or item.is_symlink() or not item.is_file()
        for item in entries
    ) or not {
        PAIR_CANONICAL_FILENAME,
        PAIR_BASE28_FILENAME,
    }.issubset(item.name for item in entries):
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
    generation_contents = {
        item.name for item in generation_dir.iterdir()
    }
    if not {
        PAIR_CANONICAL_FILENAME,
        PAIR_BASE28_FILENAME,
    }.issubset(generation_contents) or generation_contents - {
        PAIR_CANONICAL_FILENAME,
        PAIR_BASE28_FILENAME,
        PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME,
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
    if (expected_pair_generation_id is None) != (
        expected_manifest_sha256 is None
    ):
        raise RuntimeError(
            "pair publication requires both expected pointer identities or neither"
        )
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
        current = None
        if expected_pair_generation_id is None:
            if pair_manifest_path.exists() or pair_manifest_path.is_symlink():
                raise RuntimeError(
                    "pair bootstrap refused because an active pointer already exists"
                )
            require_prebuilt_pair_parent(
                lineage_contract,
                expected_parent_pair_generation_id=None,
                expected_parent_manifest_sha256=None,
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
            require_prebuilt_pair_parent(
                lineage_contract,
                expected_parent_pair_generation_id=current.pair_generation_id,
                expected_parent_manifest_sha256=current.manifest_sha256,
            )

        published_artifacts: dict[str, dict[str, object]] = {}
        for label, filename in (
            ("canonical_v3", PAIR_CANONICAL_FILENAME),
            ("base28", PAIR_BASE28_FILENAME),
        ):
            contract = dict(artifacts[label])
            contract["parquet_path"] = str(
                (final_dir / filename).absolute()
            )
            published_artifacts[label] = contract
        manifest = {
            "schema_version": PREBUILT_PAIR_SCHEMA_VERSION,
            "pair_generation_id": pair_generation_id,
            "created_utc": created_utc
            or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generation_manifest_required": True,
            "lineage": lineage_contract,
            "artifacts": published_artifacts,
        }
        encoded = (
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        staging_generation_manifest = (
            staging_resolved / PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME
        )
        with staging_generation_manifest.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(staging_resolved)

        created_final_dir = False
        if final_dir.exists() or final_dir.is_symlink():
            if final_dir.is_symlink() or not final_dir.is_dir():
                raise RuntimeError(
                    f"immutable pair generation path is invalid: {final_dir}"
                )
            if set(item.name for item in final_dir.iterdir()) != {
                PAIR_CANONICAL_FILENAME,
                PAIR_BASE28_FILENAME,
                PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME,
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
            existing_generation = read_prebuilt_pair_manifest(
                final_dir / PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME,
                generation_root=generation_root,
            )
            if existing_generation.pair_generation_id != pair_generation_id:
                raise RuntimeError(
                    f"immutable pair generation manifest mismatch: {final_dir}"
                )
            encoded = (
                final_dir / PREBUILT_PAIR_GENERATION_MANIFEST_FILENAME
            ).read_bytes()
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
            pointer_tmp = manifest_parent / (
                f".{pair_manifest_path.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with pointer_tmp.open("xb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())

                # Prove the exact temp pointer and both final immutable files
                # are admissible before the active pointer can observe them.
                candidate = read_prebuilt_pair_manifest(
                    pointer_tmp,
                    generation_root=generation_root,
                )
                if candidate.pair_generation_id != pair_generation_id:
                    raise RuntimeError(
                        "candidate pair pointer failed identity pre-admission"
                    )
                candidate_cv3_verified, candidate_base28_verified = (
                    verify_prebuilt_pair(candidate)
                )
                if current is not None:
                    current_cv3_verified, current_base28_verified = (
                        verify_prebuilt_pair(current)
                    )
                    if (
                        candidate_cv3_verified.arrow_schema
                        != current_cv3_verified.arrow_schema
                    ):
                        raise RuntimeError(
                            "CANONICAL_V3_SUCCESSOR_ARROW_SCHEMA_MISMATCH"
                        )
                    if (
                        candidate_base28_verified.arrow_schema
                        != current_base28_verified.arrow_schema
                    ):
                        raise RuntimeError(
                            "BASE28_SUCCESSOR_ARROW_SCHEMA_MISMATCH"
                        )
                    current_cv3, _ = _load_verified_prebuilt(
                        current_cv3_verified.binding
                    )
                    current_base28, _ = _load_verified_prebuilt(
                        current_base28_verified.binding
                    )
                    candidate_cv3, _ = _load_verified_prebuilt(
                        candidate_cv3_verified.binding
                    )
                    candidate_base28, _ = _load_verified_prebuilt(
                        candidate_base28_verified.binding
                    )
                    require_prebuilt_successor_frame(
                        current_cv3,
                        candidate_cv3,
                        label="CANONICAL_V3",
                    )
                    require_prebuilt_successor_frame(
                        current_base28,
                        candidate_base28,
                        label="BASE28",
                    )

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


def _sha256_regular_file(path: Path, *, label: str) -> str:
    """Hash one canonical regular file without following a symlink."""

    path = Path(path)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
    ):
        raise RuntimeError(f"{label}_PATH_INVALID: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_bundle_cas_snapshot(
    descriptor: dict[str, object],
    *,
    timeframe: str,
) -> tuple[str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Capture cheap byte-CAS evidence after the full bundle was validated.

    ``canonical_xau_source_descriptor_v1`` already performs the expensive
    semantic validation.  Loading the parquet frame must still detect a
    concurrent mutation, but decoding every historical OANDA response again
    is unnecessary: the immutable manifest hash, every year-part hash and
    every source-chunk byte hash provide the required identity proof.
    """

    normalized = str(timeframe).upper()
    root = Path(str(descriptor.get("root") or "")).expanduser()
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
    ):
        raise RuntimeError(
            f"PAIR_BOOTSTRAP_NATIVE_{normalized}_CAS_ROOT_INVALID: {root}"
        )
    manifest_path = root / "MANIFEST.json"
    expected_manifest_sha = str(descriptor.get("manifest_sha256") or "")
    observed_manifest_sha = _sha256_regular_file(
        manifest_path,
        label=f"PAIR_BOOTSTRAP_NATIVE_{normalized}_MANIFEST",
    )
    if observed_manifest_sha != expected_manifest_sha:
        raise RuntimeError(
            f"PAIR_BOOTSTRAP_NATIVE_{normalized}_CAS_MANIFEST_MISMATCH"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"PAIR_BOOTSTRAP_NATIVE_{normalized}_CAS_MANIFEST_INVALID"
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(
            f"PAIR_BOOTSTRAP_NATIVE_{normalized}_CAS_MANIFEST_INVALID"
        )

    year_hashes = descriptor.get("year_sha256")
    if not isinstance(year_hashes, dict) or not year_hashes:
        raise RuntimeError(
            f"PAIR_BOOTSTRAP_NATIVE_{normalized}_CAS_YEARS_INVALID"
        )
    year_snapshot: list[tuple[str, str]] = []
    for key, expected in sorted(year_hashes.items()):
        if not isinstance(key, str) or not isinstance(expected, str):
            raise RuntimeError(
                f"PAIR_BOOTSTRAP_NATIVE_{normalized}_CAS_YEARS_INVALID"
            )
        part = root / key / "part-000.parquet"
        observed = _sha256_regular_file(
            part,
            label=f"PAIR_BOOTSTRAP_NATIVE_{normalized}_YEAR",
        )
        if observed != expected:
            raise RuntimeError(
                f"PAIR_BOOTSTRAP_NATIVE_{normalized}_CAS_YEAR_MISMATCH: {key}"
            )
        year_snapshot.append((key, observed))

    source_chunks = manifest.get("source_chunks")
    if not isinstance(source_chunks, list) or not source_chunks:
        raise RuntimeError(
            f"PAIR_BOOTSTRAP_NATIVE_{normalized}_CAS_CHUNKS_INVALID"
        )
    chunk_snapshot: list[tuple[str, str]] = []
    for position, metadata in enumerate(source_chunks):
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"PAIR_BOOTSTRAP_NATIVE_{normalized}_CAS_CHUNKS_INVALID"
            )
        relative = Path(str(metadata.get("relative_path") or ""))
        expected = str(metadata.get("sha256") or "")
        expected_relative = (
            Path("source_chunks") / f"chunk-{position:06d}.json.gz"
        )
        if (
            relative != expected_relative
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise RuntimeError(
                f"PAIR_BOOTSTRAP_NATIVE_{normalized}_CAS_CHUNK_PATH_INVALID"
            )
        chunk = root / relative
        observed = _sha256_regular_file(
            chunk,
            label=f"PAIR_BOOTSTRAP_NATIVE_{normalized}_CHUNK",
        )
        if observed != expected:
            raise RuntimeError(
                f"PAIR_BOOTSTRAP_NATIVE_{normalized}_CAS_CHUNK_MISMATCH: {relative}"
            )
        chunk_snapshot.append((str(relative), observed))
    return (
        observed_manifest_sha,
        tuple(year_snapshot),
        tuple(chunk_snapshot),
    )


def _load_native_source_frame(
    descriptor: dict[str, object],
    *,
    timeframe: str,
) -> pd.DataFrame:
    root = Path(str(descriptor["root"]))
    cas_before = _native_bundle_cas_snapshot(
        descriptor,
        timeframe=timeframe,
    )
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
    # The complete bundle was semantically validated before this load.  Check
    # byte-CAS identity after reading so a concurrent mutation cannot be
    # silently admitted, without decoding every historical source response a
    # second time.
    cas_after = _native_bundle_cas_snapshot(
        descriptor,
        timeframe=timeframe,
    )
    if cas_after != cas_before:
        raise RuntimeError(
            f"PAIR_BOOTSTRAP_NATIVE_{timeframe}_CHANGED_DURING_LOAD"
        )
    return frame


def _apply_local_canonical_v3_augment(v2: pd.DataFrame) -> pd.DataFrame:
    """Apply only local transforms; V4-owned cross-TF momentum comes later."""

    v3 = v2.copy()
    if "time" in v3.columns and not isinstance(v3.index, pd.DatetimeIndex):
        v3["time"] = pd.to_datetime(v3["time"], utc=True)
        v3 = v3.set_index("time")
    v3 = v3.drop(columns=[name for name in DROP_COLUMNS if name in v3.columns])
    v3 = add_cyclic_time_features(v3)
    return v3


def _canonical_cache_paths(
    checkpoint_dir: Path,
    checkpoint_key: str,
) -> tuple[Path, Path]:
    if (
        not isinstance(checkpoint_key, str)
        or len(checkpoint_key) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_key)
    ):
        raise RuntimeError("PAIR_CANONICAL_CACHE_KEY_INVALID")
    root = Path(checkpoint_dir).resolve(strict=True)
    return (
        root / f"canonical_model_agnostic_{checkpoint_key}.parquet",
        root / f"canonical_model_agnostic_{checkpoint_key}.manifest.json",
    )


def _canonical_columns_sha256(columns: object) -> str:
    return hashlib.sha256(
        json.dumps(
            list(columns),
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _validate_model_agnostic_canonical(canonical: pd.DataFrame) -> pd.DataFrame:
    if (
        canonical.empty
        or not canonical.index.is_unique
        or not canonical.index.is_monotonic_increasing
        or canonical.index.tz is None
        or len(canonical.columns) != len(set(canonical.columns))
    ):
        raise RuntimeError("PAIR_BOOTSTRAP_CANONICAL_INDEX_OR_SCHEMA_INVALID")
    if "m5h1_momentum" not in canonical.columns:
        raise RuntimeError("PAIR_BOOTSTRAP_CANONICAL_V4_MOMENTUM_MISSING")
    basic_fields = tuple(
        name for name in canonical.columns if name.startswith("_v1_")
    )
    if len(basic_fields) != len(BASIC_V1_FEATURES) or set(basic_fields) != set(
        BASIC_V1_FEATURES
    ):
        raise RuntimeError(
            "PAIR_BOOTSTRAP_CANONICAL_BASIC_V1_SURFACE_INVALID: "
            f"actual={basic_fields} expected={BASIC_V1_FEATURES}"
        )
    if "_v1h1_vwap_drift" in canonical.columns:
        raise RuntimeError("PAIR_BOOTSTRAP_CANONICAL_DUPLICATE_H1_OWNER")
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


def _load_model_agnostic_canonical_cache(
    *,
    checkpoint_dir: Path,
    checkpoint_key: str,
) -> pd.DataFrame | None:
    parquet_path, manifest_path = _canonical_cache_paths(
        checkpoint_dir,
        checkpoint_key,
    )
    present = (parquet_path.exists(), manifest_path.exists())
    if not any(present):
        return None
    if not all(present) or parquet_path.is_symlink() or manifest_path.is_symlink():
        raise RuntimeError("PAIR_CANONICAL_CACHE_INCOMPLETE")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("PAIR_CANONICAL_CACHE_MANIFEST_INVALID") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MODEL_AGNOSTIC_CACHE_SCHEMA
        or manifest.get("basic_v1_schema_version") != BASIC_V1_SCHEMA_VERSION
        or manifest.get("basic_v1_features_sha256")
        != BASIC_V1_FEATURES_SHA256
        or manifest.get("basic_v1_formula_sha256")
        != BASIC_V1_FORMULA_SHA256
        or manifest.get("checkpoint_key") != checkpoint_key
        or manifest.get("parquet_path") != str(parquet_path)
        or manifest.get("parquet_sha256")
        != _sha256_regular_file(parquet_path, label="PAIR_CANONICAL_CACHE")
        or manifest.get("columns_sha256")
        != _canonical_columns_sha256(manifest.get("columns", []))
    ):
        raise RuntimeError("PAIR_CANONICAL_CACHE_BINDING_INVALID")
    loaded = pd.read_parquet(parquet_path)
    if "time" not in loaded.columns:
        raise RuntimeError("PAIR_CANONICAL_CACHE_TIME_FIELD_MISSING")
    loaded["time"] = pd.to_datetime(loaded["time"], utc=True, errors="coerce")
    loaded = loaded.set_index("time")
    if tuple(loaded.columns) != tuple(manifest.get("columns", [])):
        raise RuntimeError("PAIR_CANONICAL_CACHE_COLUMN_ORDER_INVALID")
    if int(manifest.get("rows", -1)) != len(loaded):
        raise RuntimeError("PAIR_CANONICAL_CACHE_ROW_COUNT_INVALID")
    return _validate_model_agnostic_canonical(loaded)


def _write_model_agnostic_canonical_cache(
    *,
    canonical: pd.DataFrame,
    checkpoint_dir: Path,
    checkpoint_key: str,
) -> None:
    parquet_path, manifest_path = _canonical_cache_paths(
        checkpoint_dir,
        checkpoint_key,
    )
    if parquet_path.exists() or manifest_path.exists():
        raise RuntimeError("PAIR_CANONICAL_CACHE_ALREADY_EXISTS")
    frame = canonical.reset_index()
    temp_parquet = parquet_path.with_name(
        f".{parquet_path.name}.tmp-{uuid.uuid4().hex}"
    )
    temp_manifest = manifest_path.with_name(
        f".{manifest_path.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        frame.to_parquet(temp_parquet, index=False)
        _fsync_file(temp_parquet)
        os.replace(temp_parquet, parquet_path)
        _fsync_directory(parquet_path.parent)
        manifest = {
            "schema_version": MODEL_AGNOSTIC_CACHE_SCHEMA,
            "basic_v1_schema_version": BASIC_V1_SCHEMA_VERSION,
            "basic_v1_features_sha256": BASIC_V1_FEATURES_SHA256,
            "basic_v1_formula_sha256": BASIC_V1_FORMULA_SHA256,
            "checkpoint_key": checkpoint_key,
            "parquet_path": str(parquet_path),
            "parquet_sha256": _sha256_regular_file(
                parquet_path,
                label="PAIR_CANONICAL_CACHE",
            ),
            "rows": int(len(canonical)),
            "columns": list(canonical.columns),
            "columns_sha256": _canonical_columns_sha256(canonical.columns),
        }
        temp_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _fsync_file(temp_manifest)
        os.replace(temp_manifest, manifest_path)
        _fsync_directory(manifest_path.parent)
    finally:
        for path in (temp_parquet, temp_manifest):
            if path.exists():
                path.unlink()


def _build_model_agnostic_canonical(
    native_m5: pd.DataFrame,
    *,
    checkpoint_dir: Path,
    checkpoint_key: str,
    workers: int,
) -> pd.DataFrame:
    """Build the complete shared M5 surface once, without TRAIN-fit state."""

    cached = _load_model_agnostic_canonical_cache(
        checkpoint_dir=checkpoint_dir,
        checkpoint_key=checkpoint_key,
    )
    if cached is not None:
        LOG.info("using exact model-agnostic canonical cache: %s", checkpoint_key)
        return cached

    v2 = build_canonical_v2(native_m5)
    canonical = _apply_local_canonical_v3_augment(v2)
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
        attach_model_native_mtf_scalars_v4,
        build_multi_tf_per_bar_features_v4,
    )
    from gx1.execution.v12_ctx_augment_live import (
        augment_canonical_v3_model_agnostic_from_v4,
    )
    from gx1.scripts.augment_forward_outcome_v2 import (
        attach_group_a_ctx_columns_parallel,
        trim_causal_context_warmup_prefix,
    )

    from gx1.execution.v12_state_from_prebuilt import (
        _require_v29_registry_constants_from_bound_cache,
        _require_volatility_squeeze_artifacts_from_bound_cache,
    )

    multi_tf = build_multi_tf_per_bar_features_v4(
        indexed_m5.loc[
            canonical.index,
            ["open", "high", "low", "close", "volume"],
        ],
        v29_registry_constants=_require_v29_registry_constants_from_bound_cache(),
        volatility_squeeze_artifacts=(
            _require_volatility_squeeze_artifacts_from_bound_cache()
        ),
    )
    attach_model_native_mtf_scalars_v4(
        canonical,
        multi_tf=multi_tf,
        decision_bar_duration=pd.Timedelta(minutes=5),
    )
    canonical = augment_canonical_v3_model_agnostic_from_v4(
        canonical,
        base_bar_duration=pd.Timedelta(minutes=5),
    )
    canonical = add_cross_tf_momentum(
        canonical,
        decision_bar_duration=pd.Timedelta(minutes=5),
    )
    # Group-A long-memory state (60-D1 liquidity, trailing-1yr ATR
    # percentiles, pivots) must see the full causal native prehistory, not the
    # warmup-trimmed decision slice — resetting it at the trim boundary was
    # the exact V11 failure mode.
    pre_attach_columns = set(canonical.columns)
    canonical = attach_group_a_ctx_columns_parallel(
        canonical,
        multi_tf=multi_tf,
        journal_label="native_pair_generation",
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
    canonical = _validate_model_agnostic_canonical(canonical)
    _write_model_agnostic_canonical_cache(
        canonical=canonical,
        checkpoint_dir=checkpoint_dir,
        checkpoint_key=checkpoint_key,
    )
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


def _require_native_pair_compatibility(
    m1_descriptor: dict[str, object],
    m5_descriptor: dict[str, object],
    *,
    vedtak: str,
    label: str,
) -> None:
    """Require one M1/M5 source decision and one exact requested interval."""

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
                f"PAIR_{label}_NATIVE_{field.upper()}_MISMATCH"
            )
    if m1_descriptor.get("explicit_vedtak_id") != vedtak:
        raise RuntimeError(f"PAIR_{label}_VEDTAK_SOURCE_MISMATCH")


def _native_frame_view(
    frame: pd.DataFrame,
    *,
    timeframe: str,
) -> pd.DataFrame:
    """Return a timestamp-indexed exact native frame for bit-prefix proof."""

    required = list(CANONICAL_NATIVE_REQUIRED_COLUMNS)
    if (
        not isinstance(frame, pd.DataFrame)
        or list(frame.columns) != required
        or frame.empty
    ):
        raise RuntimeError(f"PAIR_NATIVE_{timeframe}_FRAME_INVALID")
    indexed = frame.copy()
    indexed["time"] = pd.to_datetime(indexed["time"], utc=True, errors="coerce")
    indexed = indexed.set_index("time")
    indexed.index.name = "time"
    return indexed


def _aggregate_native_m1_to_m5(native_m1: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every observed M1 bucket without session-time assumptions."""

    required = list(CANONICAL_NATIVE_REQUIRED_COLUMNS)
    if (
        not isinstance(native_m1, pd.DataFrame)
        or native_m1.empty
        or list(native_m1.columns) != required
    ):
        raise RuntimeError("PAIR_NATIVE_M1_AGGREGATION_INPUT_INVALID")
    indexed = native_m1.copy()
    indexed["time"] = pd.to_datetime(
        indexed["time"],
        utc=True,
        errors="coerce",
    )
    if indexed["time"].isna().any():
        raise RuntimeError("PAIR_NATIVE_M1_AGGREGATION_TIME_INVALID")
    indexed["_m5_time"] = indexed["time"].dt.floor("5min")
    aggregations: dict[str, str] = {}
    for name in required[1:]:
        if name == "volume":
            aggregations[name] = "sum"
        elif name.endswith("open") or name == "open":
            aggregations[name] = "first"
        elif name.endswith("high") or name == "high":
            aggregations[name] = "max"
        elif name.endswith("low") or name == "low":
            aggregations[name] = "min"
        elif name.endswith("close") or name == "close":
            aggregations[name] = "last"
        else:
            raise RuntimeError(
                f"PAIR_NATIVE_M1_AGGREGATION_FIELD_UNOWNED: {name}"
            )
    aggregated = indexed.groupby("_m5_time", sort=True).agg(aggregations)
    aggregated.index = pd.DatetimeIndex(aggregated.index, name="time")
    return aggregated.loc[:, required[1:]]


def _require_native_m1_m5_aggregation_identity(
    native_m1: pd.DataFrame,
    native_m5: pd.DataFrame,
) -> None:
    """Prove every native M5 row is the exact aggregation of native M1.

    Sparse/reopen buckets are accepted only when the separately sourced M5 row
    is still bit-exact. No UTC session hour is encoded in this contract.
    """

    required = list(CANONICAL_NATIVE_REQUIRED_COLUMNS)
    if (
        not isinstance(native_m5, pd.DataFrame)
        or native_m5.empty
        or list(native_m5.columns) != required
    ):
        raise RuntimeError("PAIR_NATIVE_M5_AGGREGATION_INPUT_INVALID")
    observed = native_m5.copy()
    observed["time"] = pd.to_datetime(
        observed["time"],
        utc=True,
        errors="coerce",
    )
    if observed["time"].isna().any():
        raise RuntimeError("PAIR_NATIVE_M5_AGGREGATION_TIME_INVALID")
    observed = observed.set_index("time")
    observed.index.name = "time"
    expected = _aggregate_native_m1_to_m5(native_m1)
    expected_through_m5_tail = expected.loc[expected.index <= observed.index[-1]]
    if not expected_through_m5_tail.index.equals(observed.index):
        missing = observed.index.difference(expected_through_m5_tail.index)
        unexpected = expected_through_m5_tail.index.difference(observed.index)
        raise RuntimeError(
            "PAIR_NATIVE_M1_M5_BUCKET_IDENTITY_MISMATCH: "
            f"missing_from_m1={list(missing[:3])} "
            f"missing_from_m5={list(unexpected[:3])}"
        )
    for name in required[1:]:
        expected_values = expected_through_m5_tail[name].to_numpy()
        observed_values = observed[name].to_numpy()
        if (
            expected_values.dtype != observed_values.dtype
            or expected_values.tobytes() != observed_values.tobytes()
        ):
            mismatch = np.flatnonzero(
                expected_values != observed_values
            )
            position = int(mismatch[0]) if len(mismatch) else 0
            raise RuntimeError(
                "PAIR_NATIVE_M1_M5_VALUE_IDENTITY_MISMATCH: "
                f"field={name} time={observed.index[position]} "
                f"expected={expected_values[position]!r} "
                f"observed={observed_values[position]!r}"
            )


def _require_native_successor_descriptor_binding(
    *,
    parent_descriptor: dict[str, object],
    successor_descriptor: dict[str, object],
    timeframe: str,
) -> None:
    """Require a schema-v4 successor bound to the active pair's native source."""

    if (
        successor_descriptor.get("schema_version")
        != CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA
        or successor_descriptor.get("publication_mode")
        != CANONICAL_NATIVE_SUCCESSOR_MODE
    ):
        raise RuntimeError(
            f"PAIR_SUCCESSOR_NATIVE_{timeframe}_SCHEMA_OR_MODE_INVALID"
        )
    successor_parent = successor_descriptor.get("parent_source")
    if not isinstance(successor_parent, dict):
        raise RuntimeError(
            f"PAIR_SUCCESSOR_NATIVE_{timeframe}_PARENT_SOURCE_INVALID"
        )
    for field in ("root", "manifest_path", "manifest_sha256"):
        if successor_parent.get(field) != parent_descriptor.get(field):
            raise RuntimeError(
                f"PAIR_SUCCESSOR_NATIVE_{timeframe}_PARENT_SOURCE_{field.upper()}_MISMATCH"
            )


def _require_native_source_successor(
    *,
    parent_descriptor: dict[str, object],
    successor_descriptor: dict[str, object],
    successor_frame: pd.DataFrame,
    timeframe: str,
) -> None:
    """Prove that a new immutable native source is an exact strict child."""

    _require_native_successor_descriptor_binding(
        parent_descriptor=parent_descriptor,
        successor_descriptor=successor_descriptor,
        timeframe=timeframe,
    )

    # The active pair lineage is the already-admitted semantic identity of
    # the parent source.  Re-decoding every historical response here adds no
    # new authority; the manifest hash, year hashes and source-chunk hashes
    # prove that the bytes still are exactly that admitted parent.
    _native_bundle_cas_snapshot(parent_descriptor, timeframe=timeframe)
    parent_frame = _load_native_source_frame(
        parent_descriptor,
        timeframe=timeframe,
    )
    for field in (
        "instrument",
        "timeframe",
        "explicit_vedtak_id",
        "source_environment",
        "source_base_url",
        "requested_start_utc",
        "time_min_utc",
    ):
        if parent_descriptor.get(field) != successor_descriptor.get(field):
            raise RuntimeError(
                f"PAIR_SUCCESSOR_NATIVE_{timeframe}_{field.upper()}_MISMATCH"
            )
    try:
        parent_end = pd.Timestamp(parent_descriptor["requested_end_utc_exclusive"])
        successor_end = pd.Timestamp(
            successor_descriptor["requested_end_utc_exclusive"]
        )
        parent_max = pd.Timestamp(parent_descriptor["time_max_utc"])
        successor_max = pd.Timestamp(successor_descriptor["time_max_utc"])
    except Exception as exc:
        raise RuntimeError(
            f"PAIR_SUCCESSOR_NATIVE_{timeframe}_TIME_IDENTITY_INVALID"
        ) from exc
    if (
        successor_end <= parent_end
        or successor_max <= parent_max
        or int(successor_descriptor["row_count"])
        <= int(parent_descriptor["row_count"])
    ):
        raise RuntimeError(
            f"PAIR_SUCCESSOR_NATIVE_{timeframe}_NOT_STRICTLY_ADVANCING"
        )
    require_prebuilt_successor_frame(
        _native_frame_view(parent_frame, timeframe=timeframe),
        _native_frame_view(successor_frame, timeframe=timeframe),
        label=f"NATIVE_{timeframe}",
    )


def _derive_pair_frames(
    *,
    native_m1: pd.DataFrame,
    native_m5: pd.DataFrame,
    checkpoint_dir: Path,
    checkpoint_key: str,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the complete canonical M5 and raw M1 pair from loaded sources."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers != 1:
        raise RuntimeError("PAIR_FEATURE_WORKERS_MUST_EQUAL_ONE")

    canonical = _build_model_agnostic_canonical(
        native_m5,
        checkpoint_dir=checkpoint_dir,
        checkpoint_key=checkpoint_key,
        workers=workers,
    )
    m1 = native_m1.set_index("time").sort_index()
    closed_keys = closed_m5_start_for_m1_bar_labels(m1.index)
    keep = closed_keys.isin(canonical.index)
    base28 = _build_raw_base28_owned_frame(m1.loc[keep])
    if base28.empty:
        raise RuntimeError("PAIR_RAW_BASE28_EMPTY")
    if closed_m5_start_for_m1_bar_labels(base28.index)[-1] != canonical.index[-1]:
        raise RuntimeError("PAIR_RAW_BASE28_TAIL_MISMATCH")
    return canonical, base28


def _build_pair_lineage(
    *,
    vedtak: str,
    commit: str,
    source_inventory: list[dict[str, str]],
    m1_descriptor: dict[str, object],
    m5_descriptor: dict[str, object],
    native_m1: pd.DataFrame,
    native_m5: pd.DataFrame,
    canonical: pd.DataFrame,
    base28: pd.DataFrame,
    parent_pair_generation_id: str | None,
    parent_pair_manifest_sha256: str | None,
) -> dict[str, object]:
    """Build the single lineage envelope used by bootstrap and successors."""

    return {
        "schema_version": PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION,
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
            "canonical_builder": PREBUILT_CANONICAL_BUILDER_CONTRACT,
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
            "formula_contract": dict(PREBUILT_PAIR_FORMULA_CONTRACT),
            "timing_contract": dict(PREBUILT_PAIR_TIMING_CONTRACT),
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
            "canonical_warmup_prefix_rows_trimmed": len(native_m5)
            - len(canonical),
        },
        "parent_pair_generation_id": parent_pair_generation_id,
        "parent_pair_manifest_sha256": parent_pair_manifest_sha256,
    }


def bootstrap_prebuilt_pair(
    *,
    native_m1_root: Path,
    native_m5_root: Path,
    vedtak_id: str,
    checkpoint_dir: Path,
    pair_manifest_path: Path = PREBUILT_PAIR_MANIFEST_PATH,
    generation_root: Path = PREBUILT_PAIR_ROOT,
    repo_root: Path = REPO_ROOT,
    workers: int = 1,
) -> str:
    """Derive and publish the first raw pair from two immutable native sources."""

    from gx1_guards.gates import require_retrain_vedtak

    require_offline_scope("featurebase_build")
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
    _require_native_pair_compatibility(
        m1_descriptor,
        m5_descriptor,
        vedtak=vedtak,
        label="BOOTSTRAP",
    )
    checkpoint = Path(checkpoint_dir)
    if not checkpoint.is_absolute() or checkpoint.is_symlink():
        raise RuntimeError("PAIR_BOOTSTRAP_CHECKPOINT_DIR_INVALID")
    checkpoint.mkdir(parents=True, exist_ok=True)
    if checkpoint.resolve() != checkpoint:
        raise RuntimeError("PAIR_BOOTSTRAP_CHECKPOINT_DIR_NOT_CANONICAL")

    native_m1 = _load_native_source_frame(m1_descriptor, timeframe="M1")
    native_m5 = _load_native_source_frame(m5_descriptor, timeframe="M5")
    _require_native_m1_m5_aggregation_identity(native_m1, native_m5)
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
    canonical, base28 = _derive_pair_frames(
        native_m1=native_m1,
        native_m5=native_m5,
        checkpoint_dir=checkpoint,
        checkpoint_key=source_key,
        workers=workers,
    )

    if _clean_repository_commit(Path(repo_root)) != commit:
        raise RuntimeError("PAIR_BOOTSTRAP_REPOSITORY_CHANGED_DURING_BUILD")
    source_inventory = _pair_producer_source_inventory(Path(repo_root))
    lineage = _build_pair_lineage(
        vedtak=vedtak,
        commit=commit,
        source_inventory=source_inventory,
        m1_descriptor=m1_descriptor,
        m5_descriptor=m5_descriptor,
        native_m1=native_m1,
        native_m5=native_m5,
        canonical=canonical,
        base28=base28,
        parent_pair_generation_id=None,
        parent_pair_manifest_sha256=None,
    )

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


def publish_prebuilt_pair_successor(
    *,
    native_m1_root: Path,
    native_m5_root: Path,
    vedtak_id: str,
    checkpoint_dir: Path,
    expected_pair_generation_id: str,
    expected_manifest_sha256: str,
    pair_manifest_path: Path = PREBUILT_PAIR_MANIFEST_PATH,
    generation_root: Path = PREBUILT_PAIR_ROOT,
    repo_root: Path = REPO_ROOT,
    workers: int = 1,
) -> str:
    """Derive and CAS-publish one strict child of the active pair.

    Both native source bundles must themselves be immutable, full-history,
    strict bit-prefix successors of the exact M1/M5 sources recorded in the
    active pair. The derived canonical and BASE28 artifacts must then be strict
    bit-prefix successors too. Any source, formula, schema, history, pointer or
    repository mismatch aborts before the serving pointer moves.
    """

    from gx1_guards.gates import require_retrain_vedtak

    require_offline_scope("featurebase_build")
    vedtak = require_retrain_vedtak(vedtak_id)
    commit = _clean_repository_commit(Path(repo_root))
    current = read_prebuilt_pair_manifest(
        pair_manifest_path,
        generation_root=generation_root,
    )
    if (
        current.pair_generation_id != expected_pair_generation_id
        or current.manifest_sha256 != expected_manifest_sha256
    ):
        raise RuntimeError(
            "PAIR_SUCCESSOR_EXPECTED_POINTER_IDENTITY_MISMATCH"
        )
    verify_prebuilt_pair(current)
    if current.lineage["explicit_vedtak_id"] != vedtak:
        raise RuntimeError("PAIR_SUCCESSOR_VEDTAK_PARENT_MISMATCH")

    m1_descriptor = canonical_xau_source_descriptor_v1(
        native_m1_root,
        timeframe="M1",
    )
    m5_descriptor = canonical_xau_source_descriptor_v1(
        native_m5_root,
        timeframe="M5",
    )
    _require_native_pair_compatibility(
        m1_descriptor,
        m5_descriptor,
        vedtak=vedtak,
        label="SUCCESSOR",
    )
    parent_sources = current.lineage["native_sources"]
    if not isinstance(parent_sources, dict):
        raise RuntimeError("PAIR_SUCCESSOR_PARENT_NATIVE_SOURCES_INVALID")
    for label, timeframe, descriptor in (
        ("m1", "M1", m1_descriptor),
        ("m5", "M5", m5_descriptor),
    ):
        raw_parent = parent_sources.get(label)
        if not isinstance(raw_parent, dict):
            raise RuntimeError(
                f"PAIR_SUCCESSOR_PARENT_NATIVE_{timeframe}_DESCRIPTOR_INVALID"
            )
        _require_native_successor_descriptor_binding(
            parent_descriptor=raw_parent,
            successor_descriptor=descriptor,
            timeframe=timeframe,
        )
    native_m1 = _load_native_source_frame(m1_descriptor, timeframe="M1")
    native_m5 = _load_native_source_frame(m5_descriptor, timeframe="M5")
    native_cas_after_load = {
        "M1": _native_bundle_cas_snapshot(m1_descriptor, timeframe="M1"),
        "M5": _native_bundle_cas_snapshot(m5_descriptor, timeframe="M5"),
    }
    _require_native_m1_m5_aggregation_identity(native_m1, native_m5)
    for label, timeframe, descriptor, frame in (
        ("m1", "M1", m1_descriptor, native_m1),
        ("m5", "M5", m5_descriptor, native_m5),
    ):
        raw_parent = parent_sources[label]
        _require_native_source_successor(
            parent_descriptor=raw_parent,
            successor_descriptor=descriptor,
            successor_frame=frame,
            timeframe=timeframe,
        )

    checkpoint = Path(checkpoint_dir)
    if not checkpoint.is_absolute() or checkpoint.is_symlink():
        raise RuntimeError("PAIR_SUCCESSOR_CHECKPOINT_DIR_INVALID")
    checkpoint.mkdir(parents=True, exist_ok=True)
    if checkpoint.resolve() != checkpoint:
        raise RuntimeError("PAIR_SUCCESSOR_CHECKPOINT_DIR_NOT_CANONICAL")
    source_key = hashlib.sha256(
        json.dumps(
            {
                "m1": m1_descriptor["manifest_sha256"],
                "m5": m5_descriptor["manifest_sha256"],
                "commit": commit,
                "parent_pair_generation_id": current.pair_generation_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    canonical, base28 = _derive_pair_frames(
        native_m1=native_m1,
        native_m5=native_m5,
        checkpoint_dir=checkpoint,
        checkpoint_key=source_key,
        workers=workers,
    )

    current_cv3, _ = _load_verified_prebuilt(current.canonical_v3)
    current_base28, _ = _load_verified_prebuilt(current.base28)
    require_prebuilt_successor_frame(
        current_cv3,
        canonical,
        label="CANONICAL_V3",
    )
    require_prebuilt_successor_frame(
        current_base28,
        base28,
        label="BASE28",
    )
    for timeframe, descriptor in (
        ("M1", m1_descriptor),
        ("M5", m5_descriptor),
    ):
        observed_cas = _native_bundle_cas_snapshot(
            descriptor,
            timeframe=timeframe,
        )
        if observed_cas != native_cas_after_load[timeframe]:
            raise RuntimeError(
                f"PAIR_SUCCESSOR_NATIVE_{timeframe}_CHANGED_DURING_BUILD"
            )
    if _clean_repository_commit(Path(repo_root)) != commit:
        raise RuntimeError("PAIR_SUCCESSOR_REPOSITORY_CHANGED_DURING_BUILD")
    source_inventory = _pair_producer_source_inventory(Path(repo_root))
    lineage = _build_pair_lineage(
        vedtak=vedtak,
        commit=commit,
        source_inventory=source_inventory,
        m1_descriptor=m1_descriptor,
        m5_descriptor=m5_descriptor,
        native_m1=native_m1,
        native_m5=native_m5,
        canonical=canonical,
        base28=base28,
        parent_pair_generation_id=current.pair_generation_id,
        parent_pair_manifest_sha256=current.manifest_sha256,
    )

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
            expected_pair_generation_id=expected_pair_generation_id,
            expected_manifest_sha256=expected_manifest_sha256,
            lineage_contract=lineage,
        )
    finally:
        _discard_pair_staging_dir(
            staging_dir,
            generation_root=generation_root,
        )


def pair_publication_evidence(
    *,
    pair_manifest_path: Path,
    generation_root: Path,
) -> dict[str, object]:
    """Return terminal, re-admitted identity and coverage evidence."""

    admitted = read_prebuilt_pair_manifest(
        pair_manifest_path,
        generation_root=generation_root,
    )
    verify_prebuilt_pair(admitted)
    return {
        "schema_version": "gx1_prebuilt_pair_publication_evidence_v1",
        "pair_generation_id": admitted.pair_generation_id,
        "pair_manifest_path": str(admitted.manifest_path),
        "pair_manifest_sha256": admitted.manifest_sha256,
        "generation_manifest_path": (
            str(admitted.generation_manifest_path)
            if admitted.generation_manifest_path is not None
            else None
        ),
        "generation_manifest_sha256": (
            admitted.manifest_sha256
            if admitted.generation_manifest_path is not None
            else None
        ),
        "parent_pair_generation_id": admitted.lineage[
            "parent_pair_generation_id"
        ],
        "native_sources": admitted.lineage["native_sources"],
        "coverage": admitted.lineage["coverage"],
        "canonical_v3": {
            "parquet_sha256": admitted.canonical_v3.parquet_sha256,
            "rows": admitted.canonical_v3.rows,
            "cols_total": admitted.canonical_v3.cols_total,
        },
        "base28": {
            "parquet_sha256": admitted.base28.parquet_sha256,
            "rows": admitted.base28.rows,
            "cols_total": admitted.base28.cols_total,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build one immutable canonical-v3/raw-BASE28 pair"
    )
    p.add_argument(
        "--publication-mode",
        choices=("bootstrap", "successor"),
        required=True,
    )
    p.add_argument("--native-m1-root", type=Path)
    p.add_argument("--native-m5-root", type=Path)
    p.add_argument("--vedtak")
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--expected-pair-generation-id")
    p.add_argument("--expected-manifest-sha256")
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
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")

    missing_build_args = [
        name
        for name, value in (
            ("--native-m1-root", args.native_m1_root),
            ("--native-m5-root", args.native_m5_root),
            ("--vedtak", args.vedtak),
            ("--checkpoint-dir", args.checkpoint_dir),
        )
        if value is None
    ]
    if missing_build_args:
        p.error(
            f"{args.publication_mode} requires explicit "
            + ", ".join(missing_build_args)
        )
    if args.publication_mode == "bootstrap":
        if (
            args.expected_pair_generation_id is not None
            or args.expected_manifest_sha256 is not None
        ):
            p.error(
                "bootstrap forbids successor pointer arguments"
            )
        generation_id = bootstrap_prebuilt_pair(
            native_m1_root=args.native_m1_root,
            native_m5_root=args.native_m5_root,
            vedtak_id=args.vedtak,
            checkpoint_dir=args.checkpoint_dir,
            pair_manifest_path=args.pair_manifest,
            generation_root=args.generation_root,
            workers=args.workers,
        )
    else:
        missing = [
            name
            for name, value in (
                (
                    "--expected-pair-generation-id",
                    args.expected_pair_generation_id,
                ),
                ("--expected-manifest-sha256", args.expected_manifest_sha256),
            )
            if value is None
        ]
        if missing:
            p.error(
                "successor requires explicit " + ", ".join(missing)
            )
        generation_id = publish_prebuilt_pair_successor(
            native_m1_root=args.native_m1_root,
            native_m5_root=args.native_m5_root,
            vedtak_id=args.vedtak,
            checkpoint_dir=args.checkpoint_dir,
            expected_pair_generation_id=args.expected_pair_generation_id,
            expected_manifest_sha256=args.expected_manifest_sha256,
            pair_manifest_path=args.pair_manifest,
            generation_root=args.generation_root,
            workers=args.workers,
        )
    evidence = pair_publication_evidence(
        pair_manifest_path=args.pair_manifest,
        generation_root=args.generation_root,
    )
    if evidence["pair_generation_id"] != generation_id:
        raise RuntimeError("PAIR_PUBLICATION_TERMINAL_IDENTITY_MISMATCH")
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
