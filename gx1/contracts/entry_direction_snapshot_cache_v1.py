"""Explicit immutable-prior-proof cache reuse for the 193 snapshot reference.

This is not a replacement for load_multi_tf_v4_cache. It freshly verifies the
bound manifest and all fifteen NPY files, then reuses the exact earlier 055/
full-input-liveness proof instead of recomputing matrix liveness. Squeeze,
registry and M5-source qualification are inherited from that same-byte proof;
their external artifacts are not loaded here. The caller owns admission,
resource caps and fresh selected-row value/domain/closed-time qualification.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

import numpy as np

from gx1.features import htf_features as htf
from gx1.contracts.entry_direction_snapshot_data_v1 import prepare_direction_mtf_surface

SCHEMA_VERSION = "gx1_direction_snapshot_prior_proof_cache_v1"
PROOF_MODE = "immutable_prior_full_liveness_reused_no_fresh_fullmatrix_recomputation"
_HTF_OWNER = "gx1/features/htf_features.py"


def _require(condition, reason):
    if not condition:
        raise RuntimeError("DIRECTION_SNAPSHOT_CACHE_" + reason)


def _binding(value):
    _require(isinstance(value, Mapping) and set(value) == {"path", "sha256"}
        and isinstance(value["path"], str) and Path(value["path"]).is_absolute(),
        "BINDING")
    htf._exact_cache_sha256(value["sha256"], label="snapshot binding")
    return dict(value)


def _file_stat(path, *, directory=False):
    path = Path(path)
    _require(path.is_absolute() and not htf._cache_path_has_symlink_component(path)
        and path.resolve(strict=True) == path, "CANONICAL_PATH")
    info = path.lstat()
    _require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
        "FILE_KIND")
    return {name: int(getattr(info, "st_" + name)) for name in
        ("dev", "ino", "mode", "size", "mtime_ns", "ctime_ns")}


@contextmanager
def _directory(path):
    before = _file_stat(path, directory=True)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        _require(all(int(getattr(info, "st_" + key)) == value
            for key, value in before.items()), "DIRECTORY_CHANGED")
        yield fd
    finally:
        os.close(fd)


def _remember(bookends, path, *, directory=False):
    observed = _file_stat(path, directory=directory)
    item = {"directory": directory, "stat": observed}
    previous = bookends.setdefault(str(path), item)
    _require(previous == item, "FILE_CHANGED")
    return observed


def _reject_json_constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def _read_json(binding, *, cap, bookends):
    binding = _binding(binding)
    path = Path(binding["path"])
    before = _remember(bookends, path)
    _require(before["size"] <= cap, "METADATA_BYTE_CAP")
    with _directory(path.parent) as fd:
        raw = htf._read_cache_file_bytes(fd, path.name,
            expected_sha256=binding["sha256"], expected_size_bytes=before["size"],
            label="direction snapshot " + path.name)
    try:
        value = json.loads(raw.decode("utf-8"),
            object_pairs_hook=htf._json_object_without_duplicate_keys,
            parse_constant=_reject_json_constant)
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("DIRECTION_SNAPSHOT_CACHE_JSON") from exc
    _require(isinstance(value, dict), "JSON_ROOT")
    _remember(bookends, path)
    return value


def _require_source_match(binding, bookends):
    _require(isinstance(binding, Mapping)
        and set(binding) == {"source_head", "sha256"}
        and isinstance(binding["source_head"], str)
        and len(binding["source_head"]) == 40
        and all(c in "0123456789abcdef" for c in binding["source_head"]),
        "HISTORICAL_SOURCE_BINDING")
    htf._exact_cache_sha256(binding["sha256"], label="historical HTF owner")
    owner = Path(htf.__file__).absolute()
    _require(owner.name == "htf_features.py", "HTF_OWNER_PATH")
    _remember(bookends, owner)
    current = owner.read_bytes()
    historical = subprocess.check_output(
        ["git", "-C", str(owner.parents[2]), "show",
         binding["source_head"] + ":" + _HTF_OWNER], stderr=subprocess.PIPE)
    _require(hashlib.sha256(current).hexdigest() == binding["sha256"]
        and hashlib.sha256(historical).hexdigest() == binding["sha256"]
        and current == historical, "HTF_OWNER_CHANGED")
    _remember(bookends, owner)
    return {"path": str(owner), **dict(binding), "historical_current_bytes_equal": True}


def _prior_chain(normalization, result, terminal, full, *, normalization_binding,
        prior_result_binding, full_input_liveness_binding, source_head):
    _require(normalization.get("schema_version")
        == "gx1_entry_only_dynamic_m5_base_normalization_v1"
        and normalization.get("decision") == "PASS"
        and normalization.get("test_accessed") is False
        and normalization.get("val_fit_rows") == 0
        and normalization.get("test_fit_rows") == 0, "NORMALIZATION_PRIOR")
    contract = normalization.get("contract")
    _require(isinstance(contract, dict) and isinstance(contract.get("lineage"), dict),
        "NORMALIZATION_CONTRACT")
    contract_sha = htf._exact_cache_sha256(normalization.get("contract_sha256"),
        label="normalization contract")
    _require(result.get("schema_version")
        == "gx1_whole_train_dynamic_m5_normalization_result_v1"
        and result.get("decision") == "PASS_WHOLE_TRAIN_DYNAMIC_M5_NORMALIZATION"
        and result.get("source_head") == source_head
        and result.get("normalization") == normalization_binding
        and result.get("contract_sha256") == contract_sha
        and result.get("market_normalization_fit_complete") is True
        and result.get("test_data_used") is False
        and result.get("val_fit_rows") == 0 and result.get("test_fit_rows") == 0,
        "RESULT_PRIOR")
    _require(terminal.get("status") == "COMPLETE"
        and type(terminal.get("exit_code")) is int and terminal["exit_code"] == 0
        and terminal.get("source_unchanged") is True
        and terminal.get("source_head") == source_head
        and terminal.get("result_sha256") == prior_result_binding["sha256"]
        and terminal.get("normalization_binding") == normalization_binding
        and terminal.get("market_normalization_fit_complete") is True
        and terminal.get("test_data_used") is False, "TERMINAL_PRIOR")
    _require(normalization.get("full_input_liveness") == full_input_liveness_binding
        and result.get("full_input_liveness") == full_input_liveness_binding
        and full.get("schema_version") == "entry_full_input_liveness_contract_v13"
        and full.get("decision") == "PASS" and full.get("failures") == [],
        "FULL_LIVENESS_PRIOR")
    mtf = full.get("multi_tf_liveness")
    _require(isinstance(mtf, dict), "MTF_LIVENESS_PRIOR")
    liveness = htf.require_multi_tf_v4_liveness_contract(mtf.get("contract"))
    return contract, liveness, mtf.get("cache_binding")


def _manifest_headers(manifest, *, lineage, liveness, cache_binding, manifest_binding):
    _require(set(manifest) == htf._HTF_V4_CACHE_MANIFEST_KEYS, "MANIFEST_KEYS")
    expected = {
        "schema_version": htf.HTF_V4_CACHE_SCHEMA_VERSION,
        "builder_version": htf.HTF_V4_CACHE_BUILDER_VERSION,
        "smc_causal_replay_schema_version": htf.SMC_CAUSAL_REPLAY_SCHEMA_VERSION,
        "technical_indicator_owner": htf.technical_indicator_contract_metadata(),
        "native_primitive_owner": htf.native_mtf_full_primitive_contract_metadata_v4(),
        "swing_structure_owner": htf.swing_structure_contract_metadata(),
        "feature_count": htf.MULTI_TF_FEATURE_COUNT_V4,
        "feature_names": list(htf.MULTI_TF_PER_BAR_FEATURES_V4),
        "shift_contract": {tf: str(value) for tf, value in htf.MULTI_TF_SHIFT.items()},
        "resample_origin_contract": dict(htf.MULTI_TF_RESAMPLE_ORIGIN_CONTRACT),
        "m5_prebuilt_source": lineage.get("m5_prebuilt_path"),
        "m5_prebuilt_source_sha256": lineage.get("m5_prebuilt_sha256"),
        "full_input_liveness": liveness,
    }
    _require(all(manifest.get(key) == value for key, value in expected.items()),
        "MANIFEST_HEADERS")
    _binding({"path": manifest["m5_prebuilt_source"],
        "sha256": manifest["m5_prebuilt_source_sha256"]})
    identity = htf._exact_cache_sha256(manifest["cache_identity_sha256"],
        label="snapshot cache identity")
    _require(htf.compute_htf_v4_cache_identity(manifest) == identity
        and cache_binding == {"cache_identity_sha256": identity,
            "manifest_path": manifest_binding["path"],
            "manifest_sha256": manifest_binding["sha256"],
            "observed_manifest_sha256": manifest_binding["sha256"]},
        "CACHE_PRIOR_IDENTITY")
    tfs = tuple(htf.MULTI_TF_RESAMPLE_RULES)
    _require(isinstance(manifest["tfs"], dict) and tuple(manifest["tfs"]) == tfs,
        "TIMEFRAME_ORDER")
    _require(lineage.get("mtf_builder_version") == htf.HTF_V4_CACHE_BUILDER_VERSION
        and lineage.get("mtf_feature_names_sha256") == htf.MULTI_TF_FEATURE_NAMES_SHA256_V4
        and lineage.get("per_tf_shift_seconds")
        == {tf: int(htf.MULTI_TF_SHIFT[tf].value // 10**9) for tf in tfs}
        and isinstance(lineage.get("per_tf_seq_lens"), dict)
        and set(lineage["per_tf_seq_lens"]) == set(tfs), "NORMALIZATION_MTF_LINEAGE")
    return tfs


def _load_timeframe(fd, tf, info, *, liveness, population, sequence_bars):
    _require(isinstance(info, dict) and set(info) == htf._HTF_V4_CACHE_TF_KEYS,
        "TF_KEYS")
    rows = htf._exact_cache_int(info["n_bars"], label=tf + ".n_bars", minimum=1)
    width = htf._exact_cache_int(info["feature_count"], label=tf + ".width", minimum=1)
    _require(width == htf.MULTI_TF_FEATURE_COUNT_V4, "TF_WIDTH")
    kinds = ("feats", "ts", "model_native_scalars")
    arrays = []
    for kind in kinds:
        name = tf + "_" + kind + ".npy"
        _require(info[kind + "_npy"] == name, "TF_FILENAME")
        array = htf._load_verified_cache_npy(fd, name,
            expected_sha256=htf._exact_cache_sha256(info[kind + "_npy_sha256"],
                label=name),
            expected_size_bytes=htf._exact_cache_int(info[kind + "_npy_size_bytes"],
                label=name, minimum=1), label="direction snapshot " + name)
        _require(isinstance(array, np.memmap) and not array.flags.writeable
            and array.flags.c_contiguous, "READONLY_MMAP")
        arrays.append(array)
    features, times, scalars = arrays
    scalar_fields = htf.MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[tf]
    scalar_count = htf._exact_cache_int(info["model_native_scalar_count"],
        label=tf + ".scalar_count", minimum=0)
    _require(features.dtype == np.float32 and features.shape == (rows, width)
        and times.dtype == np.int64 and times.shape == (rows,)
        and scalars.dtype == np.float32 and scalars.shape == (rows, scalar_count)
        and scalar_count == len(scalar_fields)
        and info["model_native_scalar_fields"] == list(scalar_fields)
        and info["model_native_scalar_contract"] == htf.MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4,
        "TF_ARRAY_HEADERS")
    warmup = htf._exact_cache_int(info["causal_warmup_rows"],
        label=tf + ".warmup", minimum=0)
    scalar_warmup = htf._exact_cache_int(info["model_native_scalar_warmup_rows"],
        label=tf + ".scalar_warmup", minimum=0)
    first = htf._exact_cache_int(info["first_ts_ns"], label=tf + ".first", minimum=0)
    last = htf._exact_cache_int(info["last_ts_ns"], label=tf + ".last", minimum=0)
    _require(first == int(times[0]) and last == int(times[-1])
        and warmup < rows and scalar_warmup <= rows
        and liveness["rows"] == rows and liveness["warmup_rows"] == warmup
        and isinstance(population, dict) and population.get("source_row_count") == rows
        and population.get("source_warmup_rows") == warmup, "TF_ENDPOINT_WARMUP_JOIN")
    surface = prepare_direction_mtf_surface(timeframe=tf, timestamps_ns=times,
        features=features, warmup_rows=warmup, sequence_bars=sequence_bars)
    return surface, arrays, {"rows": rows, "feature_count": width,
        "scalar_count": scalar_count, "first_ts_ns": first, "last_ts_ns": last,
        "warmup_rows": warmup, "scalar_warmup_rows": scalar_warmup,
        "sequence_bars": sequence_bars}


def load_direction_snapshot_cache(*, normalization_binding, prior_result_binding,
        prior_terminal_binding, full_input_liveness_binding, historical_htf_source,
        max_metadata_bytes, max_manifest_bytes):
    """Load only an exact prior-qualified cache; no liveness fit or model call.

    Each artifact binding is {path, sha256}; historical_htf_source is
    {source_head, sha256} for the fixed HTF source owner. Caps are mandatory
    per-file byte limits. No hash memo survives across invocations.
    """
    _require(type(max_metadata_bytes) is int and max_metadata_bytes > 0
        and type(max_manifest_bytes) is int and max_manifest_bytes > 0, "BYTE_CAPS")
    refs = [_binding(value) for value in (normalization_binding, prior_result_binding,
        prior_terminal_binding, full_input_liveness_binding)]
    bookends = {}
    source_proof = _require_source_match(historical_htf_source, bookends)
    normalization, result, terminal, full = [
        _read_json(ref, cap=max_metadata_bytes, bookends=bookends) for ref in refs]
    contract, liveness, cache_binding = _prior_chain(normalization, result, terminal,
        full, normalization_binding=refs[0], prior_result_binding=refs[1],
        full_input_liveness_binding=refs[3], source_head=source_proof["source_head"])
    lineage = contract["lineage"]
    manifest_binding = _binding({"path": lineage.get("mtf_cache_manifest_path"),
        "sha256": lineage.get("mtf_cache_manifest_sha256")})
    manifest_path = Path(manifest_binding["path"])
    _require(manifest_path.name == "manifest.json", "MANIFEST_FILENAME")
    cache_dir = manifest_path.parent
    _remember(bookends, cache_dir, directory=True)
    manifest = _read_json(manifest_binding, cap=max_manifest_bytes, bookends=bookends)
    tfs = _manifest_headers(manifest, lineage=lineage, liveness=liveness,
        cache_binding=cache_binding, manifest_binding=manifest_binding)
    inventory = {"manifest.json"} | {
        f"{tf}_{kind}.npy" for tf in tfs
        for kind in ("feats", "ts", "model_native_scalars")}
    surfaces, arrays, headers = {}, [], {}
    populations = normalization.get("fit_population_proof", {}).get("mtf_populations", {})
    with _directory(cache_dir) as fd:
        _require(set(os.listdir(fd)) == inventory, "CACHE_INVENTORY")
        for name in sorted(inventory):
            _remember(bookends, cache_dir / name)
        for tf in tfs:
            surface, mapped, header = _load_timeframe(fd, tf, manifest["tfs"][tf],
                liveness=liveness["timeframes"][tf], population=populations.get(tf),
                sequence_bars=lineage["per_tf_seq_lens"][tf])
            surfaces[tf], headers[tf] = surface, header
            arrays.extend(mapped)
        _require(set(os.listdir(fd)) == inventory, "CACHE_INVENTORY_CHANGED")
    cache = {"surfaces": surfaces, "normalization_contract": contract,
        "_disk_backing_arrays": tuple(arrays),
        "proof": {"schema_version": SCHEMA_VERSION, "mode": PROOF_MODE,
            "normalization": refs[0], "prior_result": refs[1], "prior_terminal": refs[2],
            "full_input_liveness": refs[3], "htf_source": source_proof,
            "manifest": manifest_binding,
            "cache_identity_sha256": manifest["cache_identity_sha256"],
            "liveness_contract_sha256": liveness["contract_sha256"],
            "fresh_verified_npy_files": len(arrays), "inventory": sorted(inventory),
            "timeframes": headers, "bookends": bookends,
            "reused_checks": ["full_matrix_liveness_and_categorical_validation",
                "squeeze_external_artifact_qualification",
                "registry_external_provenance", "m5_source_file_qualification"],
            "fresh_selected_rows_still_required": True,
            "native_loader_equivalence_claimed": False,
            "current_capacity_qualified": False}}
    require_direction_snapshot_cache_preserved(cache)
    return cache


def require_direction_snapshot_cache_preserved(cache):
    """Stat/inventory bookends for this invocation only; never a new SHA proof."""
    _require(isinstance(cache, Mapping) and isinstance(cache.get("proof"), Mapping)
        and cache["proof"].get("schema_version") == SCHEMA_VERSION
        and cache["proof"].get("mode") == PROOF_MODE, "CACHE_PROOF")
    proof = cache["proof"]
    for path, before in proof["bookends"].items():
        _require(_file_stat(Path(path), directory=before["directory"]) == before["stat"],
            "FILE_CHANGED")
    with _directory(Path(proof["manifest"]["path"]).parent) as fd:
        _require(set(os.listdir(fd)) == set(proof["inventory"]), "CACHE_INVENTORY_CHANGED")
    return proof
