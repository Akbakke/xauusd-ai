"""Generated immutable-proof fixtures only; no production cache or data read.

The prior metadata below models an already accepted proof. These tests verify
the reuse protocol and fresh NPY/header checks, not the truth of a new liveness
computation or predictive value.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from gx1.contracts import entry_direction_snapshot_cache_v1 as cache
from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS

htf = cache.htf
HEAD = "a" * 40
CAP = 8 * 1024 * 1024


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    return {"path": str(path), "sha256": _sha(path.read_bytes())}


def _prior_liveness():
    timeframes = {}
    for tf in htf.MULTI_TF_RESAMPLE_RULES:
        fields = {}
        for field in htf.MULTI_TF_PER_BAR_FEATURES_V4:
            constant = htf.MULTI_TF_SOURCE_CONSTANT_BINARY_EVENTS_V4.get(tf, {}).get(field)
            fields[field] = {"unique_count": 2, "mean": 0.5, "std": 0.5,
                "minimum": 0.0, "maximum": 1.0, "nonzero_fraction": 0.5,
                "values_sha256": _sha((tf + ":" + field).encode())}
            if constant is not None:
                fields[field].update(unique_count=1, mean=constant, std=0.0,
                    minimum=constant, maximum=constant,
                    nonzero_fraction=float(constant != 0.0))
        timeframes[tf] = {"rows": 4, "warmup_rows": 0, "live_rows": 4,
            "constant_fields": [], "exact_duplicate_pairs": [], "fields": fields}
    proof = {"schema_version": htf.HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION,
        "matrix_contract": htf.HTF_V4_MATRIX_CONTRACT,
        "feature_names_sha256": htf.MULTI_TF_FEATURE_NAMES_SHA256_V4,
        "timeframe_order": list(htf.MULTI_TF_RESAMPLE_RULES),
        "decision": "PASS", "failures": [], "timeframes": timeframes}
    proof["contract_sha256"] = _sha(json.dumps(proof, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode())
    return proof


def _fixture(tmp_path, monkeypatch, *, manifest_change=None, full_change=None,
        normalization_change=None, result_change=None, terminal_change=None,
        array_change=None):
    owner = tmp_path / "gx1" / "features" / "htf_features.py"
    owner.parent.mkdir(parents=True)
    owner.write_bytes(b"generated historical HTF owner\n")
    monkeypatch.setattr(htf, "__file__", str(owner))

    def git_show(args, *, stderr):
        assert args == ["git", "-C", str(tmp_path), "show",
            HEAD + ":gx1/features/htf_features.py"]
        return owner.read_bytes()

    monkeypatch.setattr(cache.subprocess, "check_output", git_show)

    def forbidden(*args, **kwargs):
        raise AssertionError("fresh native/liveness/external-artifact load forbidden")

    for name in ("load_multi_tf_v4_cache", "build_multi_tf_v4_liveness_contract",
            "validate_causal_feature_matrix", "require_model_native_mtf_scalar_owner_v4",
            "require_volatility_squeeze_artifact_binding", "require_v29_registry_constants"):
        monkeypatch.setattr(htf, name, forbidden)
    directory = tmp_path / "cache"
    directory.mkdir()
    tfs = {}
    for tf in htf.MULTI_TF_RESAMPLE_RULES:
        start = htf.multi_tf_bar_label(htf.pd.Timestamp("2024-01-08T00:00:00Z"), tf).value
        times = start + np.arange(4, dtype=np.int64) * htf.MULTI_TF_SHIFT[tf].value
        fields = htf.MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[tf]
        arrays = {"feats": np.arange(4 * htf.MULTI_TF_FEATURE_COUNT_V4,
                dtype=np.float32).reshape(4, htf.MULTI_TF_FEATURE_COUNT_V4),
            "ts": times, "model_native_scalars": np.ones((4, len(fields)), dtype=np.float32)}
        if array_change:
            array_change(tf, arrays)
        info = {"n_bars": 4, "feature_count": htf.MULTI_TF_FEATURE_COUNT_V4,
            "first_ts_ns": int(arrays["ts"][0]), "last_ts_ns": int(arrays["ts"][-1]),
            "causal_warmup_rows": 0, "model_native_scalar_fields": list(fields),
            "model_native_scalar_count": len(fields), "model_native_scalar_warmup_rows": 0,
            "model_native_scalar_contract": htf.MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4}
        for kind, array in arrays.items():
            path = directory / f"{tf}_{kind}.npy"
            np.save(path, array, allow_pickle=False)
            info.update({kind + "_npy": path.name,
                kind + "_npy_sha256": _sha(path.read_bytes()),
                kind + "_npy_size_bytes": path.stat().st_size})
        tfs[tf] = info
    liveness = _prior_liveness()
    manifest = {"schema_version": htf.HTF_V4_CACHE_SCHEMA_VERSION,
        "builder_version": htf.HTF_V4_CACHE_BUILDER_VERSION,
        "smc_causal_replay_schema_version": htf.SMC_CAUSAL_REPLAY_SCHEMA_VERSION,
        "technical_indicator_owner": htf.technical_indicator_contract_metadata(),
        "native_primitive_owner": htf.native_mtf_full_primitive_contract_metadata_v4(),
        "swing_structure_owner": htf.swing_structure_contract_metadata(),
        "feature_count": htf.MULTI_TF_FEATURE_COUNT_V4,
        "feature_names": list(htf.MULTI_TF_PER_BAR_FEATURES_V4),
        "shift_contract": {tf: str(value) for tf, value in htf.MULTI_TF_SHIFT.items()},
        "resample_origin_contract": dict(htf.MULTI_TF_RESAMPLE_ORIGIN_CONTRACT),
        "m5_prebuilt_source": str(tmp_path / "never_opened.parquet"),
        "m5_prebuilt_source_sha256": "b" * 64,
        "v29_registry_constants": {"fixture": "prior-qualified opaque binding"},
        "volatility_squeeze_artifact_set": {"fixture": "prior-qualified opaque binding"},
        "full_input_liveness": liveness, "tfs": tfs}
    if manifest_change:
        manifest_change(manifest)
    manifest["cache_identity_sha256"] = htf.compute_htf_v4_cache_identity(manifest)
    manifest_ref = _write(directory / "manifest.json", manifest)
    full = {"schema_version": "entry_full_input_liveness_contract_v13",
        "decision": "PASS", "failures": [],
        "multi_tf_liveness": {"contract": liveness, "cache_binding": {
            "cache_identity_sha256": manifest["cache_identity_sha256"],
            "manifest_path": manifest_ref["path"],
            "manifest_sha256": manifest_ref["sha256"],
            "observed_manifest_sha256": manifest_ref["sha256"]}}}
    if full_change:
        full_change(full)
    full_ref = _write(tmp_path / "FULL_INPUT.json", full)
    lineage = {"m5_prebuilt_path": manifest["m5_prebuilt_source"],
        "m5_prebuilt_sha256": manifest["m5_prebuilt_source_sha256"],
        "mtf_cache_manifest_path": manifest_ref["path"],
        "mtf_cache_manifest_sha256": manifest_ref["sha256"],
        "mtf_builder_version": htf.HTF_V4_CACHE_BUILDER_VERSION,
        "mtf_feature_names_sha256": htf.MULTI_TF_FEATURE_NAMES_SHA256_V4,
        "per_tf_seq_lens": dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
        "per_tf_shift_seconds": {
            tf: int(value.value // 10**9) for tf, value in htf.MULTI_TF_SHIFT.items()}}
    normalization = {"schema_version": "gx1_entry_only_dynamic_m5_base_normalization_v1",
        "decision": "PASS", "test_accessed": False, "val_fit_rows": 0, "test_fit_rows": 0,
        "contract": {"lineage": lineage}, "contract_sha256": "c" * 64,
        "full_input_liveness": full_ref, "fit_population_proof": {"mtf_populations": {
            tf: {"source_row_count": 4, "source_warmup_rows": 0}
            for tf in htf.MULTI_TF_RESAMPLE_RULES}}}
    if normalization_change:
        normalization_change(normalization)
    normal_ref = _write(tmp_path / "NORMALIZATION.json", normalization)
    result = {"schema_version": "gx1_whole_train_dynamic_m5_normalization_result_v1",
        "decision": "PASS_WHOLE_TRAIN_DYNAMIC_M5_NORMALIZATION", "source_head": HEAD,
        "normalization": normal_ref, "contract_sha256": normalization["contract_sha256"],
        "market_normalization_fit_complete": True, "test_data_used": False,
        "val_fit_rows": 0, "test_fit_rows": 0, "full_input_liveness": full_ref}
    if result_change:
        result_change(result)
    result_ref = _write(tmp_path / "RESULT.json", result)
    terminal = {"status": "COMPLETE", "exit_code": 0, "source_unchanged": True,
        "source_head": HEAD, "result_sha256": result_ref["sha256"],
        "normalization_binding": normal_ref, "market_normalization_fit_complete": True,
        "test_data_used": False}
    if terminal_change:
        terminal_change(terminal)
    terminal_ref = _write(tmp_path / "TERMINAL.json", terminal)
    return {"normalization_binding": normal_ref, "prior_result_binding": result_ref,
        "prior_terminal_binding": terminal_ref, "full_input_liveness_binding": full_ref,
        "historical_htf_source": {"source_head": HEAD, "sha256": _sha(owner.read_bytes())},
        "max_metadata_bytes": CAP, "max_manifest_bytes": CAP}


def test_prior_proof_reuse_maps_all_fifteen_and_preserves_readonly_views(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    loaded = cache.load_direction_snapshot_cache(**args)
    proof = cache.require_direction_snapshot_cache_preserved(loaded)
    assert proof["fresh_verified_npy_files"] == 15
    assert len(proof["inventory"]) == 16
    assert proof["mode"] == cache.PROOF_MODE
    assert proof["fresh_selected_rows_still_required"] is True
    assert proof["native_loader_equivalence_claimed"] is False
    assert proof["current_capacity_qualified"] is False
    assert "squeeze_external_artifact_qualification" in proof["reused_checks"]
    assert loaded["normalization_contract"]["lineage"]["mtf_cache_manifest_path"] == proof["manifest"]["path"]
    for tf, surface in loaded["surfaces"].items():
        assert not surface["features"].flags.writeable
        assert not surface["times"].flags.writeable
        feature_map = loaded["_disk_backing_arrays"][list(loaded["surfaces"]).index(tf) * 3]
        assert isinstance(feature_map, np.memmap)
        assert np.shares_memory(surface["features"], feature_map)


@pytest.mark.parametrize("change", [
    {"result_change": lambda d: d.update(decision="FAIL")},
    {"result_change": lambda d: d.update(normalization={"path": "/wrong", "sha256": "0" * 64})},
    {"terminal_change": lambda d: d.update(exit_code=True)},
    {"terminal_change": lambda d: d.update(source_unchanged=False)},
    {"terminal_change": lambda d: d.update(result_sha256="0" * 64)},
    {"full_change": lambda d: d.update(decision="FAIL")},
    {"full_change": lambda d: d["multi_tf_liveness"]["cache_binding"].update(
        cache_identity_sha256="0" * 64)},
    {"normalization_change": lambda d: d.update(full_input_liveness={"path": "/wrong", "sha256": "0" * 64})},
    {"normalization_change": lambda d: d["fit_population_proof"]["mtf_populations"]["H1"].update(source_warmup_rows=1)},
])
def test_prior_chain_must_match_exact_bound_pass(tmp_path, monkeypatch, change):
    args = _fixture(tmp_path, monkeypatch, **change)
    with pytest.raises(RuntimeError):
        cache.load_direction_snapshot_cache(**args)


@pytest.mark.parametrize("change", [
    lambda d: d["feature_names"].reverse(),
    lambda d: d["tfs"]["M5"].update(causal_warmup_rows=1),
    lambda d: d["tfs"]["M5"].update(model_native_scalar_contract="wrong"),
    lambda d: d["tfs"]["M5"].update(first_ts_ns=0),
    lambda d: d["tfs"]["M5"].update(feats_npy="../M5_feats.npy"),
])
def test_prior_reuse_keeps_strict_current_manifest_and_warmup_guards(tmp_path, monkeypatch, change):
    args = _fixture(tmp_path, monkeypatch, manifest_change=change)
    with pytest.raises(RuntimeError):
        cache.load_direction_snapshot_cache(**args)


@pytest.mark.parametrize("kind", ["dtype", "shape", "grid", "order"])
def test_fresh_npy_headers_and_clock_are_checked(tmp_path, monkeypatch, kind):
    def change(tf, arrays):
        if tf != "M5":
            return
        if kind == "dtype":
            arrays["feats"] = arrays["feats"].astype(np.float64)
        elif kind == "shape":
            arrays["feats"] = arrays["feats"][:, :-1]
        elif kind == "grid":
            arrays["ts"] = arrays["ts"] + 1
        else:
            arrays["ts"] = arrays["ts"][::-1].copy()
    args = _fixture(tmp_path, monkeypatch, array_change=change)
    with pytest.raises(RuntimeError):
        cache.load_direction_snapshot_cache(**args)


@pytest.mark.parametrize("kind", ["tampered_bytes", "extra_file", "symlink"])
def test_fresh_file_bytes_and_exact_inventory_are_required(tmp_path, monkeypatch, kind):
    args = _fixture(tmp_path, monkeypatch)
    path = tmp_path / "cache" / "M5_feats.npy"
    if kind == "tampered_bytes":
        with path.open("ab") as stream:
            stream.write(b"x")
    elif kind == "extra_file":
        (path.parent / "unexpected.npy").write_bytes(b"x")
    else:
        moved = tmp_path / "moved.npy"
        path.rename(moved)
        path.symlink_to(moved)
    with pytest.raises(RuntimeError):
        cache.load_direction_snapshot_cache(**args)


@pytest.mark.parametrize("kind", ["current_hash", "historical_bytes"])
def test_metadata_bytes_and_historical_source_are_not_stat_only(tmp_path, monkeypatch, kind):
    args = _fixture(tmp_path, monkeypatch)
    if kind == "current_hash":
        args["historical_htf_source"]["sha256"] = "0" * 64
    else:
        monkeypatch.setattr(cache.subprocess, "check_output",
            lambda *args, **kwargs: b"different historical owner")
    with pytest.raises(RuntimeError, match="HTF_OWNER_CHANGED"):
        cache.load_direction_snapshot_cache(**args)
    args["historical_htf_source"]["sha256"] = _sha(Path(htf.__file__).read_bytes())
    monkeypatch.setattr(cache.subprocess, "check_output",
        lambda *args, **kwargs: Path(htf.__file__).read_bytes())
    Path(args["prior_result_binding"]["path"]).write_bytes(b"{}")
    with pytest.raises(RuntimeError):
        cache.load_direction_snapshot_cache(**args)


@pytest.mark.parametrize("cap", [True, 0, 1])
def test_metadata_caps_are_mandatory_before_parse(tmp_path, monkeypatch, cap):
    args = _fixture(tmp_path, monkeypatch)
    args["max_metadata_bytes"] = cap
    with pytest.raises(RuntimeError):
        cache.load_direction_snapshot_cache(**args)


def test_stat_bookend_rejects_later_change_in_same_invocation(tmp_path, monkeypatch):
    loaded = cache.load_direction_snapshot_cache(**_fixture(tmp_path, monkeypatch))
    path = tmp_path / "cache" / "H4_ts.npy"
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    with pytest.raises(RuntimeError, match="FILE_CHANGED"):
        cache.require_direction_snapshot_cache_preserved(loaded)
