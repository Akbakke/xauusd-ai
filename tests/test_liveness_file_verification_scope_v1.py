from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path

import pytest

from gx1.contracts import entry_full_input_liveness_v1 as liveness
from gx1.contracts import local_random_access_campaign_v2 as owner
from tests.entry_full_input_liveness_support import write_full_input_liveness_fixture
from tests.test_scoped_binding_file_hash_v1 import physical_hash_calls


def test_liveness_hash_wrapper_preserves_unscoped_path_and_error_behavior(
    tmp_path, monkeypatch, physical_hash_calls,
):
    path = tmp_path / "payload"
    data = b"genuine small fixture bytes"
    path.write_bytes(data)
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    monkeypatch.chdir(tmp_path)
    expected = hashlib.sha256(data).hexdigest()

    assert owner._FILE_VERIFICATION_MEMO.get() is None
    for candidate in (path, Path("payload"), alias):
        assert liveness.sha256_file(candidate) == expected
    assert physical_hash_calls == [path, Path("payload"), alias]
    with pytest.raises(FileNotFoundError):
        liveness.sha256_file(tmp_path / "missing")
    with pytest.raises(IsADirectoryError):
        liveness.sha256_file(tmp_path)
    assert owner._FILE_VERIFICATION_MEMO.get() is None


def test_real_liveness_validation_reuses_digests_without_reusing_semantic_admission(
    tmp_path, physical_hash_calls,
):
    path, artifact, manifests = write_full_input_liveness_fixture(tmp_path)
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    arguments = {"expected_sha256": expected_sha}
    baseline = liveness.validate_full_input_liveness_artifact(path, **arguments)
    assert baseline["ok"] is True
    direct_paths = {
        path,
        Path(artifact["multi_tf_liveness"]["cache_binding"]["manifest_path"]),
        *(Path(manifests[split]["path"]) for split in ("train", "val")),
        *(Path(artifact["input_bindings"]["fullscan_proof"][split]["parquet_path"])
          for split in ("train", "val")),
    }
    assert len(direct_paths) == 6
    physical_hash_calls.clear()
    parquet = Path(artifact["input_bindings"]["fullscan_proof"]["train"]["parquet_path"])

    with owner.file_verification_scope():
        assert liveness.validate_full_input_liveness_artifact(path, **arguments) == baseline
        assert liveness.validate_full_input_liveness_artifact(path, **arguments) == baseline
        assert all(Counter(physical_hash_calls)[item] == 1 for item in direct_paths)

        wrong_expected = liveness.validate_full_input_liveness_artifact(path, expected_sha256="0" * 64)
        assert wrong_expected["ok"] is False
        assert any(row["code"] == "artifact_sha256_mismatch" for row in wrong_expected["failures"])
        assert all(Counter(physical_hash_calls)[item] == 1 for item in direct_paths)

        before = parquet.stat()
        original = parquet.read_bytes()
        parquet.write_bytes(bytes(value ^ 1 for value in original))
        os.utime(parquet, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert parquet.stat().st_size == before.st_size
        assert parquet.stat().st_mtime_ns == before.st_mtime_ns
        changed = liveness.validate_full_input_liveness_artifact(path, **arguments)
        assert changed["ok"] is False
        failure = next(row for row in changed["failures"]
            if row["code"] == "fullscan_proof_invalid" and row.get("split") == "train")
        assert failure["recorded_parquet_sha256"] != failure["observed_parquet_sha256"]
        counts = Counter(physical_hash_calls)
        assert counts[parquet] == 2
        assert all(counts[item] == 1 for item in direct_paths - {parquet})

    with owner.file_verification_scope():
        assert liveness.validate_full_input_liveness_artifact(path, **arguments)["ok"] is False
    counts = Counter(physical_hash_calls)
    assert counts[parquet] == 3
    assert all(counts[item] == 2 for item in direct_paths - {parquet})
    assert liveness.sha256_file(path) == expected_sha
    assert Counter(physical_hash_calls)[path] == 3
