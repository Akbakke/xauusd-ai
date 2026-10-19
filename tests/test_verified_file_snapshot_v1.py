"""Physical identity proofs stay local to one completed metadata validation."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from gx1.contracts import local_random_access_campaign_v2 as files


def _binding(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def physical_reads(monkeypatch):
    calls = []
    original = files._file_sha256_uncached
    def counted(path, **kwargs):
        calls.append(str(path))
        return original(path, **kwargs)
    monkeypatch.setattr(files, "_file_sha256_uncached", counted)
    return calls


def test_verified_snapshot_captures_actual_hashes_and_never_becomes_a_digest_cache(tmp_path, physical_reads):
    source = tmp_path / "input.json"
    source.write_text('{"qualified":true}')
    binding = _binding(source)
    checkpoint = tmp_path / "mutable_checkpoint.pt"
    checkpoint.write_bytes(b"original mutable state")
    with files.fresh_file_verification_scope():
        for _ in range(2):
            assert files.require_binding(binding, label="immutable input") == binding
            assert files.read_bound_json(source, binding["sha256"]) == {"qualified": True}
        with files.file_verification_scope():
            files.require_binding(binding, label="nested input")
        snapshot = files.capture_verified_file_snapshot()
        assert snapshot.file_count == 1
        assert physical_reads == [str(source)]
        with pytest.raises(files.RandomAccessCampaignError, match="SHA-256 mismatch"):
            files.require_binding({**binding, "sha256": "0" * 64}, label="wrong expected SHA")
    checkpoint.write_bytes(b"new mutable state")
    snapshot.require_unchanged()
    # A snapshot is not consulted by ordinary future digest operations, even
    # when their file and expected SHA happen to equal the captured input.
    files.require_binding(binding, label="new unscoped operation")
    with files.fresh_file_verification_scope():
        files.require_binding(binding, label="fresh root")
    assert physical_reads == [str(source)] * 3
    with pytest.raises(files.RandomAccessCampaignError, match="nonempty active scope"):
        files.capture_verified_file_snapshot()


@pytest.mark.parametrize("fault", ["same_size_restored_mtime", "inode", "symlink", "parent_symlink", "mode", "delete"])
def test_verified_snapshot_rejects_changed_identity_without_refresh(tmp_path, physical_reads, fault):
    parent = tmp_path / "inputs"
    parent.mkdir()
    source = parent / "bound.npy"
    source.write_bytes(b"first immutable physical bytes")
    binding = _binding(source)
    with files.fresh_file_verification_scope():
        files.require_binding(binding, label="immutable input")
        snapshot = files.capture_verified_file_snapshot()
    before = source.stat()
    if fault == "same_size_restored_mtime":
        source.write_bytes(b"x" * before.st_size)
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert source.stat().st_size == before.st_size
        assert source.stat().st_mtime_ns == before.st_mtime_ns
        assert source.stat().st_ctime_ns != before.st_ctime_ns
    elif fault in {"inode", "symlink"}:
        replacement = parent / "replacement.npy"
        replacement.write_bytes(source.read_bytes())
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        if fault == "inode":
            os.replace(replacement, source)
        else:
            source.unlink()
            source.symlink_to(replacement)
    elif fault == "parent_symlink":
        moved = tmp_path / "moved"
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
    elif fault == "mode":
        source.chmod(before.st_mode ^ 0o100)
    else:
        source.unlink()
    for _ in range(2):
        with pytest.raises(files.RandomAccessCampaignError):
            snapshot.require_unchanged()
    assert physical_reads == [str(source)]  # no implicit refresh or rehash


def test_verified_snapshot_seals_after_audit_and_resets_failed_or_nested_roots(tmp_path, physical_reads):
    source = tmp_path / "physical.parquet"
    source.write_bytes(b"original pointer-proof bytes")
    binding = _binding(source)
    with pytest.raises(files.RandomAccessCampaignError, match="identity changed"):
        with files.fresh_file_verification_scope():
            files.require_binding(binding, label="before pointer audit")
            before = source.stat()
            source.write_bytes(b"x" * before.st_size)
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
            files.capture_verified_file_snapshot()
    current = _binding(source)
    with files.fresh_file_verification_scope():
        files.require_binding(current, label="new operation after failed capture")
        with pytest.raises(files.RandomAccessCampaignError, match="no outer scope"):
            with files.fresh_file_verification_scope():
                pytest.fail("nested fresh root must not enter")
        snapshot = files.capture_verified_file_snapshot()
    snapshot.require_unchanged()
    assert physical_reads == [str(source)] * 2
    with pytest.raises(LookupError):
        with files.fresh_file_verification_scope():
            files.require_binding(current, label="failed operation")
            raise LookupError("consumer failed")
    with files.fresh_file_verification_scope():
        files.require_binding(current, label="next operation")
    assert physical_reads == [str(source)] * 4
