from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import gx1.contracts.local_random_access_campaign_v2 as owner


def _binding(path: Path, data: bytes) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def physical_hash_calls(monkeypatch):
    """Count genuine content streams while retaining the complete real owner."""
    original = owner._file_sha256_uncached
    calls: list[Path] = []

    def traced(path, *, identity, scoped):
        calls.append(path)
        return original(path, identity=identity, scoped=scoped)

    monkeypatch.setattr(owner, "_file_sha256_uncached", traced)
    return calls


def test_duplicate_bound_reads_share_only_the_actual_digest(tmp_path, physical_hash_calls):
    path = tmp_path / "bound.json"
    data = b'{"value": 7}\n'
    path.write_bytes(data)
    binding = _binding(path, data)
    with owner.file_verification_scope():
        assert owner.require_binding(binding, label="first") == binding
        assert owner.read_bound_json(path, binding["sha256"]) == {"value": 7}
        assert owner.require_binding(binding, label="again") == binding
    assert physical_hash_calls == [path]


def test_expected_sha_is_checked_on_every_memo_hit(tmp_path, physical_hash_calls):
    path = tmp_path / "bound"
    data = b"actual bytes"
    path.write_bytes(data)
    binding = _binding(path, data)
    with owner.file_verification_scope():
        owner.require_binding(binding, label="valid")
        with pytest.raises(owner.RandomAccessCampaignError, match="SHA-256 mismatch"):
            owner.require_binding({**binding, "sha256": "0" * 64}, label="wrong")
        with pytest.raises(owner.RandomAccessCampaignError, match="SHA-256 mismatch"):
            owner.read_bound_json(path, "0" * 64)
        assert owner.require_binding(binding, label="still valid") == binding
    assert physical_hash_calls == [path]


def test_same_size_edit_with_restored_mtime_cannot_reuse_digest(tmp_path, physical_hash_calls):
    path = tmp_path / "bound"
    original, changed = b"before", b"AFTER!"
    path.write_bytes(original)
    before = path.stat()
    with owner.file_verification_scope():
        owner.require_binding(_binding(path, original), label="before")
        path.write_bytes(changed)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = path.stat()
        assert after.st_size == before.st_size
        assert after.st_mtime_ns == before.st_mtime_ns
        assert after.st_ctime_ns != before.st_ctime_ns
        with pytest.raises(owner.RandomAccessCampaignError, match="SHA-256 mismatch"):
            owner.require_binding(_binding(path, original), label="stale")
        assert owner.require_binding(_binding(path, changed), label="changed")
    assert physical_hash_calls == [path, path]


def test_inode_replacement_with_same_size_and_mtime_is_rehashed(tmp_path, physical_hash_calls):
    path = tmp_path / "bound"
    original, changed = b"before", b"AFTER!"
    path.write_bytes(original)
    before = path.stat()
    replacement = tmp_path / "replacement"
    replacement.write_bytes(changed)
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    with owner.file_verification_scope():
        owner.require_binding(_binding(path, original), label="before")
        replacement.replace(path)
        assert path.stat().st_ino != before.st_ino
        with pytest.raises(owner.RandomAccessCampaignError, match="SHA-256 mismatch"):
            owner.require_binding(_binding(path, original), label="replaced")
        assert owner.require_binding(_binding(path, changed), label="current")
    assert physical_hash_calls == [path, path]


def test_mode_change_rechecks_file_identity(tmp_path, physical_hash_calls):
    path = tmp_path / "bound"
    data = b"same content"
    path.write_bytes(data)
    path.chmod(0o600)
    with owner.file_verification_scope():
        owner.require_binding(_binding(path, data), label="private")
        path.chmod(0o640)
        owner.require_binding(_binding(path, data), label="group readable")
    assert physical_hash_calls == [path, path]


@pytest.mark.parametrize("direct_hash", [False, True])
def test_symlink_replacement_cannot_use_cached_regular_file(tmp_path, physical_hash_calls, direct_hash):
    path, target = tmp_path / "bound", tmp_path / "target"
    data = b"same content"
    path.write_bytes(data)
    target.write_bytes(data)
    with owner.file_verification_scope():
        owner.require_binding(_binding(path, data), label="before")
        path.unlink()
        path.symlink_to(target)
        with pytest.raises(owner.RandomAccessCampaignError, match="absolute normalized|regular file"):
            if direct_hash:
                owner.file_sha256(path)
            else:
                owner.require_binding(_binding(path, data), label="symlink")
    assert physical_hash_calls == [path]


def test_noncanonical_parent_symlink_is_rejected_in_scope(tmp_path, physical_hash_calls):
    directory = tmp_path / "real"
    directory.mkdir()
    path = directory / "bound"
    data = b"same content"
    path.write_bytes(data)
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    with owner.file_verification_scope():
        owner.require_binding(_binding(path, data), label="canonical")
        with pytest.raises(owner.RandomAccessCampaignError, match="absolute normalized"):
            owner.file_sha256(alias / "bound")
    assert physical_hash_calls == [path]


@pytest.mark.parametrize("mutation", ["content", "inode"])
@pytest.mark.parametrize("scoped", [False, True])
def test_mutation_during_first_read_is_rejected_and_not_cached(
    tmp_path, monkeypatch, physical_hash_calls, mutation, scoped
):
    path = tmp_path / "bound"
    original, changed = b"before", b"AFTER!"
    path.write_bytes(original)
    before = path.stat()
    real_open = Path.open
    fired = False

    class MutatingReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            nonlocal fired
            block = self.handle.read(size)
            if block and not fired:
                fired = True
                if mutation == "content":
                    path.write_bytes(changed)
                    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                else:
                    replacement = tmp_path / "replacement"
                    replacement.write_bytes(changed)
                    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
                    replacement.replace(path)
            return block

    def mutating_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if self == path and args and args[0] == "rb":
            return MutatingReader(handle)
        return handle

    monkeypatch.setattr(Path, "open", mutating_open)

    def check():
        with pytest.raises(owner.RandomAccessCampaignError, match="file changed"):
            owner.file_sha256(path)
        assert fired
        assert owner.file_sha256(path) == hashlib.sha256(changed).hexdigest()

    if scoped:
        with owner.file_verification_scope():
            check()
    else:
        check()
    assert physical_hash_calls == [path, path]


def test_nested_scope_reuses_digest_but_next_root_and_unscoped_calls_do_not(
    tmp_path, physical_hash_calls
):
    path = tmp_path / "bound"
    data = b"same content"
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    with owner.file_verification_scope():
        assert owner.file_sha256(path) == expected
        with owner.file_verification_scope():
            assert owner.file_sha256(path) == expected
        assert owner.file_sha256(path) == expected
        assert physical_hash_calls == [path]
    with owner.file_verification_scope():
        assert owner.file_sha256(path) == expected
    assert physical_hash_calls == [path, path]
    assert owner.file_sha256(path) == expected
    assert owner.file_sha256(path) == expected
    assert physical_hash_calls == [path] * 4


def test_root_and_nested_exceptions_do_not_leak_scope(tmp_path, physical_hash_calls):
    path = tmp_path / "bound"
    data = b"same content"
    path.write_bytes(data)
    with pytest.raises(RuntimeError, match="root failure"):
        with owner.file_verification_scope():
            owner.file_sha256(path)
            with pytest.raises(ValueError, match="nested failure"):
                with owner.file_verification_scope():
                    owner.file_sha256(path)
                    raise ValueError("nested failure")
            owner.file_sha256(path)
            assert physical_hash_calls == [path]
            raise RuntimeError("root failure")
    with owner.file_verification_scope():
        owner.file_sha256(path)
    owner.file_sha256(path)
    assert physical_hash_calls == [path] * 3


def test_unscoped_relative_and_symlink_paths_preserve_public_behavior(
    tmp_path, monkeypatch, physical_hash_calls
):
    path = tmp_path / "bound"
    data = b"same content"
    path.write_bytes(data)
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    monkeypatch.chdir(tmp_path)
    expected = hashlib.sha256(data).hexdigest()
    assert owner.file_sha256(Path("bound")) == expected
    assert owner.file_sha256(alias) == expected
    assert len(physical_hash_calls) == 2
