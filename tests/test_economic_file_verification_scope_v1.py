"""Small-file verification integration, without importing model/training owners."""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path

import pytest

from gx1.contracts import local_random_access_campaign_v2 as files
from gx1.contracts import unified_exit_broker_evidence_v1 as broker


NO_CAP_SOURCE = Path(__file__).resolve().parents[1] / "gx1/contracts/unified_exit_no_cap_economic_authority_v1.py"


def _no_cap_file_sha256():
    # The module's legacy fitted-Q import reaches torch. Execute only the
    # actual hash wrapper definition, preserving its production delegation.
    tree = ast.parse(NO_CAP_SOURCE.read_text(), filename=str(NO_CAP_SOURCE))
    function = next(node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "file_sha256")
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace = {"Path": Path}
    exec(compile(ast.fix_missing_locations(isolated), str(NO_CAP_SOURCE), "exec"), namespace)
    return namespace["file_sha256"]


def _binding(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def raw_streams(monkeypatch):
    calls = []
    raw = files._file_sha256_uncached

    def counted(path, **kwargs):
        calls.append((path, kwargs["scoped"]))
        return raw(path, **kwargs)

    monkeypatch.setattr(files, "_file_sha256_uncached", counted)
    return calls


def test_economic_wrappers_deliver_actual_unscoped_hashes_to_common_owner(tmp_path, raw_streams):
    source = tmp_path / "cost-evidence.json"
    source.write_bytes(b'{"fixture_cost": 1}\n')
    binding = _binding(source)
    assert _no_cap_file_sha256()(source) == binding["sha256"]
    assert broker._require_local_file(binding, "FIXTURE", True) is None
    assert raw_streams == [(source, False), (source, False)]


def test_economic_wrappers_share_digest_only_inside_one_finite_scope(tmp_path, raw_streams):
    source = tmp_path / "cost-evidence.json"
    source.write_bytes(b'{"fixture_cost": 2}\n')
    binding = _binding(source)
    no_cap_hash = _no_cap_file_sha256()
    for operation in (1, 2):
        with files.file_verification_scope():
            assert no_cap_hash(source) == binding["sha256"]
            broker._require_local_file(binding, "FIXTURE", True)
            assert no_cap_hash(source) == binding["sha256"]
        assert len(raw_streams) == operation
    broker._require_local_file(binding, "FIXTURE", True)
    assert raw_streams == [(source, True), (source, True), (source, False)]


def test_broker_expected_sha_is_still_checked_on_shared_digest_hit(tmp_path, raw_streams):
    source = tmp_path / "cost-evidence.json"
    source.write_bytes(b'{"fixture_cost": 3}\n')
    binding = _binding(source)
    with files.file_verification_scope():
        assert _no_cap_file_sha256()(source) == binding["sha256"]
        with pytest.raises(RuntimeError, match="^UNIFIED_EXIT_BROKER_EVIDENCE_FIXTURE_SOURCE_HASH_MISMATCH$"):
            broker._require_local_file({**binding, "sha256": "0" * 64}, "FIXTURE", True)
        broker._require_local_file(binding, "FIXTURE", True)
    assert raw_streams == [(source, True)]


@pytest.mark.parametrize("mutation", ["content", "replacement_inode"])
def test_changed_cost_file_cannot_reuse_old_digest_or_expected_sha(tmp_path, raw_streams, mutation):
    source = tmp_path / "cost-evidence.json"
    source.write_bytes(b'{"fixture_cost": 4}\n')
    binding = _binding(source)
    no_cap_hash = _no_cap_file_sha256()
    with files.file_verification_scope():
        assert no_cap_hash(source) == binding["sha256"]
        before = source.stat()
        changed = b'{"fixture_cost": 5}\n'
        if mutation == "content":
            source.write_bytes(changed)
        else:
            replacement = tmp_path / "replacement.json"
            replacement.write_bytes(changed)
            replacement.replace(source)
            assert source.stat().st_ino != before.st_ino
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert source.stat().st_size == before.st_size
        assert source.stat().st_mtime_ns == before.st_mtime_ns
        with pytest.raises(RuntimeError, match="^UNIFIED_EXIT_BROKER_EVIDENCE_FIXTURE_SOURCE_HASH_MISMATCH$"):
            broker._require_local_file(binding, "FIXTURE", True)
        assert no_cap_hash(source) == hashlib.sha256(changed).hexdigest()
    assert raw_streams == [(source, True), (source, True)]


@pytest.mark.parametrize("kind", ["relative", "symlink"])
def test_no_cap_unscoped_relative_and_symlink_semantics_are_preserved(tmp_path, monkeypatch, raw_streams, kind):
    source = tmp_path / "source.bin"
    source.write_bytes(b"unscoped source bytes")
    monkeypatch.chdir(tmp_path)
    path = Path("source.bin")
    if kind == "symlink":
        path = tmp_path / "alias.bin"
        path.symlink_to(source)
    assert _no_cap_file_sha256()(path) == _binding(source)["sha256"]
    assert raw_streams == [(path, False)]


@pytest.mark.parametrize("kind", ["relative", "symlink", "directory", "path_object"])
def test_broker_keeps_local_file_path_and_type_guards(tmp_path, monkeypatch, raw_streams, kind):
    source = tmp_path / "source.bin"
    source.write_bytes(b"source bytes")
    binding = _binding(source)
    monkeypatch.chdir(tmp_path)
    if kind == "relative":
        binding["path"] = "source.bin"
    elif kind == "symlink":
        alias = tmp_path / "alias.bin"
        alias.symlink_to(source)
        binding["path"] = str(alias)
    elif kind == "directory":
        binding["path"] = str(tmp_path)
    else:
        binding["path"] = source
    with pytest.raises(RuntimeError, match="^UNIFIED_EXIT_BROKER_EVIDENCE_FIXTURE_SOURCE_INVALID$"):
        broker._require_local_file(binding, "FIXTURE", True)
    assert raw_streams == []


def test_broker_metadata_only_check_does_not_hash_missing_source(tmp_path, raw_streams):
    broker._require_local_file({"path": str(tmp_path / "missing.json"), "sha256": "1" * 64}, "FIXTURE", False)
    assert raw_streams == []


def test_verified_snapshot_includes_both_cost_wrappers_without_caching_model_work(tmp_path, raw_streams):
    no_cap_source, broker_source = tmp_path / "component.json", tmp_path / "broker-evidence.json"
    no_cap_source.write_bytes(b"component source")
    broker_source.write_bytes(b"broker evidence")
    with files.fresh_file_verification_scope():
        assert _no_cap_file_sha256()(no_cap_source) == _binding(no_cap_source)["sha256"]
        broker._require_local_file(_binding(broker_source), "FIXTURE", True)
        snapshot = files.capture_verified_file_snapshot()
    assert snapshot.file_count == 2
    snapshot.require_unchanged()
    assert raw_streams == [(no_cap_source, True), (broker_source, True)]
    # The snapshot retains identities, never a digest memo for later work.
    broker._require_local_file(_binding(broker_source), "FIXTURE", True)
    assert raw_streams[-1] == (broker_source, False)
    broker_source.write_bytes(b"changed evidence")
    with pytest.raises(files.RandomAccessCampaignError, match="verified immutable input identity changed"):
        snapshot.require_unchanged()
    assert len(raw_streams) == 3
