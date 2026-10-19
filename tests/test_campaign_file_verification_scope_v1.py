from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pytest

import gx1.contracts.local_random_access_campaign_v2 as owner
import gx1.scripts.local_random_access_campaign_v2 as cli
import gx1.scripts.materialize_local_random_access_campaign_v2 as materializer
from tests.test_local_random_access_campaign_v2 import _boot, _gpu_plan, _write


@pytest.fixture
def physical_hash_calls(monkeypatch):
    """Count actual byte streams, retaining the real file and SHA checks."""
    calls: Counter[Path] = Counter()
    original = owner._file_sha256_uncached

    def traced(path, *, identity, scoped):
        calls[path] += 1
        return original(path, identity=identity, scoped=scoped)

    monkeypatch.setattr(owner, "_file_sha256_uncached", traced)
    return calls


def _inspect_arguments(tmp_path, monkeypatch):
    plan, path, *_ = _gpu_plan(tmp_path)
    boot_path = tmp_path / "CURRENT_BOOT.json"
    _write(boot_path, _boot(101, 1))
    # As in the existing running-CLI test, bind the fixture's actual source file.
    monkeypatch.setattr(cli, "__file__", plan["controller_sources"]["campaign_cli"]["path"])
    arguments = [
        "inspect", "--plan-json", str(path), "--plan-file-sha256",
        hashlib.sha256(path.read_bytes()).hexdigest(), "--boot-json", str(boot_path),
    ]
    return plan, path, arguments


def test_inspect_cli_reuses_hashes_across_real_plan_binding_chain(
    tmp_path, monkeypatch, capsys, physical_hash_calls
):
    plan, path, arguments = _inspect_arguments(tmp_path, monkeypatch)
    # Fixture construction is outside the operation being measured.
    physical_hash_calls.clear()
    for operation in (1, 2):
        assert cli.main(arguments) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["ok"] is True
        assert result["action"]["decision"] == "LAUNCH"
        assert result["cuda_started"] is False
        assert result["test_authorized"] is False
        assert result["plan_sha256"] == plan["plan_sha256"]
        # _require_running_cli and _load_plan both verify the actual plan bytes.
        assert physical_hash_calls[path] == operation
        # Three genuine invocation/manifest chains share this physical input.
        assert physical_hash_calls[tmp_path / "PRELAUNCH.json"] == operation
        assert physical_hash_calls[Path(plan["signed_guard_sources"]["certificate"]["path"])] == operation


def test_inspect_cli_failed_sha_check_does_not_leak_scope(
    tmp_path, monkeypatch, capsys, physical_hash_calls
):
    _, path, arguments = _inspect_arguments(tmp_path, monkeypatch)
    physical_hash_calls.clear()
    wrong = list(arguments)
    wrong[wrong.index("--plan-file-sha256") + 1] = "0" * 64
    assert cli.main(wrong) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert "SHA-256 mismatch" in result["error"]
    assert physical_hash_calls[path] == 1
    assert cli.main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert physical_hash_calls[path] == 2


class _SourceBoundaryStop(RuntimeError):
    pass


def _invoke_native(tmp_path, *, entrypoint):
    inputs = dict(
        repo=tmp_path / "repo", output=tmp_path / "output", runtime=tmp_path / "runtime",
        gpu_uuid="fixture-gpu", prepared_boot_path=tmp_path / "BOOT.json",
        prepared_boot_file_sha256="0" * 64, certificate_path=tmp_path / "CERTIFICATE.json",
        recipe_path=tmp_path / "RECIPE.json", recipe_file_sha256="0" * 64, window_count=1,
    )
    if entrypoint == "direct":
        return materializer.materialize_native_candidate_campaign(**inputs)
    return materializer.main([
        "--phase", "native-candidate", "--source-repo", str(inputs["repo"]),
        "--output-root", str(inputs["output"]), "--runtime-root", str(inputs["runtime"]),
        "--gpu-uuid", inputs["gpu_uuid"], "--prepared-boot", str(inputs["prepared_boot_path"]),
        "--prepared-boot-file-sha256", inputs["prepared_boot_file_sha256"],
        "--telemetry-certificate", str(inputs["certificate_path"]),
        "--native-recipe", str(inputs["recipe_path"]),
        "--native-recipe-file-sha256", inputs["recipe_file_sha256"],
        "--native-window-count", "1",
    ])


@pytest.mark.parametrize("entrypoint", ["direct", "cli"])
def test_native_materialization_scope_covers_nested_real_binding_reads_and_exception(
    tmp_path, monkeypatch, physical_hash_calls, entrypoint
):
    path = tmp_path / "BOUND.json"
    data = b'{"source": "actual synthetic bytes"}\n'
    path.write_bytes(data)
    binding = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}

    def source_boundary(repo):
        # Isolate before recipe/admission/publication while exercising real owners.
        assert repo == tmp_path / "repo"
        assert owner.require_binding(binding, label="source") == binding
        assert owner.read_bound_json(path, binding["sha256"]) == {"source": "actual synthetic bytes"}
        assert owner.require_binding(binding, label="nested source") == binding
        raise _SourceBoundaryStop("before native admission")

    monkeypatch.setattr(materializer, "_source_commit", source_boundary)
    for operation in (1, 2):
        with pytest.raises(_SourceBoundaryStop, match="before native admission"):
            _invoke_native(tmp_path, entrypoint=entrypoint)
        assert physical_hash_calls[path] == operation
        assert not (tmp_path / "output").exists()
        assert not (tmp_path / "runtime").exists()
    owner.require_binding(binding, label="outside completed operations")
    assert physical_hash_calls[path] == 3


@pytest.mark.parametrize("entrypoint", ["direct", "cli"])
@pytest.mark.parametrize("fault", ["wrong_sha", "replaced_inode"])
def test_native_materialization_rechecks_expected_sha_and_file_identity(
    tmp_path, monkeypatch, physical_hash_calls, entrypoint, fault
):
    path = tmp_path / "BOUND.json"
    before_data, after_data = b'{"value": 1}', b'{"value": 2}'
    path.write_bytes(before_data)
    binding = {"path": str(path), "sha256": hashlib.sha256(before_data).hexdigest()}

    def source_boundary(repo):
        assert owner.require_binding(binding, label="first source") == binding
        if fault == "wrong_sha":
            owner.require_binding({**binding, "sha256": "0" * 64}, label="wrong source")
        else:
            before = path.stat()
            replacement = tmp_path / "replacement"
            replacement.write_bytes(after_data)
            os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
            replacement.replace(path)
            assert path.stat().st_size == before.st_size
            assert path.stat().st_mtime_ns == before.st_mtime_ns
            assert path.stat().st_ino != before.st_ino
            owner.require_binding(binding, label="stale source")
        pytest.fail("invalid source binding must fail before native admission")

    monkeypatch.setattr(materializer, "_source_commit", source_boundary)
    with pytest.raises(owner.RandomAccessCampaignError, match="SHA-256 mismatch"):
        _invoke_native(tmp_path, entrypoint=entrypoint)
    expected_reads = 1 if fault == "wrong_sha" else 2
    assert physical_hash_calls[path] == expected_reads
    actual_data = before_data if fault == "wrong_sha" else after_data
    current = {"path": str(path), "sha256": hashlib.sha256(actual_data).hexdigest()}
    assert owner.require_binding(current, label="after failed operation") == current
    assert physical_hash_calls[path] == expected_reads + 1
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "runtime").exists()
