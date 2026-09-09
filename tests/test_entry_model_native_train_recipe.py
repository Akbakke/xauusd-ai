from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

import pytest

from gx1.contracts import entry_model_native_train_launch_v1 as launch
from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV,
    MODEL_NATIVE_RECIPE_ENV_KEYS,
    MODEL_NATIVE_RECIPE_ENV_SHA256,
    SCHEMA_VERSION as MODEL_NATIVE_RECIPE_ENV_SCHEMA_VERSION,
    model_native_recipe_env_contract_metadata,
    require_model_native_recipe_env,
)
from gx1.scripts import (
    materialize_entry_model_native_seq513_train_recipe_audit_v1 as producer,
)
from tests.entry_model_native_train_wrapper_support import (
    DATASET_RUN_ID,
    build_wrapper_contract,
)


REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts/run_entry_model_native_seq513_train.sh"


def test_recipe_source_closure_binds_feature_usefulness_native_and_capped_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots_seen: list[Path] = []
    import_closure = launch._recipe_local_python_import_closure

    def capture_roots(*, repo, roots):
        roots_seen.extend(roots)
        return import_closure(repo=repo, roots=roots)

    monkeypatch.setattr(launch, "_recipe_local_python_import_closure", capture_roots)
    bindings = launch.recipe_source_binding_paths(repo=REPO, wrapper_path=WRAPPER)
    audit = REPO / "gx1/scripts/audit_entry_exit_feature_usefulness_v1.py"
    native = REPO / "gx1/scripts/entry_exit_feature_usefulness_native_v1.py"
    capped = REPO / "gx1/contracts/gx1_capped_execution_v1.py"

    assert audit in roots_seen
    assert native not in roots_seen
    assert capped not in roots_seen
    for path in (audit, native, capped):
        assert bindings[f"python:{path.relative_to(REPO).as_posix()}"] == path
    assert bindings["control_surface"] == REPO / "scripts/entry_next_edge_control.sh"
    assert bindings["capped_runner"] == REPO / "scripts/gx1_capped_run.sh"


def test_launch_artifact_binding_rehashes_same_stat_byte_mutation(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "large-looking.parquet"
    artifact.write_bytes(b"AAAA")
    expected = launch.sha256_file(artifact)
    prior = artifact.stat()
    artifact.write_bytes(b"BBBB")
    os.utime(artifact, ns=(prior.st_atime_ns, prior.st_mtime_ns))

    with pytest.raises(
        launch.LaunchContractError,
        match="artifact content hash mismatch before launch",
    ):
        launch.artifact_binding(artifact, content_sha256=expected)


def test_recipe_source_provenance_is_durable_and_hash_bound() -> None:
    bindings = {
        "python:gx1/__init__.py": {
            "path": "/tmp/gx1/__init__.py",
            "sha256": "a" * 64,
            "size_bytes": 1,
            "mtime_ns": 1,
            "device": 1,
            "inode": 1,
        }
    }
    provenance = {
        "schema_version": launch.TRAINING_RECIPE_SOURCE_PROVENANCE_SCHEMA,
        "recipe_audit_path": "/tmp/ENTRY_TRAIN_RECIPE_20260828T120000Z.json",
        "recipe_audit_sha256": "b" * 64,
        "source_commit": "c" * 40,
        "source_bindings": bindings,
        "source_bindings_sha256": launch.canonical_json_sha256(bindings),
    }
    assert launch.require_training_recipe_source_provenance_metadata(
        provenance,
        context="TEST",
    ) == provenance

    provenance["source_bindings_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="BINDINGS_SHA_INVALID"):
        launch.require_training_recipe_source_provenance_metadata(
            provenance,
            context="TEST",
        )


def test_recipe_env_is_one_exact_complete_value_source_contract() -> None:
    metadata = model_native_recipe_env_contract_metadata()

    # Wave C leaves only diagnostics/admission and optimization switches.
    # Registry operator parameters come only from immutable TRAIN-fit artifacts.
    from gx1.contracts.entry_model_native_train_recipe_v1 import (
        MODEL_NATIVE_STABILITY_DAMPER_RECIPE_ENV_KEYS,
        MODEL_NATIVE_V29_REGISTRY_RECIPE_ENV_KEYS,
        MODEL_NATIVE_WEIGHT_EMA_DECAY_DECLARED_VALUES,
        MODEL_NATIVE_WEIGHT_EMA_DECAY_DISABLED_VALUE,
        MODEL_NATIVE_WEIGHT_EMA_DECAY_EPOCH_HORIZON_VALUE,
    )

    # The key COUNT is owned and enforced by the recipe module itself
    # (``_EXPECTED_RECIPE_ENV_KEY_COUNT`` raises at import).  Restating it here
    # was a second origin that went stale the moment Wave C retired keys, so
    # the count is derived and what is asserted is the live invariant: the
    # published key tuple is exactly the declared mapping's key set, with no
    # duplicate, omission or extra.
    expected_count = len(MODEL_NATIVE_RECIPE_ENV)
    assert len(MODEL_NATIVE_RECIPE_ENV_KEYS) == expected_count
    assert set(MODEL_NATIVE_RECIPE_ENV_KEYS) == set(MODEL_NATIVE_RECIPE_ENV)
    assert list(MODEL_NATIVE_RECIPE_ENV_KEYS) == sorted(MODEL_NATIVE_RECIPE_ENV)
    assert "ENTRY_DIRECTION_LOGIT_ADJUST_TAU" not in MODEL_NATIVE_RECIPE_ENV
    assert "ENTRY_DIRECTION_CE_SCALE" not in MODEL_NATIVE_RECIPE_ENV
    assert "ENTRY_TAIL_DIRECTION_CE_WEIGHT" not in MODEL_NATIVE_RECIPE_ENV
    assert "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT" not in MODEL_NATIVE_RECIPE_ENV
    assert "ENTRY_HIER_SLICE_SIDE_CE_WEIGHT" not in MODEL_NATIVE_RECIPE_ENV
    assert "ENTRY_HIER_BAD_PATH_POS_WEIGHT_CAP" not in MODEL_NATIVE_RECIPE_ENV
    assert not any(key.endswith("_POS_WEIGHT_CAP") for key in MODEL_NATIVE_RECIPE_ENV)
    # V30 package 5. The cosine switch carries no magnitude at all (T_max is
    # the declared --epochs budget, eta_min the library default 0.0), so it is
    # adopted ON.  V30 package 6: the weight-EMA key declares a HORIZON, not a
    # magnitude — "epoch" selects the owner's derivation (one epoch of
    # optimizer steps), "0.0" is the OFF sentinel, and nothing else is a
    # declared value, so no bare decay can ever be pinned here.
    assert set(MODEL_NATIVE_STABILITY_DAMPER_RECIPE_ENV_KEYS) <= set(
        MODEL_NATIVE_RECIPE_ENV_KEYS
    )
    assert MODEL_NATIVE_RECIPE_ENV["ENTRY_TRAIN_LR_COSINE_DECAY"] == "1"
    assert MODEL_NATIVE_RECIPE_ENV["ENTRY_TRAIN_WEIGHT_EMA_DECAY"] == (
        MODEL_NATIVE_WEIGHT_EMA_DECAY_EPOCH_HORIZON_VALUE
    )
    assert MODEL_NATIVE_WEIGHT_EMA_DECAY_DISABLED_VALUE == "0.0"
    assert MODEL_NATIVE_WEIGHT_EMA_DECAY_EPOCH_HORIZON_VALUE == "epoch"
    assert MODEL_NATIVE_WEIGHT_EMA_DECAY_DECLARED_VALUES == ("0.0", "epoch")
    # Registry tolerance/lifecycle decisions are immutable TRAIN artifacts, so
    # the owner declares an empty registry key tuple and no registry operator
    # key may reappear on the trainer recipe surface by either route.
    assert MODEL_NATIVE_V29_REGISTRY_RECIPE_ENV_KEYS == ()
    assert not any(
        "REGISTRY" in key for key in MODEL_NATIVE_RECIPE_ENV
    )
    assert metadata == {
        "schema_version": MODEL_NATIVE_RECIPE_ENV_SCHEMA_VERSION,
        "count": expected_count,
        "sha256": MODEL_NATIVE_RECIPE_ENV_SHA256,
        "keys": list(MODEL_NATIVE_RECIPE_ENV_KEYS),
    }
    assert require_model_native_recipe_env(MODEL_NATIVE_RECIPE_ENV) == dict(
        MODEL_NATIVE_RECIPE_ENV
    )


def test_every_recipe_value_is_consumed_by_the_only_trainer_source() -> None:
    trainer_path = REPO / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
    trainer = trainer_path.read_text(encoding="utf-8")

    assert "_CANONICAL_ENTRY_TRAIN_ENV_DEFAULTS" not in trainer
    assert "MODEL_NATIVE_RECIPE_ENV_KEYS" in trainer
    assert "for key in MODEL_NATIVE_RECIPE_ENV_KEYS" in trainer
    assert "require_model_native_recipe_env(recipe_env)" in trainer
    assert "[ENTRY_TRAIN_AMBIENT_CONTROL_FORBIDDEN]" in trainer


def test_launch_reads_current_m1_feature_surface_schema_from_owner() -> None:
    source = Path(launch.__file__).read_text(encoding="utf-8")

    assert "ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION" in source
    assert '"gx1_entry_exit_m1_feature_surface_v1"' not in source


def test_candidate_binding_requires_the_refreshed_current_liveness_before_training() -> None:
    """Retained V46 evidence must never replace the current reviewed dataset."""

    state = json.loads(
        (REPO / "PROJECT_STATE_xau_direction_launch.json").read_text(
            encoding="utf-8"
        )
    )
    evidence = state["current_audited_dataset_evidence"]
    reports = evidence["reports"]
    artifacts = {
        artifact_key: Path(reports[report_name]["path"])
        for artifact_key, report_name in launch._CURRENT_AUDITED_CANDIDATE_REPORTS.items()
    }
    arguments = {
        "repo": REPO,
        "dataset_dir": Path(evidence["dataset_dir"]),
        "dataset_run_id": str(evidence["dataset_run_id"]),
        "artifacts": artifacts,
    }
    if "current_pretest_trainability_readiness" in state:
        # The current TRAIN/VAL owner supersedes the retained three-split V46
        # reports. A safety hold does not turn this back into the earlier
        # target-correction boundary or make the old dataset current again.
        current = launch.require_blocked_launch_state_with_current_audited_dataset(state)
        selected = state["current_source_technical_recipe"]
        recipe = json.loads(Path(selected["recipe_path"]).read_text(encoding="utf-8"))
        assert current["dataset_dir"] == recipe["dataset_dir"]
        assert current["dataset_run_id"] == recipe["dataset_run_id"]
        assert Path(current["dataset_dir"]) != arguments["dataset_dir"]
        without_hold = dict(state)
        without_hold.pop("pretraining_review_hold", None)
        # CPU evidence selection only; never remove the real launch hold.
        assert launch.require_blocked_launch_state_with_current_audited_dataset(
            without_hold,
        ) == current
        with pytest.raises(
            launch.LaunchContractError,
            match="candidate dataset does not match current audited dataset",
        ):
            launch._candidate_current_audited_dataset_binding(**arguments)
        return
    if "pretraining_review_hold" in state:
        # Valid feature liveness cannot rescue a superseded target/causality
        # contract. Retained V46 evidence must not bind a new candidate.
        with pytest.raises(launch.LaunchContractError, match="EXECUTION_CAUSALITY_EXPECTATION_INVALID"):
            launch._candidate_current_audited_dataset_binding(**arguments)
        return
    binding = launch._candidate_current_audited_dataset_binding(**arguments)
    assert binding["reports"]["full_input_liveness_audit_json"] == {
        "path": str(reports["full_input_liveness"]["path"]),
        "sha256": str(reports["full_input_liveness"]["sha256"]),
    }


@pytest.mark.parametrize("mutation", ("missing", "extra", "changed"))
def test_recipe_env_rejects_every_non_exact_surface(mutation: str) -> None:
    candidate = dict(MODEL_NATIVE_RECIPE_ENV)
    # The mutated key comes from the owner's own key tuple.  Naming a key here
    # made this test die the day Wave C retired it, and it also let the
    # "changed" arm silently degrade into a second "extra" arm because the
    # named key was no longer present to change.
    owned_key = MODEL_NATIVE_RECIPE_ENV_KEYS[0]
    if mutation == "missing":
        mutated_key = owned_key
        candidate.pop(owned_key)
    elif mutation == "extra":
        mutated_key = "ENTRY_UNAUDITED_PASS_THROUGH"
        candidate[mutated_key] = "1"
    else:
        mutated_key = owned_key
        candidate[owned_key] = f"{candidate[owned_key]}_NOT_THE_DECLARED_VALUE"
    assert candidate != dict(MODEL_NATIVE_RECIPE_ENV)

    with pytest.raises(
        RuntimeError, match="MODEL_NATIVE_RECIPE_ENV_MISMATCH"
    ) as rejected:
        require_model_native_recipe_env(candidate)
    assert mutated_key in str(rejected.value)


def test_recipe_producer_event_drives_exact_smoke_wrapper_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wrapper_argv, paths = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    out_dir = (tmp_path / "recipe_20260722T140000Z").resolve()
    producer_argv = [
        "--profile",
        "smoke",
        "--repo",
        str(REPO),
        "--wrapper-path",
        str(WRAPPER),
        *wrapper_argv,
        "--out-dir",
        str(out_dir),
        "--quiet",
    ]
    recipe_flag = producer_argv.index("--recipe-audit-json")
    del producer_argv[recipe_flag : recipe_flag + 2]
    monkeypatch.setattr(producer, "_clean_worktree", lambda _repo: None)

    event_path, event = producer.run(producer.build_parser().parse_args(producer_argv))

    assert event_path.is_file()
    assert event["json_path"] == str(event_path)
    assert event["trainer_env"] == MODEL_NATIVE_RECIPE_ENV
    assert event["dataset_run_id"] == DATASET_RUN_ID
    assert event["prefreeze_test_seal_lineage"]["dataset_run_id"] == DATASET_RUN_ID
    assert event["prefreeze_test_seal_lineage"]["access_proof"][
        "test_metrics_read"
    ] is False
    assert event["prefreeze_test_seal_lineage_sha256"] == (
        launch.canonical_json_sha256(event["prefreeze_test_seal_lineage"])
    )
    assert event["trainer_env_contract"] == model_native_recipe_env_contract_metadata()
    assert event["activation_authority"] is False
    assert event["report_only"] is True
    assert not any(event["side_effects_started"].values())
    bindings = event["source_bindings"]
    assert {
        "control_surface",
        "wrapper",
        "trainer_safety_guard",
        "capped_runner",
        "host_telemetry_query",
        "windows_power_keeper",
        "windows_power_benchmark_scope",
    }.issubset(bindings)
    python_bindings = {
        key for key in bindings if key.startswith("python:gx1/")
    }
    assert len(python_bindings) >= 70
    assert {
        "python:gx1/models/entry_v10/entry_v10_ctx_train_v3.py",
        "python:gx1/models/entry_v10/entry_v10_bundle.py",
        "python:gx1/models/entry_v10/entry_v10_input_normalization.py",
        "python:gx1/scripts/run_entry_model_native_pretest_technical_train_v1.py",
        "python:gx1/contracts/entry_pretest_candidate_launch_gate_v1.py",
        "python:gx1/contracts/local_power_benchmark_v1.py",
        "python:gx1/contracts/gx1_capped_execution_v1.py",
        "python:gx1/contracts/entry_model_native_training_objective_v1.py",
        "python:gx1/contracts/entry_model_native_joint_task_weighting_v1.py",
        # Every package initializer is executable before the corresponding
        # import target and therefore must be inside the byte-bound closure.
        "python:gx1/__init__.py",
        "python:gx1/contracts/__init__.py",
        "python:gx1/models/__init__.py",
        "python:gx1/models/entry_v10/__init__.py",
    }.issubset(python_bindings)

    original_recipe = str(paths["recipe_audit_json"])
    validated_argv = [
        str(event_path) if value == original_recipe else value
        for value in wrapper_argv
    ]
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--profile",
            "smoke",
            *validated_argv,
            "--dry-run",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Validated model-native seq513 smoke contract" in result.stdout
    assert "ENTRY_HIER_TRADE_WEIGHT" not in result.stdout
    assert "ENTRY_MTF_DIR_AUX_WEIGHT" not in result.stdout


def test_recipe_producer_rejects_smoke_run_lineage_before_large_rehash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stale run ID must fail before any large parquet binding is read."""

    wrapper_argv, _ = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    run_id_index = wrapper_argv.index("--run-id") + 1
    wrapper_argv[run_id_index] = "RECIPE_LINEAGE_MISMATCH_V1"
    out_dir = (tmp_path / "recipe_20260722T140002Z").resolve()
    producer_argv = [
        "--profile",
        "smoke",
        "--repo",
        str(REPO),
        "--wrapper-path",
        str(WRAPPER),
        *wrapper_argv,
        "--out-dir",
        str(out_dir),
        "--quiet",
    ]
    recipe_flag = producer_argv.index("--recipe-audit-json")
    del producer_argv[recipe_flag : recipe_flag + 2]
    monkeypatch.setattr(producer, "_clean_worktree", lambda _repo: None)

    def fail_if_large_binding_is_reached(*_args, **_kwargs):
        raise AssertionError("large artifact rehash reached before run-lineage gate")

    monkeypatch.setattr(launch, "artifact_binding", fail_if_large_binding_is_reached)

    with pytest.raises(
        launch.LaunchContractError,
        match="trainability smoke run lineage mismatch",
    ):
        producer.run(producer.build_parser().parse_args(producer_argv))
    assert not out_dir.exists()


@pytest.mark.parametrize("hold", [None, {}, {"decision": "BLOCK"}, {"decision": "PASS"}])
def test_direct_execution_rejects_any_present_review_hold(tmp_path: Path, hold) -> None:
    (tmp_path / "PROJECT_STATE_xau_direction_launch.json").write_text(
        json.dumps({"pretraining_review_hold": hold}), encoding="utf-8"
    )
    with pytest.raises(launch.LaunchContractError, match="review hold blocks execution"):
        launch._require_training_review_hold_cleared(tmp_path)


def test_direct_execution_review_hold_requires_readable_state(tmp_path: Path) -> None:
    with pytest.raises(launch.LaunchContractError):
        launch._require_training_review_hold_cleared(tmp_path)
    (tmp_path / "PROJECT_STATE_xau_direction_launch.json").write_text("{}", encoding="utf-8")
    launch._require_training_review_hold_cleared(tmp_path)


def _commit_source_fixture(repo: Path) -> str:
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(repo), "-c", "core.hooksPath=/dev/null",
            "-c", "user.name=GX1 fixture",
            "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-qm", "source fixture",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _held_source_repo(tmp_path: Path) -> Path:
    """Real clean Git/source-binding fixture; no provenance checks are mocked."""
    repo = tmp_path / "source-repo"
    repo.mkdir()
    # Derive all required roots/package initializers from the current owner.
    # The fixture modules deliberately import nothing and execute no GX1 work.
    for path in launch.recipe_source_binding_paths(repo=REPO, wrapper_path=WRAPPER).values():
        target = repo / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# immutable source fixture\n", encoding="utf-8")
    (repo / "PROJECT_STATE_xau_direction_launch.json").write_text(
        json.dumps({"pretraining_review_hold": {
            "schema_version": "gx1_pretraining_review_hold_v1",
            "decision": "BLOCK",
            "reason": "successor dataset review is pending",
            "activation_authority": False,
            "report_path": "docs/PREMIERE_CODE_REVIEW_20260905.md",
        }}),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _commit_source_fixture(repo)
    return repo


def _held_source_provenance_fixture(tmp_path: Path) -> tuple[dict, Path]:
    repo = _held_source_repo(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    bundle = tmp_path / "bundle"
    bindings = launch.recipe_source_bindings(
        repo=repo, wrapper_path=repo / launch.TRAIN_WRAPPER_RELATIVE_PATH
    )
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps({
        "schema_version": launch.RECIPE_AUDIT_SCHEMA,
        "decision": "PASS",
        "failures": [],
        "profile": "smoke",
        "run_id": "SOURCE_REVIEW_SMOKE_V1",
        "dataset_run_id": "SOURCE_REVIEW_DATASET_V1",
        "dataset_dir": str(dataset),
        "out_bundle_dir": str(bundle),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_bindings": bindings,
        "source_bindings_sha256": launch.canonical_json_sha256(bindings),
    }), encoding="utf-8")
    return {
        "recipe_audit_path": recipe_path,
        "recipe_audit_sha256": launch.sha256_file(recipe_path),
        "repo": repo,
        "profile": "smoke",
        "run_id": "SOURCE_REVIEW_SMOKE_V1",
        "dataset_run_id": "SOURCE_REVIEW_DATASET_V1",
        "dataset_dir": dataset,
        "out_bundle_dir": bundle,
    }, repo / launch.TRAINER_RELATIVE_PATH


def test_read_only_source_provenance_passes_under_hold_but_execution_does_not(
    tmp_path: Path,
) -> None:
    kwargs, _trainer = _held_source_provenance_fixture(tmp_path)
    state = kwargs["repo"] / "PROJECT_STATE_xau_direction_launch.json"
    original_state = state.read_bytes()
    result = launch.require_training_recipe_source_provenance(**kwargs)
    assert result["recipe_audit_sha256"] == kwargs["recipe_audit_sha256"]
    assert state.read_bytes() == original_state
    with pytest.raises(launch.LaunchContractError, match="review hold blocks execution"):
        launch.require_training_recipe_execution_provenance(**kwargs)

    # Execution must check the hold before even reading the recipe.
    kwargs["recipe_audit_path"] = tmp_path / "missing-recipe.json"
    with pytest.raises(launch.LaunchContractError, match="review hold blocks execution"):
        launch.require_training_recipe_execution_provenance(**kwargs)


@pytest.mark.parametrize("mutation", ["dirty", "untracked", "committed-source", "recipe-bytes"])
def test_read_only_source_provenance_retains_all_freshness_checks_under_hold(
    tmp_path: Path, mutation: str,
) -> None:
    kwargs, trainer = _held_source_provenance_fixture(tmp_path)
    if mutation in {"dirty", "committed-source"}:
        trainer.write_text("# modified source fixture\n", encoding="utf-8")
        if mutation == "committed-source":
            _commit_source_fixture(kwargs["repo"])
            expected = "recipe source binding mismatch"
        else:
            expected = "trainer source worktree must be clean"
    elif mutation == "untracked":
        (kwargs["repo"] / "unreviewed.py").write_text("# unknown source\n", encoding="utf-8")
        expected = "trainer source worktree must be clean"
    else:
        recipe = kwargs["recipe_audit_path"]
        recipe.write_bytes(recipe.read_bytes() + b"\n")
        expected = "trainer recipe audit bytes do not match declared sha256"
    with pytest.raises(launch.LaunchContractError, match=expected):
        launch.require_training_recipe_source_provenance(**kwargs)


def test_cleared_execution_and_read_only_check_share_identical_source_proof(
    tmp_path: Path,
) -> None:
    kwargs, _trainer = _held_source_provenance_fixture(tmp_path)
    (kwargs["repo"] / "PROJECT_STATE_xau_direction_launch.json").write_text(
        "{}", encoding="utf-8"
    )
    _commit_source_fixture(kwargs["repo"])
    assert launch.require_training_recipe_execution_provenance(**kwargs) == (
        launch.require_training_recipe_source_provenance(**kwargs)
    )


def test_execution_provenance_accepts_clean_descendant_with_exact_source_bindings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Docs-only descendants must not force a repeat large-input recipe audit."""

    recipe_path = tmp_path / "recipe.json"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    out_bundle_dir = tmp_path / "bundle"
    recipe = {
        "schema_version": launch.RECIPE_AUDIT_SCHEMA,
        "decision": "PASS",
        "failures": [],
        "profile": "smoke",
        "run_id": "RECIPE_DESCENDANT_SMOKE_V1",
        "dataset_run_id": "RECIPE_DESCENDANT_DATASET_V1",
        "dataset_dir": str(dataset_dir),
        "out_bundle_dir": str(out_bundle_dir),
        "source_commit": "a" * 40,
        "source_bindings": {"trainer": {"sha256": "b" * 64}},
        "source_bindings_sha256": "c" * 64,
    }
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    recipe_sha256 = producer._sha256_file(recipe_path)
    observed: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        observed.append(tuple(command))
        return subprocess.CompletedProcess(command, 0)

    def fake_check_output(command, **_kwargs):
        observed.append(tuple(command))
        assert command[-2:] == ["--porcelain=v1", "--untracked-files=all"]
        return ""

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    # This isolated test models an operator-cleared state, not the current
    # worktree's deliberate semantic-rebuild hold (tested separately below).
    monkeypatch.setattr(launch, "_require_training_review_hold_cleared", lambda _repo: None)
    monkeypatch.setattr(launch.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(launch, "_validate_source_bindings", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        launch,
        "require_training_recipe_source_provenance_metadata",
        lambda value, **_kwargs: value,
    )

    result = launch.require_training_recipe_execution_provenance(
        recipe_audit_path=recipe_path,
        recipe_audit_sha256=recipe_sha256,
        repo=REPO,
        profile="smoke",
        run_id="RECIPE_DESCENDANT_SMOKE_V1",
        dataset_run_id="RECIPE_DESCENDANT_DATASET_V1",
        dataset_dir=dataset_dir,
        out_bundle_dir=out_bundle_dir,
    )

    assert result["source_commit"] == "a" * 40
    assert any(
        command[-4:] == ("merge-base", "--is-ancestor", "a" * 40, "HEAD")
        for command in observed
    )
    assert not any(command[-1:] == ("HEAD",) and "rev-parse" in command for command in observed)


@pytest.mark.parametrize("invalid_dropout", ("-0.01", "1.0", "nan"))
def test_train_launch_rejects_invalid_explicit_dropout(
    tmp_path: Path,
    invalid_dropout: str,
) -> None:
    wrapper_argv, _ = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    dropout_index = wrapper_argv.index("--dropout") + 1
    wrapper_argv[dropout_index] = invalid_dropout
    args = launch.build_parser().parse_args(
        [
            "--profile",
            "smoke",
            "--repo",
            str(REPO),
            "--wrapper-path",
            str(WRAPPER),
            *wrapper_argv,
        ]
    )

    with pytest.raises(launch.LaunchContractError, match="dropout"):
        launch._trainer_cli_contract(args)


def test_time_window_is_hash_bound_in_trainer_cli_contract(tmp_path: Path) -> None:
    wrapper_argv, _ = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
        train_time_window=(
            "2024-12-01T00:00:00Z",
            "2025-06-01T00:00:00Z",
        ),
    )
    device_index = wrapper_argv.index("--device") + 1
    wrapper_argv[device_index] = "cuda"
    args = launch.build_parser().parse_args(
        [
            "--profile",
            "smoke",
            "--repo",
            str(REPO),
            "--wrapper-path",
            str(WRAPPER),
            *wrapper_argv,
        ]
    )
    trainer_cli = launch._trainer_cli_contract(args)
    assert trainer_cli["train_time_window"] == {
        "start_utc": "2024-12-01T00:00:00+00:00",
        "end_utc": "2025-06-01T00:00:00+00:00",
    }


def test_train_launch_rejects_cpu_time_window_before_wrapper_boundary(
    tmp_path: Path,
) -> None:
    wrapper_argv, _ = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
        train_time_window=(
            "2024-12-01T00:00:00Z",
            "2025-06-01T00:00:00Z",
        ),
    )
    args = launch.build_parser().parse_args(
        [
            "--profile",
            "smoke",
            "--repo",
            str(REPO),
            "--wrapper-path",
            str(WRAPPER),
            *wrapper_argv,
        ]
    )

    with pytest.raises(
        launch.LaunchContractError,
        match="train time-window requires --device cuda",
    ):
        launch._trainer_cli_contract(args)


def test_train_launch_rejects_every_noncanonical_wrapper_path(
    tmp_path: Path,
) -> None:
    wrapper_argv, _ = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    noncanonical = tmp_path / "noncanonical_train.sh"
    noncanonical.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    args = launch.build_parser().parse_args(
        [
            "--profile",
            "smoke",
            "--repo",
            str(REPO),
            "--wrapper-path",
            str(noncanonical),
            *wrapper_argv,
        ]
    )

    with pytest.raises(
        launch.LaunchContractError,
        match="canonical profile-explicit trainer wrapper",
    ):
        launch.validate_launch(args)


def test_candidate_launch_rejects_training_subsample(tmp_path: Path) -> None:
    wrapper_argv, _ = build_wrapper_contract(
        tmp_path,
        profile="candidate",
        wrapper=WRAPPER,
    )
    subsample_index = wrapper_argv.index("--subsample-rows") + 1
    wrapper_argv[subsample_index] = "512"
    args = launch.build_parser().parse_args(
        [
            "--profile",
            "candidate",
            "--repo",
            str(REPO),
            "--wrapper-path",
            str(WRAPPER),
            *wrapper_argv,
        ]
    )

    with pytest.raises(
        launch.LaunchContractError,
        match="candidate training requires full TRAIN population",
    ):
        launch._trainer_cli_contract(args)


def test_recipe_producer_fails_before_publication_when_source_is_dirty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wrapper_argv, _ = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    out_dir = (tmp_path / "recipe_20260722T140001Z").resolve()
    producer_argv = [
        "--profile",
        "smoke",
        "--repo",
        str(REPO),
        "--wrapper-path",
        str(WRAPPER),
        *wrapper_argv,
        "--out-dir",
        str(out_dir),
        "--quiet",
    ]
    recipe_flag = producer_argv.index("--recipe-audit-json")
    del producer_argv[recipe_flag : recipe_flag + 2]

    def reject(_repo: Path) -> None:
        raise RuntimeError("TRAIN_RECIPE_SOURCE_WORKTREE_DIRTY")

    monkeypatch.setattr(producer, "_clean_worktree", reject)

    with pytest.raises(RuntimeError, match="TRAIN_RECIPE_SOURCE_WORKTREE_DIRTY"):
        producer.run(producer.build_parser().parse_args(producer_argv))
    assert not out_dir.exists()


@pytest.mark.parametrize("mutation", ("missing", "wrong_equation", "unknown"))
def test_launch_rejects_non_exact_split_aux_target_emission_proof(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, paths = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    manifest_path = paths["train_manifest_json"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    emission = payload["extra"]["aux_head_target_contract"]
    if mutation == "missing":
        emission.pop("complete_rows_emitted")
    elif mutation == "wrong_equation":
        emission["candidate_rows_before_completeness"] += 1
    else:
        emission["allow_incomplete_rows"] = False

    with pytest.raises(
        launch.LaunchContractError,
        match="split manifest aux-target emission contract invalid",
    ):
        launch._validate_split_manifest(
            payload,
            path=manifest_path,
            parquet=paths["train_parquet"],
            m5_prebuilt=paths["m5_prebuilt_path"],
        )


def test_launch_rejects_tampered_train_sequence_integrity_proof(
    tmp_path: Path,
) -> None:
    wrapper_argv, paths = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    integrity_path = paths["train_sequence_integrity_audit_json"]
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["checks"]["every_overlap_eligible_pair_has_exact_physical_overlap"] = False
    integrity_path.write_text(
        json.dumps(integrity, sort_keys=True) + "\n", encoding="utf-8"
    )
    args = launch.build_parser().parse_args(
        [
            "--profile",
            "smoke",
            "--repo",
            str(REPO),
            "--wrapper-path",
            str(WRAPPER),
            *wrapper_argv,
        ]
    )

    with pytest.raises(
        launch.LaunchContractError,
        match="train sequence-integrity audit invalid",
    ):
        launch.validate_launch(args)


def test_launch_rejects_tampered_train_sequence_source_reconstruction_proof(
    tmp_path: Path,
) -> None:
    wrapper_argv, paths = build_wrapper_contract(
        tmp_path,
        profile="candidate",
        wrapper=WRAPPER,
    )
    proof_path = paths["train_sequence_source_audit_json"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["checks"]["every_sequence_equals_exact_source_surface_history_bit_identical"] = False
    proof_path.write_text(
        json.dumps(proof, sort_keys=True) + "\n", encoding="utf-8"
    )
    args = launch.build_parser().parse_args(
        [
            "--profile",
            "candidate",
            "--repo",
            str(REPO),
            "--wrapper-path",
            str(WRAPPER),
            *wrapper_argv,
        ]
    )

    with pytest.raises(
        launch.LaunchContractError,
        match="train sequence-source reconstruction audit invalid",
    ):
        launch.validate_launch(args)


def test_launch_derives_dataset_run_id_from_post_rebuild_and_all_splits(
    tmp_path: Path,
) -> None:
    _, paths = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    payloads = {
        f"{split}_manifest_json": json.loads(
            paths[f"{split}_manifest_json"].read_text(encoding="utf-8")
        )
        for split in ("train", "val", "test")
    }
    post_rebuild = json.loads(
        paths["post_rebuild_readiness_json"].read_text(encoding="utf-8")
    )

    assert (
        launch._dataset_run_id_from_launch_evidence(
            post_rebuild=post_rebuild,
            payloads=payloads,
        )
        == DATASET_RUN_ID
    )

    payloads["val_manifest_json"]["extra"]["entry_run_id"] = (
        "DIFFERENT_DATASET_RUN_ID"
    )
    with pytest.raises(
        launch.LaunchContractError,
        match="val split dataset run lineage mismatch",
    ):
        launch._dataset_run_id_from_launch_evidence(
            post_rebuild=post_rebuild,
            payloads=payloads,
        )


def test_recipe_rejects_training_run_id_equal_to_dataset_run_id(
    tmp_path: Path,
) -> None:
    wrapper_argv, _ = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    run_id_index = wrapper_argv.index("--run-id") + 1
    wrapper_argv[run_id_index] = DATASET_RUN_ID
    args = launch.build_parser().parse_args(
        [
            "--profile",
            "smoke",
            "--repo",
            str(REPO),
            "--wrapper-path",
            str(WRAPPER),
            *wrapper_argv,
        ]
    )

    with pytest.raises(
        launch.LaunchContractError,
        match="training run_id must differ from immutable dataset_run_id",
    ):
        launch.validate_launch(args)


def test_smoke_launch_rejects_otherwise_valid_evidence_with_mixed_run_lineage(
    tmp_path: Path,
) -> None:
    wrapper_argv, paths = build_wrapper_contract(
        tmp_path,
        profile="smoke",
        wrapper=WRAPPER,
    )
    readiness_path = paths["smoke_readiness_json"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["entry_run_id"] = "MODEL_NATIVE_SEQ513_OTHER_SMOKE_PYTEST"
    readiness_path.write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args = launch.build_parser().parse_args(
        [
            "--profile",
            "smoke",
            "--repo",
            str(REPO),
            "--wrapper-path",
            str(WRAPPER),
            *wrapper_argv,
        ]
    )

    with pytest.raises(
        launch.LaunchContractError,
        match="smoke readiness run lineage mismatch",
    ):
        launch.validate_launch(args)


def test_trainer_has_no_shadow_default_for_any_recipe_value() -> None:
    """The recipe owner must be the single origin of every training value.

    The trainer used to carry its own literal default at all 160 call sites.
    62 had drifted from the recipe and 30 of those to zero, which silently
    deletes the direction-balance, slice-guard and prior-match loss families.
    A second origin can never be allowed back.
    """
    import re
    from gx1.contracts.entry_model_native_train_recipe_v1 import (
        MODEL_NATIVE_V29_REGISTRY_RECIPE_ENV_KEYS,
    )

    source = (REPO / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py").read_text(
        encoding="utf-8"
    )
    with_default = re.findall(r'_env_str\(\s*"[A-Z0-9_]+"\s*,', source)
    assert with_default == [], f"trainer regained shadow defaults: {with_default}"
    single_arg = re.findall(r'_env_str\(\s*"([A-Z0-9_]+)"\s*\)', source)
    indirect_keys = {
        *MODEL_NATIVE_V29_REGISTRY_RECIPE_ENV_KEYS,
        "GX1_CTX_CONTRACT",
    }
    assert set(single_arg) == set(MODEL_NATIVE_RECIPE_ENV_KEYS) - indirect_keys


def test_trainer_cli_has_no_numeric_execution_defaults() -> None:
    source = (REPO / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py").read_text(
        encoding="utf-8"
    )
    required_flags = (
        "--seed",
        "--device",
        "--batch_size",
        "--epochs",
        "--lr",
        "--seq_len",
        "--num-workers",
        "--early-stopping-patience",
        "--early-stopping-min-delta",
        "--multi-tf-scale",
        "--subsample-rows",
        "--grad-clip-norm",
        "--weight-decay",
    )
    tree = ast.parse(source)
    observed: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in required_flags
        ):
            continue
        observed[str(node.args[0].value)] = node
    assert set(observed) == set(required_flags)
    for flag, node in observed.items():
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        assert isinstance(keywords.get("required"), ast.Constant)
        assert keywords["required"].value is True, flag
        assert "default" not in keywords, flag


def test_env_reader_rejects_a_key_the_recipe_does_not_own() -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    with pytest.raises(RuntimeError, match="RECIPE_KEY_UNKNOWN"):
        trainer._env_str("ENTRY_NOT_A_RECIPE_KEY")
