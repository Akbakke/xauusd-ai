from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gx1.scripts.report_training_efficiency_v1 import (
    EfficiencyEvidenceError, _read_bound, parse_measurements,
)

# Small synthetic logs exercise parser arithmetic and refusal boundaries only.
# They are never training-throughput, numerical-parity, or safety evidence.
TRAIN = """2026-09-08 08:47:48,000 [INFO] [TRAIN_RSS] epoch_start rss_gib=5.35
2026-09-08 08:47:48,001 [INFO] [TRAIN_RSS] first_batch_fetched rss_gib=5.37
2026-09-08 08:47:50,000 [INFO] [UNIFIED_EXIT_PROFILE] eligible_entries=8 chunk_rows=8 chunks=1 materialize_s=0.05 online_forward_s=0.5 target_forward_s=0.2 bellman_backward_s=0.2 post_backward_s=0.05 total_s=1.0
2026-09-08 08:47:50,000 [INFO] [TRAIN_PROFILE] batch=1 entry_online_forward_s=0.25 entry_target_forward_s=0.25 exit_train_s=1.0 post_exit_backward_s=0.5 total_s=2.0 peak_cuda_mib=6000
2026-09-08 08:47:54,000 [INFO] [TRAIN_LR_SCHEDULE] epoch=1 next_lr=0
"""
GUARD = """2026-09-08T08:42:00Z event=telemetry phase=preflight core_temp_c=35 memory_temp_c=40 memory_observed=true power_draw_w=20 power_limit_w=160 memory_used_mib=300
2026-09-08T08:42:00Z event=start execution_mode=canonical device=cuda
2026-09-08T08:48:00Z event=exit child_status=0 telemetry_samples=300 peak_core_temp_c=50 peak_memory_temp_c=52 peak_power_draw_w=159 peak_memory_used_mib=9400
"""


def test_cold_samples_cannot_become_steady_state_or_average_power() -> None:
    report = parse_measurements(TRAIN, GUARD, rows=32, batch_size=8)
    assert report["train_log_window"]["optimizer_steps"] == 4
    assert report["train_log_window"]["samples_per_second"] == pytest.approx(32 / 6)
    assert report["cold_first_batch"]["exit_share_of_compute"] == 0.5
    assert report["cold_first_batch"]["exit_materialization_share_of_compute"] == 0.025
    assert report["guard_run"]["gpu_resident_peak_mib"] == 9400
    assert report["cold_first_batch"]["peak_cuda_mib"] == 6000
    assert "post_warmup_train_window" not in report
    assert "power_draw_avg_w" not in report["guard_run"]
    assert "epoch_eta" not in report


@pytest.mark.parametrize("guard", [
    GUARD.replace("child_status=0", "child_status=1"),
    GUARD + "2026-09-08T08:47:00Z event=stop reason=thermal\n",
    GUARD.replace("memory_observed=true", "memory_observed=false"),
    GUARD.replace("power_limit_w=160", "power_limit_w=nan"),
    GUARD + GUARD.splitlines()[-1] + "\n",
])
def test_failed_or_ambiguous_guard_cannot_produce_baseline(guard: str) -> None:
    with pytest.raises(EfficiencyEvidenceError):
        parse_measurements(TRAIN, guard, rows=32, batch_size=8)


@pytest.mark.parametrize("train", [
    TRAIN.replace("total_s=2.0", "total_s=3.0"),
    TRAIN.replace("total_s=2.0", "total_s=nan"),
    TRAIN.replace("batch=1 ", "batch=2 "),
    TRAIN.replace("08:47:54,000", "08:47:47,000"),
    TRAIN.replace("[TRAIN_LR_SCHEDULE]", "[UNRELATED]"),
])
def test_bad_or_mixed_timing_rejected(train: str) -> None:
    with pytest.raises(EfficiencyEvidenceError):
        parse_measurements(train, GUARD, rows=32, batch_size=8)


def test_measured_window_uses_measured_rows_and_steps_not_entire_sample() -> None:
    value = {"report_only": True, "warmup_optimizer_steps": 1,
             "measured_optimizer_steps": 3, "measured_train_rows": 24,
             "measured_train_seconds": 4}
    train = TRAIN + "[TRAIN_EFFICIENCY_WINDOW] " + json.dumps(value) + "\n"
    result = parse_measurements(train, GUARD, rows=32, batch_size=8)
    assert result["post_warmup_train_window"]["samples_per_second"] == 6
    assert result["post_warmup_train_window"]["optimizer_steps_per_second"] == 0.75
    value["measured_optimizer_steps"] = 4
    with pytest.raises(EfficiencyEvidenceError, match="geometry"):
        parse_measurements(TRAIN + "[TRAIN_EFFICIENCY_WINDOW] " + json.dumps(value),
                           GUARD, rows=32, batch_size=8)


def test_changed_input_and_symlink_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "evidence.log"
    path.write_text(TRAIN)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _read_bound(path, digest)[0].decode() == TRAIN
    link = tmp_path / "alias.log"
    link.symlink_to(path)
    with pytest.raises(EfficiencyEvidenceError):
        _read_bound(link, digest)
    path.write_text(TRAIN + "changed")
    with pytest.raises(EfficiencyEvidenceError, match="SHA-256 mismatch"):
        _read_bound(path, digest)


def _bound_report_fixture(tmp_path: Path):
    from argparse import Namespace
    from gx1.contracts.entry_model_native_bundle_commit_v1 import write_bundle_commit_manifest
    from gx1.scripts.report_training_efficiency_v1 import build_report

    bundle = tmp_path / "fixture_BUNDLE"
    bundle.mkdir()
    recipe_path = tmp_path / "recipe.json"
    recipe = {"profile": "smoke", "out_bundle_dir": str(bundle),
              "source_commit": "a" * 40, "run_id": "fixture", "dataset_run_id": "data",
              "trainer_cli": {"epochs": 1, "grad_accum_steps": 1, "device": "cuda",
                              "execution_tier": "canonical", "batch_size": 8}}
    recipe_path.write_text(json.dumps(recipe))
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {"recipe_audit_path": str(recipe_path),
                  "recipe_audit_sha256": digest(recipe_path), "source_commit": "a" * 40}
    lineage = {"training_profile": "smoke", "training_run_id": "fixture",
               "dataset_run_id": "data", "effective_train_rows": 32}
    shared = {"recipe_source_provenance": provenance, "run_lineage": lineage}
    meta = {**shared, "seq_len": 96, "seq_input_dim": 238, "ctx_cont_dim": 71,
            "ctx_cat_dim": 1, "specialist_fusion": {"group_feature_counts": {}}}
    (bundle / "bundle_metadata.json").write_text(json.dumps(meta))
    (bundle / "MASTER_TRANSFORMER_LOCK.json").write_text(json.dumps(shared))
    (bundle / "model_state_dict.pt").write_bytes(b"parser fixture only")
    commit = write_bundle_commit_manifest(bundle_dir=bundle,
        artifact_names=("bundle_metadata.json", "MASTER_TRANSFORMER_LOCK.json", "model_state_dict.pt"),
        bundle_kind="trained", created_at_utc="2026-09-08T12:00:00+00:00")
    trainer_log = tmp_path / ".fixture_BUNDLE.trainer.explicit.log"
    trainer_log.write_text(TRAIN + f"[DONE] immutable bundle atomically published: {bundle}\n")
    guard_log = tmp_path / ".fixture_BUNDLE.guard.explicit.log"
    guard_log.write_text(GUARD)
    args = Namespace(recipe=str(recipe_path), recipe_sha256=digest(recipe_path),
                     trainer_log=str(trainer_log), trainer_log_sha256=digest(trainer_log),
                     guard_log=str(guard_log), guard_log_sha256=digest(guard_log),
                     bundle=str(bundle), bundle_commit_sha256=commit["commit_sha256"])
    return args, build_report


def test_report_requires_exact_bundle_and_denies_authority(tmp_path: Path) -> None:
    args, build_report = _bound_report_fixture(tmp_path)
    report = build_report(args)
    assert report["decision"] == "INCOMPLETE_BASELINE_NOT_OPTIMIZATION_QUALIFIED"
    assert set(report["authority"].values()) == {False}
    assert report["test_accessed"] is False
    assert "synchronized_post_warmup_throughput" in report["missing_measurements"]
    args.bundle_commit_sha256 = "0" * 64
    with pytest.raises(EfficiencyEvidenceError, match="bundle commit"):
        build_report(args)


def test_changed_recipe_cannot_borrow_unchanged_bundle(tmp_path: Path) -> None:
    args, build_report = _bound_report_fixture(tmp_path)
    recipe = Path(args.recipe)
    value = json.loads(recipe.read_text())
    value["source_commit"] = "b" * 40
    recipe.write_text(json.dumps(value))
    args.recipe_sha256 = hashlib.sha256(recipe.read_bytes()).hexdigest()
    with pytest.raises(EfficiencyEvidenceError, match="provenance"):
        build_report(args)


def test_valid_log_hash_cannot_borrow_other_run_sidecar(tmp_path: Path) -> None:
    args, build_report = _bound_report_fixture(tmp_path)
    other = tmp_path / "other.trainer.log"
    other.write_bytes(Path(args.trainer_log).read_bytes())
    args.trainer_log = str(other)
    with pytest.raises(EfficiencyEvidenceError, match="sidecars"):
        build_report(args)
