import json
from pathlib import Path

from gx1.scripts import rebuild_truth_downstream_canonical_v2 as rebuild_mod


def test_rebuild_truth_downstream_canonical_v2_best_effort_writes_audits(tmp_path, monkeypatch):
    reports_root = tmp_path / "truth_root"
    reports_root.mkdir()

    monkeypatch.setattr(rebuild_mod, "write_shadow_meta_v2_split_manifest", lambda root: {"manifest": str(root / "split.json")})
    monkeypatch.setattr(
        rebuild_mod,
        "write_shadow_meta_v2_prefreeze_threshold_artifacts",
        lambda root: (_ for _ in ()).throw(RuntimeError("derive_feature_spec missing")),
    )
    monkeypatch.setattr(rebuild_mod, "write_shadow_meta_v2_prefreeze_shield_artifacts", lambda root: {"shield": str(root / "shield.json")})
    monkeypatch.setattr(rebuild_mod, "write_shadow_meta_v2_prefreeze_pocket_artifacts", lambda root: {"pocket": str(root / "pocket.json")})
    monkeypatch.setattr(
        rebuild_mod,
        "write_shadow_meta_v2_prefreeze_promote_semantics_artifact",
        lambda root: {"promote": str(root / "promote.json")},
    )
    monkeypatch.setattr(rebuild_mod, "write_shadow_meta_v2_audit_only_holdout_artifacts", lambda root: {"holdout": str(root / "holdout.json")})
    monkeypatch.setattr(rebuild_mod, "write_shadow_meta_v2_activation_history_artifacts", lambda root: {"history": str(root / "history.json")})
    monkeypatch.setattr(
        rebuild_mod,
        "write_shadow_meta_v2_contact_week_descriptive_artifacts",
        lambda root: {"contact_week": str(root / "contact_week.csv")},
    )
    monkeypatch.setattr(
        rebuild_mod,
        "write_shadow_meta_v2_contact_cohort_baseline_artifacts",
        lambda root: {"contact_cohort": str(root / "contact_cohort.csv")},
    )
    monkeypatch.setattr(rebuild_mod, "write_shadow_meta_v2_parallel_test_artifacts", lambda root: {"parallel": str(root / "parallel.json")})
    monkeypatch.setattr(
        rebuild_mod,
        "write_all_trade_review_ledger_closed_trades",
        lambda root, out_dir=None: {"out_dir": str((out_dir or (root / "ledger")).resolve())},
    )
    monkeypatch.setattr(
        rebuild_mod,
        "build_skipability_pressure_summary",
        lambda root, sample_limit=10: {"completed_zero_trade_runs": 2, "candidate_rich_zero_trade_runs": 2},
    )
    monkeypatch.setattr(
        rebuild_mod,
        "build_truth_management_rl_readiness_summary",
        lambda root, review_dir=None, sample_limit=10: {"downstream_management_ready": False},
    )
    monkeypatch.setattr(
        rebuild_mod,
        "build_trade_foundation_quality_summary",
        lambda root, sample_limit=10: {"trade_count": 12, "outlook_v1": "POSITIVE_EDGE_HIGH_REGRET"},
    )
    monkeypatch.setattr(
        rebuild_mod,
        "build_continuous_market_opportunity_summary",
        lambda root, sample_limit=10: {"opportunity_rich_zero_trade_runs_anchor": ["r1", "r2"]},
    )

    result = rebuild_mod.rebuild_truth_downstream_canonical_v2(reports_root)

    assert Path(result["summary_path"]).exists()
    assert Path(result["skipability_path"]).exists()
    assert Path(result["readiness_path"]).exists()
    assert Path(result["foundation_path"]).exists()
    assert Path(result["market_opportunity_path"]).exists()

    payload = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert payload["ledger_dir"] == str((reports_root / "ledger").resolve())
    assert payload["headline"]["blocked_step_count"] == 1
    assert payload["headline"]["opportunity_rich_zero_trade_runs"] == 2
    blocked = [row for row in payload["steps"] if row.get("status") == "blocked"]
    assert len(blocked) == 1
    assert blocked[0]["step"] == "prefreeze_threshold"
    assert blocked[0]["error_type"] == "RuntimeError"
    assert "derive_feature_spec missing" in blocked[0]["error"]
