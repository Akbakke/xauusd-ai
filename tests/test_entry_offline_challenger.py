from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gx1.contracts.entry_offline_challenger_v1 import (
    OFFLINE_CHALLENGER_CONTRACT,
    OFFLINE_CHALLENGER_RESULT_EVENT_PREFIX,
    OFFLINE_CHALLENGER_RESULT_SCHEMA_VERSION,
    OfflineChampionChallengerError,
    publish_offline_challenger_comparison,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _result(
    root: Path,
    *,
    candidate_id: str,
    bundle: str,
    train_end: datetime,
    evaluation_start: datetime,
    evaluation_end: datetime,
    pnl: float,
    test_seal_used: bool = False,
    cost_model: str = "costs",
) -> dict[str, str]:
    created = evaluation_end + timedelta(hours=1)
    payload = {
        "schema_version": OFFLINE_CHALLENGER_RESULT_SCHEMA_VERSION,
        "created_utc": created.isoformat().replace("+00:00", "Z"),
        "json_path": "",
        "decision": "PASS",
        "failures": [],
        "contract": OFFLINE_CHALLENGER_CONTRACT,
        "candidate_id": candidate_id,
        "bundle_sha256": _sha(bundle),
        "feature_contract_sha256": _sha("features"),
        "target_contract_sha256": _sha("targets"),
        "decision_contract_sha256": _sha("raw-q"),
        "cost_model_sha256": _sha(cost_model),
        "training_window": {
            "start_utc": "2024-01-01T00:00:00Z",
            "end_utc": train_end.isoformat().replace("+00:00", "Z"),
        },
        "evaluation_window": {
            "start_utc": evaluation_start.isoformat().replace("+00:00", "Z"),
            "end_utc": evaluation_end.isoformat().replace("+00:00", "Z"),
        },
        "evaluation_scope": "rolling_oos",
        "test_seal_used": test_seal_used,
        "metrics": {
            "net_pnl_bps": pnl,
            "win_rate": 0.55,
            "max_drawdown_loss_bps": 12.0,
            "trade_count": 30,
            "mean_mae_bps": 4.0,
            "mean_mfe_bps": 7.0,
            "mae_before_mfe_rate": 0.25,
        },
        "activation_authority": False,
        "promotion_allowed": False,
        "online_weight_updates_allowed": False,
        "background_scheduler_allowed": False,
    }
    path, _ = write_immutable_json_event(
        root, OFFLINE_CHALLENGER_RESULT_EVENT_PREFIX, payload
    )
    return {"json_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_publishes_review_only_comparison_with_exact_shared_oos_window(tmp_path: Path) -> None:
    evaluation_start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    evaluation_end = datetime(2025, 7, 8, tzinfo=timezone.utc)
    champion = _result(
        tmp_path / "champion",
        candidate_id="CHAMPION_0001",
        bundle="champion",
        train_end=evaluation_start,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        pnl=5.0,
    )
    challenger = _result(
        tmp_path / "challenger",
        candidate_id="CHALLENGER_0001",
        bundle="challenger",
        train_end=evaluation_start,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        pnl=8.0,
    )

    path, report = publish_offline_challenger_comparison(
        out_dir=tmp_path / "comparison",
        champion_result=champion,
        challenger_result=challenger,
        created_utc="2025-07-08T01:00:00Z",
    )

    assert path.is_file()
    assert report["decision"] == "READY_FOR_HUMAN_REVIEW"
    assert report["metric_deltas"]["net_pnl_bps"] == 3.0
    assert report["review_required"] is True
    assert report["activation_authority"] is False
    assert report["promotion_allowed"] is False
    assert report["online_weight_updates_allowed"] is False
    assert report["background_scheduler_allowed"] is False


def test_rejects_noncausal_training_window(tmp_path: Path) -> None:
    evaluation_start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    evaluation_end = datetime(2025, 7, 8, tzinfo=timezone.utc)
    champion = _result(
        tmp_path / "champion",
        candidate_id="CHAMPION_0001",
        bundle="champion",
        train_end=evaluation_start,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        pnl=5.0,
    )
    challenger = _result(
        tmp_path / "challenger",
        candidate_id="CHALLENGER_0001",
        bundle="challenger",
        train_end=evaluation_start + timedelta(seconds=1),
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        pnl=8.0,
    )

    with pytest.raises(OfflineChampionChallengerError, match="training window"):
        publish_offline_challenger_comparison(
            out_dir=tmp_path / "comparison",
            champion_result=champion,
            challenger_result=challenger,
            created_utc="2025-07-08T01:00:00Z",
        )


def test_rejects_sealed_test_and_economic_mismatch(tmp_path: Path) -> None:
    evaluation_start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    evaluation_end = datetime(2025, 7, 8, tzinfo=timezone.utc)
    champion = _result(
        tmp_path / "champion",
        candidate_id="CHAMPION_0001",
        bundle="champion",
        train_end=evaluation_start,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        pnl=5.0,
    )
    sealed_challenger = _result(
        tmp_path / "sealed_challenger",
        candidate_id="CHALLENGER_0001",
        bundle="challenger",
        train_end=evaluation_start,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        pnl=8.0,
        test_seal_used=True,
    )
    with pytest.raises(OfflineChampionChallengerError, match="review-only rolling-OOS"):
        publish_offline_challenger_comparison(
            out_dir=tmp_path / "comparison_sealed",
            champion_result=champion,
            challenger_result=sealed_challenger,
            created_utc="2025-07-08T01:00:00Z",
        )

    different_cost_challenger = _result(
        tmp_path / "different_cost_challenger",
        candidate_id="CHALLENGER_0002",
        bundle="challenger_2",
        train_end=evaluation_start,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        pnl=8.0,
        cost_model="different-costs",
    )
    with pytest.raises(OfflineChampionChallengerError, match="cost_model_sha256"):
        publish_offline_challenger_comparison(
            out_dir=tmp_path / "comparison_economics",
            champion_result=champion,
            challenger_result=different_cost_challenger,
            created_utc="2025-07-08T01:00:00Z",
        )


def test_rejects_a_different_unseen_window(tmp_path: Path) -> None:
    evaluation_start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    evaluation_end = datetime(2025, 7, 8, tzinfo=timezone.utc)
    champion = _result(
        tmp_path / "champion",
        candidate_id="CHAMPION_0001",
        bundle="champion",
        train_end=evaluation_start,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        pnl=5.0,
    )
    challenger = _result(
        tmp_path / "challenger",
        candidate_id="CHALLENGER_0001",
        bundle="challenger",
        train_end=evaluation_start,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end + timedelta(days=1),
        pnl=8.0,
    )

    with pytest.raises(OfflineChampionChallengerError, match="evaluation_window"):
        publish_offline_challenger_comparison(
            out_dir=tmp_path / "comparison",
            champion_result=champion,
            challenger_result=challenger,
            created_utc="2025-07-09T01:00:00Z",
        )
