from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from gx1.analysis import shadow_meta_v1 as mod


def _write_candidate(path: Path, *, accepted: bool = True, trainable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "accepted": accepted,
                "trainable_mask_v1": trainable,
                "meta_allow_label_v1": True,
                "run_id": path.parent.name,
            }
        ]
    ).to_parquet(path, index=False)


def test_parse_truth_e2e_run_window_accepts_monday_and_orderfix_ids() -> None:
    assert mod._parse_truth_e2e_run_window("E2E_SANITY_ORDERFIX_20260211_20260218") == (
        date(2026, 2, 11),
        date(2026, 2, 18),
    )
    assert mod._parse_truth_e2e_run_window("TRUTH_MONFRI_WEEK_20260302_20260309") == (
        date(2026, 3, 2),
        date(2026, 3, 9),
    )
    assert mod._parse_truth_e2e_run_window("TRUTH_MONFRI_WEEK_20260302_20260309_PREFLIGHT") is None


def test_build_shadow_meta_v2_split_manifest_supports_monday_runs(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    reports_root.mkdir()

    run_ids = [
        "TRUTH_MONFRI_WEEK_20260202_20260209",
        "TRUTH_MONFRI_WEEK_20260209_20260216",
        "TRUTH_MONFRI_WEEK_20260216_20260223",
        "TRUTH_MONFRI_WEEK_20260223_20260302",
        "TRUTH_MONFRI_WEEK_20260302_20260309",
        "TRUTH_MONFRI_WEEK_20260309_20260316",
    ]
    for run_id in run_ids:
        run_dir = reports_root / run_id
        run_dir.mkdir()
        _write_candidate(run_dir / f"shadow_meta_candidates_{run_id}_MERGED.parquet")

    manifest = mod.build_shadow_meta_v2_split_manifest(
        reports_root=reports_root,
        control_date=date(2026, 4, 11),
        freeze_cut=date(2026, 2, 25),
    )

    assert manifest["pre_freeze_train_runs"] == ["TRUTH_MONFRI_WEEK_20260202_20260209"]
    assert manifest["pre_freeze_val_runs"] == [
        "TRUTH_MONFRI_WEEK_20260209_20260216",
        "TRUTH_MONFRI_WEEK_20260216_20260223",
    ]
    assert manifest["post_freeze_holdout_runs"] == [
        "TRUTH_MONFRI_WEEK_20260302_20260309",
        "TRUTH_MONFRI_WEEK_20260309_20260316",
    ]
    assert manifest["excluded_freeze_overlap_runs"] == ["TRUTH_MONFRI_WEEK_20260223_20260302"]
    assert manifest["verification"]["excluded_freeze_overlap_run_count"] == 1
    assert manifest["verification"]["prefreeze_val_run_count"] == 2
    assert manifest["verification"]["post_freeze_holdout_run_count"] == 2


def test_sanitize_ml_feature_frame_v1_keeps_sklearn_safe_with_pd_na() -> None:
    raw = pd.DataFrame(
        {
            "numeric_feature": [1.5, pd.NA, 3.0, 4.5],
            "categorical_feature": ["LONDON", pd.NA, "NY", "ASIA"],
        }
    )
    sanitized = mod._sanitize_ml_feature_frame_v1(raw, ["numeric_feature", "categorical_feature"])

    assert sanitized.loc[1, "categorical_feature"] is None
    assert pd.isna(sanitized.loc[1, "numeric_feature"])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), ["numeric_feature"]),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["categorical_feature"],
            ),
        ],
        remainder="drop",
    )
    model = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=200, random_state=0))])

    model.fit(sanitized, pd.Series(["TAKE", "WAIT", "TAKE", "WAIT"]))
    preds = model.predict(sanitized)

    assert len(preds) == 4
