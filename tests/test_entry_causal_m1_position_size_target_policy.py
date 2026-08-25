from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gx1.contracts.entry_causal_m1_position_size_target_policy_v1 import (
    causal_m1_position_size_targets_from_policy,
    causal_m1_position_size_target_policy_contract,
    fit_causal_m1_position_size_target_policy,
    require_causal_m1_position_size_target_manifest_binding,
    require_causal_m1_position_size_target_policy,
)
from tests.test_entry_causal_m1_target_policy import _fit, _m1, _sha


def test_fit_and_apply_use_only_exact_m1_selected_side_paths(tmp_path: Path) -> None:
    m1 = _m1()
    m5 = pd.DataFrame({"time": m1["time"].iloc[::5].reset_index(drop=True)})
    direction = _fit()
    policy = fit_causal_m1_position_size_target_policy(
        closed_m5=m5,
        closed_m1=m1,
        entry_causal_m1_target_policy=direction,
        source_parquet_sha256=_sha("m5"),
        tape_provenance_sha256=_sha("tape"),
        m1_source_sha256=_sha("m1"),
        ecdf_artifact_path=tmp_path / "m1_ecdf.npy",
    )
    assert require_causal_m1_position_size_target_policy(policy) == policy
    assert policy["fit_long_rows"] > 0
    assert policy["fit_short_rows"] > 0
    assert causal_m1_position_size_target_policy_contract(policy)[
        "position_size_target_source"
    ] == "train_fitted_exact_m1_selected_side_path_ecdf"
    binding = {
        **causal_m1_position_size_target_policy_contract(policy),
    }
    assert require_causal_m1_position_size_target_manifest_binding(
        binding,
        expected_source_parquet_sha256=_sha("m5"),
        expected_tape_provenance_sha256=_sha("tape"),
        expected_m1_source_sha256=_sha("m1"),
        expected_direction_policy_sha256=direction["policy_sha256"],
    ) == policy
    output = causal_m1_position_size_targets_from_policy(
        policy=policy,
        mfe_first_n_bps=np.array([20.0, 30.0, 5.0]),
        mae_first_n_bps=np.array([2.0, 3.0, 1.0]),
        selected_side=np.array([0, 1, -1]),
        trade_mask=np.array([True, True, False]),
    )
    assert output["mask"].tolist() == [1.0, 1.0, 0.0]
    assert output["target"][2] == 0.0
    assert np.all((output["target"][:2] > 0.0) & (output["target"][:2] <= 1.0))
