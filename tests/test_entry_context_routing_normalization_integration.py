from __future__ import annotations

import numpy as np

from gx1.contracts.entry_model_native_input_normalization_v1 import (
    fit_surface_normalization,
    share_temporal_alias_stats_from_ctx,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    model_native_context_temporal_alias_policy,
)


def test_ctx_cont_is_sole_train_statistics_owner_for_temporal_alias() -> None:
    signal_names = ["ctx_cont.atr_bps", "momentum_fixture"]
    ctx_names = ["atr_bps", "spread_fixture"]
    alias = model_native_context_temporal_alias_policy(signal_names)["aliases"]
    signal = fit_surface_normalization(
        np.column_stack(
            [
                np.linspace(100.0, 900.0, 64),
                np.linspace(-2.0, 3.0, 64),
            ]
        ),
        surface="signal",
        field_names=signal_names,
    )
    ctx_cont = fit_surface_normalization(
        np.column_stack(
            [
                np.linspace(1.0, 9.0, 64),
                np.linspace(10.0, 40.0, 64),
            ]
        ),
        surface="ctx_cont",
        field_names=ctx_names,
    )

    shared = share_temporal_alias_stats_from_ctx(
        signal,
        ctx_cont,
        temporal_aliases=alias,
    )
    signal_index = alias[0]["signal_index"]
    ctx_index = alias[0]["ctx_cont_index"]

    assert np.float32(shared["center"][signal_index]).tobytes() == np.float32(
        ctx_cont["center"][ctx_index]
    ).tobytes()
    assert np.float32(shared["scale"][signal_index]).tobytes() == np.float32(
        ctx_cont["scale"][ctx_index]
    ).tobytes()
    assert shared["scale_source"][signal_index] == ctx_cont["scale_source"][
        ctx_index
    ]
