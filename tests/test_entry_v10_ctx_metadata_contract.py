from gx1.contracts.signal_bridge_v3 import ORDERED_CTX_CONT_NAMES_V3
from gx1.models.entry_v10.entry_v10_bundle import _infer_entry_bundle_capabilities
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    _build_active_head_names,
    _build_ordered_ctx_cont_names,
)


def test_v10_metadata_ctx_cont_names_use_full_v3_contract() -> None:
    legacy_base_names = list(ORDERED_CTX_CONT_NAMES_V3[:21])

    got = _build_ordered_ctx_cont_names(len(ORDERED_CTX_CONT_NAMES_V3), legacy_base_names)

    assert got == list(ORDERED_CTX_CONT_NAMES_V3)
    assert len(got) == len(ORDERED_CTX_CONT_NAMES_V3)


def test_v10_metadata_active_heads_include_enabled_smart_heads() -> None:
    got = _build_active_head_names(
        enable_tf_agreement_head=True,
        enable_path_quality_variance_head=True,
        enable_position_size_head=True,
        enable_hold_horizon_head=True,
        enable_mtf_direction_head=True,
        enable_dip_head=True,
        enable_forecast_head=True,
        enable_timing_head=True,
        enable_tail_risk_head=True,
        enable_vol_forecast_head=True,
    )

    assert got == [
        "direction",
        "tradable",
        "path_quality",
        "mfe_first_n",
        "bad_path",
        "clean_edge",
        "survival",
        "tf_agreement",
        "path_quality_log_var",
        "position_size",
        "hold_horizon",
        "mtf_direction",
        "dip",
        "forecast",
        "timing",
        "tail_risk",
        "vol_forecast",
    ]


def test_v10_bundle_capabilities_accept_declared_smart_heads_from_state_dict() -> None:
    state_dict = {}
    for prefix in [
        "head_direction",
        "head_path_quality",
        "head_mfe_first_n",
        "head_tradable",
        "head_bad_path",
        "head_clean_edge",
        "head_survival",
        "head_dip",
        "head_forecast",
        "head_timing",
        "head_tail_risk",
        "head_vol_forecast",
        "head_mtf_direction",
    ]:
        state_dict[f"{prefix}.weight"] = object()
        state_dict[f"{prefix}.bias"] = object()
    state_dict["mtf_dir_scale"] = object()
    meta = {
        "supports_context_features": True,
        "train_recipe": {
            "active_heads": [
                "direction",
                "dip",
                "forecast",
                "timing",
                "tail_risk",
                "vol_forecast",
                "mtf_direction",
            ]
        },
    }

    got = _infer_entry_bundle_capabilities(meta, state_dict)

    assert "dip" in got["supported_heads"]
    assert "forecast" in got["supported_heads"]
    assert "timing" in got["supported_heads"]
    assert "tail_risk" in got["supported_heads"]
    assert "vol_forecast" in got["supported_heads"]
    assert "mtf_direction" in got["supported_heads"]
    assert got["supports_context_features"] is True
