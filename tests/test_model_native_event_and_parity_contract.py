from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts import model_native_serve_gate_v1 as serve_gate
from gx1.models.entry_v10 import direction_decision_contract
from gx1.scripts import audit_model_native_direction_pockets_v1 as pocket_audit
from gx1.scripts import verify_model_native_serve_parity_v1 as serve_parity


EVENT_ID = "v10_6yr_rebuild_20260716_fresh_xau_direction_repair"
EVENT_ROOT = (
    Path("/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605") / EVENT_ID
)
FULL_DATASET = EVENT_ROOT / "v10_dataset_6yr_smartctx_xau_direction_repair"
PINNED_PATH = (
    EVENT_ROOT
    / "serve_parity/selective_edge_predictions_20260716T120000123456Z.parquet"
)


def test_v4_family_timeframe_token_arithmetic_is_exact() -> None:
    assert len(serve_parity.SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS) == 40
    assert len(serve_parity.MULTI_TF_PER_BAR_FEATURES_V4) == 111
    assert len(serve_parity.SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS) == 555
    assert len(set(serve_parity.SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS)) == 555


def _softmax(values: tuple[float, ...]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    exp = np.exp(array - array.max())
    return (exp / exp.sum()).tolist()


def _fresh_pinned_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    logit_rows = ((4.0, 1.0, 0.0), (0.0, 3.0, 1.0), (0.0, 1.0, 4.0))
    specialists = tuple(serve_parity.MODEL_NATIVE_REQUIRED_SPECIALISTS)
    for offset in range(48):
        logits = logit_rows[offset % len(logit_rows)]
        public_logits = (max(logits[0], logits[1]), logits[2])
        probabilities = _softmax(logits)
        public_probabilities = _softmax(public_logits)
        direction_index = int(np.argmax(np.asarray(logits)))
        public_index = int(np.argmax(np.asarray(public_logits)))
        edge_score = max(probabilities[0], probabilities[1]) - probabilities[2]
        gate = [0.05] * len(specialists)
        gate[offset % len(specialists)] = 0.65
        tf_gate = [0.05] * 5
        tf_gate[offset % 5] = 0.80
        family_tf_width = 5 * len(specialists)
        family_tf_base = (1.0 - 0.76) / (family_tf_width - 1)
        family_tf_gate = [family_tf_base] * family_tf_width
        family_tf_gate[offset % len(family_tf_gate)] = 0.76
        family_tf_feature_gate = [
            1.0 + 0.05 * np.sin(0.2 * offset + 0.013 * index)
            for index in range(
                len(serve_parity.SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS)
            )
        ]

        def vector(width: int, base: float) -> list[float]:
            return [base + 0.01 * offset + 0.001 * index for index in range(width)]

        raw_logits = vector(3, -0.15)
        side_logits = vector(2, -0.1)
        side_probs = _softmax(tuple(side_logits))
        side_bad_path_logit = vector(2, -0.4)
        side_bad_path_probs = [1.0 / (1.0 + np.exp(-item)) for item in side_bad_path_logit]
        side_validity_logit = vector(2, 0.1)
        side_validity_probs = [1.0 / (1.0 + np.exp(-item)) for item in side_validity_logit]
        mtf_dir_logits = vector(3, -0.2)
        mtf_dir_probs = _softmax(tuple(mtf_dir_logits))
        trendline_rail_logits = vector(6, -0.5)
        trendline_rail_probs = [
            1.0 / (1.0 + np.exp(-item)) for item in trendline_rail_logits
        ]
        path_log_var = -0.3 + 0.01 * offset
        action_value = vector(9, -0.25)
        expectile_value = [
            -0.1 + 0.005 * offset + 0.001 * index
            for index in range(3)
        ]
        action_advantage = [
            value - expectile_value[index % 3]
            for index, value in enumerate(action_value)
        ]

        rows.append(
            {
                "time": pd.Timestamp("2026-02-02T00:00:00Z") + pd.Timedelta(minutes=5 * offset),
                "split": "test",
                "model": "candidate",
                "selection_score_mode": direction_decision_contract.MODEL_DIRECTION_SELECTION_MODE,
                "selection_score": edge_score,
                "pred_direction": direction_index,
                "trade_side": direction_index,
                "public_trade_flat_hard_decision": public_index,
                "direction_logits": list(logits),
                "raw_direction_logits": raw_logits,
                "direction_probs": probabilities,
                "public_trade_flat_decision_logits": list(public_logits),
                "public_trade_flat_decision_probs": public_probabilities,
                "p_long": probabilities[0],
                "p_short": probabilities[1],
                "p_flat": probabilities[2],
                "edge_score": edge_score,
                "public_trade_probability": public_probabilities[0],
                "public_flat_probability": public_probabilities[1],
                "path_quality_pred": 11.0 + offset,
                "path_quality_raw": 10.0 + offset,
                "path_quality": 11.0 + offset,
                "model_native_logits": vector(3, -0.3),
                "tradable_logit": -0.2 + 0.01 * offset,
                "tradable_prob": 0.60 + 0.005 * offset,
                "mfe_first_n": 15.0 + offset,
                "mfe_first_n_pred": 15.0 + offset,
                "bad_path_logit_raw": -1.0 + 0.01 * offset,
                "bad_path_logit": -0.9 + 0.01 * offset,
                "bad_path_prob": 0.10 + 0.003 * offset,
                "clean_edge_logit": 0.1 + 0.01 * offset,
                "clean_edge_prob": 0.55 + 0.004 * offset,
                "survival_logit": 0.2 + 0.01 * offset,
                "survival_prob": 0.58 + 0.004 * offset,
                "tf_agreement_logit": 0.05 + 0.01 * offset,
                "tf_agreement_prob": 0.50 + 0.005 * offset,
                "path_quality_log_var": path_log_var,
                "path_quality_std": float(np.exp(0.5 * path_log_var)),
                "position_size_logit": -0.4 + 0.02 * offset,
                "position_size_pred": 0.40 + 0.01 * offset,
                "dip_pred": vector(18, 0.1),
                "forecast_pred": vector(4, 0.2),
                "timing_pred": vector(12, 0.3),
                "tail_risk_pred": vector(6, 0.4),
                "vol_forecast_pred": vector(3, 0.5),
                "action_value": action_value,
                "expectile_value": expectile_value,
                "action_advantage": action_advantage,
                "mtf_dir_logits": mtf_dir_logits,
                "mtf_dir_probs": mtf_dir_probs,
                "p_trade": public_probabilities[0],
                "p_flat_hier": public_probabilities[1],
                "p_long_given_trade": side_probs[0],
                "p_short_given_trade": side_probs[1],
                "side_logits": side_logits,
                "side_probs": side_probs,
                "side_utility": vector(2, 0.2),
                "side_bad_path_logit": side_bad_path_logit,
                "long_bad_path_prob": side_bad_path_probs[0],
                "short_bad_path_prob": side_bad_path_probs[1],
                "side_mae": vector(2, 0.3),
                "trendline_rail_logits": trendline_rail_logits,
                "trendline_rail_probs": trendline_rail_probs,
                "side_validity_logit": side_validity_logit,
                "long_validity_prob": side_validity_probs[0],
                "short_validity_prob": side_validity_probs[1],
                "trade_logit": -0.1 + 0.01 * offset,
                "specialist_gate": gate,
                "tf_gate": tf_gate,
                "family_tf_cooperation_gate": family_tf_gate,
                "family_tf_feature_gate": family_tf_feature_gate,
            }
        )
    return pd.DataFrame(rows)


def test_model_native_parity_requires_explicit_event_inputs() -> None:
    assert not hasattr(serve_parity, "FRESH_XAU_DIRECTION_EVENT_ROOT")
    assert not hasattr(serve_parity, "DEFAULT_DATASET_DIR")
    assert not hasattr(serve_parity, "DEFAULT_CV3_PATH")
    assert not hasattr(serve_parity, "DEFAULT_PINNED_PREDICTIONS")
    assert not hasattr(serve_parity, "DEFAULT_OUT_DIR")
    source = Path(serve_parity.__file__).read_text(encoding="utf-8")
    assert 'ap.add_argument("--dataset-dir", type=Path, required=True)' in source
    assert 'ap.add_argument("--pair-manifest-path", type=Path, required=True)' in source
    assert 'ap.add_argument("--pair-generation-root", type=Path, required=True)' in source
    assert "PrebuiltStateLoader(canonical_v3_path=" not in source
    assert '"--out-dir",' in source
    for retired_option in (
        '"--split"',
        '"--n-bars"',
        '"--min-ts"',
        '"--max-ts"',
        '"--state-tol"',
        '"--forward-tol"',
    ):
        assert retired_option not in source
    assert serve_parity.SERVE_PARITY_SAMPLE_COUNT == 256
    assert serve_parity.SERVE_PARITY_STATE_TOL == 1e-5
    assert serve_parity.SERVE_PARITY_FORWARD_TOL == 1e-3


def test_serve_parity_resolves_test_dataset_only_from_prediction_report(
    tmp_path: Path,
) -> None:
    dataset_dir = (tmp_path / "dataset").resolve()
    dataset_dir.mkdir()
    parquet_path = dataset_dir / "model_native_test.parquet"
    parquet_path.write_bytes(b"bound-test-artifact")
    manifest_path = dataset_dir / "model_native_test.manifest.json"
    manifest_path.write_text(
        json.dumps({"output_data_path": str(parquet_path)}, sort_keys=True),
        encoding="utf-8",
    )
    (dataset_dir / "decoy_test.parquet").write_bytes(b"unbound-decoy")
    report = {
        "dataset_signal_contract": {
            "splits": {
                "test": {
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": serve_parity.sha256_file(manifest_path),
                    "parquet_path": str(parquet_path),
                    "parquet_sha256": serve_parity.sha256_file(parquet_path),
                }
            }
        }
    }

    assert (
        serve_parity._prediction_report_test_parquet(report, dataset_dir)
        == parquet_path
    )

    parquet_path.write_bytes(b"changed-after-report")
    with pytest.raises(RuntimeError, match="TEST parquet hash mismatch"):
        serve_parity._prediction_report_test_parquet(report, dataset_dir)


def test_serve_parity_has_no_dataset_glob_fallback() -> None:
    source = Path(serve_parity.__file__).read_text(encoding="utf-8")

    assert "dataset_dir.glob" not in source
    assert "_prediction_report_test_parquet(" in source
    assert "_load_offline_rows(dataset_parquet, targets)" in source


def test_serve_parity_uses_shared_model_direction_contract() -> None:
    assert (
        serve_parity.MODEL_DIRECTION_SELECTION_MODE
        == direction_decision_contract.MODEL_DIRECTION_SELECTION_MODE
        == "model_direction_argmax"
    )
    assert (
        serve_parity.require_model_direction_decision_contract
        is direction_decision_contract.require_model_direction_decision_contract
    )
    source = Path(serve_parity.__file__).read_text(encoding="utf-8")
    assert "SEQ_LEN_MODEL_NATIVE" in source
    assert 'report["model_native_state_contract"]' in source
    assert "SEQ_LEN_SMART520" not in source
    assert 'report["smart520_state_contract"]' not in source


def test_fresh_pinned_contract_accepts_only_canonical_model_direction_rows() -> None:
    validated = serve_parity._validate_pinned_prediction_contract(
        _fresh_pinned_frame(),
        dataset_dir=FULL_DATASET,
        pinned_path=PINNED_PATH,
    )

    assert isinstance(validated.index, pd.DatetimeIndex)
    assert len(validated) == 48
    assert validated["pred_direction"].tolist()[:3] == [0, 1, 2]
    assert validated["public_trade_flat_hard_decision"].tolist()[:3] == [0, 0, 1]


def test_forward_delta_contract_covers_logits_public_pair_and_heads() -> None:
    validated = serve_parity._validate_pinned_prediction_contract(
        _fresh_pinned_frame(),
        dataset_dir=FULL_DATASET,
        pinned_path=PINNED_PATH,
    )
    pinned = validated.iloc[0]
    head = {
        live_key: pinned[pinned_column]
        for pinned_column, live_key in serve_parity.FORWARD_FIELD_MAP.items()
    }
    head["time"] = validated.index[0]

    deltas = serve_parity._forward_row_deltas(head, pinned)

    assert set(deltas) == set(serve_parity.FORWARD_COLS)
    assert all(delta == 0.0 for delta in deltas.values())


def test_full_test_prediction_liveness_proves_every_head_and_specialist_gate() -> None:
    report = serve_parity._test_prediction_liveness_contract(_fresh_pinned_frame())

    assert report["decision"] == "PASS"
    assert report["failures"] == []
    assert report["active_heads"] == list(serve_parity.MODEL_NATIVE_ACTIVE_HEADS)
    assert tuple(report["active_head_evidence"]) == tuple(
        serve_parity.MODEL_NATIVE_ACTIVE_HEADS
    )
    assert report["specialist_gate"]["decision"] == "PASS"
    assert report["tf_gate"]["decision"] == "PASS"
    assert report["family_tf_cooperation_gate"]["decision"] == "PASS"
    assert report["family_tf_feature_gate"]["decision"] == "PASS"
    assert all(
        count > 0
        for count in report["specialist_gate"]["top_rank_count"].values()
    )


def test_full_test_prediction_liveness_fails_closed_on_constant_head_or_gate() -> None:
    frame = _fresh_pinned_frame()
    frame["position_size_pred"] = 0.5
    frame["specialist_gate"] = [
        [1.0 / len(serve_parity.MODEL_NATIVE_REQUIRED_SPECIALISTS)]
        * len(serve_parity.MODEL_NATIVE_REQUIRED_SPECIALISTS)
        for _ in range(len(frame))
    ]

    report = serve_parity._test_prediction_liveness_contract(frame)

    assert report["decision"] == "FAIL"
    assert report["active_head_evidence"]["position_size"]["decision"] == "FAIL"
    assert report["specialist_gate"]["decision"] == "FAIL"


@pytest.mark.parametrize("column", ("tf_gate", "family_tf_cooperation_gate"))
def test_full_test_prediction_liveness_fails_closed_on_constant_cooperation_gate(
    column: str,
) -> None:
    frame = _fresh_pinned_frame()
    width = len(frame[column].iloc[0])
    frame[column] = [[1.0 / width] * width for _ in range(len(frame))]

    report = serve_parity._test_prediction_liveness_contract(frame)

    assert report["decision"] == "FAIL"
    assert report[column]["decision"] == "FAIL"


def test_vector_head_liveness_requires_every_component_to_vary() -> None:
    frame = _fresh_pinned_frame()
    frame["dip_pred"] = [
        [0.0 if index == 7 else value for index, value in enumerate(row)]
        for row in frame["dip_pred"]
    ]

    report = serve_parity._test_prediction_liveness_contract(frame)

    assert report["decision"] == "FAIL"
    dip_metric = report["active_head_evidence"]["dip"]["fields"]["dip_pred"]
    fusion_metric = report["active_head_evidence"][
        "model_native_evidence_fusion"
    ]["fields"]["dip_pred"]
    assert dip_metric["component_std"][7] == 0.0
    assert dip_metric["min_component_std"] == 0.0
    assert fusion_metric["min_component_std"] == 0.0


def test_fusion_reference_uses_only_finite_candidate_val_rows_in_exact_layout() -> None:
    test = _fresh_pinned_frame()
    val = test.copy()
    val["split"] = "val"
    val["time"] = val["time"] - pd.Timedelta(days=1)
    unrelated = val.copy()
    unrelated["model"] = "baseline"
    unrelated["model_native_logits"] = [[999.0, 999.0, 999.0]] * len(unrelated)

    reference_frame = serve_parity._validate_fusion_reference_prediction_contract(
        pd.concat([test, unrelated, val], ignore_index=True)
    )
    report = serve_parity._direction_evidence_fusion_reference_contract(
        reference_frame
    )

    assert len(reference_frame) == len(val)
    assert report["split"] == "val"
    assert report["input_dim"] == serve_parity.DIRECTION_EVIDENCE_FUSION_INPUT_DIM
    assert tuple(report["mean_by_input"]) == tuple(
        name for name, _width in serve_parity.DIRECTION_EVIDENCE_FUSION_INPUTS
    )
    assert len(report["ordered_mean_sha256"]) == 64


def test_fusion_reference_fails_closed_without_candidate_val_or_on_nonfinite_input() -> None:
    with pytest.raises(RuntimeError, match="no candidate VAL rows"):
        serve_parity._validate_fusion_reference_prediction_contract(
            _fresh_pinned_frame()
        )

    val = _fresh_pinned_frame()
    val["split"] = "val"
    val.at[0, "model_native_logits"] = [0.0, float("nan"), 1.0]
    with pytest.raises(RuntimeError, match="model_native_logits must be finite"):
        serve_parity._validate_fusion_reference_prediction_contract(val)


def test_fusion_reference_rejects_impossible_action_value_manifold() -> None:
    val = _fresh_pinned_frame()
    val["split"] = "val"
    broken = list(val.at[0, "action_advantage"])
    broken[0] += 0.25
    val.at[0, "action_advantage"] = broken

    with pytest.raises(
        RuntimeError,
        match="violates action_advantage=Q-V",
    ):
        serve_parity._validate_fusion_reference_prediction_contract(val)


def test_candidate_specific_context_tf_and_fusion_ablation_execution(
    tmp_path: Path,
) -> None:
    class AssertActionValueManifold(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            layouts = {
                str(row["name"]): row
                for row in (
                    serve_parity.DIRECTION_EVIDENCE_FUSION_ORDERED_INPUT_LAYOUT
                )
            }
            q_layout = layouts["action_value"]
            v_layout = layouts["expectile_value"]
            a_layout = layouts["action_advantage"]
            q = value[:, q_layout["start"]:q_layout["stop"]]
            v = value[:, v_layout["start"]:v_layout["stop"]]
            advantage = value[:, a_layout["start"]:a_layout["stop"]]
            expected = (q.reshape(-1, 3, 3) - v.unsqueeze(1)).reshape(-1, 9)
            if not torch.allclose(advantage, expected, rtol=0.0, atol=1e-6):
                raise RuntimeError("impossible Q/V/A fusion state")
            return value

    class InfluenceModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.evidence_fusion_norm = AssertActionValueManifold()
            width = serve_parity.DIRECTION_EVIDENCE_FUSION_INPUT_DIM
            weights = torch.stack(
                (
                    torch.full((width,), 0.50),
                    torch.full((width,), -0.30),
                    torch.full((width,), 0.10),
                )
            )
            self.register_buffer("direction_weights", weights)
            self.register_buffer(
                "direction_bias", torch.tensor([0.20, -0.10, 0.05])
            )

        def forward(
            self,
            seq: torch.Tensor,
            snap: torch.Tensor,
            *,
            ctx_cat: torch.Tensor,
            ctx_cont: torch.Tensor,
            **multi_tf: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            base = (
                seq.float().mean(dim=(1, 2))
                + snap.float().mean(dim=1)
                + ctx_cont.float().mean(dim=1)
                + ctx_cat.float().mean(dim=1)
            )
            for timeframe in serve_parity.SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES:
                value = multi_tf[f"seq_{timeframe.lower()}"]
                base = base + value.float().mean(dim=tuple(range(1, value.ndim)))
            offsets = torch.arange(
                1,
                serve_parity.DIRECTION_EVIDENCE_FUSION_INPUT_DIM + 1,
                device=base.device,
                dtype=base.dtype,
            ).reshape(1, -1) * 0.001
            fusion = base.reshape(-1, 1) + offsets
            layouts = {
                str(row["name"]): row
                for row in (
                    serve_parity.DIRECTION_EVIDENCE_FUSION_ORDERED_INPUT_LAYOUT
                )
            }
            q_layout = layouts["action_value"]
            v_layout = layouts["expectile_value"]
            a_layout = layouts["action_advantage"]
            q = fusion[:, q_layout["start"]:q_layout["stop"]]
            v = fusion[:, v_layout["start"]:v_layout["stop"]]
            fusion[:, a_layout["start"]:a_layout["stop"]] = (
                q.reshape(-1, 3, 3) - v.unsqueeze(1)
            ).reshape(-1, 9)
            fusion = self.evidence_fusion_norm(fusion)
            raw = torch.nn.functional.linear(fusion, self.direction_weights)
            final = raw / 1.25 + self.direction_bias
            return {
                "raw_direction_logits": raw,
                "direction_logits": final,
            }

    class InfluenceAdapter:
        def __init__(self, bundle_dir: Path, times: pd.DatetimeIndex) -> None:
            self.bundle_dir = bundle_dir
            self.device = torch.device("cpu")
            self._model = InfluenceModel().eval()
            self._meta = {
                "model_native_direction_evidence_fusion": (
                    serve_parity.direction_evidence_fusion_metadata()
                )
            }
            self._positions = {
                pd.Timestamp(value): index for index, value in enumerate(times)
            }

        def _multi_tf_window_tensors(
            self, timestamp: pd.Timestamp
        ) -> dict[str, torch.Tensor]:
            row = self._positions[pd.Timestamp(timestamp)]
            result: dict[str, torch.Tensor] = {}
            for offset, timeframe in enumerate(
                serve_parity.SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES,
                start=1,
            ):
                result[f"seq_{timeframe.lower()}"] = torch.full(
                    (
                        1,
                        2,
                        len(serve_parity.MULTI_TF_PER_BAR_FEATURES_V4),
                    ),
                    0.25 * offset + 0.001 * row,
                    dtype=torch.float32,
                )
            return result

    times = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=serve_parity.SERVE_PARITY_SAMPLE_COUNT,
        freq="5min",
    )
    row = np.arange(serve_parity.SERVE_PARITY_SAMPLE_COUNT, dtype=np.float32)
    states: dict[str, object] = {
        "seq": np.repeat((0.10 + 0.001 * row)[:, None, None], 4, axis=1).reshape(
            -1, 2, 2
        ),
        "snap": np.repeat((0.20 + 0.001 * row)[:, None], 2, axis=1),
        "ctx_cont": np.repeat((0.30 + 0.001 * row)[:, None], 2, axis=1),
        "ctx_cat": np.stack(
            ((row.astype(np.int64) % 3) + 1, (row.astype(np.int64) % 4) + 1),
            axis=1,
        ),
        "times": times.to_numpy(dtype=object),
    }
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    fusion_metadata = serve_parity.direction_evidence_fusion_metadata()
    payload = {"model_native_direction_evidence_fusion": fusion_metadata}
    (bundle_dir / "bundle_metadata.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (bundle_dir / "MASTER_TRANSFORMER_LOCK.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    adapter = InfluenceAdapter(bundle_dir, times)
    candidate_val = _fresh_pinned_frame()
    candidate_val["split"] = "val"
    candidate_val = serve_parity._validate_fusion_reference_prediction_contract(
        candidate_val
    )

    upstream = serve_parity._upstream_context_decision_influence_contract(
        adapter=adapter,
        states=states,
        parity_targets=times,
    )
    multi_tf = serve_parity._multi_tf_decision_influence_contract(
        adapter=adapter,
        states=states,
        parity_targets=times,
    )
    family_tf = serve_parity._family_tf_decision_influence_contract(
        adapter=adapter,
        states=states,
        parity_targets=times,
    )
    fusion = serve_parity._direction_evidence_fusion_influence_contract(
        adapter=adapter,
        states=states,
        parity_targets=times,
        val_reference_frame=candidate_val,
    )

    assert upstream["decision"] == "PASS", upstream["failures"]
    assert multi_tf["decision"] == "PASS", multi_tf["failures"]
    assert family_tf["decision"] == "PASS", family_tf["failures"]
    for report in (upstream, multi_tf, family_tf):
        for metric in report["metrics"].values():
            assert metric["max_abs_class_centered_raw_logit_delta"] > 0.0
            assert metric["raw_changed_rows"] > 0
            assert metric["max_abs_class_centered_logit_delta"] > 0.0
            assert metric["changed_rows"] > 0
    assert set(multi_tf["metrics"]) == set(
        serve_parity.SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES
    )
    assert set(family_tf["metrics"]) == set(
        serve_parity.SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS
    )
    assert fusion["decision"] == "PASS", fusion["failures"]
    assert set(fusion["groups"]) == {
        name for name, _width in serve_parity.DIRECTION_EVIDENCE_FUSION_INPUTS
    }
    for group in fusion["groups"].values():
        assert group["max_abs_class_centered_raw_logit_delta"] > (
            serve_parity.SERVE_PARITY_FUSION_INFLUENCE_EPSILON
        )
        assert group["raw_changed_rows"] >= (
            serve_parity.SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS
        )
        assert group["max_abs_class_centered_logit_delta"] > (
            serve_parity.SERVE_PARITY_FUSION_INFLUENCE_EPSILON
        )
        assert group["changed_rows"] >= (
            serve_parity.SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS
        )
    assert fusion["bundle_metadata_exact_match"] is True
    assert fusion["master_transformer_lock_exact_match"] is True


def test_every_retained_numeric_and_categorical_input_reaches_direction_margins(
    tmp_path: Path,
) -> None:
    signal_names = [
        f"signal_{index:03d}"
        for index in range(serve_parity.MODEL_NATIVE_SIGNAL_DIM)
    ]

    class AllInputModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ctx_cat_embeddings = torch.nn.ModuleList(
                torch.nn.Embedding(4, 1)
                for _name in serve_parity.MODEL_NATIVE_CTX_CAT_FIELDS
            )
            with torch.no_grad():
                for index, embedding in enumerate(self.ctx_cat_embeddings):
                    embedding.weight[:, 0] = (
                        torch.arange(4, dtype=torch.float32)
                        * (0.01 + 0.01 * index)
                    )

        @staticmethod
        def _weighted_sum(value: torch.Tensor) -> torch.Tensor:
            width = int(value.shape[-1])
            weights = torch.linspace(
                0.25,
                1.25,
                width,
                dtype=value.dtype,
                device=value.device,
            )
            return (value * weights).sum(dim=tuple(range(1, value.ndim)))

        def forward(
            self,
            seq: torch.Tensor,
            snap: torch.Tensor,
            *,
            ctx_cat: torch.Tensor,
            ctx_cont: torch.Tensor,
            **multi_tf: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            base = self._weighted_sum(seq.float())
            base = base + self._weighted_sum(snap.float())
            base = base + self._weighted_sum(ctx_cont.float())
            for value in multi_tf.values():
                base = base + self._weighted_sum(value.float())
            for index, embedding in enumerate(self.ctx_cat_embeddings):
                base = base + embedding(ctx_cat[:, index]).reshape(-1)
            raw = torch.stack((0.50 * base, -0.30 * base, 0.10 * base), dim=1)
            return {
                "raw_direction_logits": raw,
                "direction_logits": raw / 1.25,
            }

    class AllInputAdapter:
        def __init__(self, bundle_dir: Path, times: pd.DatetimeIndex) -> None:
            self.bundle_dir = bundle_dir
            self.device = torch.device("cpu")
            self._model = AllInputModel().eval()
            self._meta = {
                "ordered_signal_names": signal_names,
                "ordered_ctx_cont_names": list(
                    serve_parity.MODEL_NATIVE_CTX_CONT_FIELDS
                ),
                "ordered_ctx_cat_names": list(
                    serve_parity.MODEL_NATIVE_CTX_CAT_FIELDS
                ),
            }
            self._positions = {
                pd.Timestamp(value): index for index, value in enumerate(times)
            }

        def _multi_tf_window_tensors(
            self, timestamp: pd.Timestamp
        ) -> dict[str, torch.Tensor]:
            row = self._positions[pd.Timestamp(timestamp)]
            return {
                f"seq_{timeframe.lower()}": torch.full(
                    (
                        1,
                        2,
                        len(serve_parity.MULTI_TF_PER_BAR_FEATURES_V4),
                    ),
                    0.01 + 0.0001 * row,
                    dtype=torch.float32,
                )
                for timeframe in (
                    serve_parity.SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES
                )
            }

    times = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=serve_parity.SERVE_PARITY_SAMPLE_COUNT,
        freq="5min",
    )
    row = np.arange(
        serve_parity.SERVE_PARITY_SAMPLE_COUNT, dtype=np.float32
    )
    signal_dim = serve_parity.MODEL_NATIVE_SIGNAL_DIM
    ctx_cont_dim = len(serve_parity.MODEL_NATIVE_CTX_CONT_FIELDS)
    states: dict[str, object] = {
        "seq": np.repeat(
            (0.10 + 0.0001 * row)[:, None, None],
            2 * signal_dim,
            axis=1,
        ).reshape(-1, 2, signal_dim),
        "snap": np.repeat(
            (0.20 + 0.0001 * row)[:, None], signal_dim, axis=1
        ),
        "ctx_cont": np.repeat(
            (0.30 + 0.0001 * row)[:, None], ctx_cont_dim, axis=1
        ),
        "ctx_cat": np.stack(
            [
                (row.astype(np.int64) + index) % 4
                for index in range(
                    len(serve_parity.MODEL_NATIVE_CTX_CAT_FIELDS)
                )
            ],
            axis=1,
        ),
        "times": times.to_numpy(dtype=object),
    }
    adapter = AllInputAdapter(tmp_path, times)
    report = serve_parity._individual_input_decision_influence_contract(
        adapter=adapter,
        states=states,
        parity_targets=times,
    )

    assert report["decision"] == "PASS", report["failures"]
    assert report["numeric_input_count"] == (2 * 513) + 142 + (5 * 111)
    assert report["categorical_input_count"] == 5
    assert set(report["numeric"]) == {
        "seq_signal",
        "snap_signal",
        "ctx_cont",
        "seq_m5",
        "seq_m15",
        "seq_h1",
        "seq_h4",
        "seq_d1",
    }

    dead_seq_route = json.loads(json.dumps(report))
    first_signal = signal_names[0]
    dead_seq_route["numeric"]["seq_signal"]["metrics"][first_signal][
        "max_abs_raw_class_margin_gradient"
    ] = 0.0
    assert any(
        "numeric.seq_signal" in failure and "is dead" in failure
        for failure in (
            serve_gate._individual_input_decision_influence_contract_failures(
                dead_seq_route
            )
        )
    )


def test_pinned_contract_rejects_direction_only_partial_head_artifact() -> None:
    partial = _fresh_pinned_frame().drop(columns=["specialist_gate", "dip_pred"])

    with pytest.raises(RuntimeError, match="missing required model-direction columns"):
        serve_parity._validate_pinned_prediction_contract(
            partial,
            dataset_dir=FULL_DATASET,
            pinned_path=PINNED_PATH,
        )


def test_pinned_contract_rejects_non_timestamped_prediction_identity() -> None:
    old_path = Path(
        "/home/andre2/GX1_DATA/reports/joint_smart_policy_replay_20260708/"
        "heads_rerun/selective_edge_predictions.parquet"
    )
    with pytest.raises(RuntimeError, match="microsecond-stamped immutable prediction parquet"):
        serve_parity._validate_pinned_prediction_contract(
            _fresh_pinned_frame(),
            dataset_dir=FULL_DATASET,
            pinned_path=old_path,
        )


def test_pinned_contract_rejects_expected_utility_mode_and_columns() -> None:
    old_mode = _fresh_pinned_frame()
    old_mode["selection_score_mode"] = "expected_utility"
    with pytest.raises(RuntimeError, match="must be exactly 'model_direction_argmax'"):
        serve_parity._validate_pinned_prediction_contract(
            old_mode,
            dataset_dir=FULL_DATASET,
            pinned_path=PINNED_PATH,
        )

    old_schema = _fresh_pinned_frame()
    old_schema["expected_utility_long_bps"] = 1.0
    with pytest.raises(RuntimeError, match="legacy expected-utility decision columns are forbidden"):
        serve_parity._validate_pinned_prediction_contract(
            old_schema,
            dataset_dir=FULL_DATASET,
            pinned_path=PINNED_PATH,
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (
            "public_trade_flat_decision_logits",
            [0.0, 4.0],
            "not the canonical pair",
        ),
        ("direction_logits", [4.0, float("nan"), 0.0], "contains non-finite"),
        ("pred_direction", 0.5, "not an exact integer"),
    ],
)
def test_pinned_contract_rejects_malformed_direction_ssot(
    column: str,
    value: object,
    message: str,
) -> None:
    frame = _fresh_pinned_frame()
    frame[column] = frame[column].astype(object)
    frame.at[0, column] = value
    with pytest.raises(RuntimeError, match=message):
        serve_parity._validate_pinned_prediction_contract(
            frame,
            dataset_dir=FULL_DATASET,
            pinned_path=PINNED_PATH,
        )


def test_pinned_contract_rejects_tied_top_direction_logits() -> None:
    frame = _fresh_pinned_frame()
    logits = (4.0, 4.0, 0.0)
    probabilities = _softmax(logits)
    frame.at[0, "direction_logits"] = list(logits)
    frame.at[0, "direction_probs"] = probabilities
    frame.loc[0, ["p_long", "p_short", "p_flat"]] = probabilities
    frame.loc[0, "edge_score"] = max(probabilities[:2]) - probabilities[2]

    with pytest.raises(RuntimeError, match="no unique top class"):
        serve_parity._validate_pinned_prediction_contract(
            frame,
            dataset_dir=FULL_DATASET,
            pinned_path=PINNED_PATH,
        )


def test_mutable_prediction_mirror_fails_closed_without_fallback() -> None:
    mirror = EVENT_ROOT / "serve_parity/selective_edge_predictions.parquet"
    with pytest.raises(RuntimeError, match="not a timestamped authoritative predictions path"):
        serve_parity._load_pinned_predictions(
            dataset_dir=FULL_DATASET,
            pinned_path=mirror,
            prediction_report_path=EVENT_ROOT / "serve_parity/ENTRY_CANDIDATE_SELECTIVE_EDGE_20260716T120000123456Z.json",
        )


def test_parity_sampling_is_exact_deterministic_full_test_span() -> None:
    positions = serve_parity._deterministic_sample_positions(10_001)

    assert len(positions) == 256
    assert len(np.unique(positions)) == 256
    assert positions[0] == 0
    assert positions[-1] == 10_000
    with pytest.raises(RuntimeError, match="at least 256"):
        serve_parity._deterministic_sample_positions(255)


def test_pinned_contract_uses_candidate_test_rows_only() -> None:
    frame = _fresh_pinned_frame()
    val = frame.copy()
    val["split"] = "val"
    val["time"] = val["time"] - pd.Timedelta(days=1)

    validated = serve_parity._validate_pinned_prediction_contract(
        pd.concat([val, frame], ignore_index=True),
        dataset_dir=FULL_DATASET,
        pinned_path=PINNED_PATH,
    )

    assert len(validated) == 48
    assert set(validated["split"]) == {"test"}


def test_direction_pocket_audit_rejects_mutable_prediction_mirror_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror = tmp_path / "selective_edge_predictions.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_model_native_direction_pockets_v1",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--dataset-parquet",
            str(tmp_path / "dataset/entry_model_native_test.parquet"),
            "--predictions-parquet",
            str(mirror),
            "--prediction-report-json",
            str(tmp_path / "ENTRY_CANDIDATE_SELECTIVE_EDGE_20260716T120000123456Z.json"),
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--bundle-metadata-json",
            str(tmp_path / "bundle/bundle_metadata.json"),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(RuntimeError, match="not a timestamped authoritative predictions path"):
        pocket_audit.main()
