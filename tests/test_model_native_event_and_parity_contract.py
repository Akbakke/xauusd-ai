from __future__ import annotations

import json
import os
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
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    model_native_context_temporal_alias_policy,
)


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
    assert len(serve_parity.SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS) == 32
    per_tf_width = len(serve_parity.MULTI_TF_PER_BAR_FEATURES_V4)
    expected_tokens = 4 * per_tf_width
    assert len(serve_parity.SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS) == (
        expected_tokens
    )
    assert len(set(serve_parity.SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS)) == (
        expected_tokens
    )


def test_serve_gate_requires_actual_runtime_cache_and_pair_bindings() -> None:
    cache = {
        "cache_dir": "/tmp/cache",
        "cache_identity_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "m5_prebuilt_source": "/tmp/m5.parquet",
        "m5_prebuilt_source_sha256": "c" * 64,
    }
    pair = {
        "pair_generation_id": "d" * 64,
        "pair_manifest_sha256": "e" * 64,
        "pair_generation_manifest_path": "/tmp/pair.json",
        "canonical_v3_path": "/tmp/canonical.parquet",
        "canonical_v3_sha256": "f" * 64,
        "base28_path": "/tmp/base28.parquet",
        "base28_sha256": "0" * 64,
    }
    assert serve_gate._runtime_mtf_cache_binding_failures(cache) == []
    assert serve_gate._runtime_prebuilt_pair_binding_failures(pair) == []

    cache["manifest_sha256"] = "invalid"
    pair["base28_path"] = "relative.parquet"
    assert serve_gate._runtime_mtf_cache_binding_failures(cache)
    assert serve_gate._runtime_prebuilt_pair_binding_failures(pair)


def _softmax(values: tuple[float, ...]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    exp = np.exp(array - array.max())
    return (exp / exp.sum()).tolist()


def _pinned_column_widths() -> dict[str, int]:
    """Exact per-column widths, read from the serve-gate owner only."""
    widths = dict(serve_gate.SERVE_PARITY_FORWARD_FIELD_WIDTHS)
    for fields in serve_gate.SERVE_PARITY_ACTIVE_HEAD_EVIDENCE_FIELDS.values():
        for column, width in fields.items():
            if widths.setdefault(column, int(width)) != int(width):
                raise AssertionError(
                    f"serve-gate owner declares two widths for {column}"
                )
    return {str(name): int(width) for name, width in widths.items()}


def _fresh_pinned_frame() -> pd.DataFrame:
    """Build a fitted-Q pinned prediction frame from the contract owners.

    Every column, width and semantic comes from the serve-gate/direction
    contract owners; the only literals here are the raw-bps Q rows that make
    each of the three actions win once, which is what the argmax contract
    under test needs.
    """
    rows: list[dict[str, object]] = []
    q_rows = ((4.0, 1.0, 0.0), (0.0, 3.0, 1.0), (0.0, 1.0, 4.0))
    specialists = tuple(serve_parity.MODEL_NATIVE_REQUIRED_SPECIALISTS)
    widths = _pinned_column_widths()
    handled = {
        "entry_action_q_bps",
        "entry_action_q_margin_bps",
        "edge_score",
        "specialist_gate",
        "tf_gate",
        "family_tf_cooperation_gate",
        "family_tf_feature_gate",
    }
    for offset in range(48):
        def vector(width: int, base: float) -> list[float]:
            return [
                base + 0.01 * offset + 0.001 * index for index in range(width)
            ]

        entry_q = [
            value + 0.001 * offset * (index + 1)
            for index, value in enumerate(q_rows[offset % len(q_rows)])
        ]
        action_index = int(np.argmax(np.asarray(entry_q)))
        ordered = sorted(entry_q, reverse=True)
        margin = ordered[0] - ordered[1]

        gate = [0.05] * len(specialists)
        gate[offset % len(specialists)] = 1.0 - 0.05 * (len(specialists) - 1)
        tf_width = len(serve_parity.SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES)
        tf_gate = [0.05] * tf_width
        tf_gate[offset % tf_width] = 1.0 - 0.05 * (tf_width - 1)
        family_tf_width = tf_width * len(specialists)
        family_tf_base = (1.0 - 0.76) / (family_tf_width - 1)
        family_tf_gate = [family_tf_base] * family_tf_width
        family_tf_gate[offset % len(family_tf_gate)] = 0.76
        family_tf_feature_gate = [
            1.0 + 0.05 * np.sin(0.2 * offset + 0.013 * index)
            for index in range(
                len(serve_parity.SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS)
            )
        ]

        row: dict[str, object] = {
            "time": pd.Timestamp("2026-02-02T00:00:00Z")
            + pd.Timedelta(minutes=5 * offset),
            "split": "test",
            "model": "candidate",
            "selection_score_mode": (
                direction_decision_contract.MODEL_DIRECTION_SELECTION_MODE
            ),
            "selection_score": float(max(entry_q)),
            "pred_direction": action_index,
            "trade_side": action_index,
            "entry_action_q_bps": entry_q,
            "entry_action_q_margin_bps": float(margin),
            "edge_score": float(margin),
            "specialist_gate": gate,
            "tf_gate": tf_gate,
            "family_tf_cooperation_gate": family_tf_gate,
            "family_tf_feature_gate": family_tf_feature_gate,
        }
        for column, width in widths.items():
            if column in handled:
                continue
            row[column] = (
                float(0.1 + 0.01 * offset)
                if width == 1
                else vector(width, 0.1 + 0.001 * len(column))
            )
        rows.append(row)
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
    # The decision authority is the unique argmax of the raw-bps Q surface.
    assert validated["trade_side"].tolist() == validated["pred_direction"].tolist()
    for entry_q, action in zip(
        validated["entry_action_q_bps"], validated["pred_direction"]
    ):
        assert int(np.argmax(np.asarray(entry_q))) == int(action)


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
    assert dip_metric["component_std"][7] == 0.0
    assert dip_metric["min_component_std"] == 0.0


def test_retired_direction_evidence_fusion_reference_cannot_reenter() -> None:
    """The VAL fusion-reference subsystem is retired with the direction head.

    ``_validate_fusion_reference_prediction_contract``,
    ``_direction_evidence_fusion_reference_contract``,
    ``_direction_evidence_fusion_influence_contract``,
    ``_require_action_value_manifold`` and
    ``direction_evidence_fusion_metadata`` were the VAL-reference owners of a
    counterfactual Q/V/A manifold that no longer exists: the Entry action
    value is the frozen fitted-Q teacher (gx1/contracts/entry_fitted_q_v1.py).
    No serve-parity route may reintroduce them.
    """
    for retired in (
        "_validate_fusion_reference_prediction_contract",
        "_direction_evidence_fusion_reference_contract",
        "_direction_evidence_fusion_influence_contract",
        "_require_action_value_manifold",
        "_batched_fusion_input_margin_gradients",
        "direction_evidence_fusion_metadata",
        "DIRECTION_EVIDENCE_FUSION_INPUTS",
        "DIRECTION_EVIDENCE_FUSION_INPUT_DIM",
        "DIRECTION_EVIDENCE_FUSION_ORDERED_INPUT_LAYOUT",
    ):
        assert not hasattr(serve_parity, retired)
    source = Path(serve_parity.__file__).read_text(encoding="utf-8")
    assert "action_advantage" not in source
    assert "expectile_value" not in source


def test_candidate_specific_context_tf_and_family_ablation_execution(
    tmp_path: Path,
) -> None:
    class InfluenceModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "action_weights", torch.tensor([0.50, -0.30, 0.10])
            )
            self.register_buffer(
                "action_bias", torch.tensor([0.20, -0.10, 0.05])
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
            entry_q = (
                base.reshape(-1, 1) * self.action_weights.reshape(1, -1)
                + self.action_bias.reshape(1, -1)
            )
            return {"entry_action_q_bps": entry_q}

    class InfluenceAdapter:
        def __init__(self, bundle_dir: Path, times: pd.DatetimeIndex) -> None:
            self.bundle_dir = bundle_dir
            self.device = torch.device("cpu")
            self._model = InfluenceModel().eval()
            self._meta: dict[str, object] = {}
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
    adapter = InfluenceAdapter(bundle_dir, times)

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

    assert upstream["decision"] == "PASS", upstream["failures"]
    assert multi_tf["decision"] == "PASS", multi_tf["failures"]
    assert family_tf["decision"] == "PASS", family_tf["failures"]
    for report in (upstream, multi_tf, family_tf):
        for metric in report["metrics"].values():
            assert metric["max_abs_entry_action_q_delta_bps"] > 0.0
            assert metric["raw_changed_rows"] > 0
            assert metric["max_abs_entry_action_q_margin_delta_bps"] > 0.0
            assert metric["changed_rows"] > 0
    assert set(multi_tf["metrics"]) == set(
        serve_parity.SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES
    )
    assert set(family_tf["metrics"]) == set(
        serve_parity.SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS
    )


def test_every_retained_numeric_and_categorical_input_reaches_direction_margins(
    tmp_path: Path,
) -> None:
    signal_names = [
        f"signal_{index:03d}"
        for index in range(serve_parity.MODEL_NATIVE_SIGNAL_DIM)
    ]
    nominal_ctx_fields = list(
        serve_gate.CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS
    )
    for index, field in enumerate(nominal_ctx_fields):
        signal_names[index] = f"ctx_cont.{field}"
    numeric_alias_field = "atr_bps"
    signal_names[len(nominal_ctx_fields)] = (
        f"ctx_cont.{numeric_alias_field}"
    )
    alias_policy = model_native_context_temporal_alias_policy(signal_names)

    class AllInputModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ctx_cat_embeddings = torch.nn.ModuleList(
                torch.nn.Embedding(
                    len(serve_gate.MODEL_NATIVE_CTX_CAT_DOMAINS[name]), 1
                )
                for name in serve_parity.MODEL_NATIVE_CTX_CAT_FIELDS
            )
            with torch.no_grad():
                for index, embedding in enumerate(self.ctx_cat_embeddings):
                    embedding.weight[:, 0] = (
                        torch.arange(embedding.num_embeddings, dtype=torch.float32)
                        * (0.01 + 0.01 * index)
                    )
            signal_width = len(signal_names)
            ctx_width = len(serve_parity.MODEL_NATIVE_CTX_CONT_FIELDS)
            self.register_buffer(
                "input_norm_signal_center",
                torch.zeros(signal_width, dtype=torch.float32),
            )
            self.register_buffer(
                "input_norm_signal_scale",
                torch.ones(signal_width, dtype=torch.float32),
            )
            self.register_buffer(
                "input_norm_signal_binary_mask",
                torch.zeros(signal_width, dtype=torch.bool),
            )
            self.register_buffer(
                "input_norm_signal_categorical_mask",
                torch.zeros(signal_width, dtype=torch.bool),
            )
            self.register_buffer(
                "input_norm_ctx_cont_binary_mask",
                torch.zeros(ctx_width, dtype=torch.bool),
            )
            self.register_buffer(
                "input_norm_ctx_cont_categorical_mask",
                torch.zeros(ctx_width, dtype=torch.bool),
            )
            for alias in alias_policy["aliases"]:
                if alias["ctx_cont_field"] in nominal_ctx_fields:
                    self.input_norm_signal_categorical_mask[
                        int(alias["signal_index"])
                    ] = True
                    self.input_norm_ctx_cont_categorical_mask[
                        int(alias["ctx_cont_index"])
                    ] = True
            self._alias_pairs = tuple(
                (
                    int(alias["signal_index"]),
                    int(alias["ctx_cont_index"]),
                )
                for alias in alias_policy["aliases"]
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
            for signal_index, ctx_index in self._alias_pairs:
                if not torch.equal(seq[:, -1, signal_index], snap[:, signal_index]):
                    raise RuntimeError("test temporal alias seq/snap left manifold")
                if not torch.equal(
                    snap[:, signal_index], ctx_cont[:, ctx_index]
                ):
                    raise RuntimeError("test temporal alias snap/ctx left manifold")
            base = self._weighted_sum(seq.float())
            base = base + self._weighted_sum(snap.float())
            base = base + self._weighted_sum(ctx_cont.float())
            for value in multi_tf.values():
                base = base + self._weighted_sum(value.float())
            for index, embedding in enumerate(self.ctx_cat_embeddings):
                base = base + embedding(ctx_cat[:, index]).reshape(-1)
            entry_q = torch.stack(
                (0.50 * base, -0.30 * base, 0.10 * base), dim=1
            )
            return {"entry_action_q_bps": entry_q}

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
            windows = {
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
            # V30 (2026-08-14): `regime_class_id` is retired from the MTF lane;
            # the ternary EMA-stack alignment is the remaining exact-domain
            # per-timeframe field.
            stack_index = serve_parity.MULTI_TF_PER_BAR_FEATURES_V4.index(
                "ema_stack_aligned_v2"
            )
            for tensor in windows.values():
                tensor[..., stack_index] = float(row % 3) - 1.0
            return windows

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
    ctx_cont_values = np.repeat(
        (0.30 + 0.0001 * row)[:, None], ctx_cont_dim, axis=1
    )
    for field in serve_gate.CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS:
        index = list(serve_parity.MODEL_NATIVE_CTX_CONT_FIELDS).index(field)
        ctx_cont_values[:, index] = row.astype(np.int64) % 5
    seq_values = np.repeat(
        (0.10 + 0.0001 * row)[:, None, None],
        2 * signal_dim,
        axis=1,
    ).reshape(-1, 2, signal_dim)
    snap_values = np.repeat(
        (0.20 + 0.0001 * row)[:, None], signal_dim, axis=1
    )
    for alias in alias_policy["aliases"]:
        signal_index = int(alias["signal_index"])
        ctx_index = int(alias["ctx_cont_index"])
        alias_values = ctx_cont_values[:, ctx_index]
        seq_values[:, -1, signal_index] = alias_values
        snap_values[:, signal_index] = alias_values
    states: dict[str, object] = {
        "seq": seq_values,
        "snap": snap_values,
        "ctx_cont": ctx_cont_values,
        "ctx_cat": np.stack(
            [
                (row.astype(np.int64) + index) % 4
                % len(
                    serve_gate.MODEL_NATIVE_CTX_CAT_DOMAINS[
                        serve_parity.MODEL_NATIVE_CTX_CAT_FIELDS[index]
                    ]
                )
                for index in range(len(serve_parity.MODEL_NATIVE_CTX_CAT_FIELDS))
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
    expected_ownership = serve_gate.individual_input_influence_layout(signal_names)
    assert report["numeric_input_count"] == sum(
        len(row["tokens"])
        for row in expected_ownership["numeric"].values()
    )
    assert report["categorical_input_count"] == len(
        expected_ownership["categorical"]
    )
    assert report["continuous_manifold_input_count"] == len(
        expected_ownership["continuous_manifold"]
    )
    assert set(report["numeric"]) == {
        "seq_signal",
        "snap_signal",
        "ctx_cont",
        "seq_m15",
        "seq_h1",
        "seq_h4",
        "seq_d1",
    }
    numeric_alias_token = f"temporal_alias.{numeric_alias_field}"
    assert report["continuous_manifold"][numeric_alias_token][
        "decision"
    ] == "PASS"
    assert all(
        report["categorical"][f"temporal_alias.{field}"]["decision"]
        == "PASS"
        for field in nominal_ctx_fields
    )

    dead_seq_route = json.loads(json.dumps(report))
    first_signal = report["numeric"]["seq_signal"]["tokens"][0]
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

    alias_misclassified = json.loads(json.dumps(report))
    numeric_alias = next(
        row
        for row in expected_ownership["continuous_manifold"]
        if row["token"] == numeric_alias_token
    )
    alias_misclassified["input_ownership"]["numeric"]["snap_signal"][
        "tokens"
    ].append(str(numeric_alias["signal_field"]))
    alias_misclassified["input_ownership"]["numeric"]["snap_signal"][
        "source_indices"
    ].append(int(numeric_alias["signal_index"]))
    assert any(
        "input_ownership mismatch" in failure
        for failure in (
            serve_gate._individual_input_decision_influence_contract_failures(
                alias_misclassified
            )
        )
    )

    # V30 (2026-08-14): the per-timeframe `regime_class_id` categorical is
    # retired, so a fabricated extra MTF token is the remaining ownership
    # mismatch this owner must reject.
    mtf_misclassified = json.loads(json.dumps(report))
    mtf_misclassified["input_ownership"]["numeric"]["seq_m15"]["tokens"].append(
        "m15:not_an_owned_token"
    )
    mtf_misclassified["input_ownership"]["numeric"]["seq_m15"][
        "source_indices"
    ].append(len(serve_parity.MULTI_TF_PER_BAR_FEATURES_V4))
    assert any(
        "input_ownership mismatch" in failure
        for failure in (
            serve_gate._individual_input_decision_influence_contract_failures(
                mtf_misclassified
            )
        )
    )


def test_individual_input_layout_uses_physical_owners_and_nominal_manifolds() -> None:
    signal_names = [
        f"signal_{index:03d}" for index in range(MODEL_NATIVE_SIGNAL_DIM)
    ]
    nominal_ctx = list(serve_gate.CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS)
    for index, field in enumerate(nominal_ctx):
        signal_names[index] = f"ctx_cont.{field}"
    numeric_alias = "atr_bps"
    signal_names[len(nominal_ctx)] = f"ctx_cont.{numeric_alias}"
    signal_names[-1] = "smc_swing_state"

    layout = serve_gate.individual_input_influence_layout(signal_names)
    numeric = layout["numeric"]
    continuous_manifold = layout["continuous_manifold"]
    categorical = layout["categorical"]
    snap_tokens = set(numeric["snap_signal"]["tokens"])
    seq_tokens = set(numeric["seq_signal"]["tokens"])
    ctx_tokens = set(numeric["ctx_cont"]["tokens"])

    assert f"ctx_cont.{numeric_alias}" not in seq_tokens
    assert f"ctx_cont.{numeric_alias}" not in snap_tokens
    assert numeric_alias not in ctx_tokens
    numeric_owner = next(
        row
        for row in continuous_manifold
        if row["token"] == f"temporal_alias.{numeric_alias}"
    )
    assert numeric_owner["manifold"] == "joint_seq_last_snap_ctx_cont_alias"
    assert numeric_owner["perturbation"] == (
        "binary_flip_or_train_frozen_center_else_one_scale"
    )
    for field in nominal_ctx:
        assert f"ctx_cont.{field}" not in seq_tokens
        assert f"ctx_cont.{field}" not in snap_tokens
        assert field not in ctx_tokens
        owner = next(
            row for row in categorical if row["token"] == f"temporal_alias.{field}"
        )
        assert owner["manifold"] == "joint_seq_last_snap_ctx_cont_alias"
    assert "smc_swing_state" not in seq_tokens
    assert "smc_swing_state" not in snap_tokens
    signal_owner = next(
        row for row in categorical if row["token"] == "signal.smc_swing_state"
    )
    assert signal_owner == {
        "token": "signal.smc_swing_state",
        "owner": "signal_nominal_embedding",
        "surface": "signal",
        "source_index": MODEL_NATIVE_SIGNAL_DIM - 1,
        "field": "smc_swing_state",
        "domain": [0, 1, 2, 3, 4],
        "manifold": "causal_local_history_category",
    }
    # V30 (2026-08-14): the per-timeframe `regime_class_id` categorical is
    # retired, so every multi-TF token is numeric.  The one declared local
    # signal category stays an embedding owner alongside ctx_cat.
    assert not serve_gate.MTF_SEMANTIC_CATEGORICAL_DOMAINS
    mtf_tokens = {
        token
        for timeframe in serve_parity.SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES
        for token in numeric[f"seq_{timeframe.lower()}"]["tokens"]
    }
    assert not [token for token in mtf_tokens if token.endswith(":regime_class_id")]
    assert {row["surface"] for row in categorical} == {"ctx_cat", "signal"}
    assert {row["token"] for row in categorical} == {
        f"ctx_cat.{field}" for field in serve_parity.MODEL_NATIVE_CTX_CAT_FIELDS
    } | {"signal.smc_swing_state"}


def test_pinned_contract_rejects_direction_only_partial_head_artifact() -> None:
    partial = _fresh_pinned_frame().drop(columns=["specialist_gate", "dip_pred"])

    with pytest.raises(RuntimeError, match="missing fitted-Q columns"):
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
    with pytest.raises(
        RuntimeError,
        match=(
            "selection_score_mode must be exactly "
            f"{serve_parity.MODEL_DIRECTION_SELECTION_MODE!r}"
        ),
    ):
        serve_parity._validate_pinned_prediction_contract(
            old_mode,
            dataset_dir=FULL_DATASET,
            pinned_path=PINNED_PATH,
        )

    old_schema = _fresh_pinned_frame()
    old_schema["expected_utility_long_bps"] = 1.0
    with pytest.raises(
        RuntimeError, match="retired Entry decision columns are forbidden"
    ):
        serve_parity._validate_pinned_prediction_contract(
            old_schema,
            dataset_dir=FULL_DATASET,
            pinned_path=PINNED_PATH,
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (
            "entry_action_q_bps",
            [0.0, 4.0],
            r"entry_action_q_bps must have shape \(3,\)",
        ),
        (
            "entry_action_q_bps",
            [4.0, float("nan"), 0.0],
            "contains non-finite",
        ),
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


def test_pinned_contract_rejects_tied_top_entry_action_q() -> None:
    frame = _fresh_pinned_frame()
    tied = [4.0, 4.0, 0.0]
    frame["entry_action_q_bps"] = frame["entry_action_q_bps"].astype(object)
    frame.at[0, "entry_action_q_bps"] = tied
    frame.loc[0, "selection_score"] = max(tied)
    frame.loc[0, "entry_action_q_margin_bps"] = 0.0

    with pytest.raises(RuntimeError, match="no unique top action"):
        serve_parity._validate_pinned_prediction_contract(
            frame,
            dataset_dir=FULL_DATASET,
            pinned_path=PINNED_PATH,
        )


def test_mutable_prediction_mirror_fails_closed_without_fallback(
    tmp_path: Path,
) -> None:
    mirror = tmp_path / "selective_edge_predictions.parquet"
    report = tmp_path / "ENTRY_CANDIDATE_SELECTIVE_EDGE_20260716T120000123456Z.json"
    bundle_dir = tmp_path / "candidate_bundle"
    bundle_dir.mkdir()
    with pytest.raises(RuntimeError, match="not a timestamped authoritative predictions path"):
        serve_parity._load_pinned_predictions(
            dataset_dir=FULL_DATASET,
            bundle_dir=bundle_dir,
            pinned_path=mirror,
            prediction_report_path=report,
            expected_predictions_sha256="0" * 64,
        )


class _ResolverReached(Exception):
    """Raised by the spies below once the call under test has been observed."""


def test_parity_loader_hands_the_operator_bundle_to_the_evidence_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`bundle_dir` reaching the resolver must be the explicit candidate bundle.

    The gate used to pass `bundle_dir=None`, which (a) raised `TypeError`
    inside the resolver so the gate could not reach any verdict at all, and
    (b) would, if it had resolved, have let the prediction report nominate its
    own bundle instead of being checked against the operator's `--bundle-dir`.
    The resolver binds that directory's `bundle_metadata.json` hash,
    `state_dict_sha256` and direction-decision contract to the prediction
    event, so the value must be the caller's, unmodified.
    """
    captured: dict[str, object] = {}

    def _spy(requested_path, **kwargs):
        captured["requested_path"] = requested_path
        captured.update(kwargs)
        raise _ResolverReached()

    monkeypatch.setattr(
        serve_parity, "resolve_and_validate_prediction_evidence", _spy
    )
    bundle_dir = (tmp_path / "candidate_bundle").resolve()
    bundle_dir.mkdir()
    dataset_dir = (tmp_path / "dataset").resolve()
    pinned = tmp_path / "selective_edge_predictions_20260716T120000123456Z.parquet"
    report = tmp_path / "ENTRY_CANDIDATE_SELECTIVE_EDGE_20260716T120000123456Z.json"

    with pytest.raises(_ResolverReached):
        serve_parity._load_pinned_predictions(
            dataset_dir=dataset_dir,
            bundle_dir=bundle_dir,
            pinned_path=pinned,
            prediction_report_path=report,
            expected_predictions_sha256="0" * 64,
        )

    assert captured["bundle_dir"] == bundle_dir
    assert captured["dataset_dir"] == dataset_dir
    assert captured["expected_stage"] == (
        serve_parity.MODEL_NATIVE_REQUIRED_EVIDENCE_STAGE
    )
    assert captured["expected_splits"] == (
        serve_parity.MODEL_NATIVE_REQUIRED_TEST_SPLIT,
    )
    assert captured["expected_model"] == serve_parity.MODEL_NATIVE_REQUIRED_MODEL_NAME


def test_parity_main_forwards_the_explicit_bundle_dir_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--bundle-dir` must reach the pinned-prediction loader from `main`.

    This is the wiring the `bundle_dir=None` defect bypassed: `main` already
    required `--bundle-dir` and only compared it against the report *after*
    the prediction evidence had been resolved without it.
    """
    captured: dict[str, object] = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        raise _ResolverReached()

    # `main` assigns the parity env pins process-wide; register them with
    # monkeypatch first so this test cannot leak CUDA_VISIBLE_DEVICES="" into
    # the rest of the suite.
    for pin_name, pin_value in serve_gate.SERVE_PARITY_ENV_PINS.items():
        monkeypatch.setenv(pin_name, os.environ.get(pin_name, pin_value))

    monkeypatch.setattr(serve_parity, "_load_pinned_predictions", _spy)
    bundle_dir = (tmp_path / "candidate_bundle").resolve()
    bundle_dir.mkdir()
    dataset_dir = (tmp_path / "dataset").resolve()
    dataset_dir.mkdir()
    pinned = tmp_path / "selective_edge_predictions_20260716T120000123456Z.parquet"
    report = tmp_path / "ENTRY_CANDIDATE_SELECTIVE_EDGE_20260716T120000123456Z.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_model_native_serve_parity_v1",
            "--dataset-dir",
            str(dataset_dir),
            "--pair-manifest-path",
            str(tmp_path / "pair_manifest.json"),
            "--pair-generation-root",
            str(tmp_path / "generations"),
            "--pinned-predictions",
            str(pinned),
            "--pinned-predictions-sha256",
            "0" * 64,
            "--prediction-report-json",
            str(report),
            "--bundle-dir",
            str(bundle_dir),
            "--max-trades",
            "1",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(_ResolverReached):
        serve_parity.main()

    assert captured["bundle_dir"] == bundle_dir
    assert captured["dataset_dir"] == dataset_dir
    assert captured["prediction_report_path"] == report


def test_parity_does_not_second_guess_the_evidence_bundle_comparison() -> None:
    """The bundle_dir comparison has exactly one owner: the evidence resolver.

    `main` previously re-implemented `report['bundle_dir'] == --bundle-dir`
    itself. With the resolver receiving the real bundle that copy is dead and
    is a second owner of one contract (rule 13), so it must not come back.
    """
    source = Path(serve_parity.__file__).read_text(encoding="utf-8")

    assert "bundle_dir=None" not in source
    assert "prediction_bundle_dir" not in source
    assert (
        "prediction evidence bundle does not equal the explicit" not in source
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
            "--predictions-sha256",
            "0" * 64,
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
