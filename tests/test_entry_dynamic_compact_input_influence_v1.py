from __future__ import annotations

import copy
import random
import unittest

import numpy as np
import torch

from gx1.contracts import entry_dynamic_compact_input_influence_v1 as influence


class _AnalyticDynamicForward(torch.nn.Module):
    """Known derivatives and a unique final-horizon maximum, no native model."""

    def __init__(self, coordinate, *, cancel_candidate=False):
        super().__init__()
        self.coordinate = coordinate
        self.cancel_candidate = cancel_candidate
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.unused = torch.nn.Parameter(torch.tensor(2.0))
        self.child = torch.nn.Identity()
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        x = self.coordinate(inputs) * self.weight
        h = torch.arange(1, 481, dtype=x.dtype, device=x.device)[None, :]
        x = x[:, None]
        terminal = torch.stack((4 * x + h, 2 * x + 2 * h,
                                torch.zeros_like(x + h)), dim=-1)
        mae = torch.stack(((x + 10).expand_as(x + h),
                           (x + 10).expand_as(x + h)), dim=-1)
        if self.cancel_candidate:
            terminal = torch.cat((mae + torch.stack((h.expand_as(x + h),
                                                     (2 * h).expand_as(x + h)), dim=-1),
                                  torch.zeros_like(terminal[..., :1])), dim=-1)
        quality = terminal[..., :2] - mae
        best = quality.max(dim=1).values
        flat = torch.zeros_like(best[..., :1])
        components = {
            "dynamic_terminal_net_markout_bps": terminal,
            "dynamic_mfe_bps": torch.stack(((7 * x).expand_as(x + h),
                                             (-3 * x).expand_as(x + h)), dim=-1),
            "dynamic_pre_mfe_mae_upper_bps": mae,
            "dynamic_closed_mfe_bps": torch.stack(((-2 * x).expand_as(x + h),
                                                    (5 * x).expand_as(x + h)), dim=-1),
            "dynamic_closed_pre_mfe_mae_upper_bps": torch.stack(
                ((x + 20).expand_as(x + h), (2 * x + 20).expand_as(x + h)), dim=-1),
        }
        return {
            **components,
            "dynamic_entry_action_quality_bps_candidate": torch.cat((best, flat), dim=-1),
            "dynamic_entry_quality_bps_candidate": torch.cat(
                (quality, torch.zeros_like(quality[..., :1])), dim=-1),
            "dynamic_entry_quality_best_horizon_mask": quality == best[:, None, :],
            "dynamic_query_horizon_minutes": torch.arange(1, 481, device=x.device),
        }


class TestDynamicCompactInputInfluenceV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = influence.compile_dynamic_compact_influence_layout()

    def inputs(self, batch=2):
        domains = self.layout["domains"]
        flat = {}
        for surface, spec in domains.items():
            width = len(spec["field_names"])
            shape = (batch, 96, width) if surface == "seq_x" else (
                (batch, 3, width) if surface.startswith("seq_") else (batch, width))
            flat[surface] = torch.zeros(shape, dtype=torch.int64 if surface == "ctx_cat" else torch.float32)
            for index, value in spec["source_constants"].items():
                flat[surface][..., index] = value
        return {**{k: v for k, v in flat.items() if k in ("seq_x", "snap_x", "ctx_cont", "ctx_cat")},
                "mtf": {k: v for k, v in flat.items() if k not in ("seq_x", "snap_x", "ctx_cont", "ctx_cat")}}

    @staticmethod
    def flat(inputs):
        return {**{k: inputs[k] for k in ("seq_x", "snap_x", "ctx_cont", "ctx_cat")}, **inputs["mtf"]}

    def set_value(self, inputs, surface, index, value, *, time_index=-1):
        flat = self.flat(inputs)
        if surface.startswith("seq_"):
            flat[surface][:, time_index, index] = value
            if surface != "seq_x" or time_index not in (-1, 95):
                return
        elif surface == "snap_x":
            flat[surface][:, index] = value
        elif surface == "ctx_cont":
            flat[surface][:, index] = value
            matches = [a for a in self.layout["aliases"] if a["ctx_cont_index"] == index]
            if matches:
                index = matches[0]["signal_index"]
            else:
                return
        else:
            flat[surface][:, index] = value
            return
        flat["seq_x"][:, -1, index] = value
        flat["snap_x"][:, index] = value
        for alias in self.layout["aliases"]:
            if alias["signal_index"] == index:
                flat["ctx_cont"][:, alias["ctx_cont_index"]] = value

    def numeric(self):
        spec = self.layout["domains"]["seq_x"]
        constrained = set(spec["binary_indices"]) | set(spec["enum_domains"]) | set(spec["source_constants"])
        constrained |= {i for pair in spec["masking_pairs"] for i in pair}
        return next(row for row in self.layout["numeric"]
                    if row["surface"] == "seq_x" and row["source_index"] not in constrained)

    def masked_alias(self, *, binary=False):
        spec = self.layout["domains"]["ctx_cont"]
        by_index = {row["ctx_cont_index"]: row for row in self.layout["aliases"]}
        masks = {mask for _, mask in spec["masking_pairs"]}
        for value, mask in spec["masking_pairs"]:
            if value not in by_index or mask not in by_index or value in masks:
                continue
            name = spec["field_names"][value]
            if bool(value in spec["binary_indices"]) == binary and value not in spec["enum_domains"] and "range_fraction" not in name:
                return by_index[value], by_index[mask]
        self.fail("Source layout has no declared masked value of the requested kind")

    def observed(self, *, binary=False, value=2):
        inputs = self.inputs()
        owner, mask = self.masked_alias(binary=binary)
        self.set_value(inputs, "ctx_cont", mask["ctx_cont_index"], 1)
        self.set_value(inputs, "ctx_cont", owner["ctx_cont_index"], value)
        return inputs, owner, mask

    def normalization(self):
        result = {"surfaces": {}}
        for surface, spec in self.layout["domains"].items():
            key = "signal" if surface in ("seq_x", "snap_x") else (
                "mtf_" + surface[4:] if surface.startswith("seq_") else surface)
            result["surfaces"][key] = {"field_names": tuple(spec["field_names"]),
                "center": torch.zeros(len(spec["field_names"])), "scale": torch.ones(len(spec["field_names"]))}
        return result

    def case(self, inputs, owner, *, kind="value", normalization=None, time_index=None):
        return influence.make_dynamic_compact_counterfactual(inputs, layout=self.layout,
            token=owner["token"], witness_row=0, kind=kind,
            normalization=self.normalization() if normalization is None else normalization, time_index=time_index)

    def assert_vector(self, actual, expected):
        self.assertTrue(torch.equal(actual, torch.full_like(actual, expected)), (actual, expected))

    def test_compiled_active_physical_routes_are_exact_and_parked_are_excluded(self):
        layout = self.layout
        self.assertEqual(layout["counts"], {"numeric": 882, "aliases": 2525, "categorical": 7, "families": 32})
        self.assertEqual({k: len(v) for k, v in layout["physical_field_names"].items()},
                         {"signal": 2725, "ctx_cont": 2542, "ctx_cat": 1, "mtf": 253})
        profile = layout["profile"]
        for row in layout["numeric"]:
            active = profile["signal_indices"] if row["surface"] == "seq_x" else (
                profile["generic_snap_signal_indices"] if row["surface"] == "snap_x" else profile["mtf_indices"])
            self.assertIn(row["source_index"], active)
        self.assertEqual(sum(row["surface"] == "local" for row in layout["families"]), 7)
        self.assertEqual({lane: sum(row["surface"] == lane for row in layout["families"])
                          for lane in ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1")},
                         {lane: 5 for lane in ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1")})
        tokens = [row["token"] for key in ("numeric", "aliases", "categorical", "families") for row in layout[key]]
        self.assertEqual(len(tokens), len(set(tokens)))

    def test_entire_current_signal_snapshot_equality_is_required(self):
        inputs = self.inputs()
        index = self.numeric()["source_index"]
        inputs["snap_x"][:, index] = 1
        with self.assertRaisesRegex(RuntimeError, "DYNAMIC_COMPACT_INFLUENCE_SEQ_LAST_SNAP_NOT_EXACT"):
            influence.require_dynamic_compact_inputs(inputs, layout=self.layout)

    def test_mask_zero_nonzero_value_is_rejected_before_any_forward(self):
        inputs, owner, mask = self.observed()
        self.set_value(inputs, "ctx_cont", mask["ctx_cont_index"], 0)
        model = _AnalyticDynamicForward(lambda x: x["seq_x"][:, -1, owner["signal_index"]])
        with self.assertRaisesRegex(RuntimeError, "DYNAMIC_COMPACT_INFLUENCE_MASKED_STORAGE"):
            influence.dynamic_compact_input_vjps(model, inputs, layout=self.layout)
        self.assertEqual(model.calls, 0)

    def test_input_binary_dtype_finite_and_history_domains_are_checked(self):
        for defect in ("binary", "dtype", "nan", "history"):
            with self.subTest(defect=defect):
                inputs = self.inputs()
                if defect == "binary":
                    index = self.layout["domains"]["seq_x"]["binary_indices"][0]
                    self.set_value(inputs, "seq_x", index, 0.5)
                elif defect == "dtype":
                    inputs["snap_x"] = inputs["snap_x"].double()
                elif defect == "nan":
                    self.set_value(inputs, "seq_x", self.numeric()["source_index"], float("nan"))
                else:
                    inputs["seq_x"] = inputs["seq_x"][:, 1:]
                with self.assertRaisesRegex(RuntimeError, "DYNAMIC_COMPACT_INFLUENCE_"):
                    influence.require_dynamic_compact_inputs(inputs, layout=self.layout)

    def test_signed_alias_cancellation_cannot_pass_from_absolute_partials(self):
        inputs, owner, _ = self.observed()
        si = owner["signal_index"]
        model = _AnalyticDynamicForward(lambda x: x["seq_x"][:, -1, si] - x["snap_x"][:, si])
        report = influence.dynamic_compact_input_vjps(model, inputs, layout=self.layout)
        for projection in report["projections"]:
            self.assert_vector(report["routes"][owner["token"]][projection]["current_abs"], 0)
        self.assertEqual(model.calls, 1)

    def test_nonalias_current_seq_snapshot_partials_also_cancel_on_legal_manifold(self):
        inputs = self.inputs()
        owner = self.numeric()
        index = owner["source_index"]
        model = _AnalyticDynamicForward(lambda x: x["seq_x"][:, -1, index] - x["snap_x"][:, index])
        report = influence.dynamic_compact_input_vjps(model, inputs, layout=self.layout)
        for projection in report["projections"]:
            row = report["routes"][owner["token"]][projection]
            self.assert_vector(row["current_abs"], 0)
        self.assertGreater(float(report["routes"][owner["token"]]["candidate.long_flat"]["current_partial_abs"].max()), 0)

    def test_all_thirteen_projection_derivatives_have_independent_hand_fasits(self):
        inputs, owner, _ = self.observed()
        model = _AnalyticDynamicForward(lambda x: x["seq_x"][:, -1, owner["signal_index"]])
        report = influence.dynamic_compact_input_vjps(model, inputs, layout=self.layout)
        expected = {"candidate.long_short": 2, "candidate.long_flat": 3, "candidate.short_flat": 1,
            "dynamic_terminal_net_markout_bps.long": 1920, "dynamic_terminal_net_markout_bps.short": 960,
            "dynamic_mfe_bps.long": 3360, "dynamic_mfe_bps.short": -1440,
            "dynamic_pre_mfe_mae_upper_bps.long": 480, "dynamic_pre_mfe_mae_upper_bps.short": 480,
            "dynamic_closed_mfe_bps.long": -960, "dynamic_closed_mfe_bps.short": 2400,
            "dynamic_closed_pre_mfe_mae_upper_bps.long": 480, "dynamic_closed_pre_mfe_mae_upper_bps.short": 960}
        self.assertEqual(set(report["projections"]), set(expected))
        self.assertEqual((report["forward_calls"], report["reverse_calls"]), (1, 13))
        for name, derivative in expected.items():
            with self.subTest(projection=name):
                row = report["routes"][owner["token"]][name]
                self.assert_vector(row["current_signed"], derivative)
                self.assert_vector(row["current_abs"], abs(derivative))
        self.assertNotIn("entry_action_q_bps", report["outputs"])
        self.assert_vector(report["outputs"]["dynamic_entry_action_quality_bps_candidate"][:, 2], 0)

    def test_component_only_response_is_not_candidate_response(self):
        inputs, owner, _ = self.observed()
        model = _AnalyticDynamicForward(lambda x: x["seq_x"][:, -1, owner["signal_index"]], cancel_candidate=True)
        report = influence.dynamic_compact_input_vjps(model, inputs, layout=self.layout)
        for name in ("candidate.long_short", "candidate.long_flat", "candidate.short_flat"):
            self.assert_vector(report["routes"][owner["token"]][name]["current_abs"], 0)
        self.assert_vector(report["routes"][owner["token"]]["dynamic_mfe_bps.long"]["current_abs"], 3360)

    def test_observed_history_cannot_fill_missing_current_evidence(self):
        inputs = self.inputs()
        owner, _ = self.masked_alias()
        si = owner["signal_index"]
        mask = next(mask for value, mask in self.layout["domains"]["seq_x"]["masking_pairs"] if value == si)
        self.set_value(inputs, "seq_x", si, 2, time_index=0)
        self.set_value(inputs, "seq_x", mask, 1, time_index=0)
        report = influence.dynamic_compact_input_vjps(_AnalyticDynamicForward(lambda x: x["seq_x"][:, 0, si]), inputs, layout=self.layout)
        row = report["routes"][owner["token"]]["candidate.long_flat"]
        self.assertFalse(bool(row["observed_current"].any()))
        self.assertTrue(bool(row["observed_history"].all()))
        self.assert_vector(row["current_abs"], 0)
        self.assert_vector(row["history_abs"], 3)
        self.assertEqual(self.case(inputs, owner)["status"], "not_exercised_missing")

    def test_current_alias_value_change_is_atomic_and_leaves_history_intact(self):
        inputs, owner, mask = self.observed()
        self.set_value(inputs, "seq_x", owner["signal_index"], 7, time_index=0)
        history_mask = next(m for v, m in self.layout["domains"]["seq_x"]["masking_pairs"] if v == owner["signal_index"])
        self.set_value(inputs, "seq_x", history_mask, 1, time_index=0)
        case = self.case(inputs, owner)
        self.assertEqual(case["status"], "valid")
        result = case["inputs"]
        self.assert_vector(result["seq_x"][:, -1, owner["signal_index"]], 0)
        self.assert_vector(result["snap_x"][:, owner["signal_index"]], 0)
        self.assert_vector(result["ctx_cont"][:, owner["ctx_cont_index"]], 0)
        self.assert_vector(result["ctx_cont"][:, mask["ctx_cont_index"]], 1)
        self.assertTrue(torch.equal(result["seq_x"][:, :-1], inputs["seq_x"][0:1, :-1]))
        self.assertEqual(len(case["changed_coordinates"]), 3)

    def test_mask_drop_is_atomic_but_never_invents_an_observation(self):
        inputs, owner, mask = self.observed()
        case = self.case(inputs, mask, kind="mask_drop")
        self.assertEqual(case["status"], "valid")
        result = case["inputs"]
        for row in (owner, mask):
            self.assert_vector(result["seq_x"][:, -1, row["signal_index"]], 0)
            self.assert_vector(result["snap_x"][:, row["signal_index"]], 0)
            self.assert_vector(result["ctx_cont"][:, row["ctx_cont_index"]], 0)
        absent = self.inputs()
        self.assertEqual(self.case(absent, mask, kind="mask_drop")["status"], "not_exercised_missing")
        self.assertEqual(self.case(absent, mask, kind="value")["status"], "invalid_counterfactual")

    def test_structural_binary_flip_preserves_genuine_own_mask(self):
        inputs, owner, mask = self.observed(binary=True, value=0)
        case = self.case(inputs, owner)
        self.assertEqual(case["status"], "valid")
        self.assert_vector(case["inputs"]["ctx_cont"][:, owner["ctx_cont_index"]], 1)
        self.assert_vector(case["inputs"]["ctx_cont"][:, mask["ctx_cont_index"]], 1)

    def test_enums_and_signed_ema_ordinal_are_source_domains(self):
        for lane in ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1"):
            spec = self.layout["domains"][lane]
            index = spec["field_names"].index("ema_stack_aligned_v2")
            self.assertEqual(tuple(spec["enum_domains"][index]), (-1, 0, 1))
            for value in (-1, 0, 1, 0.5, 2):
                with self.subTest(lane=lane, value=value):
                    inputs = self.inputs()
                    self.set_value(inputs, lane, index, value)
                    if value in (-1, 0, 1):
                        influence.require_dynamic_compact_inputs(inputs, layout=self.layout)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "DYNAMIC_COMPACT_INFLUENCE_ENUM_DOMAIN"):
                            influence.require_dynamic_compact_inputs(inputs, layout=self.layout)
        for owner in self.layout["categorical"]:
            with self.subTest(category=owner["token"]):
                case = self.case(self.inputs(), owner, kind="category")
                self.assertEqual(case["status"], "valid")
                influence.require_dynamic_compact_inputs(case["inputs"], layout=self.layout)

    def test_parked_range_fraction_is_not_an_active_counterfactual_owner(self):
        owner = next(a for a in self.layout["physical_aliases"]
                     if a["ctx_cont_field"] == "close_distance_below_high_range_fraction")
        active = {row["token"] for group in ("numeric", "aliases", "categorical", "families")
                  for row in self.layout[group]}
        self.assertNotIn(owner["token"], active)
        with self.assertRaisesRegex(RuntimeError, "DYNAMIC_COMPACT_INFLUENCE_OWNER_TOKEN"):
            self.case(self.inputs(), owner)

    def test_frozen_normalization_recipe_and_bad_stats_or_order(self):
        inputs, owner, _ = self.observed()
        norm = self.normalization()
        norm["surfaces"]["signal"]["scale"][owner["signal_index"]] = 3
        norm["surfaces"]["ctx_cont"]["scale"][owner["ctx_cont_index"]] = 3
        case = self.case(inputs, owner, normalization=norm)
        self.assert_vector(case["inputs"]["ctx_cont"][:, owner["ctx_cont_index"]], 0)
        self.set_value(inputs, "ctx_cont", owner["ctx_cont_index"], 0)
        case = self.case(inputs, owner, normalization=norm)
        self.assert_vector(case["inputs"]["ctx_cont"][:, owner["ctx_cont_index"]], 3)
        for defect in ("order", "zero_scale", "infinite_center"):
            with self.subTest(defect=defect):
                bad = copy.deepcopy(norm)
                stats = bad["surfaces"]["signal"]
                if defect == "order": stats["field_names"] = tuple(reversed(stats["field_names"]))
                elif defect == "zero_scale": stats["scale"][owner["signal_index"]] = 0
                else: stats["center"][owner["signal_index"]] = float("inf")
                with self.assertRaisesRegex(RuntimeError, "DYNAMIC_COMPACT_INFLUENCE_NORMALIZATION_"):
                    self.case(inputs, owner, normalization=bad)
        with self.subTest(defect="finite_f64_replacement_outside_float32"):
            inputs, owner, norm = self.inputs(), self.numeric(), self.normalization()
            index, maximum = owner["source_index"], torch.finfo(torch.float32).max
            self.set_value(inputs, "seq_x", index, maximum)
            norm["surfaces"]["signal"]["center"][index] = maximum
            norm["surfaces"]["signal"]["scale"][index] = maximum
            before = {name: value.clone() for name, value in self.flat(inputs).items()}
            case = self.case(inputs, owner, normalization=norm)
            self.assertEqual(case["status"], "invalid_counterfactual")
            self.assertEqual(case["reason"], "replacement_not_finite_float32")
            self.assertNotIn("inputs", case)
            for name, value in self.flat(inputs).items():
                self.assertTrue(torch.equal(value, before[name]))
                self.assertTrue(bool(torch.isfinite(value).all()))

    def test_source_constant_missing_and_sample_constant_are_distinct(self):
        lane = "seq_d1"
        index, value = next(iter(self.layout["domains"][lane]["source_constants"].items()))
        owner = next(row for row in self.layout["numeric"] if row["surface"] == lane and row["source_index"] == index)
        self.assertEqual(value, 0)
        self.assertEqual(self.case(self.inputs(), owner)["status"], "source_declared_constant")
        missing, _ = self.masked_alias()
        self.assertEqual(self.case(self.inputs(), missing)["status"], "not_exercised_missing")
        inputs, observed, _ = self.observed()
        self.assertEqual(self.case(inputs, observed)["status"], "valid")

    def test_family_ablation_keeps_full_current_and_mask_manifolds_valid(self):
        inputs, owner, _ = self.observed()
        family = next(row for row in self.layout["families"] if row["surface"] == "local"
                      and owner["signal_index"] in row["signal_indices"]
                      and owner["ctx_cont_index"] in row["ctx_cont_indices"])
        case = self.case(inputs, family, kind="family")
        self.assertEqual(case["status"], "valid")
        influence.require_dynamic_compact_inputs(case["inputs"], layout=self.layout)
        self.assertTrue(torch.equal(case["inputs"]["seq_x"][:, -1], case["inputs"]["snap_x"]))

    def test_lazy_b16_packing_case_order_padding_and_score_mask(self):
        baseline = self.inputs(1)
        owner = self.numeric()
        consumed = []
        def cases():
            for i in range(17):
                inputs = {k: v.clone() for k, v in baseline.items() if k != "mtf"}
                inputs["mtf"] = {k: v.clone() for k, v in baseline["mtf"].items()}
                self.set_value(inputs, "seq_x", owner["source_index"], i + 1)
                consumed.append(i)
                yield {"status": "valid", "token": "case_" + str(i), "witness_row": 0, "inputs": inputs}
        packed = iter(influence.pack_dynamic_compact_cases(cases(), baseline_inputs=baseline))
        first = next(packed)
        self.assertEqual(consumed, list(range(16)))
        self.assertEqual(first["case_tokens"], tuple("case_" + str(i) for i in range(16)))
        self.assertEqual(int(first["score_mask"].sum()), 16)
        self.assertTrue(torch.equal(first["inputs"]["snap_x"][:, owner["source_index"]], torch.arange(1, 17, dtype=torch.float32)))
        last = next(packed)
        self.assertEqual(consumed, list(range(17)))
        self.assertEqual(last["case_tokens"], ("case_16",) + (None,) * 15)
        self.assertEqual(int(last["score_mask"].sum()), 1)
        self.assert_vector(last["inputs"]["snap_x"][:, owner["source_index"]], 17)
        with self.assertRaises(StopIteration): next(packed)

    def outputs(self, values, *, cancel_candidate=False):
        model = _AnalyticDynamicForward(lambda inputs: inputs["x"], cancel_candidate=cancel_candidate)
        with torch.no_grad():
            return model({"x": values})

    def test_b16_inverse_permutation_parity_has_no_row_reassignment(self):
        values = torch.arange(16, dtype=torch.float32)
        permutation = torch.tensor([15, *range(15)])
        result = influence.require_dynamic_compact_output_parity(self.outputs(values), self.outputs(values[permutation]),
            inverse_permutation=torch.argsort(permutation))
        self.assertEqual((result["bit_exact"], result["rows"]), (True, 16))

    def test_parity_rejects_ulp_signedzero_query_tie_and_bad_inverse(self):
        for defect in ("ulp", "negative_zero", "query", "tie", "inverse"):
            with self.subTest(defect=defect):
                expected = self.outputs(torch.ones(2))
                actual = {k: v.clone() for k, v in expected.items()}
                inverse = None
                if defect == "ulp":
                    actual["dynamic_mfe_bps"][0, 0, 0] = torch.nextafter(actual["dynamic_mfe_bps"][0, 0, 0], torch.tensor(float("inf")))
                elif defect == "negative_zero": actual["dynamic_entry_action_quality_bps_candidate"][0, 2] = -0.0
                elif defect == "query": actual["dynamic_query_horizon_minutes"][0] = 2
                elif defect == "tie": actual["dynamic_entry_quality_best_horizon_mask"][0, 0, 0] = True
                else: inverse = torch.tensor([0, 0])
                with self.assertRaisesRegex(RuntimeError, "DYNAMIC_COMPACT_INFLUENCE_"):
                    influence.require_dynamic_compact_output_parity(expected, actual, inverse_permutation=inverse)

    def test_case_forward_and_reverse_budgets_are_exact(self):
        for count, forwards in ((0, 2), (1, 3), (16, 3), (17, 4), (3446, 218)):
            with self.subTest(count=count):
                result = influence.dynamic_compact_influence_budget(count, count)
                self.assertEqual((result["forward_calls"], result["reverse_calls"], result["execution_authorized"]), (forwards, 13, False))
        for pre, executed in ((3447, 1), (1, 2), (-1, 0), (True, 0), (1, -1)):
            with self.subTest(pre=pre, executed=executed):
                with self.assertRaisesRegex(RuntimeError, "DYNAMIC_COMPACT_INFLUENCE_CASE_BUDGET"):
                    influence.dynamic_compact_influence_budget(pre, executed)

    def test_input_only_vjps_preserve_parameters_gradients_modes_and_rng(self):
        inputs, owner, _ = self.observed()
        before_inputs = {k: v.clone() for k, v in self.flat(inputs).items()}
        model = _AnalyticDynamicForward(lambda x: x["seq_x"][:, -1, owner["signal_index"]])
        model.train(); model.child.eval(); model.weight.grad = torch.tensor(19.0)
        modes = tuple(m.training for m in model.modules())
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        py, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        influence.dynamic_compact_input_vjps(model, inputs, layout=self.layout)
        self.assertEqual(random.getstate(), py)
        after_numpy = np.random.get_state()
        self.assertEqual(numpy[0], after_numpy[0]); self.assertTrue(np.array_equal(numpy[1], after_numpy[1])); self.assertEqual(numpy[2:], after_numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu))
        self.assertEqual(tuple(m.training for m in model.modules()), modes)
        self.assertEqual(model.weight.grad.item(), 19.0); self.assertIsNone(model.unused.grad)
        for k, v in state.items(): self.assertTrue(torch.equal(v, model.state_dict()[k]))
        for k, v in self.flat(inputs).items():
            self.assertTrue(torch.equal(v, before_inputs[k])); self.assertFalse(v.requires_grad); self.assertIsNone(v.grad)

    def test_invalid_output_shapes_flat_and_mae_fail_closed(self):
        for defect in ("flat", "terminal_flat", "mae", "dtype", "missing"):
            with self.subTest(defect=defect):
                inputs, owner, _ = self.observed()
                good = _AnalyticDynamicForward(lambda x: x["seq_x"][:, -1, owner["signal_index"]])
                def forward(x):
                    result = good(x)
                    if defect == "flat": result["dynamic_entry_action_quality_bps_candidate"] = result["dynamic_entry_action_quality_bps_candidate"] + torch.tensor([0, 0, 1.0])
                    elif defect == "terminal_flat": result["dynamic_terminal_net_markout_bps"] = result["dynamic_terminal_net_markout_bps"] + torch.tensor([0, 0, 1.0])
                    elif defect == "mae": result["dynamic_pre_mfe_mae_upper_bps"] = -result["dynamic_pre_mfe_mae_upper_bps"]
                    elif defect == "dtype": result["dynamic_mfe_bps"] = result["dynamic_mfe_bps"].double()
                    else: result.pop("dynamic_entry_action_quality_bps_candidate")
                    return result
                with self.assertRaisesRegex(RuntimeError, "DYNAMIC_COMPACT_INFLUENCE_"):
                    influence.dynamic_compact_input_vjps(forward, inputs, layout=self.layout)

    def test_history_counterfactual_does_not_change_current_aliases(self):
        inputs = self.inputs()
        owner, _ = self.masked_alias()
        mask = next(m for v, m in self.layout["domains"]["seq_x"]["masking_pairs"] if v == owner["signal_index"])
        self.set_value(inputs, "seq_x", owner["signal_index"], 2, time_index=0)
        self.set_value(inputs, "seq_x", mask, 1, time_index=0)
        case = self.case(inputs, owner, time_index=-96)
        self.assertEqual((case["status"], case["route"]), ("valid", "history"))
        result = case["inputs"]
        self.assert_vector(result["seq_x"][:, 0, owner["signal_index"]], 0)
        self.assertTrue(torch.equal(result["seq_x"][:, -1], inputs["seq_x"][0:1, -1]))
        self.assertTrue(torch.equal(result["snap_x"], inputs["snap_x"][0:1]))
        self.assertTrue(torch.equal(result["ctx_cont"], inputs["ctx_cont"][0:1]))

    def test_counterfactual_is_detached_and_baseline_is_immutable(self):
        inputs, owner, _ = self.observed()
        before = {k: v.clone() for k, v in self.flat(inputs).items()}
        case = self.case(inputs, owner)
        for key, value in self.flat(case["inputs"]).items():
            self.assertFalse(value.requires_grad); self.assertIsNone(value.grad)
            self.assertNotEqual(value.data_ptr(), self.flat(inputs)[key].data_ptr())
        for key, value in self.flat(inputs).items(): self.assertTrue(torch.equal(value, before[key]))

    def test_invalid_or_non_B1_cases_never_enter_fixed_B16_packing(self):
        inputs, owner, _ = self.observed()
        case = self.case(inputs, owner)
        for defect in ("missing", "wrong_batch", "batch_width"):
            with self.subTest(defect=defect):
                bad = dict(case)
                width = 16
                if defect == "missing": bad["status"] = "not_exercised_missing"
                elif defect == "wrong_batch": bad["inputs"] = inputs
                else: width = 8
                with self.assertRaisesRegex(RuntimeError, "DYNAMIC_COMPACT_INFLUENCE_"):
                    next(iter(influence.pack_dynamic_compact_cases([bad], baseline_inputs=inputs, batch_rows=width)))


    def test_packed_response_keeps_candidate_components_and_padding_separate(self):
        for cancel in (False, True):
            with self.subTest(component_only=cancel):
                expected = self.outputs(torch.tensor([1.0, 2.0]), cancel_candidate=cancel)
                actual = self.outputs(torch.tensor([1.0, 3.0] + [999.0] * 14), cancel_candidate=cancel)
                result = influence.dynamic_compact_output_response(expected, actual,
                    witness_rows=(0, 1) + (None,) * 14,
                    score_mask=torch.tensor([True, True] + [False] * 14))
                self.assertEqual(result["witness_rows"], (0, 1))
                self.assertTrue(torch.equal(result["packed_rows"], torch.tensor([0, 1])))
                self.assertTrue(torch.equal(result["candidate_margin_delta_bps"], torch.tensor([0.0, 0.0 if cancel else 3.0])))
                hand = {"dynamic_terminal_net_markout_bps": 1 if cancel else 4,
                        "dynamic_mfe_bps": 7, "dynamic_pre_mfe_mae_upper_bps": 1,
                        "dynamic_closed_mfe_bps": 5, "dynamic_closed_pre_mfe_mae_upper_bps": 2}
                self.assertEqual(set(result["component_delta_bps"]), set(hand))
                for key, value in hand.items():
                    self.assertTrue(torch.equal(result["component_delta_bps"][key], torch.tensor([0.0, float(value)])))

    def test_packed_response_rejects_scored_padding_and_wrong_witness_map(self):
        expected = self.outputs(torch.ones(2))
        actual = self.outputs(torch.ones(16))
        for defect in ("padding_witness", "bad_witness", "mask_dtype"):
            with self.subTest(defect=defect):
                witnesses = [0] + [None] * 15
                score = torch.tensor([True] + [False] * 15)
                if defect == "padding_witness": witnesses[-1] = 0
                elif defect == "bad_witness": witnesses[0] = 2
                else: score = score.float()
                with self.assertRaisesRegex(RuntimeError, "DYNAMIC_COMPACT_INFLUENCE_"):
                    influence.dynamic_compact_output_response(expected, actual, witness_rows=witnesses, score_mask=score)
