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



class TestDynamicCompactCaseManifestV1(unittest.TestCase):
    """Small source-shaped inputs: no native constructor, dataset or forward."""

    lanes = ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1")

    def fixture(self, batch=4):
        signal = ("free", "value", "mask", "binary", "ordinal", "constant")
        mtf = ("free", "mask", "value", "ordinal", "constant")
        def domain(names, binary=(), enums=None, pairs=(), constants=None):
            return {"field_names": names, "binary_indices": binary, "enum_domains": enums or {},
                    "masking_pairs": pairs, "source_constants": constants or {}}
        domains = {
            "seq_x": domain(signal, (2, 3), {4: (-1, 0, 1)}, ((1, 2),), {5: 0}),
            "snap_x": domain(signal, (2, 3), {4: (-1, 0, 1)}, ((1, 2),), {5: 0}),
            "ctx_cont": domain(("value", "mask"), (1,), pairs=((0, 1),)),
            "ctx_cat": domain(("regime",), enums={0: (0, 1, 2)}),
            **{lane: domain(mtf, (1,), {3: (-1, 0, 1)}, ((2, 1),),
                            {4: 0} if lane == "seq_d1" else {}) for lane in self.lanes},
        }
        numeric = [
            {"token": "seq.free", "surface": "seq_x", "source_index": 0, "field": "free"},
            {"token": "snap.free", "surface": "snap_x", "source_index": 0, "field": "free"},
            {"token": "m5.value", "surface": "seq_m5", "source_index": 2, "field": "value"},
            {"token": "m5.mask", "surface": "seq_m5", "source_index": 1, "field": "mask"},
            {"token": "m5.ordinal", "surface": "seq_m5", "source_index": 3, "field": "ordinal"},
            {"token": "d1.constant", "surface": "seq_d1", "source_index": 4, "field": "constant"},
            {"token": "seq.binary", "surface": "seq_x", "source_index": 3, "field": "binary"},
        ]
        aliases = [
            {"token": "alias.value", "signal_index": 1, "ctx_cont_index": 0, "ctx_cont_field": "value"},
            {"token": "alias.mask", "signal_index": 2, "ctx_cont_index": 1, "ctx_cont_field": "mask"},
        ]
        categorical = [
            {"token": "category.ctx", "surface": "ctx_cat", "source_index": 0, "domain": (0, 1, 2)},
            {"token": "category.signal", "surface": "signal", "source_index": 4, "domain": (-1, 0, 1)},
        ]
        families = [
            {"token": "family.local", "surface": "local", "signal_indices": (0, 1, 2, 3, 4),
             "ctx_cont_indices": (0, 1), "ctx_cat_indices": (0,)},
            {"token": "family.m5", "surface": "seq_m5", "source_indices": (0, 1, 2, 3)},
        ]
        layout = {"schema_version": influence.SCHEMA_VERSION, "domains": domains,
                  "numeric": numeric, "aliases": aliases, "physical_aliases": aliases,
                  "categorical": categorical, "families": families,
                  "counts": {"numeric": 7, "aliases": 2, "categorical": 2, "families": 2}}
        inputs = {"seq_x": torch.zeros(batch, 96, 6), "snap_x": torch.zeros(batch, 6),
                  "ctx_cont": torch.zeros(batch, 2), "ctx_cat": torch.zeros(batch, 1, dtype=torch.int64),
                  "mtf": {lane: torch.zeros(batch, 3, 5) for lane in self.lanes}}
        for row in range(batch):
            inputs["seq_x"][row, :, 0] = row + 1
            inputs["snap_x"][row, 0] = row + 1
        for lane in self.lanes:
            inputs["mtf"][lane][:, :, 3] = 1
        if batch > 1:
            self.alias_observed(inputs, 1, True)
            inputs["mtf"]["seq_m5"][1, -1, 1] = 1  # Legitimate observed value0.
        normalization = {"surfaces": {}}
        for surface in ("seq_x", "ctx_cont", *self.lanes):
            key = "signal" if surface == "seq_x" else "mtf_" + surface[4:] if surface in self.lanes else surface
            names = domains[surface]["field_names"]
            normalization["surfaces"][key] = {"field_names": names,
                "center": torch.zeros(len(names), dtype=torch.float64),
                "scale": torch.ones(len(names), dtype=torch.float64)}
        return inputs, layout, normalization

    def alias_observed(self, inputs, row, observed, value=0.0):
        inputs["seq_x"][row, -1, 1:3] = torch.tensor([value if observed else 0, int(observed)])
        inputs["snap_x"][row, 1:3] = inputs["seq_x"][row, -1, 1:3]
        inputs["ctx_cont"][row] = inputs["snap_x"][row, 1:3]

    def build(self, inputs, layout, normalization, *, parents=(100, 200, 50), times=(30, 10, 20)):
        return influence.build_dynamic_compact_case_manifest(inputs, layout=layout, normalization=normalization,
            parent_rows=list(parents), decision_times_ns=list(times), genuine_rows=3)

    @staticmethod
    def indexed(manifest):
        return {case["token"]: case for case in manifest["cases"]}

    def test_f32_view_matches_model_buffer_rounding_and_never_mutates_source_stats(self):
        _, layout, normalization = self.fixture()
        normalization["surfaces"]["signal"]["center"][0] = 0.1
        view = influence.dynamic_compact_normalization_f32_view(normalization, layout=layout)
        self.assertEqual(view["surfaces"]["signal"]["center"].dtype, torch.float32)
        self.assertEqual(int(view["surfaces"]["signal"]["center"][0].view(torch.int32)), 0x3DCCCCCD)
        view["surfaces"]["signal"]["center"][0] = 99
        self.assertEqual(normalization["surfaces"]["signal"]["center"].dtype, torch.float64)
        self.assertEqual(float(normalization["surfaces"]["signal"]["center"][0]), 0.1)
        normalization["surfaces"]["signal"]["scale"][0] = 1e-300
        with self.assertRaisesRegex(RuntimeError, "NORMALIZATION_STATS"):
            influence.dynamic_compact_normalization_f32_view(normalization, layout=layout)

    def test_input_digest_has_a_handframed_golden_and_preserves_signed_zero(self):
        tiny = {"seq_x": torch.zeros(1, 1, 1), "snap_x": torch.zeros(1, 1),
                "ctx_cont": torch.zeros(1, 1), "ctx_cat": torch.zeros(1, 1, dtype=torch.int64),
                "mtf": {lane: torch.zeros(1, 1, 1) for lane in reversed(self.lanes)}}
        # Independent stdlib struct-packed nine-scalar little-endian hand stream.
        expected = "db688a330cef890c36db8d38a8a995161e74b517347161799aab1185e8ba0cb4"
        self.assertEqual(influence.dynamic_compact_input_sha256(tiny), expected)
        tiny["ctx_cont"][0, 0] = -0.0
        self.assertNotEqual(influence.dynamic_compact_input_sha256(tiny), expected)
        tiny["ctx_cat"] = tiny["ctx_cat"].float()
        with self.assertRaisesRegex(RuntimeError, "BIT_DIGEST_SHAPE"):
            influence.dynamic_compact_input_sha256(tiny)

    def test_mask_only_witness_order_observed_zero_and_parent_ties(self):
        inputs, layout, normalization = self.fixture()
        self.alias_observed(inputs, 1, False)
        self.alias_observed(inputs, 0, True)
        self.alias_observed(inputs, 2, True)  # Value0 is genuine; no value!=0 selection.
        manifest = self.build(inputs, layout, normalization)
        cases = self.indexed(manifest)
        self.assertEqual(manifest["witness_order"], [1, 2, 0])
        self.assertEqual(cases["seq.free"]["witness_row"], 1)
        self.assertEqual(cases["alias.value"]["witness_row"], 2)
        self.assertEqual(cases["alias.mask"]["kind"], "mask_drop")
        self.assertEqual(cases["alias.value"]["status"], "valid")
        tied = self.build(inputs, layout, normalization, times=(30, 10, 10))
        self.assertEqual(self.indexed(tied)["seq.free"]["physical_parent_row"], 50)

    def test_padding_never_repairs_missing_and_source_constant_is_distinct(self):
        inputs, layout, normalization = self.fixture()
        for row in range(3):
            self.alias_observed(inputs, row, False)
        self.alias_observed(inputs, 3, True, 8)  # Padding must never become witness4.
        cases = self.indexed(self.build(inputs, layout, normalization))
        for token in ("alias.value", "alias.mask"):
            self.assertEqual(cases[token]["status"], "not_exercised_missing")
            self.assertIsNone(cases[token]["witness_row"])
            self.assertIsNone(cases[token]["input_sha256"])
        self.assertEqual(cases["d1.constant"]["status"], "source_declared_constant")
        self.assertEqual(cases["d1.constant"]["witness_row"], 1)

    def test_invalid_recipe_gets_one_attempt_and_no_more_favorable_witness(self):
        from unittest import mock
        inputs, layout, normalization = self.fixture()
        maximum = torch.finfo(torch.float32).max
        inputs["seq_x"][1, -1, 0] = inputs["snap_x"][1, 0] = maximum
        normalization["surfaces"]["signal"]["center"][0] = maximum
        normalization["surfaces"]["signal"]["scale"][0] = maximum
        original = influence.make_dynamic_compact_counterfactual
        with mock.patch.object(influence, "make_dynamic_compact_counterfactual", wraps=original) as calls:
            manifest = self.build(inputs, layout, normalization)
        case = self.indexed(manifest)["seq.free"]
        selected = [item for item in calls.call_args_list if item.kwargs["token"] == "seq.free"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].kwargs["witness_row"], 1)
        self.assertEqual(case["status"], "invalid_counterfactual")
        self.assertEqual(case["reason"], "replacement_not_finite_float32")
        self.assertIsNone(case["input_sha256"])

    def test_valid_case_bits_rebuild_enum_drop_and_family_using_existing_owner(self):
        inputs, layout, normalization = self.fixture()
        manifest = self.build(inputs, layout, normalization)
        view = influence.dynamic_compact_normalization_f32_view(normalization, layout=layout)
        cases = self.indexed(manifest)
        for case in manifest["cases"]:
            if case["status"] != "valid":
                continue
            with self.subTest(token=case["token"]):
                made = influence.make_dynamic_compact_counterfactual(inputs, layout=layout, normalization=view,
                    token=case["token"], witness_row=case["witness_row"], kind=case["kind"], time_index=case["time_index"])
                self.assertEqual(influence.dynamic_compact_input_sha256(made["inputs"]), case["input_sha256"])
        self.assertEqual(cases["m5.ordinal"]["kind"], "category")
        enum = influence.make_dynamic_compact_counterfactual(inputs, layout=layout, normalization=view,
            token="m5.ordinal", witness_row=1, kind="category")
        self.assertEqual(float(enum["inputs"]["mtf"]["seq_m5"][0, -1, 3]), -1)
        self.assertEqual(float(enum["inputs"]["mtf"]["seq_m5"][0, 0, 3]), 1)
        self.assertEqual(cases["family.local"]["route"], "family_all_consumed_bars")
        self.assertEqual(cases["m5.mask"]["kind"], "mask_drop")
        self.assertEqual(cases["m5.mask"]["status"], "valid")

    def test_manifest_is_serializable_deterministic_bounded_and_baseline_immutable(self):
        import json
        inputs, layout, normalization = self.fixture()
        before = influence.dynamic_compact_input_sha256(inputs)
        first = self.build(inputs, layout, normalization)
        second = self.build(inputs, layout, normalization)
        self.assertEqual(json.dumps(first, sort_keys=True, allow_nan=False),
                         json.dumps(second, sort_keys=True, allow_nan=False))
        self.assertEqual(influence.dynamic_compact_input_sha256(inputs), before)
        self.assertEqual(first["inventory_label_count"], 13)
        self.assertEqual(first["owner_counts"], layout["counts"])
        self.assertEqual(sum(first["status_counts"].values()), 13)
        self.assertEqual(first["valid_case_count"], first["status_counts"]["valid"])
        self.assertEqual(first["budget"]["forward_calls"], 2 + (first["valid_case_count"] + 15) // 16)
        self.assertFalse(first["execution_authorized"])
        cases = self.indexed(first)
        self.assertEqual(cases["seq.free"]["input_sha256"], cases["snap.free"]["input_sha256"])
        self.assertTrue(first["physical_route_labels_are_not_independent_current_manifolds"])
        self.assertIn("no separate historical value-attribution proof", first["history_scope"])
        self.assertEqual(cases["seq.free"]["time_index"], None)
        self.assertEqual(cases["seq.free"]["effective_time_index"], -1)

    def test_malformed_identity_layout_and_normalizer_fail_closed(self):
        inputs, layout, normalization = self.fixture()
        for parents, times in (((1, 1, 2), (1, 2, 3)), ((1, True, 2), (1, 2, 3)), ((1, 2), (1, 2))):
            with self.subTest(parents=parents):
                with self.assertRaisesRegex(RuntimeError, "CASE_WITNESS_IDENTITY"):
                    self.build(inputs, layout, normalization, parents=parents, times=times)
        bad_layout = copy.deepcopy(layout)
        bad_layout["numeric"][1]["token"] = bad_layout["numeric"][0]["token"]
        with self.assertRaisesRegex(RuntimeError, "CASE_OWNER_INVENTORY"):
            self.build(inputs, bad_layout, normalization)
        bad_stats = copy.deepcopy(normalization)
        bad_stats["surfaces"]["signal"]["field_names"] = tuple(reversed(bad_stats["surfaces"]["signal"]["field_names"]))
        with self.assertRaisesRegex(RuntimeError, "NORMALIZATION_FIELD_ORDER"):
            self.build(inputs, layout, bad_stats)

    def test_family_history_observation_is_not_misreported_as_current_missing(self):
        inputs, layout, normalization = self.fixture()
        layout["families"][1]["source_indices"] = (1, 2)
        m5 = inputs["mtf"]["seq_m5"]
        m5[:, :, 1:3] = 0
        m5[1, 0, 1:3] = torch.tensor([1.0, 7.0])
        cases = self.indexed(self.build(inputs, layout, normalization))
        self.assertEqual(cases["family.m5"]["witness_row"], 1)
        self.assertEqual(cases["family.m5"]["status"], "valid")
        self.assertEqual(cases["family.m5"]["route"], "family_all_consumed_bars")
        self.assertEqual(cases["family.m5"]["replacement_recipe"]["eligibility_scope"],
                         "all_consumed_family_coordinates")
        self.assertEqual(cases["m5.value"]["status"], "not_exercised_missing")
        self.assertEqual(cases["m5.mask"]["status"], "not_exercised_missing")


class _DirectionWitnessAlgebra(torch.nn.Module):
    """Small pointwise graph with native output/label shapes; no native model."""

    def __init__(self, *, zero_forecast_route=False):
        super().__init__()
        from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        from gx1.contracts.entry_model_native_joint_task_weighting_v1 import DYNAMIC_M5_JOINT_TASK_NAMES
        self.entry_training_phase = ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        self.family_tf_context_gate = torch.nn.Linear(2, 2)
        self.family_tf_token_gate = torch.nn.Linear(2, 1)
        self.entry_q_joint_norm = torch.nn.Identity()
        self.entry_q_joint_in = torch.nn.Linear(2, 2)
        self.entry_q_joint_in.register_parameter("unused", torch.nn.Parameter(torch.tensor(3.0)))
        self.dynamic_m5_query_projection = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.GELU())
        self.head_dynamic_terminal_net_markout = torch.nn.Linear(2, 2)
        self.head_dynamic_mfe = torch.nn.Linear(2, 2)
        self.head_dynamic_pre_mfe_mae_upper = torch.nn.Linear(2, 2)
        self.head_forecast = torch.nn.Linear(2, 10)
        self.task_log_variances = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.tensor(float(index) / 4))
            for index, name in enumerate(DYNAMIC_M5_JOINT_TASK_NAMES)})
        with torch.no_grad():
            for layer in (self.family_tf_context_gate, self.entry_q_joint_in):
                layer.weight.copy_(torch.eye(2))
                layer.bias.zero_()
            self.dynamic_m5_query_projection[0].weight.copy_(torch.tensor([[1., 0., 2., 3.], [0., 1., -1., 1.]]))
            self.dynamic_m5_query_projection[0].bias.fill_(0.25)
            self.head_dynamic_terminal_net_markout.weight.copy_(torch.tensor([[1., 0.], [0., -2.]]))
            self.head_dynamic_mfe.weight.copy_(torch.tensor([[0.5, 1.], [1., 0.]]))
            self.head_dynamic_pre_mfe_mae_upper.weight.copy_(torch.eye(2))
            for layer in (self.head_dynamic_terminal_net_markout, self.head_dynamic_mfe,
                          self.head_dynamic_pre_mfe_mae_upper, self.head_forecast):
                layer.bias.zero_()
            self.head_forecast.weight.fill_(0. if zero_forecast_route else 0.5)
        self.calls = 0
        self.eval()

    def forward(self, x):
        from gx1.contracts.entry_causal_m1_outcomes_v1 import (
            DYNAMIC_M1_PREFIX_ENDPOINT_KINDS, DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT,
            DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT,
        )
        self.calls += 1
        z = self.family_tf_context_gate(x) + 0 * self.family_tf_token_gate(x)
        hidden = self.entry_q_joint_in(self.entry_q_joint_norm(z))
        queries, curves = [], {}
        for endpoint, endpoint_kind in enumerate(DYNAMIC_M1_PREFIX_ENDPOINT_KINDS):
            query = ((torch.arange(1, 481, dtype=torch.float32) - 240.5) / ((480**2 - 1) / 12)**0.5)[None, :, None].expand(len(x), -1, -1)
            source = torch.cat((hidden[:, None].expand(-1, 480, -1), query,
                                torch.full_like(query, endpoint)), dim=-1)
            decoded = self.dynamic_m5_query_projection(source)
            queries.append(source)
            curves[endpoint_kind] = (self.head_dynamic_terminal_net_markout(decoded), self.head_dynamic_mfe(decoded),
                                     torch.nn.functional.softplus(self.head_dynamic_pre_mfe_mae_upper(decoded)))
        terminal, mfe, mae = curves[DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT]
        _, closed_mfe, closed_mae = curves[DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT]
        flat = torch.zeros_like(terminal[..., :1])
        quality = terminal - mae
        best = quality.max(dim=1).values
        outputs = {
            "dynamic_terminal_net_markout_bps": torch.cat((terminal, flat), dim=-1),
            "dynamic_mfe_bps": mfe, "dynamic_pre_mfe_mae_upper_bps": mae,
            "dynamic_closed_mfe_bps": closed_mfe, "dynamic_closed_pre_mfe_mae_upper_bps": closed_mae,
            "dynamic_entry_quality_bps_candidate": torch.cat((quality, flat), dim=-1),
            "dynamic_entry_action_quality_bps_candidate": torch.cat((best, flat[:, 0]), dim=-1),
            "dynamic_entry_quality_best_horizon_mask": quality == best[:, None],
            "dynamic_query_horizon_minutes": torch.arange(1, 481),
            "entry_q_joint_hidden": hidden, "forecast_pred": self.head_forecast(z),
        }
        return outputs, queries, z


class TestDirectionWitnessGradientArithmetic(unittest.TestCase):
    headlines = (1, 5, 15, 60, 120, 240, 480)

    def fixture(self, *, zero_forecast_route=False):
        from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import (
            DYNAMIC_M5_LOSS_SCALE_BATCH_KEY, dynamic_m5_loss_scale_contract,
        )
        from gx1.contracts.entry_observed_market_v1 import (
            DYNAMIC_M5_MARKOUT_TARGET_KEY, DYNAMIC_M5_MARKOUT_VALID_KEY,
            DYNAMIC_M5_MFE_TARGET_KEY, DYNAMIC_M5_MAE_TARGET_KEY, DYNAMIC_M5_PATH_VALID_KEY,
            DYNAMIC_M5_CLOSED_MFE_TARGET_KEY, DYNAMIC_M5_CLOSED_MAE_TARGET_KEY, DYNAMIC_M5_CLOSED_PATH_VALID_KEY,
        )
        from gx1.contracts.entry_observed_market_horizons_v1 import (
            M5_FORECAST_VALID_BATCH_KEY, SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY,
        )
        from gx1.contracts.entry_model_native_aux_targets_v3 import MODEL_NATIVE_AUX_FORECAST_HORIZONS
        model = _DirectionWitnessAlgebra(zero_forecast_route=zero_forecast_route)
        x = torch.tensor([[1., 2.], [-1., 0.5]])
        outputs, queries, z = model(x)
        terminal_valid = torch.ones(2, 480, 3, dtype=torch.bool)
        terminal_valid[0, ::2, 0] = False
        terminal_valid[1, ::3, 1] = False
        terminal_valid[..., 2] = terminal_valid[..., 0] & terminal_valid[..., 1]
        terminal_target = torch.ones(2, 480, 3)
        terminal_target[..., 1] = -2
        terminal_target[..., 2] = 0
        terminal_target[~terminal_valid] = 0
        inclusive_valid = torch.ones(2, 480, 2, dtype=torch.bool)
        inclusive_valid[0, ::2, 0] = False
        closed_valid = torch.ones(2, 480, 2, dtype=torch.bool)
        closed_valid[:, ::3] = False
        rms = {name: torch.full(shape, float(index + 2))
               for index, (name, shape) in enumerate(dynamic_m5_loss_scale_contract()["surface_shapes"].items())}
        batch = {"seq_x": x[:, None],
            DYNAMIC_M5_MARKOUT_TARGET_KEY: terminal_target, DYNAMIC_M5_MARKOUT_VALID_KEY: terminal_valid,
            DYNAMIC_M5_PATH_VALID_KEY: inclusive_valid, DYNAMIC_M5_CLOSED_PATH_VALID_KEY: closed_valid,
            DYNAMIC_M5_LOSS_SCALE_BATCH_KEY: {"identity": {}, "rms": rms}}
        for target_key, valid in ((DYNAMIC_M5_MFE_TARGET_KEY, inclusive_valid),
                                  (DYNAMIC_M5_MAE_TARGET_KEY, inclusive_valid),
                                  (DYNAMIC_M5_CLOSED_MFE_TARGET_KEY, closed_valid),
                                  (DYNAMIC_M5_CLOSED_MAE_TARGET_KEY, closed_valid)):
            batch[target_key] = torch.where(valid, torch.ones(2, 480, 2), torch.zeros(2, 480, 2))
        base_valid = torch.ones(2, 4, dtype=torch.bool)
        base_valid[0, 0] = False
        batch[M5_FORECAST_VALID_BATCH_KEY] = base_valid
        for index, h in enumerate(MODEL_NATIVE_AUX_FORECAST_HORIZONS):
            batch["y_forecast_ret_K" + str(h)] = base_valid[:, index].float()
        support_valid = torch.ones(2, 6, dtype=torch.bool)
        support_valid[1, 1] = False
        batch[SUPPORT_FORECAST_VALID_BATCH_KEY] = support_valid
        batch[SUPPORT_FORECAST_TARGET_BATCH_KEY] = support_valid.float()
        return model, outputs, batch, queries, z

    def witness(self, fixture):
        model, outputs, batch, queries, z = fixture
        return influence.dynamic_m5_direction_witness_gradients(model=model, outputs=outputs, batch=batch,
            query_inputs=queries, forecast_input=z, headline_horizons=self.headlines)

    def test_native_masks_rms_flat_denominator_and_shared_endpoint_counts(self):
        from gx1.contracts.entry_observed_market_v1 import (
            DYNAMIC_M5_MARKOUT_TARGET_KEY, DYNAMIC_M5_MARKOUT_VALID_KEY,
            DYNAMIC_M5_MFE_TARGET_KEY, DYNAMIC_M5_CLOSED_MFE_TARGET_KEY,
            DYNAMIC_M5_PATH_VALID_KEY, DYNAMIC_M5_CLOSED_PATH_VALID_KEY,
        )
        from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import DYNAMIC_M5_LOSS_SCALE_BATCH_KEY
        fixture = self.fixture()
        _, outputs, batch, _, _ = fixture
        result = self.witness(fixture)
        name = "dynamic_terminal_net_markout_bps"
        target, valid = batch[DYNAMIC_M5_MARKOUT_TARGET_KEY], batch[DYNAMIC_M5_MARKOUT_VALID_KEY]
        scales = batch[DYNAMIC_M5_LOSS_SCALE_BATCH_KEY]["rms"]
        side_square = ((outputs[name][..., :2] - target[..., :2]) / scales[name]).square()
        expected = float(side_square[valid[..., :2]].sum() / int(valid.sum()))
        self.assertEqual(result["task_statistics"][name]["observed_task_cells"], int(valid.sum()))
        self.assertEqual(result["task_statistics"][name]["train_rms_mse"], expected)
        self.assertGreater(int(valid.sum()), int(valid[..., :2].sum()))
        total, count = 0.0, 0
        for output_key, target_key, valid_key in (("dynamic_mfe_bps", DYNAMIC_M5_MFE_TARGET_KEY, DYNAMIC_M5_PATH_VALID_KEY),
            ("dynamic_closed_mfe_bps", DYNAMIC_M5_CLOSED_MFE_TARGET_KEY, DYNAMIC_M5_CLOSED_PATH_VALID_KEY)):
            mask = batch[valid_key]
            total += float((((outputs[output_key] - batch[target_key]) / scales[output_key])[mask]).square().sum())
            count += int(mask.sum())
        self.assertEqual(result["task_statistics"]["dynamic_mfe_bps"]["observed_task_cells"], count)
        self.assertAlmostEqual(result["task_statistics"]["dynamic_mfe_bps"]["train_rms_mse"], total / count, places=6)
        for partition in result["terminal_partitions"].values():
            self.assertEqual(partition["native_task_denominator"], int(valid.sum()))
            self.assertEqual(partition["partition_observed_cells_including_flat"],
                             partition["observed_side_cells"] + partition["partition_flat_cells"])
        self.assertAlmostEqual(sum(p["unit_loss_contribution"] for p in result["terminal_partitions"].values()), expected, places=6)
        group = "shared_forecast_input"
        total_norm = result["gradient_summaries"][name][group]["l2_norm"]
        partition_dots = result["gradient_pairs"]["terminal_partitions_to_native_terminal"]
        self.assertAlmostEqual(sum(pair[group]["dot"] for pair in partition_dots.values()), total_norm ** 2, places=6)

    def test_query_derivative_uses_actual_coordinate_and_endpoint_in_bps_per_minute(self):
        import math
        from gx1.contracts.entry_causal_m1_outcomes_v1 import (
            DYNAMIC_M1_PREFIX_ENDPOINT_KINDS, DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT,
            DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT,
        )
        self.assertEqual(DYNAMIC_M1_PREFIX_ENDPOINT_KINDS.index(DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT), 0)
        self.assertEqual(DYNAMIC_M1_PREFIX_ENDPOINT_KINDS.index(DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT), 1)
        fixture = self.fixture()
        model, _, _, queries, _ = fixture
        result = self.witness(fixture)
        weight = model.dynamic_m5_query_projection[0].weight.detach()
        bias = model.dynamic_m5_query_projection[0].bias.detach()
        for endpoint_kind, component, head, positive in (
            (DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT, "dynamic_terminal_net_markout_bps", model.head_dynamic_terminal_net_markout, False),
            (DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT, "dynamic_mfe_bps", model.head_dynamic_mfe, False),
            (DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT, "dynamic_closed_mfe_bps", model.head_dynamic_mfe, False),
            (DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT, "dynamic_pre_mfe_mae_upper_bps", model.head_dynamic_pre_mfe_mae_upper, True),
            (DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT, "dynamic_closed_pre_mfe_mae_upper_bps", model.head_dynamic_pre_mfe_mae_upper, True)):
            endpoint = DYNAMIC_M1_PREFIX_ENDPOINT_KINDS.index(endpoint_kind)
            pre = queries[endpoint].detach() @ weight.T + bias
            gelu_derivative = 0.5 * (1 + torch.erf(pre / math.sqrt(2))) + pre * torch.exp(-pre.square() / 2) / math.sqrt(2 * math.pi)
            expected = (gelu_derivative * weight[:, -2]) @ head.weight.detach().T / math.sqrt((480**2 - 1) / 12)
            if positive:
                logits = torch.nn.functional.gelu(pre) @ head.weight.detach().T + head.bias.detach()
                expected = expected * torch.sigmoid(logits)
            for side, label in enumerate(("long", "short")):
                row = result["query_sensitivity"][component + "." + label]
                self.assertAlmostEqual(row["all_queries"]["mean"], float(expected[..., side].double().mean()), places=8)
                self.assertAlmostEqual(row["headline_queries"]["1"]["mean"], float(expected[:, 0, side].double().mean()), places=8)
                self.assertEqual(row["endpoint_kind"], endpoint_kind)
                self.assertEqual(row["query_coordinate_units"], "standardized_fixed_query_grid_v1")
                self.assertEqual(row["query_coordinate_center_minutes"], 240.5)
                self.assertEqual(row["query_coordinate_scale_minutes"], math.sqrt((480**2 - 1) / 12))
                self.assertNotEqual(row["endpoint_input_connectivity"][endpoint], "unused")
                self.assertEqual(row["endpoint_input_connectivity"][1 - endpoint], "unused")
        first = result["query_sensitivity"]["dynamic_mfe_bps.long"]["all_queries"]["mean"]
        closed = result["query_sensitivity"]["dynamic_closed_mfe_bps.long"]["all_queries"]["mean"]
        self.assertNotEqual(first, closed)

    def test_shared_forecast_route_unused_and_connected_zero_are_distinct(self):
        result = self.witness(self.fixture())
        terminal = result["gradient_summaries"]["dynamic_terminal_net_markout_bps"]
        forecast = result["gradient_summaries"]["forecast_return_bps"]
        self.assertEqual(terminal["forecast_head"]["status"], "unused")
        self.assertIsNone(terminal["forecast_head"]["l2_norm"])
        self.assertEqual(forecast["entry_joint_hidden"]["status"], "unused")
        self.assertEqual(forecast["shared_forecast_input"]["status"], "nonzero")
        self.assertIn("entry_q_joint_in.unused", terminal["entry_joint_projection"]["unused_parameter_names"])
        self.assertIn("family_tf_token_gate.weight", terminal["shared_mtf_gates"]["connected_zero_parameter_names"])
        zero = self.witness(self.fixture(zero_forecast_route=True))
        row = zero["gradient_summaries"]["forecast_return_bps"]["shared_forecast_input"]
        self.assertEqual(row["status"], "connected_zero")
        self.assertEqual(row["l2_norm"], 0.0)
        pair = next(value for key, value in zero["gradient_pairs"]["tasks"].items() if "forecast_return_bps" in key)
        self.assertIsNone(pair["shared_forecast_input"]["cosine"])

    def test_exact_reverse_budget_native_reuse_no_forward_grad_rng_or_state_mutation(self):
        import json
        from unittest import mock
        from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
        from gx1.contracts import entry_dynamic_m5_loss_scale_v1 as loss_units
        fixture = self.fixture()
        model = fixture[0]
        before = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with mock.patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as reverse, \
                mock.patch.object(trainer, "_dynamic_m5_task_loss_surfaces", wraps=trainer._dynamic_m5_task_loss_surfaces) as native, \
                mock.patch.object(trainer, "_joint_task_loss", wraps=trainer._joint_task_loss) as joint, \
                mock.patch.object(loss_units, "dynamic_m5_scaled_square_sum", wraps=loss_units.dynamic_m5_scaled_square_sum) as squared, \
                mock.patch.object(model, "forward", side_effect=AssertionError("Unexpected extra forward")):
            result = self.witness(fixture)
        self.assertEqual(reverse.call_count, 22)
        self.assertEqual(native.call_count, 2)
        self.assertEqual(joint.call_count, 13)  # Full diagnostic plus12 native scalarized projections.
        self.assertEqual(squared.call_count, 14)  # Six native surfaces plus8 terminal partitions.
        for call in squared.call_args_list[-8:]:
            self.assertEqual(call.args[1].shape[-1], 3)
            self.assertTrue(torch.equal(call.args[1][~call.args[2]], torch.zeros_like(call.args[1][~call.args[2]])))
        self.assertEqual(model.calls, 1)
        self.assertEqual(result["forward_calls"], 0)
        self.assertEqual(result["reverse_calls"], 22)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(torch.equal(before[name], value) for name, value in model.state_dict().items()))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        json.dumps(result, allow_nan=False)

    def test_unobserved_partition_is_explicit_and_bad_mode_query_or_storage_fail_closed(self):
        from gx1.contracts.entry_observed_market_v1 import DYNAMIC_M5_MARKOUT_TARGET_KEY, DYNAMIC_M5_MARKOUT_VALID_KEY
        fixture = self.fixture()
        fixture[2][DYNAMIC_M5_MARKOUT_VALID_KEY][:, 4] = False
        fixture[2][DYNAMIC_M5_MARKOUT_TARGET_KEY][:, 4] = 0
        result = self.witness(fixture)
        self.assertEqual(result["terminal_partitions"]["terminal.h5"]["role"], "unobserved_partition_not_evidence")
        self.assertEqual(result["terminal_partitions"]["terminal.h5"]["observed_side_cells"], 0)
        fixture = self.fixture()
        fixture[0].train()
        with self.assertRaisesRegex(RuntimeError, "DIRECTION_REQUIRES_EVAL_GRAD_GRAPH"):
            self.witness(fixture)
        fixture = self.fixture()
        fixture[3].reverse()
        with self.assertRaisesRegex(RuntimeError, "DIRECTION_QUERY_ORDER_OR_IDENTITY"):
            self.witness(fixture)
        fixture = self.fixture()
        fixture[2][DYNAMIC_M5_MARKOUT_TARGET_KEY][0, 0, 0] = 7
        with self.assertRaisesRegex(RuntimeError, "TARGET_STORAGE_INVALID"):
            self.witness(fixture)



class TestDirectionWitnessNative170(unittest.TestCase):
    fixture = TestDirectionWitnessGradientArithmetic.fixture
    headlines = TestDirectionWitnessGradientArithmetic.headlines

    def witness(self, fixture):
        model, outputs, batch, queries, z = fixture
        return influence.dynamic_m5_direction_witness_gradients(model=model, outputs=outputs,
            batch=batch, query_inputs=queries, forecast_input=z, headline_horizons=self.headlines,
            terminal_objective="native_170_groupmean_v1")

    def group_case(self):
        amplitude = torch.tensor(1., requires_grad=True)
        coefficient = torch.zeros(1, 480, 3)
        coefficient[0, 0, :2] = torch.tensor([2., 8.])
        coefficient[0, 15, 0] = 2.
        valid = torch.zeros(1, 480, 3, dtype=torch.bool)
        valid[0, 0] = True  # L/S plus true analytic FLAT: N=3, scaled SSE=20.
        valid[0, 15, 0] = True
        valid[0, 16, 1] = True  # Genuine observed zero: later N=2, SSE=4.
        rms = torch.ones(480, 2)
        rms[0, 1] = 2.
        rms[16, 1] = 3.
        return amplitude, coefficient * amplitude, torch.zeros_like(coefficient), valid, rms

    def test_asymmetric_rms_masks_flat_and_observed_zero_give_13_over_3(self):
        from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import dynamic_m5_terminal_horizon_objective
        amplitude, prediction, target, valid, rms = self.group_case()
        losses, stats = influence._direction_witness_terminal_group_losses(
            prediction=prediction, target=target, valid=valid, rms=rms)
        total = sum(losses.values())
        canonical, counts = dynamic_m5_terminal_horizon_objective(prediction, target, valid, rms=rms)
        self.assertEqual(counts, {"early_1_15": 3, "later_16_480": 2})
        self.assertEqual(stats["early_1_15"]["group_flat_cells"], 1)
        self.assertEqual(stats["later_16_480"]["observed_side_cells"], 2)
        self.assertAlmostEqual(float(total), 13 / 3, places=6)
        self.assertEqual(float(total), float(canonical))
        derivative, cell_derivative = torch.autograd.grad(total, (amplitude, prediction))
        self.assertAlmostEqual(float(derivative), 26 / 3, places=5)
        self.assertTrue(torch.equal(cell_derivative[..., 2], torch.zeros_like(cell_derivative[..., 2])))
        self.assertEqual(float(cell_derivative[0, 16, 1]), 0.)
        self.assertEqual(float(cell_derivative[~valid].abs().sum()), 0.)

    def test_one_or_no_observed_group_has_no_epsilon_or_fake_observation(self):
        from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import dynamic_m5_terminal_horizon_objective
        _, prediction, target, valid, rms = self.group_case()
        valid[:, :15] = False
        losses, stats = influence._direction_witness_terminal_group_losses(
            prediction=prediction, target=target, valid=valid, rms=rms)
        self.assertEqual(float(sum(losses.values())), 2.)
        self.assertIsNone(stats["early_1_15"]["unit_group_mean"])
        self.assertEqual(stats["early_1_15"]["role"], "unobserved_group_not_evidence")
        self.assertEqual(stats["later_16_480"]["optimizer_group_coefficient"], 1.)
        valid[:] = False
        losses, stats = influence._direction_witness_terminal_group_losses(
            prediction=prediction, target=target, valid=valid, rms=rms)
        canonical, counts = dynamic_m5_terminal_horizon_objective(prediction, target, valid, rms=rms)
        self.assertIsNone(canonical)  # Native omission includes omission of its +s.
        self.assertEqual(sum(counts.values()), 0)
        self.assertEqual(float(sum(losses.values())), 0.)
        self.assertTrue(all(x["nonempty_groups"] == x["optimizer_group_coefficient"] == 0 for x in stats.values()))

    def test_full_validation_precedes_any_group_slice(self):
        _, prediction, target, valid, rms = self.group_case()
        target[0, -1, 0] = 7.
        with self.assertRaisesRegex(RuntimeError, "LOSS_SCALE_RUNTIME_INVALID"):
            influence._direction_witness_terminal_group_losses(
                prediction=prediction, target=target, valid=valid, rms=rms)

    def test_nonunit_precision_native_task_and_group_vector_sum(self):
        from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
        fixture = self.fixture()
        model, outputs, batch, _, _ = fixture
        term = "dynamic_terminal_net_markout_bps"
        with torch.no_grad():
            model.task_log_variances[term].fill_(float(np.log(2.)))
        pooled, _ = trainer._dynamic_m5_task_loss_surfaces(outputs, batch, torch.device("cpu"), loss_units="train_rms")
        optimizer, counts = trainer._dynamic_m5_optimizer_task_losses(
            outputs, batch, torch.device("cpu"), pooled_task_losses=pooled)
        scalar = trainer._joint_task_loss(model, {term: optimizer[term]}, loss_units="train_rms")[0]
        parameters = tuple(model.head_dynamic_terminal_net_markout.parameters())
        expected = torch.autograd.grad(scalar, parameters, retain_graph=True)
        expected_norm = float(torch.linalg.vector_norm(torch.cat([g.double().reshape(-1) for g in expected])))
        result = self.witness(fixture)
        self.assertEqual(result["terminal_group_counts"], counts)
        self.assertAlmostEqual(result["task_statistics"][term]["effective_precision"], .5, places=6)
        self.assertEqual(result["task_statistics"][term]["pooled_train_rms_mse"], float(pooled[term]))
        self.assertEqual(result["task_statistics"][term]["optimizer_objective"], float(optimizer[term]))
        self.assertAlmostEqual(result["gradient_summaries"][term]["terminal_head"]["l2_norm"], expected_norm, places=6)
        self.assertAlmostEqual(result["terminal_group_sum_minus_native_optimizer_objective"], 0., places=6)
        for group, residual in result["terminal_group_vector_sum_residual"].items():
            if residual["l2_norm"] is not None:
                self.assertLessEqual(residual["l2_norm"], 2e-6)
        self.assertEqual(result["terminal_group_vector_sum_residual"]["forecast_head"]["status"], "unused_not_evidence")

    def test_native170_schedule_is_four_tasks_two_groups_ten_queries_without_mutation(self):
        import json
        from unittest import mock
        from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
        fixture = self.fixture()
        model = fixture[0]
        state = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with mock.patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as reverse, \
                mock.patch.object(trainer, "_dynamic_m5_optimizer_task_losses",
                                  wraps=trainer._dynamic_m5_optimizer_task_losses) as native, \
                mock.patch.object(model, "forward", side_effect=AssertionError("extra forward")):
            result = self.witness(fixture)
        self.assertEqual(native.call_count, 1)
        self.assertEqual(reverse.call_count, 16)
        self.assertEqual(result["reverse_calls"], 16)
        self.assertEqual(result["forward_calls"], 0)
        self.assertEqual(result["schema_version"], "gx1_dynamic_m5_direction_witness_gradients_v2")
        self.assertEqual(len(result["active_native_tasks"]), 4)
        self.assertEqual(len(result["terminal_partitions"]), 2)
        self.assertEqual(len(result["query_sensitivity"]), 10)
        self.assertEqual(model.calls, 1)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(all(torch.equal(state[k], v) for k, v in model.state_dict().items()))
        json.dumps(result, allow_nan=False)

    def test_two_active_tasks_cannot_fill_a_four_task_physical_schedule(self):
        from unittest import mock
        from gx1.contracts.entry_observed_market_v1 import DYNAMIC_M5_MARKOUT_TARGET_KEY, DYNAMIC_M5_MARKOUT_VALID_KEY
        from gx1.contracts.entry_observed_market_horizons_v1 import (
            M5_FORECAST_VALID_BATCH_KEY, SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY,
        )
        fixture = self.fixture()
        batch = fixture[2]
        batch[DYNAMIC_M5_MARKOUT_TARGET_KEY].zero_()
        batch[DYNAMIC_M5_MARKOUT_VALID_KEY].zero_()
        batch[M5_FORECAST_VALID_BATCH_KEY].zero_()
        for h in (1, 5, 12, 24): batch["y_forecast_ret_K" + str(h)].zero_()
        batch[SUPPORT_FORECAST_TARGET_BATCH_KEY].zero_()
        batch[SUPPORT_FORECAST_VALID_BATCH_KEY].zero_()
        with mock.patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as reverse:
            with self.assertRaisesRegex(RuntimeError, "DIRECTION_TASK_SUPERVISION_OR_DENOMINATOR"):
                self.witness(fixture)
        self.assertEqual(reverse.call_count, 0)
