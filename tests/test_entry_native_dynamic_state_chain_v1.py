"""Small exact state-chain mathematics and synthetic wiring, no native model.

The shell bypasses native construction. These checks cannot qualify genuine
initialization, features, optimizer/resume, latency, learning or prediction.
"""
from copy import deepcopy
import ast
import hashlib
import inspect
import itertools
import math
from pathlib import Path

import pytest
import torch

from gx1.contracts import entry_native_direction_research_v1 as owner
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as targets
from gx1.features.entry_specialist_feature_groups_v1 import FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
from tests.test_entry_native_direction_research_v1 import _native_shell, _inputs
from tests.test_entry_native_direction_feature_view_v1 import bound_case


def _chain(length=4):
    first = torch.tensor([[[-.7, .2, .6]]], requires_grad=True)
    raw = torch.arange((length - 1) * 9, dtype=torch.float32)
    transitions = ((raw.remainder(11) - 5) / 3).reshape(1, 1, length - 1, 3, 3).requires_grad_()
    return first, transitions


def _enumerated(first, transitions, labels, observed):
    initial = first.double().softmax(-1)[0, 0]
    matrices = transitions.double().softmax(-1)[0, 0]
    paths, probabilities = [], []
    for path in itertools.product(range(3), repeat=len(labels)):
        probability = initial[path[0]]
        for at in range(1, len(path)):
            probability = probability * matrices[at - 1, path[at - 1], path[at]]
        paths.append(path)
        probabilities.append(probability)
    evidence = sum(p for path, p in zip(paths, probabilities)
        if all(not observed[at] or path[at] == labels[at] for at in range(len(labels))))
    marginals = torch.stack([torch.stack([sum(p for path, p in zip(paths, probabilities)
        if path[at] == state) for state in range(3)]) for at in range(len(labels))])
    return -evidence.log(), marginals


@pytest.mark.parametrize("length", [1, 2, 4, 5])
def test_parallel_marginals_match_all_paths_and_sequential_recursion(length):
    first, transitions = _chain(length)
    actual = owner.dynamic_mtf_state_marginal_probabilities(first, transitions)[0, 0]
    _, expected = _enumerated(first, transitions, [-1] * length, [False] * length)
    torch.testing.assert_close(actual.double(), expected, atol=4e-7, rtol=3e-6)
    sequential = [first.softmax(-1)[0, 0]]
    for matrix in transitions.softmax(-1)[0, 0]:
        sequential.append(sequential[-1] @ matrix)
    torch.testing.assert_close(actual, torch.stack(sequential), atol=4e-7, rtol=3e-6)
    torch.testing.assert_close(actual.sum(-1), torch.ones(length), atol=2e-7, rtol=2e-6)
    if length > 1:
        pairs = actual[:-1, :, None] * transitions.softmax(-1)[0, 0]
        torch.testing.assert_close(pairs.sum(-2), actual[1:], atol=4e-7, rtol=3e-6)


@pytest.mark.parametrize("mask", [
    (True, True, True, True), (False, True, False, True),
    (True, False, False, True), (False, False, True, False),
    (True, True, False, False), (False, False, False, False)])
def test_joint_nll_and_gradients_match_enumeration_with_unknown_states(mask):
    first, transitions = _chain()
    labels = [0, 1, 2, 0]
    state = torch.tensor([[labels]], dtype=torch.int8)
    observed = torch.tensor([[mask]], dtype=torch.bool)
    state[~observed] = -1
    actual = owner.state_chain_negative_log_likelihood(first, transitions, state=state, observed=observed)[0, 0]
    expected, _ = _enumerated(first, transitions, labels, mask)
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=3e-6)
    got = torch.autograd.grad(actual, (first, transitions), retain_graph=True)
    want = torch.autograd.grad(expected, (first, transitions))
    for a, b in zip(got, want):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=3e-6, rtol=5e-5)


def test_complete_fast_path_matches_general_forward_likelihood_and_flat_is_a_class():
    first, transitions = _chain(5)
    state = torch.full((1, 1, 5), targets.FLAT, dtype=torch.int8)
    observed = torch.ones_like(state, dtype=torch.bool)
    fast = owner.state_chain_negative_log_likelihood(first, transitions, state=state, observed=observed)
    general = owner._partial_state_chain_nll(first.log_softmax(-1).reshape(1, 3),
        transitions.log_softmax(-1).reshape(1, 4, 3, 3), state.reshape(1, 5), observed.reshape(1, 5))
    torch.testing.assert_close(fast.flatten(), general, atol=2e-6, rtol=3e-6)
    uniform = owner.state_chain_negative_log_likelihood(torch.zeros_like(first), torch.zeros_like(transitions),
        state=state, observed=observed)
    torch.testing.assert_close(uniform, torch.full((1, 1), 5 * math.log(3)), atol=2e-6, rtol=3e-6)


def test_single_query_and_missing_label_values_do_not_create_flat_labels():
    first, transitions = _chain(1)
    missing = torch.zeros((1, 1, 1), dtype=torch.bool)
    for stored in (-1, targets.DOWN, targets.FLAT, targets.UP):
        result = owner.state_chain_negative_log_likelihood(first, transitions,
            state=torch.full((1, 1, 1), stored, dtype=torch.int8), observed=missing)
        assert result.item() == 0.
    with pytest.raises(RuntimeError, match="STATE_LABEL_DOMAIN"):
        owner.state_chain_negative_log_likelihood(first, transitions,
            state=torch.full((1, 1, 1), -1, dtype=torch.int8), observed=~missing)


def test_objective_binds_target_owner_all_five_tf_full_day_and_no_sampling():
    contract = owner.dynamic_mtf_state_research_contract()
    assert contract["target_contract"] == targets.dynamic_mtf_state_target_contract()
    assert tuple(contract["timeframes"]) == targets.TIMEFRAMES
    assert tuple(contract["class_names"]) == targets.STATE_NAMES
    assert tuple(contract["lead_minutes"]) == targets.default_trial_lead_minutes()
    assert contract["lead_minutes"] == list(range(1, 1441))
    assert contract["query_sampling"] is False
    assert contract["future_targets_or_clocks_as_forward_inputs"] is False
    assert len(contract["lead_minutes"]) * len(contract["timeframes"]) == 7200
    contract["class_names"].reverse()
    assert owner.dynamic_mtf_state_research_contract()["class_names"] == list(targets.STATE_NAMES)


def _small_native_loss(monkeypatch):
    contract = owner.dynamic_mtf_state_research_contract()
    # A deliberately small synthetic query contract exercises masks and clocks.
    # The separate metadata test above checks the genuine declared 1440 grid.
    contract["lead_minutes"] = [1, 2, 3]
    monkeypatch.setattr(owner, "dynamic_mtf_state_research_contract", lambda: deepcopy(contract))
    minute = 60 * 10**9
    issue = torch.tensor([100 * minute], dtype=torch.int64)
    validity = issue[:, None] + torch.tensor([[1, 2, 3]]) * minute
    return {
        "initial_state_logits": torch.zeros(1, 5, 3, requires_grad=True),
        "state_transition_logits": torch.zeros(1, 5, 2, 3, 3, requires_grad=True),
        "state": torch.full((1, 5, 3), targets.FLAT, dtype=torch.int8),
        "observed": torch.ones(1, 5, 3, dtype=torch.bool),
        "issue_ns": issue, "validity_ns": validity,
        "maturity_ns": validity[:, None].expand(1, 5, 3).clone(),
        "cutoff_ns": 103 * minute}


def test_native_loss_maturity_is_strict_flat_counts_and_sequence_evidence(monkeypatch):
    args = _small_native_loss(monkeypatch)
    result = owner.native_dynamic_mtf_state_loss(**args)
    assert result["update_allowed"] is True
    assert result["counts"]["eligible_flat_cells"] == result["counts"]["eligible_cells"] == 10
    assert result["counts"]["immature_observed_cells"] == 5
    assert result["sequence_nll"].shape == (1, 5)
    torch.testing.assert_close(result["sequence_nll"], torch.full((1, 5), 2 * math.log(3)), atol=2e-6, rtol=3e-6)
    torch.testing.assert_close(result["loss"], torch.tensor(math.log(3)), atol=2e-6, rtol=3e-6)
    result["loss"].backward()
    assert torch.isfinite(args["initial_state_logits"].grad).all()
    assert torch.isfinite(args["state_transition_logits"].grad).all()


@pytest.mark.parametrize("fault", ["validity", "maturity", "future_issue", "unknown_observed", "unknown_storage"])
def test_native_loss_rejects_incompatible_clock_or_label_metadata(monkeypatch, fault):
    args = _small_native_loss(monkeypatch)
    if fault == "validity": args["validity_ns"][0, 0] += 1
    if fault == "maturity": args["maturity_ns"][0, 0, 0] += 1
    if fault == "future_issue": args["issue_ns"][0] = args["cutoff_ns"]
    if fault == "unknown_observed": args["state"][0, 0, 0] = -1
    if fault == "unknown_storage": args["observed"][0, 0, 0] = False
    with pytest.raises(RuntimeError):
        owner.native_dynamic_mtf_state_loss(**args)


def test_no_eligible_outcomes_mean_no_update_and_zero_sequence_information(monkeypatch):
    args = _small_native_loss(monkeypatch)
    args["state"].fill_(-1); args["observed"].fill_(False)
    result = owner.native_dynamic_mtf_state_loss(**args)
    assert result["loss"] is None and result["update_allowed"] is False
    assert result["counts"]["eligible_cells"] == 0
    assert torch.equal(result["sequence_nll"], torch.zeros(1, 5))


def _state_shell(monkeypatch):
    native = _native_shell()
    native.entry_execution_profile = {"name": FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME, "contract_sha256": "a" * 64}
    native.gc_mtf_history = {"normalization": {}}
    def representation_spy(self, seq_x, snap_x, **features):
        self._spy_calls.append({"seq_x": seq_x, "snap_x": snap_x, **features})
        assert features["research_representation_only"] is True
        return {"entry_q_joint_hidden": self._synthetic_hidden}
    monkeypatch.setattr(type(native), "forward", representation_spy)
    return owner.NativeDynamicMTFStateResearch(native), native


def test_state_wrapper_has_one_feature_only_forward_and_active_joint_decoder(monkeypatch):
    model, native = _state_shell(monkeypatch)
    output = model(**_inputs())
    assert len(native._spy_calls) == 1
    assert output["initial_state_logits"].shape == (2, 5, 3)
    assert output["state_transition_logits"].shape == (2, 5, 1439, 3, 3)
    assert set(output) == {"initial_state_logits", "state_transition_logits"}
    assert not any("target" in key or "state" in key or "maturity" in key for key in native._spy_calls[0])
    objective = output["initial_state_logits"].sum() + output["state_transition_logits"].sum()
    objective.backward()
    assert native._synthetic_hidden.grad is not None and torch.isfinite(native._synthetic_hidden.grad).all()
    for name, parameter in model.named_parameters():
        if name.startswith(("state_query_projection.", "state_initial_head.", "state_transition_head.")):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        if name.startswith(("backbone.head_", "backbone.task_log_variances.")):
            assert parameter.grad is None
    assert "state_lead_minutes" in model.state_dict()
    assert "issue_ns" not in inspect.signature(model.forward).parameters


@pytest.mark.parametrize("buffer", ["state_lead_minutes", "state_class_codes", "state_timeframe_codes",
    "state_query_coordinates", "state_objective_sha256"])
def test_state_wrapper_rejects_changed_query_class_or_objective_identity(monkeypatch, buffer):
    model, _ = _state_shell(monkeypatch)
    getattr(model, buffer).flatten()[0] += 1
    with pytest.raises(RuntimeError, match="STATE_BUFFER_BINDING"):
        model._require_binding()


def test_representation_only_addition_preserves_legacy_forward_ast():
    path = Path(inspect.getfile(owner.EntryV10CtxHybridTransformer))
    module = ast.parse(path.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "EntryV10CtxHybridTransformer")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    index = next(i for i, arg in enumerate(function.args.kwonlyargs) if arg.arg == "research_representation_only")
    assert isinstance(function.args.kw_defaults[index], ast.Constant) and function.args.kw_defaults[index].value is False
    del function.args.kwonlyargs[index]; del function.args.kw_defaults[index]
    added = [node for node in function.body if isinstance(node, ast.If)
        and any(isinstance(value, ast.Name) and value.id == "research_representation_only" for value in ast.walk(node.test))]
    assert len(added) == 2
    for node in added: function.body.remove(node)
    digest = hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest()
    assert digest == "515aea516bb5f3b1edc6a15b6a62a5938b43fe0c13bc2f38548bb3c8fa386cb0"


def _state_view_arguments(bound_case):
    args = deepcopy(bound_case["arguments"])
    args["source_bindings"]["dynamic_state_target_source"] = {
        "path": "/synthetic/entry_dynamic_mtf_state_targets_v1.py", "sha256": "b" * 64}
    args.update(full_feature_profile=True, dynamic_state_supervision=True)
    return args


def test_state_feature_view_binds_own_objective_and_preserves_legacy_metadata(bound_case):
    before = views.bind_native_direction_feature_view(**bound_case["arguments"])
    args = _state_view_arguments(bound_case)
    state = views.bind_native_direction_feature_view(**args)
    assert state["research_objective"] == owner.dynamic_mtf_state_research_contract()
    assert state["entry_execution_profile"]["name"] == FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
    assert state["prefix_normalization"] == before["prefix_normalization"]
    assert state["physical_field_names_sha256"] == before["physical_field_names_sha256"]
    assert state["research_objective"]["contract_sha256"] != before["research_objective"]["contract_sha256"]
    assert views.bind_native_direction_feature_view(**bound_case["arguments"]) == before


@pytest.mark.parametrize("fault", ["missing_full", "missing_source", "legacy_dense", "false_flag", "integer_flag"])
def test_state_view_cannot_silently_enter_legacy_or_compact_mode(bound_case, fault):
    args = _state_view_arguments(bound_case)
    if fault == "missing_full": args["full_feature_profile"] = False
    if fault == "missing_source": del args["source_bindings"]["dynamic_state_target_source"]
    if fault == "legacy_dense": args["dense_minutes"] = True
    if fault == "false_flag": args["dynamic_state_supervision"] = False
    if fault == "integer_flag": args["dynamic_state_supervision"] = 1
    with pytest.raises(RuntimeError):
        views.bind_native_direction_feature_view(**args)
