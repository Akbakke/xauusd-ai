"""Frozen actual measurement coordinates, distinct TRAIN/control roles."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts import unified_exit_bounded_val_cohort_v1 as owner
from gx1.contracts.unified_exit_reference_policy_v1 import reference_policy_contract
from gx1.scripts import run_unified_exit_random_access_val_v1 as val
from tests.test_native_chronological_control_source import _binding
from tests.test_chronological_sampled_targets import _sampled_context


def _write(path, value):
    path.write_text(json.dumps(value))
    return _binding(path)


def _fixture(tmp_path):
    n = 700
    times = np.r_[pd.date_range('2026-02-01T00:00Z', periods=300, freq='5min').asi8,
                  pd.date_range('2026-03-01T00:00Z', periods=400, freq='5min').asi8]
    frame = pd.DataFrame({'entry_row_index': np.arange(n, dtype='int64'),
        'parent_entry_row_index': np.arange(n, dtype='int64') + 11,
        'entry_time_ns': times, 'first_state_time_ns': times + 300_000_000_000,
        'parent_m1_start_row': np.arange(n, dtype='int64') * 5 + 479,
        'child_m1_start_row': np.arange(n, dtype='int64') * 5 + 479,
        'successor_transition_count': np.full(n, 600, dtype='int64'),
        'lifecycle_state_count': np.full(n, 601, dtype='int64'),
        'economic_terminal': False, 'right_censored': True, 'entry_bid': 100., 'entry_ask': 100.1})
    index = tmp_path / 'index.parquet'
    frame.to_parquet(index, index=False)
    children = {'train': np.arange(256, dtype='int64')[::-1],
                'control': np.linspace(300, n - 1, 256, dtype='int64')}
    rows = {}
    for role, ids in children.items():
        path = tmp_path / (role + '.npy')
        np.save(path, ids + 11)
        rows[role] = _binding(path)
    design = {'schema_version': 'gx1_frozen_chronological_learning_design_v1',
        'status': 'DESIGN_AND_CONTROL_IDS_FROZEN_NOT_EXECUTABLE',
        'scope': {'test_sealed': True, 'existing_control_periods_are_reused_development': True,
                  'native_launch_authorized': False, 'one_experiment': True},
        'selection': {'control_entries': 256}, 'budget': {'later_control_entries': 256},
        'calendar': {'train_entry_start_inclusive': '2025-06-01T00:00Z',
            'train_control_cutoff': '2026-03-01T00:00Z',
            'development_control_entry_end_exclusive': '2026-06-01T00:00Z',
            'index': _binding(index), 'bindings': {'CONTROL256_PARENT_ROWS': rows['control']}},
        'targets': {'reference_policy': reference_policy_contract()}}
    db = _write(tmp_path / 'design.json', design)
    aux = _write(tmp_path / 'aux.json', {'frozen_design': db,
        'bindings': {'TRAIN256_PROBE_PARENT_ROWS': rows['train']}})
    arrays = {}
    bindings = {}
    for role, ids in children.items():
        offsets = np.tile(np.array([7, 4, 7, 1], dtype='int64'), (256, 1))
        data = {'parent_rows': ids + 11, 'child_rows': ids, 'entry_time_ns': times[ids],
            'sampled_state_indices': offsets,
            'sampled_reference_end_close_ns': times[ids, None] + (offsets + 126) * 60_000_000_000,
            'anchor_reference_end_close_ns': times[ids] + 126 * 60_000_000_000}
        path = tmp_path / (role + '.npz')
        np.savez(path, **data)
        arrays[role] = data
        bindings[role] = _binding(path)
    result = {'schema_version': 'gx1_prefix_measurement_coordinates_v1',
        'decision': 'TRAIN_AND_CONTROL_COORDINATES_FROZEN_NO_MODEL_MEASUREMENTS',
        'design': db, 'source_index': _binding(index), 'population_rows': n,
        'train_samples_exactly_reused': True, 'control_entry_ids_unchanged': True,
        'all_samples_preserved': True, 'test_data_used': False, 'model_forwards': 0,
        'optimizer_steps': 0, 'fits': 0, 'auxiliary_policies': aux, 'coordinates': bindings}
    rb = _write(tmp_path / 'result.json', result)
    return db, rb, result, arrays


@pytest.mark.parametrize('role', ['train', 'control'])
def test_frozen_probe_keeps_native_order_physical_ids_samples_and_correct_cutoff(tmp_path, role):
    db, rb, result, arrays = _fixture(tmp_path)
    cohort = owner.build_chronological_measurement_cohort(db, rb, role=role)
    assert owner.require_bounded_val_cohort(cohort) == cohort
    assert cohort['parent_entry_row_indices'] == arrays[role]['parent_rows'].tolist()
    assert cohort['entry_row_indices'] == arrays[role]['child_rows'].tolist()
    assert cohort['sampled_state_indices'] == arrays[role]['sampled_state_indices'].tolist()
    assert cohort['reference_cutoff_time_ns'] == int(pd.Timestamp(
        '2026-03-01T00:00Z' if role == 'train' else '2026-06-01T00:00Z').value)
    assert cohort['measurement_only'] is True and cohort['source_split'] == 'train'
    assert cohort['split'] == ('train' if role == 'train' else 'val')
    if role == 'train':
        assert cohort['entry_row_indices'][0] > cohort['entry_row_indices'][-1]


@pytest.mark.parametrize('fault', ['reorder', 'child_mapping', 'outside_lifetime', 'future_support',
                                  'shape', 'wrong_design', 'changed_bytes'])
def test_coordinate_binding_rejects_reselection_or_invalid_support(tmp_path, fault):
    db, rb, result, arrays = _fixture(tmp_path)
    data = arrays['train']
    if fault == 'reorder':
        data = {k: v[::-1] for k, v in data.items()}
    elif fault == 'child_mapping':
        data['child_rows'][0] = 299
    elif fault == 'outside_lifetime':
        data['sampled_state_indices'][0, 0] = 600
    elif fault == 'future_support':
        data['sampled_reference_end_close_ns'][0, 0] = int(pd.Timestamp('2026-03-01T00:00Z').value) + 1
    elif fault == 'shape':
        data['sampled_state_indices'] = data['sampled_state_indices'][:, :3]
    elif fault == 'wrong_design':
        result['design'] = {**db, 'sha256': '0' * 64}
    path = Path(result['coordinates']['train']['path'])
    np.savez(path, **data)
    if fault == 'changed_bytes':
        path.write_bytes(path.read_bytes() + b'changed')
    else:
        result['coordinates']['train'] = _binding(path)
    rb = _write(Path(rb['path']), result)
    with pytest.raises(RuntimeError):
        owner.build_chronological_measurement_cohort(db, rb, role='train')


def _args(tmp_path, role):
    coord_dir = tmp_path / 'coords'
    coord_dir.mkdir()
    db, rb, _, _ = _fixture(coord_dir)
    cohort = owner.build_chronological_measurement_cohort(db, rb, role=role)
    args = _sampled_context(tmp_path)
    args.update(parent_rows=cohort['parent_entry_row_indices'],
        candidate_child_rows=cohort['entry_row_indices'], evaluation_cohort=cohort)
    args.pop('candidate_sampled_state_indices')
    factory = args['candidate_state_factory']
    factory.artifact_file_sha256 = {'random_access_index': cohort['source_index']['sha256']}
    if role == 'train':
        factory.times = pd.date_range('2026-02-20T00:00Z', periods=600, freq='min')
    return args


@pytest.mark.parametrize('role', ['train', 'control'])
def test_existing_entry_measurement_automatically_uses_frozen_role_and_samples(tmp_path, monkeypatch, role):
    args = _args(tmp_path, role)
    calls = []
    def forward(model, seq, snap, **kw):
        return {val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY: seq,
                'entry_action_q_bps': seq.expand(-1, 3).contiguous()}
    def reference(**kw):
        assert kw['reference_cutoff_time_ns'] == args['evaluation_cohort']['reference_cutoff_time_ns']
        offsets = kw.get('start_state_indices')
        calls.append(offsets)
        q = torch.ones(len(kw['child_rows']), 2)
        rows = [{'entry_row_index': i, 'state_index': offset, 'target_hold_bps': x}
            for i, offset, x in zip(kw['child_rows'], offsets or [0] * len(q), q.tolist())]
        return (rows, {'hold_target_bps': q}) if kw.get('return_reference_targets') else rows
    monkeypatch.setattr(val, '_model_forward_fp32', forward)
    monkeypatch.setattr(val, '_multi_tf_kwargs_from_batch', lambda *a: {})
    monkeypatch.setattr(val, '_bounded_reference_exit_observations', reference)
    monkeypatch.setattr(val, '_new_active_head_epoch_accumulator', lambda: {})
    monkeypatch.setattr(val, '_accumulate_active_head_epoch', lambda *a: None)
    monkeypatch.setattr(val, '_active_head_epoch_diagnostics', lambda *a: ({}, None))
    monkeypatch.setattr(val, 'accumulate_route_diagnostics_v1', lambda *a: None)
    monkeypatch.setattr(val, 'finalize_route_diagnostics_v1', lambda *a: {})
    reps, diag, _ = val._entry_representations(**args)
    assert reps[:, 0].tolist() == args['parent_rows']
    assert len(diag['bounded_exit_sampled_observations']) == 1024
    assert all(x == [7, 4, 7, 1] * 64 for x in calls if x is not None)
    # V_mu weights the signed HOLD outcome by the stationary 119/120 policy.
    for row in diag['bounded_entry_observations']:
        assert row['target_q_bps'] == pytest.approx([-4. + 119./120., -4. + 119./120., 0.], abs=1e-6)


@pytest.mark.parametrize('fault', ['sample_override', 'source_index'])
def test_measurement_cannot_replace_frozen_samples_or_source_before_forward(tmp_path, monkeypatch, fault):
    args = _args(tmp_path, 'train')
    if fault == 'sample_override':
        args['candidate_sampled_state_indices'] = [[1, 1, 1, 1]] * 256
    else:
        args['candidate_state_factory'].artifact_file_sha256['random_access_index'] = '0' * 64
    def forbidden(*a, **kw):
        raise AssertionError('mismatched binding reached model')
    monkeypatch.setattr(val, '_model_forward_fp32', forbidden)
    with pytest.raises(RuntimeError, match='CHRONOLOGICAL_MEASUREMENT_'):
        val._entry_representations(**args)


@pytest.mark.parametrize('role', ['train', 'control'])
def test_measurement_cohort_cannot_enter_economic_rollout(tmp_path, role):
    db, rb, _, _ = _fixture(tmp_path)
    cohort = owner.build_chronological_measurement_cohort(db, rb, role=role)
    with pytest.raises(RuntimeError, match='DOES_NOT_AUTHORIZE_ROLLOUT'):
        val.evaluate_bound_full_val_v1(model=None, entry_dataset=None, frame=None,
            state_factory=None, checkpoint_binding={}, parent_coordinate_evidence={},
            val_sequence_audit=tmp_path / 'unused', device=torch.device('cpu'), selected_batch_size=16,
            rollout_progress_path=tmp_path / 'unused-progress', result_path=tmp_path / 'unused-result',
            max_forwards_this_invocation=1, progress_interval_forwards=1,
            compute_guard_max_model_forwards=1, compute_guard_max_materialized_state_views=1,
            compute_guard_max_wall_seconds=1., evaluation_cohort=cohort)


def test_train_rollout_binds_original_probe_and_cutoff_without_changing_measurement(tmp_path):
    import copy
    db, rb, _, arrays = _fixture(tmp_path)
    before = owner.build_chronological_measurement_cohort(db, rb, role="train")
    cohort = owner.build_chronological_train_rollout_cohort(db, rb)
    assert owner.require_bounded_val_cohort(cohort) == cohort
    assert cohort["entry_row_indices"] == sorted(arrays["train"]["child_rows"].tolist())
    assert dict(zip(cohort["entry_row_indices"], cohort["parent_entry_row_indices"])) == dict(
        zip(before["entry_row_indices"], before["parent_entry_row_indices"]))
    assert cohort["observation_cutoff_time_ns"] == before["reference_cutoff_time_ns"]
    assert cohort["split"] == "train" and cohort["measurement_only"] is False
    assert owner.build_chronological_measurement_cohort(db, rb, role="train") == before
    assert before["measurement_only"] is True
    for key, value in [("observation_cutoff_time_ns", cohort["observation_cutoff_time_ns"] + 1),
                       ("entry_row_indices", cohort["entry_row_indices"][:-1]),
                       ("measurement_only", True)]:
        changed = copy.deepcopy(cohort); changed[key] = value
        changed.pop("cohort_sha256"); changed["cohort_sha256"] = owner.canonical_sha256(changed)
        with pytest.raises(RuntimeError, match="COHORT_BINDING_MISMATCH"):
            owner.require_bounded_val_cohort(changed)
    from gx1.contracts.unified_exit_random_access_val_rollout_v1 import _require_entries
    with pytest.raises(RuntimeError, match="MEASUREMENT_DOES_NOT_AUTHORIZE_ROLLOUT"):
        _require_entries([], before)

# Physical v38: actual index/sampler contracts on synthetic clocks only.
from tests.test_native_fresh_prefix_components import physical_component_templates, physical_component_case
from tests.test_native_prefix_coordinator import physical_prepared


@pytest.fixture
def physical_measurement(physical_prepared, tmp_path):
    from types import SimpleNamespace
    from tests.test_native_chronological_control_source import _physical_control_plan
    from tests.test_unified_exit_selected_sampler_v1 import _seal, _direct_build
    from gx1.contracts.unified_exit_random_access_sampler_v1 import (
        build_random_access_sampler_contract, schedule_random_access_epoch)
    p = physical_prepared
    c, sc = p.case, p.case["sample_case"]
    directory = tmp_path / "physical-train-index"; directory.mkdir()
    _, ib, _, _, manifest, frame, _ = _physical_control_plan(
        directory, population=len(p.train),
        entry_start=c["sources"]["train"]["declared_window"]["start"], split="train")
    for kind in ("parquet", "manifest"):
        for prefix in ("entry_", "parent_entry_"):
            manifest["source_bindings"][prefix + kind] = c["sources"]["train"][kind]
    manifest["source_bindings"]["final_bindings_bundle"] = _binding(sc["bundle_path"])
    manifest["parent_entry_source_sha256"] = c["sources"]["train"]["parquet"]["sha256"]
    manifest["parent_entry_manifest_sha256"] = c["sources"]["train"]["manifest"]["sha256"]
    vc = p.context["evaluation_cohort"]
    vm = json.loads(Path(vc["source_index_manifest"]["path"]).read_text())
    for split, m, mp in (("train", manifest, sc["manifest_path"]),
                         ("val", vm, Path(vc["source_index_manifest"]["path"]))):
        times = pd.date_range(c["sources"][split]["declared_window"]["start"],
                             periods=m["entry_row_count"]*5+10, freq="min")
        source = tmp_path / (split + "-m1.parquet")
        pd.DataFrame({"time":times}).to_parquet(source, index=False)
        m["source_bindings"]["m1_child"] = _binding(source)
        m["source_bindings"]["m1_child_manifest"] = _write(tmp_path / (split + "-m1.json"), {
            "split":split, "decision":"PASS", "test_accessed":False,
            "output_parquet_sha256":_binding(source)["sha256"]})
        _seal(mp, m, "manifest_sha256")
    from gx1.contracts.unified_exit_bounded_val_cohort_v1 import build_chronological_control_cohort
    vc = build_chronological_control_cohort(p.value["design"], source_index_binding=vc["source_index"],
        source_index_manifest_binding=_binding(Path(vc["source_index_manifest"]["path"])))
    p.context["evaluation_cohort"] = vc
    p.context["state_factory"].factory_receipt["physical_control_cohort_sha256"] = vc["cohort_sha256"]
    indexes = {"train": {"index": ib, "manifest": _binding(sc["manifest_path"])},
               "val": {"index": vc["source_index"], "manifest": vc["source_index_manifest"]}}
    for split, m in (("train", manifest), ("val", vm)):
        sc["root"]["splits"][split] = {
            "entry_row_count": m["entry_row_count"], "manifest_path": indexes[split]["manifest"]["path"],
            "manifest_sha256": m["manifest_sha256"], "index_parquet_path": m["index_parquet_path"],
            "index_parquet_sha256": m["index_parquet_sha256"]}
    sc["root"]["full_train_population"]["train_manifest_sha256"] = manifest["manifest_sha256"]
    _seal(sc["root_path"], sc["root"], "root_sha256")
    sc["receipt"]["run_bindings"]["files"]["root_manifest"] = _binding(sc["root_path"])
    _seal(sc["receipt_path"], sc["receipt"], "receipt_sha256")
    c["selected"] = _direct_build(sc)
    c["coordinates"]["selected_sampler"] = _write(c["files"]["selected_sampler"], c["selected"])
    c["seal_coordinates"]()
    train_sampler = c["selected"]["selected_sampler_contract"]
    samplers = {"train": train_sampler, "val": build_random_access_sampler_contract(
        split="val", source_lineage_sha256=train_sampler["source_lineage_sha256"],
        transition_budget_per_epoch=len(p.control)*4, transitions_per_entry=4,
        entry_pair_population=len(p.control))}
    frames = {"train": frame, "val": pd.read_parquet(indexes["val"]["index"]["path"])}
    arrays, coordinate_bindings = {}, {}
    for role, split, rows in (("train", "train", c["probe"]),
                             ("control", "val", np.array(vc["parent_entry_row_indices"], dtype="int64"))):
        f = frames[split]
        groups = {}
        for sample in schedule_random_access_epoch(sampler_contract=samplers[split], epoch_index=0,
                successor_transition_count_by_entry=f.successor_transition_count.tolist()):
            groups.setdefault(sample["entry_row_index"], []).append(sample)
        offsets = np.array([[s["state_index"] for s in sorted(groups[int(row)], key=lambda s:s["sample_slot"])]
                            for row in rows], dtype="int64")
        end = (f.iloc[rows].first_state_time_ns + (f.iloc[rows].successor_transition_count+1)*60_000_000_000).to_numpy()
        arrays[role] = {"parent_rows": rows.copy(), "child_rows": rows.copy(),
            "entry_time_ns": f.iloc[rows].entry_time_ns.to_numpy(), "sampled_state_indices": offsets,
            "sampled_reference_end_close_ns": np.repeat(end[:, None], 4, axis=1),
            "anchor_reference_end_close_ns": end}
        path = tmp_path / (role + "-physical.npz"); np.savez(path, **arrays[role])
        coordinate_bindings[role] = _binding(path)
        if role == "train":
            p.train._unified_exit_lifecycle_v2._random_access_train["samples_by_entry"] = groups
    result = {"schema_version": "gx1_physical_prefix_measurement_coordinates_v1",
        "decision": "TRAIN_AND_CONTROL_COORDINATES_FROZEN_NO_MODEL_MEASUREMENTS",
        "design": p.value["design"], "chronological_prefix": dict(p.value), "source_indexes": indexes,
        "sampler_contracts": samplers, "coordinates": coordinate_bindings,
        "train_samples_exactly_reused": True, "control_entry_ids_unchanged": True,
        "all_samples_preserved": True, "test_data_used": False,
        "model_forwards": 0, "optimizer_steps": 0, "fits": 0}
    binding = _write(tmp_path / "physical-measurement.json", result)
    return SimpleNamespace(prepared=p, result=result, binding=binding, arrays=arrays, frames=frames)


@pytest.mark.parametrize("role", ["train", "control"])
def test_physical_measurement_reuses_exact_native_draws_and_separate_sources(physical_measurement, role):
    c = physical_measurement; p = c.prepared
    cohort = owner.build_chronological_measurement_cohort(p.value["design"], c.binding, role=role)
    split = "train" if role == "train" else "val"
    assert owner.require_bounded_val_cohort(cohort) == cohort
    assert cohort["source_split"] == split
    assert cohort["source_index"] == c.result["source_indexes"][split]["index"]
    assert cohort["source_index_manifest"] == c.result["source_indexes"][split]["manifest"]
    assert cohort["parent_entry_parquet"] == p.case["sources"][split]["parquet"]
    assert cohort["entry_row_indices"] == c.arrays[role]["child_rows"].tolist()
    assert cohort["sampled_state_indices"] == c.arrays[role]["sampled_state_indices"].tolist()
    assert all(len(row) == 4 and len(set(row)) < 4 for row in cohort["sampled_state_indices"])
    if role == "control":
        assert cohort["physical_control_cohort_sha256"] == p.context["evaluation_cohort"]["cohort_sha256"]
        assert cohort["cohort_sha256"] != cohort["physical_control_cohort_sha256"]


@pytest.mark.parametrize("fault", ["swapped_indexes", "wrong_manifest", "native_sample", "reorder_probe",
                                  "sampler", "changed_bytes", "changed_prefix"])
def test_physical_measurement_rejects_namespace_or_draw_drift(physical_measurement, fault):
    c = physical_measurement; r = c.result
    if fault == "swapped_indexes":
        r["source_indexes"]["train"], r["source_indexes"]["val"] = r["source_indexes"]["val"], r["source_indexes"]["train"]
    elif fault == "wrong_manifest":
        r["source_indexes"]["train"]["manifest"] = r["source_indexes"]["val"]["manifest"]
    elif fault == "sampler":
        r["sampler_contracts"]["train"] = r["sampler_contracts"]["val"]
    elif fault == "changed_prefix":
        r["chronological_prefix"].pop("native_coordinates")
    else:
        path = Path(r["coordinates"]["train"]["path"])
        data = c.arrays["train"]
        if fault == "native_sample":
            data["sampled_state_indices"][0, 0] = (data["sampled_state_indices"][0, 0] + 1) % 3
        elif fault == "reorder_probe":
            data = {k:v[::-1] for k,v in data.items()}
        np.savez(path, **data)
        if fault == "changed_bytes": path.write_bytes(path.read_bytes() + b"changed")
        else: r["coordinates"]["train"] = _binding(path)
    rb = _write(Path(c.binding["path"]), r)
    with pytest.raises(RuntimeError):
        owner.build_chronological_measurement_cohort(c.prepared.value["design"], rb, role="train")


@pytest.mark.parametrize("role", ["train", "control"])
def test_physical_measurement_consumer_routes_bound_source_before_forward(physical_measurement, tmp_path, monkeypatch, role):
    import copy
    c = physical_measurement; split = "train" if role == "train" else "val"
    cohort = owner.build_chronological_measurement_cohort(c.prepared.value["design"], c.binding, role=role)
    directory = tmp_path / "consumer"; directory.mkdir()
    args = _sampled_context(directory)
    args.pop("candidate_sampled_state_indices")
    args.update(parent_rows=cohort["parent_entry_row_indices"], candidate_child_rows=cohort["entry_row_indices"],
                evaluation_cohort=cohort)
    args["dataset"].parquet_path = Path(cohort["parent_entry_parquet"]["path"])
    factory = args["candidate_state_factory"]; f = c.frames[split]
    factory.source_split = split
    factory.artifact_file_sha256 = {"random_access_index": cohort["source_index"]["sha256"],
                                   "random_access_index_manifest": cohort["source_index_manifest"]["sha256"],
                                   "child_m1":cohort["source_m1"]["sha256"],
                                   "child_m1_manifest":cohort["source_m1_manifest"]["sha256"]}
    factory.factory_receipt = {"physical_control_cohort_sha256": cohort.get("physical_control_cohort_sha256")}
    factory.entries = [{"entry_row_index":int(row.entry_row_index),
        "entry_m1_start_row":int(row.child_m1_start_row), "available_state_count":int(row.lifecycle_state_count),
        "entry_episode_binding_sha256":"1"*64, "entry_fill_binding_sha256":"2"*64} for row in f.itertuples()]
    factory.times = pd.date_range(pd.Timestamp(int(f.entry_time_ns.iloc[0]), unit="ns", tz="UTC"),
                                  periods=len(f)*5+10, freq="min")
    calls = []
    def forbidden(*a, **kw): raise AssertionError("source mismatch reached model")
    monkeypatch.setattr(val, "_model_forward_fp32", forbidden)
    original_path = args["dataset"].parquet_path
    args["dataset"].parquet_path = c.prepared.files["entry_val_parquet" if role == "train" else "entry_train_parquet"]
    with pytest.raises(RuntimeError, match="PHYSICAL_SOURCE_MISMATCH"): val._entry_representations(**args)
    args["dataset"].parquet_path = original_path
    original = copy.deepcopy(factory.artifact_file_sha256)
    factory.artifact_file_sha256["random_access_index_manifest"] = "0"*64
    with pytest.raises(RuntimeError, match="PHYSICAL_SOURCE_MISMATCH"): val._entry_representations(**args)
    factory.artifact_file_sha256 = original
    if role == "control":
        factory.factory_receipt["physical_control_cohort_sha256"] = cohort["cohort_sha256"]
        with pytest.raises(RuntimeError, match="PHYSICAL_SOURCE_MISMATCH"): val._entry_representations(**args)
        factory.factory_receipt["physical_control_cohort_sha256"] = cohort["physical_control_cohort_sha256"]
    def forward(model, seq, snap, **kw):
        return {val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY:seq, "entry_action_q_bps":seq.expand(-1,3).contiguous()}
    def reference(**kw):
        assert kw["state_factory"] is factory
        offsets = kw.get("start_state_indices")
        if offsets is not None:
            lookup = dict(zip(cohort["entry_row_indices"], cohort["sampled_state_indices"]))
            assert offsets == [lookup[child][slot] for child,slot in zip(kw["child_rows"], list(range(4))*(len(kw["child_rows"])//4))]
            calls.extend(zip(kw["child_rows"], offsets))
        q = torch.ones(len(kw["child_rows"]), 2)
        rows = [{"entry_row_index":i, "state_index":offset, "target_hold_bps":x}
                for i,offset,x in zip(kw["child_rows"], offsets or [0]*len(q), q.tolist())]
        return (rows, {"hold_target_bps":q}) if kw.get("return_reference_targets") else rows
    monkeypatch.setattr(val, "_model_forward_fp32", forward)
    monkeypatch.setattr(val, "_multi_tf_kwargs_from_batch", lambda *a:{})
    monkeypatch.setattr(val, "_bounded_reference_exit_observations", reference)
    monkeypatch.setattr(val, "_new_active_head_epoch_accumulator", lambda:{})
    monkeypatch.setattr(val, "_accumulate_active_head_epoch", lambda *a:None)
    monkeypatch.setattr(val, "_active_head_epoch_diagnostics", lambda *a:({},None))
    monkeypatch.setattr(val, "accumulate_route_diagnostics_v1", lambda *a:None)
    monkeypatch.setattr(val, "finalize_route_diagnostics_v1", lambda *a:{})
    reps, diagnostics, _ = val._entry_representations(**args)
    assert reps[:,0].tolist() == cohort["parent_entry_row_indices"]
    assert len(calls) == len(diagnostics["bounded_exit_sampled_observations"]) == 1024


@pytest.mark.parametrize("fault", [None, "missing_factory", "swapped_factory", "native_samples", "control_failure"])
def test_physical_initial_and_final_measurement_preserve_source_teacher_and_session(
    physical_measurement, tmp_path, monkeypatch, fault,
):
    import copy
    import time
    from types import SimpleNamespace
    from tests.test_native_prefix_coordinator import PrefixHarness, equal_tree, digest, trainer
    from gx1.scripts import run_unified_exit_random_access_full_train_v1 as runner
    p = physical_measurement.prepared
    source = tmp_path / "synthetic-source.py"; source.write_text("synthetic measurement source")
    sources = {"python:synthetic-source.py":runner.launch_owner.artifact_binding(source)}
    recipe = {"chronological_prefix":p.value, "source_bindings":sources,
              "source_bindings_sha256":runner.launch_owner.canonical_json_sha256(sources)}

    monkeypatch.setattr(trainer, "_copy_frozen_prefix_reference_model",
                        lambda m, **kw:copy.deepcopy(m).eval().requires_grad_(False))
    h = PrefixHarness(tmp_path / "run", p); output = h.root / "MEASURED"
    model, pause = h.run_prefix(output, 0)
    initial_hash = trainer._model_state_sha256(model)
    scope = {"initialization":{"online_model_state_sha256":initial_hash,
                              "model_functions":dict(trainer._PREFIX_CURRENT_MODEL_FUNCTIONS)},
             "measurement":{"coordinate_result":physical_measurement.binding},
             "artifacts":{"initialization_result":{"synthetic":True}, "measurement_binding_result":{"synthetic":True},
                          "initial_measurement_audit":{"synthetic":True}}}
    factories = {"train":SimpleNamespace(source_split="train"), "control":p.context["state_factory"]}
    seen = []
    def measure(**kw):
        cohort = kw["evaluation_cohort"]; role = cohort["measurement_role"]; seen.append(role)
        assert kw["candidate_state_factory"] is factories[role]
        assert kw["dataset"] is (p.train if role == "train" else p.control)
        assert trainer._model_state_sha256(kw["candidate_target_model"]) == initial_hash
        assert not kw["candidate_target_model"].training and kw["exit_boundary_model"] is kw["candidate_target_model"]
        torch.rand(5)
        if role == "control" and fault == "control_failure": raise RuntimeError("injected control failure")
        pred = float(kw["model"].weight[0,0].detach())
        entry = [{"entry_row_index":i, "predicted_q_bps":[pred,pred,0], "target_q_bps":[1.,-1.,0.]}
                 for i in cohort["entry_row_indices"]]
        anchor = [{"entry_row_index":i, "state_index":0, "prediction_hold_bps":[pred,pred], "target_hold_bps":[1.,-1.]}
                  for i in cohort["entry_row_indices"]]
        sampled = [{**row,"state_index":offset} for row,offsets in zip(anchor,cohort["sampled_state_indices"]) for offset in offsets]
        return None, {"bounded_entry_observations":entry, "bounded_exit_anchor_observations":anchor,
                      "bounded_exit_sampled_observations":sampled}, None
    monkeypatch.setattr(runner.val, "_entry_representations", measure)
    def components():
        return {**h.last_kwargs, "train_probe_ds":p.train, "measurement_state_factories":factories}
    current = components()
    if fault == "missing_factory": current.pop("measurement_state_factories")
    elif fault == "swapped_factory": current["measurement_state_factories"] = {"train":factories["control"],"control":factories["train"]}
    elif fault == "native_samples":
        child = int(p.case["probe"][0])
        p.train._unified_exit_lifecycle_v2._random_access_train["samples_by_entry"][child][0]["state_index"] += 1
    def invoke(current, pause, steps=0):
        return runner._run_prefix_initial_measurement(components=current, scope=scope,
            recipe=recipe, output=output, device=torch.device("cpu"),
            invocation_started=time.monotonic(), pause_evidence=pause, optimizer_steps=steps)
    saved, pointer = h.state(output), h.pointer(output).read_bytes()
    rng = trainer._attended_session_rng_state(device=torch.device("cpu"))
    if fault:
        with pytest.raises(RuntimeError, match="FACTORIES_REQUIRED|SAMPLES_CHANGED|injected control failure"):
            invoke(current, pause)
        assert seen == (["train","control"] if fault == "control_failure" else [])
    else:
        before = invoke(current, pause)
        scope["initial_measurement"] = json.loads(Path(before["path"]).read_text())
        scope["artifacts"]["initial_measurement_result"] = before
        assert seen == ["train","control"]
    assert h.pointer(output).read_bytes() == pointer and model.training
    equal_tree(h.state(output), saved)
    equal_tree(trainer._attended_session_rng_state(device=torch.device("cpu")), rng)
    if fault: return
    model, pause = h.run_prefix(output, 256, expected_pointer=digest(h.pointer(output)))
    saved, pointer = h.state(output), h.pointer(output).read_bytes()
    rng = trainer._attended_session_rng_state(device=torch.device("cpu"))
    after = invoke(components(), pause, 256)
    result = json.loads(Path(after["path"]).read_text())
    assert result["frozen_targets_exactly_preserved"] is True
    assert result["native_recipe_source_bindings"] == sources
    assert result["native_recipe_source_bindings_sha256"] == recipe["source_bindings_sha256"]
    assert scope["initial_measurement"]["native_recipe_source_bindings"] == sources

    assert result["measurement_roles"] == ["train","control"] and result["optimizer_steps"] == 256
    assert seen == ["train","control","train","control"]
    assert h.pointer(output).read_bytes() == pointer and model.training
    equal_tree(h.state(output), saved)
    equal_tree(trainer._attended_session_rng_state(device=torch.device("cpu")), rng)


def test_physical_measurement_rejects_fabricated_early_support(physical_measurement):
    c = physical_measurement
    path = Path(c.result["coordinates"]["train"]["path"])
    for field in ("sampled_reference_end_close_ns", "anchor_reference_end_close_ns"):
        data = {key:value.copy() for key,value in c.arrays["train"].items()}
        data[field].flat[0] -= 60_000_000_000
        np.savez(path, **data)
        c.result["coordinates"]["train"] = _binding(path)
        rb = _write(Path(c.binding["path"]), c.result)
        with pytest.raises(RuntimeError, match="M1_SUPPORT_MISMATCH"):
            owner.build_chronological_measurement_cohort(c.prepared.value["design"], rb, role="train")
