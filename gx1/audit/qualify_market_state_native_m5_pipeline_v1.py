"""One CPU-capped qualification of fresh native M5 feature/model bindings."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.audit import build_native_m5_direction_targets_v1 as data
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = data.REPO
POLICY = REPO / 'NEXT_RUN_POLICY.json'
STAGE = 'native_m5_pipeline_20261119_001'
PLAN = REPO / 'configs/research/NATIVE_M5_PIPELINE_20261119_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_001')
FIT_START_NS = 1714521600000000000
FIT_END_NS = 1727740800000000000
PARAMETER_START_NS = 1243814400000000000
PARAMETER_END_NS = 1357077600000000000
INNER_END = '2012-04-11T22:00:00+00:00'
SEED, BATCH = 20260911, 16
ARCHITECTURE = {'dropout': .05, 'multi_tf_num_layers': 2, 'multi_tf_scale': .5,
    'specialist_num_layers': 1, 'specialist_fusion_scale': .25,
    'cross_family_fusion_scale': .25}
EXECUTION = {'class': 'audit', 'memory': '12G', 'swap': '512M',
    'outer_seconds': 3600, 'internal_seconds': 3570, 'kill_after_seconds': 10,
    'numerical_threads': 1, 'model_constructions': 1, 'model_forwards': 1,
    'optimizer_steps': 0}
INPUT_KEYS = {'squeeze_fit_plan', 'registry_fit_plan', 'native_source_plan',
              'macro_source_plan', 'target_result', 'target_terminal', 'target_tape'}
SOURCE_PATHS = (
    'gx1/audit/qualify_market_state_native_m5_pipeline_v1.py',
    'gx1/audit/build_native_m5_direction_targets_v1.py',
    'gx1/contracts/market_state_native_m5_targets_v1.py',
    'gx1/contracts/market_state_native_m5_feature_inputs_v1.py',
    'gx1/contracts/market_state_ta_integrated_model_v1.py',
    'gx1/contracts/market_state_ta_integrated_objective_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/entry_exit_production_architecture_v1.py',
    'gx1/contracts/entry_model_native_signal_v1.py',
    'gx1/contracts/registry_hyperparameter_fit_v1.py',
    'gx1/contracts/model_state_digest_v1.py',
    'gx1/features/volatility_squeeze_state_v1.py',
    'gx1/features/htf_features.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/features/entry_model_native_feature_layers_v1.py',
    'gx1/features/d1_macro_context_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
    'gx1/scripts/fit_volatility_squeeze_artifacts_v1.py',
    'gx1/scripts/research_ta_campaign_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_qualify_market_state_native_m5_pipeline_v1.py',
)
sha, ref, read, stamp = data.sha, data.ref, data.read, data.stamp


def need(ok, reason):
    if not bool(ok):
        raise ValueError('NATIVE_M5_PIPELINE_' + reason)


def registry_provenance(binding, plan):
    from gx1.features.volatility_squeeze_state_v1 import VOLATILITY_SQUEEZE_DIRECTION_CLOCKS
    native = plan['native_m5_manifest']
    return {tf: {
        'source_artifact': native['path'], 'source_sha256': native['sha256'],
        'source_schema_version': 'xau_canonical_native_source_v3', 'source_lane': tf,
        'native_m5_manifest_artifact': native['path'],
        'native_m5_manifest_sha256': native['sha256'],
        'fit_plan_artifact': binding['path'], 'fit_plan_sha256': binding['sha256'],
        'source_clock_contract': plan['source_clock_contract'], 'train_split_id': 'TRAIN',
        'declared_train_window_start': plan['declared_train_window_start'],
        'declared_train_window_end': plan['declared_train_window_end'],
    } for tf in VOLATILITY_SQUEEZE_DIRECTION_CLOCKS}


def fit_plans(inputs):
    from gx1.features.volatility_squeeze_state_v1 import (
        _require_native_m5_provenance, volatility_squeeze_bar_grid,
    )
    from gx1.contracts.registry_hyperparameter_fit_v1 import require_registry_fit_source_v1
    squeeze, registry = (read(inputs[key]) for key in ('squeeze_fit_plan', 'registry_fit_plan'))
    need(squeeze['fit_start_ns'] == registry['fit_start_ns'] == PARAMETER_START_NS
         and squeeze['fit_cutoff_ns'] == registry['fit_cutoff_ns'] == PARAMETER_END_NS
         and registry['declared_inner_fit_window_end'] == INNER_END
         and squeeze['source_spec'] == registry['source_spec'], 'PARAMETER_POPULATION')
    provenance = registry_provenance(inputs['registry_fit_plan'], registry)
    for tf, item in provenance.items():
        require_registry_fit_source_v1(item, clock=tf)
        squeeze_item = {k: v for k, v in item.items()
                        if k not in ('source_clock_contract', 'declared_train_window_start',
                                     'declared_train_window_end')}
        squeeze_item.update(clock_contract=squeeze['source_clock_contract'],
            bar_grid=volatility_squeeze_bar_grid(tf),
            fit_plan_artifact=inputs['squeeze_fit_plan']['path'],
            fit_plan_sha256=inputs['squeeze_fit_plan']['sha256'])
        _require_native_m5_provenance(squeeze_item, timeframe=tf,
            fit_cutoff_ns=PARAMETER_END_NS, fit_start_ns=PARAMETER_START_NS)
    return squeeze, registry


def macro_metadata(binding):
    from gx1.features import d1_macro_context_v1 as macro
    spec = macro.load_d1_macro_context_source_spec(REPO)
    need(binding == ref(REPO / macro.D1_MACRO_CONTEXT_SOURCE_SPEC['path'])
         and spec['test_seal_utc'] == '2026-07-01T00:00:00Z', 'MACRO_SPEC')
    dependencies = {}
    results = [spec['cot']['result'], spec['oanda_panel']['result'],
               *(spec['drivers'][key] for key in ('treasury', 'vix', 'gld'))]
    for result_ref in results:
        result = read(result_ref)
        dependencies[result_ref['path']] = result_ref
        need(type(result.get('records')) is list and len(result['records']) > 0,
             'MACRO_RECORDS')
        for record in result['records']:
            need(record['status'] == 'RETRIEVED_NOT_ADMITTED', 'MACRO_INCOMPLETE')
            pair = ('parquet_path', 'parquet_sha256') if 'parquet_path' in record else ('raw_path', 'raw_sha256')
            path, digest = record[pair[0]], record[pair[1]]
            stamp(path)
            need(isinstance(digest, str) and len(digest) == 64
                 and all(c in '0123456789abcdef' for c in digest), 'MACRO_DIGEST')
            value = {'path': path, 'sha256': digest}
            need(path not in dependencies or dependencies[path] == value, 'MACRO_SPLIT_BINDING')
            dependencies[path] = value
    clock = dict(spec['cot']['publication_clock'])
    if not Path(clock['path']).is_absolute():
        clock['path'] = str(REPO / clock['path'])
    read(clock)
    dependencies[clock['path']] = clock
    panel = read(spec['oanda_panel']['result'])
    need(set(spec['oanda_panel']['instruments']).issubset(
        {record['item'] for record in panel['records']}), 'MACRO_PANEL_COMPONENTS')
    return dependencies


def feature_spec(source_plan):
    original = source_plan['source_spec']
    need(original['issue_start_ns'] <= FIT_START_NS
         and original['cutoff_ns'] >= FIT_END_NS, 'FEATURE_SOURCE_SCOPE')
    parts = []
    for part in original['parts']:
        groups = [item['index'] for item in source_plan['source_metadata'][part['path']]['selected_groups']
                  if item['first_bar_ns'] + data.M5_NS < FIT_END_NS]
        if groups:
            parts.append({**part, 'row_groups': groups})
    need(len(parts) > 0, 'FEATURE_SOURCE_EMPTY')
    return {'manifest': original['manifest'], 'parts': parts, 'cutoff_ns': FIT_END_NS}


def build_plan(*, inputs, source_base_commit=None):
    from gx1.contracts.market_state_native_m5_targets_v1 import contract as target_contract
    from gx1.contracts.market_state_native_m5_feature_inputs_v1 import (
        RAW_COLUMNS, native_m5_feature_provider_contract,
    )
    from gx1.contracts.market_state_ta_integrated_model_v1 import native_bar_backbone_geometry
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_objective_contract
    need(type(inputs) is dict and set(inputs) == INPUT_KEYS, 'INPUT_ROLES')
    squeeze, registry = fit_plans(inputs)
    source = read(inputs['native_source_plan'])
    need(inputs['native_source_plan']['path'] == str(data.PLAN)
         and source.get('stage') == data.STAGE
         and source.get('source_spec', {}).get('issue_start_ns') == data.ISSUE_START_NS
         and source.get('source_spec', {}).get('cutoff_ns') == data.CUTOFF_NS,
         'FULL_HISTORY_003_IDENTITY')
    spec = feature_spec(source)
    target = read(inputs['target_result'])
    terminal = read(inputs['target_terminal'])
    need(target['target_tape'] == terminal['target_tape'] == inputs['target_tape']
         and target['plan'] == terminal['plan'] == inputs['native_source_plan']
         and terminal['result'] == inputs['target_result'] and terminal['status'] == 'COMPLETE'
         and terminal['exit_code'] == 0 and target['source_unchanged'] is True
         and target['target_contract'] == target_contract()
         and target['test_data_used'] is terminal['test_data_used'] is False,
         'TARGET_CHAIN')
    dependencies = macro_metadata(inputs['macro_source_plan'])
    metadata = {part['path']: data.part_metadata(part) for part in spec['parts']}
    for part in spec['parts']:
        need(all(key in pq.ParquetFile(part['path']).schema_arrow.names
                 for key in ('time', *RAW_COLUMNS)), 'RAW_FEATURE_COLUMNS')
    head = source_base_commit or subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return {
        'schema_version': 'gx1_native_m5_pipeline_plan_v1', 'stage': STAGE,
        'status': 'PREPARED_NOT_ADMITTED', 'output_root': str(ROOT),
        'source_base_commit': head, 'source_owners': {p: sha(REPO / p) for p in SOURCE_PATHS},
        'inputs': inputs, 'macro_dependencies': dependencies, 'feature_source_spec': spec,
        'feature_source_metadata': metadata, 'parameter_source_spec': squeeze['source_spec'],
        'target_source_spec': source['source_spec'], 'target_source_metadata': source['source_metadata'],
        'fit_start_ns': FIT_START_NS, 'fit_end_ns': FIT_END_NS,
        'fit_axis': 'native_M5_issue_in_window_and_input_ready_and_any_mature_supervised_target',
        'normalization_population': 'whole_exact_consumption_union_of_selected_TRAIN_issues',
        'geometry': native_bar_backbone_geometry(), 'architecture': ARCHITECTURE,
        'seed': SEED, 'batch_rows': BATCH, 'execution': EXECUTION,
        'feature_contract': native_m5_feature_provider_contract(),
        'target_contract': target_contract(), 'objective_contract': native_bar_objective_contract(),
        'metadata_scope': 'genuine_source_macro_and_target_receipts_no_price_decode',
        'model_forwards': 1, 'model_constructions': 1, 'optimizer_steps': 0,
        'training_authorized': False, 'test_data_used': False, 'economic_edge_proven': False,
    }


def source_clean(plan):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return (subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
            == 'work/gx1-current'
        and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
        and subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_base_commit'], head],
                           cwd=REPO, capture_output=True).returncode == 0
        and subprocess.run(['git', 'diff', '--quiet', plan['source_base_commit'], head,
                           '--', 'gx1', 'scripts'], cwd=REPO).returncode == 0
        and set(plan['source_owners']) == set(SOURCE_PATHS)
        and all(sha(REPO / path) == digest for path, digest in plan['source_owners'].items()))


def checked_inputs(plan):
    bindings = list(plan['inputs'].values()) + list(plan['macro_dependencies'].values())
    bindings.append(plan['feature_source_spec']['manifest'])
    for spec in (plan['feature_source_spec'], plan['parameter_source_spec'], plan['target_source_spec']):
        bindings.extend({'path': part['path'], 'sha256': part['sha256']} for part in spec['parts'])
    seen = {}
    for binding in bindings:
        path = binding['path']
        need(path not in seen or seen[path] == binding, 'INPUT_SPLIT_BINDING')
        if path not in seen:
            need(ref(path) == binding, 'INPUT_IDENTITY')
            seen[path] = binding
    for part in plan['feature_source_spec']['parts']:
        need(data.part_metadata(part) == plan['feature_source_metadata'][part['path']],
             'FEATURE_METADATA')


def inspect(plan_sha, *, admitted):
    need(Path.cwd().resolve() == REPO and not ROOT.exists(), 'CWD_OUTPUT')
    plan_ref = {'path': str(PLAN), 'sha256': plan_sha}
    plan = read(plan_ref)
    need(plan == build_plan(inputs=plan['inputs'], source_base_commit=plan['source_base_commit'])
         and source_clean(plan), 'PLAN_SOURCE')
    policy_ref = ref(POLICY)
    policy = read(policy_ref)
    gate = policy.get(STAGE, {})
    need(all(policy.get(key) is False for key in data.CLOSED_FLAGS)
         and gate.get('plan') == plan_ref and gate.get('output_root') == str(ROOT)
         and gate.get('relaunch_allowed') is False and gate.get('max_invocations') == 1
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('training_authorized') is False and gate.get('test_authorized') is False
         and gate.get('broker_access_allowed') is False
         and gate.get('model_forwards') == 1 and gate.get('optimizer_steps') == 0, 'CLOSED_POLICY')
    expected = ('ADMITTED_ONE_USE', True, 1) if admitted else ('PREPARED_NOT_ADMITTED', False, 0)
    need((gate.get('status'), gate.get('execution_authorized'),
          gate.get('remaining_invocations')) == expected, 'ONE_USE_GATE')
    checked_inputs(plan)
    return plan, plan_ref, policy_ref


def publish(name, value):
    raw = value if isinstance(value, bytes) else (
        json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    _publish_bytes(ROOT / name, raw)
    return {'path': str(ROOT / name), 'sha256': hashlib.sha256(raw).hexdigest()}


def physical_parameter_fits(plan):
    from gx1.scripts.fit_volatility_squeeze_artifacts_v1 import native_m5_fit_frames_from_plan
    from gx1.features.volatility_squeeze_state_v1 import (
        VOLATILITY_SQUEEZE_DIRECTION_CLOCKS,
        fit_volatility_squeeze_artifact_manifest,
    )
    from gx1.features.htf_features import (
        fit_v29_registry_constants_from_m5, require_native_m5_registry_constants,
    )
    inputs = plan['inputs']
    squeeze, registry = fit_plans(inputs)
    frames, provenance, proof = native_m5_fit_frames_from_plan(
        Path(inputs['squeeze_fit_plan']['path']),
        expected_sha256=inputs['squeeze_fit_plan']['sha256'])
    output = ROOT / 'SQUEEZE'
    output.mkdir()
    manifest = fit_volatility_squeeze_artifact_manifest(
        frames, declared_train_window_start=pd.Timestamp(PARAMETER_START_NS, tz='UTC'),
        declared_train_window_end=pd.Timestamp(PARAMETER_END_NS, tz='UTC'),
        source_provenance_by_clock=provenance, output_dir=output,
        clocks=VOLATILITY_SQUEEZE_DIRECTION_CLOCKS)
    squeeze_ref = ref(manifest)
    saved = read(squeeze_ref)
    for tf in VOLATILITY_SQUEEZE_DIRECTION_CLOCKS:
        actual, expected = saved['artifacts'][tf], proof['fit_frames'][tf]
        need(actual['train_ohlcv_rows'] == expected['rows']
             and actual['train_ohlcv_sha256'] == expected['ohlcv_sha256'], 'PHYSICAL_SQUEEZE_IDENTITY')
    geometry = plan['geometry']
    constants = fit_v29_registry_constants_from_m5(
        frames['M5'], declared_train_window_start=registry['declared_train_window_start'],
        declared_train_window_end=registry['declared_train_window_end'],
        declared_inner_fit_window_end=registry['declared_inner_fit_window_end'],
        source_provenance_by_clock=registry_provenance(inputs['registry_fit_plan'], registry),
        per_tf_seq_lens={tf: geometry[tf] for tf in VOLATILITY_SQUEEZE_DIRECTION_CLOCKS},
        entry_m5_seq_len=geometry['local'])
    constants = require_native_m5_registry_constants(constants)
    registry_ref = publish('REGISTRY_CONSTANTS.json', constants)
    return {'squeeze_manifest': squeeze_ref, 'registry_constants': constants,
            'registry_artifact': registry_ref, 'physical_fit_proof': proof}


def read_feature_source(plan):
    from gx1.contracts.market_state_native_m5_feature_inputs_v1 import RAW_COLUMNS
    spec = plan['feature_source_spec']
    pieces = []
    for part in spec['parts']:
        before = stamp(part['path'])
        need(before == plan['feature_source_metadata'][part['path']]['stat'], 'FEATURE_SOURCE_STAT')
        parquet = pq.ParquetFile(part['path'])
        for group in part['row_groups']:
            table = parquet.read_row_group(group, columns=['time', *RAW_COLUMNS], use_threads=False)
            need(all(table[name].null_count == 0 for name in table.column_names), 'RAW_NULL')
            frame = table.to_pandas()
            starts = pd.to_datetime(frame.pop('time'), utc=True, errors='raise')
            frame.index = pd.DatetimeIndex(starts).as_unit('ns')
            pieces.append(frame.loc[frame.index.asi8 < spec['cutoff_ns'] - data.M5_NS,
                                    list(RAW_COLUMNS)])
        need(stamp(part['path']) == before, 'FEATURE_SOURCE_CHANGED')
    need(len(pieces) > 0, 'FEATURE_SOURCE_EMPTY')
    frame = pd.concat(pieces)
    need(frame.index.is_monotonic_increasing and not frame.index.has_duplicates, 'FEATURE_SOURCE_ORDER')
    return frame


def fold_truth(truth, upper):
    value = {key: array.copy() for key, array in truth.items()}
    mature = value['target_present'] & (value['validity_ns'] < upper)
    value['mature'] = mature
    value['observed_previous'] &= mature
    value['eligible'] = mature & value['observed_previous']
    value['target_row'][~mature] = -1
    for key in ('open_sign', 'close_sign'):
        value[key][~mature] = 0
    return value


def target_axis(plan, provider):
    with np.load(plan['inputs']['target_tape']['path'], allow_pickle=False) as archive:
        source = archive['source_row']
        issues = archive['issue_ns']
        wanted = np.flatnonzero((provider.issue_ns >= FIT_START_NS) & (provider.issue_ns < FIT_END_NS))
        need(len(wanted) > 0, 'EMPTY_TRAIN_WINDOW')
        at = np.searchsorted(source, provider.source_row[wanted])
        need(np.all(at < len(source)) and np.array_equal(source[at], provider.source_row[wanted])
             and np.array_equal(issues[at], provider.issue_ns[wanted]), 'TARGET_AXIS_JOIN')
        keys = ('issue_ns', 'target_row', 'validity_ns', 'target_present', 'observed_previous',
                'mature', 'eligible', 'open_known_at_issue', 'open_sign', 'close_sign')
        truth = {key: archive['future_' + key][at] for key in keys}
    truth = fold_truth(truth, FIT_END_NS)
    supervised = truth['eligible'].any(axis=1)
    ready = provider.input_ready[wanted]
    chosen = ready & supervised
    rows = provider.source_row[wanted[chosen]]
    need(len(rows) >= BATCH and np.all(np.diff(rows) > 0), 'INSUFFICIENT_GENUINE_FIT')
    report = {
        'window_issue_rows': len(wanted), 'input_not_ready_rows': int((~ready).sum()),
        'no_mature_supervision_rows': int((~supervised).sum()), 'fit_issue_rows': len(rows),
        'fit_first_issue_ns': int(provider.issue_ns[rows[0]]),
        'fit_last_issue_ns': int(provider.issue_ns[rows[-1]]),
        'fit_source_rows_sha256': hashlib.sha256(np.ascontiguousarray(rows, dtype='<i8').tobytes()).hexdigest(),
        'fit_issue_ns_sha256': hashlib.sha256(np.ascontiguousarray(provider.issue_ns[rows], dtype='<i8').tobytes()).hexdigest(),
        'per_tf_mature': truth['mature'][chosen].sum(axis=0).tolist(),
        'per_tf_eligible': truth['eligible'][chosen].sum(axis=0).tolist(),
    }
    return rows, {key: value[chosen] for key, value in truth.items()}, report


def normalized_batch_report(model, tensors):
    report = {}
    for key, tensor in tensors.items():
        if key == 'ctx_cat':
            normalized = tensor
            role = 'categorical_domain_verified_by_native_forward'
        else:
            surface = 'signal' if key in ('seq_x', 'snap_x') else (
                'ctx_cont' if key == 'ctx_cont' else 'mtf_' + key.removeprefix('seq_'))
            normalized = model.backbone._normalize_input_surface(tensor, surface=surface)
            role = 'same_native_backbone_input_normalization'
        raw = tensor.detach().cpu().numpy()
        values = normalized.detach().cpu().numpy()
        need(np.isfinite(raw).all() and np.isfinite(values).all(), 'NORMALIZED_NONFINITE_' + key)
        report[key] = {'shape': list(raw.shape), 'dtype': str(raw.dtype),
            'finite_values': int(values.size), 'role': role,
            'raw_sha256': hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
            'normalized_sha256': hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()}
    need(len(report) == 9, 'NINE_NATIVE_SURFACES')
    return report


def qualify_physical_pipeline(plan, parameters):
    import torch
    from gx1.features import d1_macro_context_v1 as macro
    from gx1.features.volatility_squeeze_state_v1 import (
        load_volatility_squeeze_artifact_manifest, VOLATILITY_SQUEEZE_DIRECTION_CLOCKS,
    )
    from gx1.contracts import market_state_native_m5_feature_inputs_v1 as features
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_bar_direction_model
    from gx1.contracts.market_state_ta_integrated_objective_v1 import (
        native_bar_direction_loss, native_bar_direction_score_sums,
    )
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256

    native = read_feature_source(plan)
    spec = plan['feature_source_spec']
    binding = features.bind_native_m5_feature_source(
        native, native_m5_manifest=spec['manifest'], source_spec=spec)
    squeeze_ref = parameters['squeeze_manifest']
    squeeze = load_volatility_squeeze_artifact_manifest(Path(squeeze_ref['path']),
        expected_sha256=squeeze_ref['sha256'], clocks=VOLATILITY_SQUEEZE_DIRECTION_CLOCKS)
    surfaces = features.build_native_m5_feature_surfaces(
        native, source_binding=binding, v29_registry_constants=parameters['registry_constants'],
        volatility_squeeze_artifacts=squeeze)
    provider = features.NativeM5FeatureProvider(surfaces)
    rows, truth, population = target_axis(plan, provider)
    window_rows = np.flatnonzero((provider.issue_ns >= FIT_START_NS) & (provider.issue_ns < FIT_END_NS))
    readiness = provider.readiness_report(window_rows)
    support = provider.support_report(rows)
    need(all(all(count == surface['physical_rows'] for count in surface['observed_counts'])
             for surface in support['surfaces'].values())
         and len(support['families']) == 8 and len(support['macro_fields']) == 61,
         'GENUINE_CONSUMED_SUPPORT')
    normalization = features.fit_native_m5_feature_normalization(
        provider, fit_source_rows=rows, cutoff_ns=FIT_END_NS)
    normal_ref = publish('NORMALIZATION.json', normalization)
    input_binding = provider.input_binding(normalization)
    batch_rows = rows[:BATCH]
    raw = provider.gather(batch_rows, max_rows=BATCH)
    batch_truth = {key: value[:BATCH] for key, value in truth.items()}
    torch.set_num_threads(plan['execution']['numerical_threads'])
    torch.manual_seed(plan['seed'])
    model = build_native_bar_direction_model(input_normalization=normalization,
        input_binding=input_binding, architecture=plan['architecture'], device=torch.device('cpu')).eval()
    initial_sha = canonical_model_state_sha256(model.state_dict())
    tensors = {key: torch.from_numpy(value) for key, value in raw.items()}
    with torch.inference_mode():
        normalized = normalized_batch_report(model, tensors)
        output = model(tensors)
        loss = native_bar_direction_loss(model_output=output, truth=batch_truth,
            fold_upper_ns=FIT_END_NS, target_contract_sha256=plan['target_contract']['contract_sha256'])
        score = native_bar_direction_score_sums(model_output=output, truth=batch_truth,
            fold_upper_ns=FIT_END_NS, target_contract_sha256=plan['target_contract']['contract_sha256'])
    need(not loss['skip_update'] and bool(torch.isfinite(loss['loss']))
         and all(bool(torch.isfinite(value).all()) for value in output.values())
         and canonical_model_state_sha256(model.state_dict()) == initial_sha, 'FRESH_FORWARD_IDENTITY')
    checkpoint_binding = {
        'input': input_binding, 'model': model.research_objective,
        'target': plan['target_contract'], 'normalization': normal_ref,
        'architecture': plan['architecture'], 'seed': plan['seed'],
        'model_state_sha256': initial_sha}
    stream = io.BytesIO()
    torch.save({'model_state': model.state_dict(), 'bindings': checkpoint_binding}, stream)
    payload = stream.getvalue()
    loaded = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    need(loaded['bindings'] == checkpoint_binding
         and canonical_model_state_sha256(loaded['model_state']) == initial_sha, 'CHECKPOINT_ROUNDTRIP')
    model.load_state_dict(loaded['model_state'], strict=True)
    need(canonical_model_state_sha256(model.state_dict()) == initial_sha, 'CHECKPOINT_STRICT_LOAD')
    checkpoint_ref = publish('INITIAL_MODEL.pt', payload)
    capture = {'source_row': batch_rows, **{'truth_' + key: value for key, value in batch_truth.items()},
               **{key: value.detach().cpu().numpy() for key, value in output.items()}}
    stream = io.BytesIO()
    np.savez_compressed(stream, **capture)
    capture_ref = publish('INITIAL_CAPTURE.npz', stream.getvalue())
    fields = provider.profile['physical_field_names']['signal']
    missing = {field: int((~np.isfinite(surfaces['signal'][:, j])).sum())
               for j, field in enumerate(fields) if not np.isfinite(surfaces['signal'][:, j]).all()}
    return {
        'status': 'TECHNICAL_INITIAL_BATCH_ONLY', 'model_constructions': 1,
        'model_forwards': 1, 'optimizer_steps': 0, 'checkpoint': checkpoint_ref,
        'capture': capture_ref, 'normalization': normal_ref, 'input_binding': input_binding,
        'model_contract': model.research_objective, 'initial_state_sha256': initial_sha,
        'strict_load_state_sha256': initial_sha, 'forward_surfaces': normalized,
        'fit_population': population, 'normalization_population': normalization['fit_population'],
        'input_readiness': readiness, 'consumed_support': support,
        'routed_families': support['families'],
        'field_lineage': surfaces['field_lineage'], 'macro_field_count': len(macro.D1_MACRO_CONTEXT_FIELDS),
        'physical_missing_value_counts': missing, 'input_ready_rows': int(provider.input_ready.sum()),
        'batch_loss': float(loss['loss']), 'score_sums': score,
        'full_TRAIN_baseline_qualified': False, 'predictive_quality_measured': False,
        'economic_edge_proven': False,
    }


def verify_native_m5_bar_facts_003(*, source_spec, archive_ref):
    """Read pinned originals and compare every 003 NPZ array without target owners."""
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    import pyarrow.parquet as pq

    tick = 300_000_000_000
    cutoff = 1782864000000000000
    issue_start = 1243814400000000000
    native = Path('/home/andre2/GX1_DATA/data/native_xau/XAU_M5_NATIVE_2009_20260701_PAIR_WEEKCLOSED_20260927')
    manifest_ref = {'path': str(native / 'MANIFEST.json'),
        'sha256': 'ca0012e5e30f7808876d666fe4d32e3fb873fddb6fc6178bd47b97eb89194cca'}
    durations = {'M5': tick, 'M15': 3*tick, 'H1': 12*tick,
                 'H4': 48*tick, 'D1': 288*tick}
    offsets = {'M5': 0, 'M15': 0, 'H1': 0, 'H4': 24*tick, 'D1': 264*tick}
    columns = ['time', 'open', 'high', 'low', 'close']

    def need(ok, reason):
        if not bool(ok):
            raise RuntimeError('NATIVE_M5_INDEPENDENT_FACTS_' + reason)

    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        return h.hexdigest()

    def checked_ref(ref):
        need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'REF')
        p = Path(ref['path'])
        need(p.is_absolute() and not p.is_symlink() and p.resolve(strict=True) == p
             and p.is_file() and digest(p) == ref['sha256'], 'REF_HASH')
        return p

    need(type(source_spec) is dict and set(source_spec) == {
        'manifest', 'parts', 'closure_authority', 'issue_start_ns', 'cutoff_ns'}
        and source_spec['manifest'] == manifest_ref
        and source_spec['closure_authority'] is None
        and source_spec['issue_start_ns'] == issue_start
        and source_spec['cutoff_ns'] == cutoff, 'SOURCE_IDENTITY')
    manifest = json.loads(checked_ref(manifest_ref).read_bytes())
    identity = {'schema_version': 'xau_canonical_native_source_v3',
        'instrument': 'XAU_USD', 'timeframe': 'M5', 'source_granularity': 'M5',
        'source_kind': 'oanda_native_mba_candles', 'prices': 'MBA',
        'bar_duration_seconds': 300, 'decision_available_offset_seconds': 300,
        'timestamp_semantics': 'bar_start_utc', 'completion_field': 'complete',
        'completion_value': True}
    need(all(manifest.get(k) == v for k, v in identity.items()), 'MANIFEST_IDENTITY')
    need(set(manifest['year_rows']) == {f'year={year}' for year in range(2009, 2027)},
         'MANIFEST_YEARS')
    parts = source_spec['parts']
    need(type(parts) is list and len(parts) == 18, 'PART_COUNT')
    archive_path = checked_ref(archive_ref)
    starts_parts, price_parts, decoded_rows = [], [], 0
    for year, part in zip(range(2009, 2027), parts):
        key = f'year={year}'
        need(type(part) is dict and set(part) == {'path', 'sha256', 'row_groups'}
             and part['path'] == str(native / key / 'part-000.parquet')
             and part['sha256'] == manifest['year_sha256'][key], 'PART_IDENTITY')
        p = checked_ref({k: part[k] for k in ('path', 'sha256')})
        parquet = pq.ParquetFile(p)
        need(parquet.metadata.num_rows == manifest['year_rows'][key]
             and all(k in parquet.schema_arrow.names for k in columns), 'PART_METADATA')
        time_col = parquet.schema_arrow.names.index('time')
        expected_groups = []
        for group in range(parquet.num_row_groups):
            clock = parquet.metadata.row_group(group).column(time_col).statistics
            need(clock is not None and clock.has_min_max, 'RAW_CLOCK_STATISTICS')
            first_ns = int(clock.min.timestamp()) * 1_000_000_000
            if first_ns + tick < cutoff:
                expected_groups.append(group)
        need(part['row_groups'] == expected_groups, 'EXACT_PREFIX_ROW_GROUPS')
        for group in expected_groups:
            table = parquet.read_row_group(group, columns=columns, use_threads=False)
            need(all(table[k].null_count == 0 for k in columns), 'RAW_NULL')
            starts = table['time'].to_numpy(zero_copy_only=False).astype('datetime64[ns]').astype(np.int64)
            prices = np.column_stack([table[k].to_numpy(zero_copy_only=False)
                                      for k in columns[1:]]).astype(np.float64)
            take = starts + tick < cutoff
            starts_parts.append(starts[take]); price_parts.append(prices[take])
            decoded_rows += len(starts)
    starts = np.concatenate(starts_parts)
    prices = np.concatenate(price_parts)
    del starts_parts, price_parts
    need(len(starts) > 0 and np.all(starts >= 0) and np.all(starts % tick == 0)
         and np.all(np.diff(starts) > 0) and np.all(starts + tick < cutoff), 'RAW_CLOCK')
    need(np.isfinite(prices).all() and np.all(prices > 0)
         and np.all(prices[:, 1] >= np.maximum(prices[:, 0], prices[:, 3]))
         and np.all(prices[:, 2] <= np.minimum(prices[:, 0], prices[:, 3])), 'RAW_OHLC')
    gap = np.r_[False, np.diff(starts) != tick]
    selected = np.flatnonzero((starts + tick >= issue_start) & (starts + tick < cutoff))
    issues = starts[selected] + tick
    need(len(issues) > 0, 'EMPTY_ISSUES')
    shape = (len(issues), len(durations))
    future = {'issue_ns': issues.copy(),
        'target_row': np.full(shape, -1, np.int64),
        'validity_ns': np.full(shape, -1, np.int64),
        'target_present': np.zeros(shape, np.bool_),
        'observed_previous': np.zeros(shape, np.bool_),
        'mature': np.zeros(shape, np.bool_), 'eligible': np.zeros(shape, np.bool_),
        'open_known_at_issue': np.zeros(shape, np.bool_),
        'open_sign': np.zeros(shape, np.int8), 'close_sign': np.zeros(shape, np.int8),
        'm5_count': np.zeros(shape, np.int64), 'source_gap': np.zeros(shape, np.bool_)}
    bar_fields = ('bar_start_ns', 'available_ns', 'open_available_ns', 'ohlc',
        'previous_open', 'previous_close', 'open_sign', 'close_sign',
        'observed_previous', 'm5_count', 'source_gap')
    expected_keys = {'issue_ns', 'source_row', 'ohlc', 'source_gap'}
    expected_keys.update('future_' + k for k in future)
    expected_keys.update('bars_' + tf + '_' + k for tf in durations for k in ('row',) + bar_fields)
    compared, coverage = [], {}
    with np.load(archive_path, allow_pickle=False) as saved:
        need(len(expected_keys) == 76 and len(saved.files) == 76
             and set(saved.files) == expected_keys, 'ALL_76_ARRAY_KEYS')
        def compare(key, expected):
            actual = saved[key]
            need(actual.dtype == expected.dtype and actual.shape == expected.shape,
                 'ARRAY_HEADER:' + key)
            equal = np.array_equal(actual, expected, equal_nan=True) if np.issubdtype(
                expected.dtype, np.floating) else np.array_equal(actual, expected)
            need(equal, 'ARRAY_VALUES:' + key)
            compared.append(key)

        compare('issue_ns', issues)
        compare('source_row', selected.astype(np.int64))
        compare('ohlc', prices[selected])
        compare('source_gap', gap[selected])
        for j, (tf, duration) in enumerate(durations.items()):
            label = (starts - offsets[tf]) // duration * duration + offsets[tf]
            first = np.r_[0, np.flatnonzero(np.diff(label) != 0) + 1]
            last = np.r_[first[1:] - 1, len(starts) - 1]
            labels = label[first]
            keep = labels + duration < cutoff
            values = np.column_stack((prices[first, 0],
                np.maximum.reduceat(prices[:, 1], first),
                np.minimum.reduceat(prices[:, 2], first), prices[last, 3]))[keep]
            available = (labels + duration)[keep]
            native_gap = np.logical_or.reduceat(gap, first)[keep]
            if len(available) > 1:
                native_gap[1:] |= np.diff(available) != duration
            n = len(available)
            need(n > 0, 'EMPTY_TF:' + tf)
            previous_open = np.r_[np.nan, values[:-1, 0]].astype(np.float64)
            previous_close = np.r_[np.nan, values[:-1, 3]].astype(np.float64)
            opens = np.r_[0, np.sign(np.diff(values[:, 0]))].astype(np.int8)
            closes = np.r_[0, np.sign(np.diff(values[:, 3]))].astype(np.int8)
            bars = {'bar_start_ns': labels[keep], 'available_ns': available,
                'open_available_ns': starts[first[keep]] + tick, 'ohlc': values,
                'previous_open': previous_open, 'previous_close': previous_close,
                'open_sign': opens, 'close_sign': closes,
                'observed_previous': np.arange(n) > 0,
                'm5_count': (last - first + 1)[keep].astype(np.int64),
                'source_gap': native_gap}
            at = np.flatnonzero(available >= issue_start)
            compare('bars_' + tf + '_row', at.astype(np.int64))
            for key in bar_fields:
                compare('bars_' + tf + '_' + key, bars[key][at])
            next_row = np.searchsorted(available, issues, side='right')
            present = next_row < n
            future['target_present'][:, j] = present
            future['validity_ns'][present, j] = available[next_row[present]]
            future['open_known_at_issue'][present, j] = (
                bars['open_available_ns'][next_row[present]] <= issues[present])
            ready = present & (future['validity_ns'][:, j] < cutoff)
            future['mature'][:, j] = ready
            future['target_row'][ready, j] = next_row[ready]
            for key in ('observed_previous', 'open_sign', 'close_sign', 'm5_count', 'source_gap'):
                future[key][ready, j] = bars[key][next_row[ready]]
            future['eligible'][:, j] = ready & future['observed_previous'][:, j]
            years = available.astype('datetime64[ns]').astype('datetime64[Y]')
            previous_years = np.r_[years[:1], years[:-1]]
            known = bars['observed_previous']
            coverage[tf] = {
                'observed_bars': int(len(at)),
                'missing_first_previous': int((~known[at]).sum()),
                'cross_year_previous_bars': int((known[at] & (years[at] != previous_years[at])).sum()),
                'yearly': {str(year): {
                    'observed_bars': int((years[at] == year).sum()),
                    'missing_previous': int(((years[at] == year) & ~known[at]).sum()),
                    'cross_year_previous': int(((years[at] == year) & known[at]
                        & (years[at] != previous_years[at])).sum()),
                } for year in np.unique(years[at])},
                'known_equal_open': int(((opens[at] == 0) & known[at]).sum()),
                'known_equal_close': int(((closes[at] == 0) & known[at]).sum()),
                'future_present': int(present.sum()), 'future_mature': int(ready.sum()),
                'future_eligible': int(future['eligible'][:, j].sum()),
                'future_open_known_at_issue': int(future['open_known_at_issue'][:, j].sum()),
                'future_unique_mature_rows': int(np.unique(next_row[ready]).size),
            }
        for key, value in future.items():
            compare('future_' + key, value)
    need(len(compared) == 76 and set(compared) == expected_keys, 'COMPLETE_COMPARISON')
    checked_ref(manifest_ref)
    for part in parts:
        checked_ref({k: part[k] for k in ('path', 'sha256')})
    checked_ref(archive_ref)
    return {'schema_version': 'native_m5_independent_all_history_facts_verification_v1',
        'raw_parts': len(parts), 'decoded_rows': decoded_rows, 'retained_source_rows': len(starts),
        'issue_rows': len(issues), 'array_count': len(compared),
        'compared_arrays': sorted(compared), 'all_arrays_exact_equal_nan': True,
        'raw_and_archive_hash_bookends_unchanged': True,
        'aggregation': 'independent_integer_UTC_buckets_numpy_reduceat_first_open_max_high_min_low_last_close',
        'offset_ns': offsets, 'strict_exclusive_close_cutoff_ns': cutoff,
        'source_row_axis': 'own_ordered_filtered_raw_M5_prefix_ordinal',
        'coverage_by_timeframe': coverage,
        'target_metadata_never_model_input': True,
        'target_resampler_audit_owner_imported': False}


def run(plan_sha):
    started = time.monotonic()
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 12 * 1024**3
         and cap['swap'] == 512 * 1024**2 and cap['numerical_threads'] == 1
         and lock['scope'] == cap['cgroup_path'][len('/sys/fs/cgroup'):]
         and signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'CAP_LIMITS')
    plan, plan_ref, policy_ref = inspect(plan_sha, admitted=True)
    ROOT.mkdir(parents=True, exist_ok=False)
    claim = publish('CLAIM.json', {'status': 'CLAIMED_ONE_USE', 'stage': STAGE,
        'plan': plan_ref, 'policy_at_claim': policy_ref, 'remaining_invocations': 0,
        'max_invocations': 1, 'test_data_used': False})
    result_ref = None
    old_handler = signal.getsignal(signal.SIGALRM)
    def expired(*args):
        raise TimeoutError('NATIVE_M5_PIPELINE_INTERNAL_LIMIT')
    try:
        remaining = plan['execution']['internal_seconds'] - (time.monotonic() - started)
        need(remaining > 0, 'TIME_BEFORE_PHYSICAL_WORK')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        history = verify_native_m5_bar_facts_003(
            source_spec=plan['target_source_spec'], archive_ref=plan['inputs']['target_tape'])
        parameters = physical_parameter_fits(plan)
        report = qualify_physical_pipeline(plan, parameters)
        checked_inputs(plan)
        need(source_clean(plan) and ref(POLICY) == policy_ref and ref(PLAN) == plan_ref,
             'SOURCE_INPUT_POLICY_BOOKEND')
        result_ref = publish('RESULT.json', {
            'status': 'COMPLETE_NATIVE_M5_PIPELINE_TECHNICAL_QUALIFICATION', 'stage': STAGE,
            'plan': plan_ref, 'claim': claim, 'policy_at_claim': policy_ref,
            'source_unchanged': True, 'parameters': {key: value for key, value in parameters.items()
                                                   if key != 'registry_constants'},
            'history_verification': history,
            'report': report, 'model_forwards': 1, 'optimizer_steps': 0,
            'normalization_fits': 1, 'test_data_used': False, 'economic_edge_proven': False,
            'elapsed_seconds': time.monotonic() - started})
        terminal = publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'stage': STAGE, 'plan': plan_ref, 'claim': claim, 'result': result_ref,
            'remaining_invocations': 0, 'source_unchanged': True,
            'relaunch_allowed': False, 'test_data_used': False})
        return {'result': result_ref, 'terminal': terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        publish('TERMINAL.json', {'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'stage': STAGE, 'plan': plan_ref, 'claim': claim, 'result': result_ref,
            'error': type(error).__name__ + ':' + str(error)[:2000],
            'remaining_invocations': 0, 'relaunch_allowed': False, 'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    need(args.plan == str(PLAN) and args.out_dir == str(ROOT), 'CLI_SCOPE')
    if args.preflight:
        plan, plan_ref, policy_ref = inspect(args.plan_sha256, admitted=False)
        print(json.dumps({'status': 'PREFLIGHT_PASS_GENUINE_METADATA_ONLY', 'plan': plan_ref,
            'policy': policy_ref, 'claim_created': False, 'output_root_absent': True,
            'parquet_values_decoded': 0, 'model_constructions': 0,
            'model_forwards': 0, 'test_data_used': False}))
    else:
        print(json.dumps(run(args.plan_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
