"""One claimed, model-forward-free parity audit of eight physical MTF rows.

Compare the bound NumPy normalizer with the candidate model's runtime input
normalizer. This does not train, score, select indicators, or read TEST labels.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import time

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'dynamic_m5_october_native_postnorm_20261106_001'
PLAN_PATH = REPO / 'configs/research/DYNAMIC_M5_OCTOBER_NATIVE_POSTNORM_20261106_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_POSTNORM_20261106_001')
TFS = ('M5', 'M15', 'H1', 'H4', 'D1')
POSITIONS = (0, 1, 29847, 29848, 29849, 29850, 36170, 36171)
FIELDS = ('atr_bps_14', 'ema20_dist_atr', 'mtf_geometry_support_dist_atr',
          'mtf_geometry_resistance_dist_atr')
WITNESSES = ('mtf_level_above_dist_atr', 'mtf_level_above_touch_count',
             'mtf_level_above2_dist_atr', 'mtf_level_above2_present',
             'mtf_level_below_dist_atr', 'mtf_level_below_present',
             'mtf_level_below2_dist_atr', 'mtf_level_below2_present',
             'mtf_level_above_pending_retest_dist_atr',
             'mtf_level_above_pending_retest_age_bars',
             'mtf_level_below_pending_retest_dist_atr',
             'mtf_level_below_pending_retest_age_bars')
BUDGET = {'memory_max': 4294967296, 'swap': 536870912, 'pids': 64,
          'numerical_threads': 1, 'max_wall_seconds': 1800,
          'max_invocations': 1, 'max_json_bytes': 1048576}
RTOL = ATOL = 2e-6
COMPATIBILITY = {
    'key': 'direction_component_source',
    'path': '/home/andre2/src/GX1_CURRENT/gx1/contracts/entry_native_direction_research_v1.py',
    'commit': 'f574d838d7c54d5ed36881cdbb4525efe167dc15',
    'old_sha256': '627e063c51524d1bd9bfcf4f70b4c29451ddbb0e63d4ecda0906f1c327c3d497',
    'current_sha256': 'dba2d263f0d28e52e73fa38e1d7febd88b3e9a5958edc566f30df0ca69180fa3',
    'patch_sha256': '827bec7f99ecf635a0f821b63eeffbce1e4ff44072a973f5b76c0912de2a2054',
    'scope': 'f574d838 changes only confirmed_run_age_input validation and loss-contract identity in native_dynamic_mtf_state_loss; no normalization or model constructor change',
}
STAT_KEYS = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
JSON_BYTES = 0


def require(ok, reason):
    if not ok:
        raise RuntimeError('DYNAMIC_M5_POSTNORM_' + reason)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), 'PATH:' + str(path))
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), 'REGULAR:' + str(path))
    return {key: int(getattr(info, key)) for key in STAT_KEYS}


def ref(path):
    path = Path(path)
    before = regular(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    require(regular(path) == before, 'HASH_BOOKEND:' + str(path))
    return {'path': str(path), 'sha256': digest.hexdigest()}


def read(binding):
    require(set(binding) == {'path', 'sha256'}
            and regular(binding['path'])['st_size'] <= 64 * 1024**2
            and ref(binding['path']) == binding, 'JSON_REF')
    return json.loads(Path(binding['path']).read_text())


def publish(name, value):
    global JSON_BYTES
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    require(JSON_BYTES + len(data) <= BUDGET['max_json_bytes'], 'JSON_BUDGET')
    _publish_bytes(ROOT / name, data)
    (ROOT / name).chmod(0o444)
    JSON_BYTES += len(data)
    return ref(ROOT / name)


def freeze(binding, plan, snapshots):
    require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
            == 'work/gx1-current', 'BRANCH')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
            == binding['source_head'], 'HEAD')
    require(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True) == '', 'DIRTY')
    for item in (binding['plan'], binding['policy'], binding['worker'], *plan['source_owners'].values()):
        require(ref(item['path']) == item, 'SOURCE_CHANGED:' + item['path'])
    for path, before in snapshots.items():
        require(regular(path) == before, 'PAYLOAD_STAT_CHANGED:' + path)



@contextmanager
def forbid_model_work(torch):
    """Fail closed if construction or normalization enters learning or forward."""
    counts = {'model_forwards': 0, 'backward_calls': 0, 'optimizer_steps': 0}
    require(sys.getprofile() is None, 'PROFILE_ALREADY_ACTIVE')

    def profile(frame, event, _arg):
        if event != 'call':
            return
        name = frame.f_code.co_name
        if name not in ('_call_impl', '_wrapped_call_impl', 'forward',
                        'backward', 'grad', 'step'):
            return
        owner = frame.f_locals.get('self')
        if name in ('_call_impl', '_wrapped_call_impl', 'forward') and isinstance(owner, torch.nn.Module):
            counts['model_forwards'] += 1
            raise RuntimeError('DYNAMIC_M5_POSTNORM_MODEL_FORWARD_FORBIDDEN')
        if (name in ('backward', 'grad') and
                (frame.f_globals.get('__name__', '').startswith('torch.autograd')
                 or isinstance(owner, torch.Tensor))):
            counts['backward_calls'] += 1
            raise RuntimeError('DYNAMIC_M5_POSTNORM_BACKWARD_FORBIDDEN')
        if name == 'step' and isinstance(owner, torch.optim.Optimizer):
            counts['optimizer_steps'] += 1
            raise RuntimeError('DYNAMIC_M5_POSTNORM_OPTIMIZER_FORBIDDEN')

    sys.setprofile(profile)
    try:
        yield counts
    finally:
        sys.setprofile(None)



def verify_view_source_compatibility(view, plan):
    require(plan['source_compatibility'] == COMPATIBILITY, 'COMPATIBILITY_PLAN')
    bindings = view['source_bindings']
    require(COMPATIBILITY['key'] in bindings, 'COMPATIBILITY_KEY')
    checked = 0
    for key, expected in bindings.items():
        actual = ref(expected['path'])
        if key == COMPATIBILITY['key']:
            require(expected == {'path': COMPATIBILITY['path'],
                                 'sha256': COMPATIBILITY['old_sha256']}
                    and actual == {'path': COMPATIBILITY['path'],
                                   'sha256': COMPATIBILITY['current_sha256']}
                    and actual == plan['source_owners'][
                        'gx1/contracts/entry_native_direction_research_v1.py'],
                    'COMPATIBILITY_EXACT_PAIR')
        else:
            require(actual == expected, 'VIEW_SOURCE_CHANGED:' + key)
            checked += 1
    path = 'gx1/contracts/entry_native_direction_research_v1.py'
    commit = COMPATIBILITY['commit']
    pre = subprocess.check_output(['git', 'show', commit + '^:' + path], cwd=REPO)
    post = subprocess.check_output(['git', 'show', commit + ':' + path], cwd=REPO)
    patch = subprocess.check_output(['git', 'show', '--format=', '--no-ext-diff',
                                     commit, '--', path], cwd=REPO)
    require(hashlib.sha256(pre).hexdigest() == COMPATIBILITY['old_sha256']
            and hashlib.sha256(post).hexdigest() == COMPATIBILITY['current_sha256']
            and hashlib.sha256(patch).hexdigest() == COMPATIBILITY['patch_sha256'],
            'COMPATIBILITY_COMMIT_PATCH')
    return {'source_binding_count': len(bindings),
            'unchanged_physical_source_bindings_verified': checked,
            'exact_old_new_commit_patch': dict(COMPATIBILITY)}


def audit(plan, claim, binding, snapshots):
    import numpy as np
    import torch
    from gx1.contracts import entry_model_native_input_normalization_v1 as normalizer
    from gx1.contracts import entry_native_direction_feature_view_v1 as views

    raw_result = read(plan['inputs']['raw_result'])
    cache = read(plan['inputs']['cache_manifest'])
    fitted = read(plan['inputs']['prefix_normalization'])
    view = read(plan['inputs']['candidate_view'])
    compatibility = verify_view_source_compatibility(view, plan)
    require(raw_result['plan'] == plan['raw_plan']
            and raw_result['indicator_fields'] == list(FIELDS)
            and raw_result['timeframes'] == list(TFS)
            and raw_result['parent_count'] == 36172
            and raw_result['model_forwards'] == raw_result['optimizer_updates'] == 0
            and raw_result['test_data_used'] is False, 'RAW_AUDIT_IDENTITY')
    require(tuple(sorted(int(key) for key in raw_result['sample_tensor_cache_parity'])) == POSITIONS,
            'RAW_SAMPLE_POSITIONS')
    require(cache['feature_count'] == 253 and len(cache['feature_names']) == 253
            and set(cache['tfs']) == set(TFS), 'CACHE_LAYOUT')
    require(view['full_feature_profile'] is True
            and view['entry_execution_profile']['name'] == 'full_xau_gc_mtf_entry_research_v1'
            and view['physical_widths']['mtf'] == 253
            and view['prefix_normalization']['binding'] == plan['inputs']['prefix_normalization'],
            'FULL_VIEW_BINDING')
    constructor_binding = view['constructor_metadata']['binding']
    require(constructor_binding == plan['inputs']['constructor_metadata'], 'CONSTRUCTOR_REF')
    constructor = read(constructor_binding)
    surfaces = fitted['base_normalization']['surfaces']
    require(fitted['base_normalization']['fit_scope'] == 'train_only'
            and set('mtf_' + tf.lower() for tf in TFS) <= set(surfaces), 'TRAIN_NORMALIZATION')
    names = cache['feature_names']
    require(all(name in names for name in FIELDS + WITNESSES), 'FIELD_NAMES')
    field_index = {name: names.index(name) for name in FIELDS + WITNESSES}
    cache_root = Path(plan['inputs']['cache_manifest']['path']).parent
    raw_by_tf = {}
    cache_file_refs = {}
    for tf in TFS:
        info = cache['tfs'][tf]
        require(info['feature_count'] == 253, 'CACHE_WIDTH:' + tf)
        path = cache_root / info['feats_npy']
        require(path.name == tf + '_feats.npy'
                and regular(path)['st_size'] == info['feats_npy_size_bytes'], 'CACHE_FILE:' + tf)
        # A streamed hash binds the actual physical matrix to its prior manifest.
        checked = ref(path)
        require(checked['sha256'] == info['feats_npy_sha256'], 'CACHE_SHA:' + tf)
        cache_file_refs[tf] = checked
        snapshots[str(path)] = regular(path)
        matrix = np.load(path, mmap_mode='r', allow_pickle=False)
        require(matrix.dtype == np.float32
                and matrix.shape == (info['n_bars'], 253), 'CACHE_MATRIX:' + tf)
        sampled = raw_result['sample_tensor_cache_parity']
        ids = np.array([sampled[str(pos)]['timeframes'][tf]['cache_row'] for pos in POSITIONS],
                       dtype=np.int64)
        require(np.all((ids >= 0) & (ids < len(matrix))), 'CACHE_ROW_ID:' + tf)
        values = np.ascontiguousarray(matrix[ids], dtype=np.float32)
        require(np.isfinite(values).all(), 'CACHE_NONFINITE:' + tf)
        for row, pos in enumerate(POSITIONS):
            documented = sampled[str(pos)]['timeframes'][tf]
            require(documented['cache_row'] == int(ids[row]), 'ROW_BINDING:' + tf)
            for name in FIELDS:
                got = np.asarray(values[row, field_index[name]], dtype=np.float32)
                prior = np.asarray(documented['raw_indicator_values'][name], dtype=np.float32)
                require(got.view(np.uint32).item() == prior.view(np.uint32).item(),
                        'RAW_VALUE_CHANGED:' + tf + ':' + name)
        raw_by_tf[tf] = values
        del matrix

    with forbid_model_work(torch) as guard_counts:
        torch.set_num_threads(1)
        model = views.build_native_direction_research_model(
            view, prefix_normalization=fitted, constructor_metadata=constructor,
            device=torch.device('cpu'), unobserved_units_profile=view['unobserved_units_profile'])
        backbone = model.backbone
        require(backbone.entry_execution_profile == view['entry_execution_profile'], 'MODEL_PROFILE')
        model.eval()
        reports = {}
        with torch.no_grad():
            for tf in TFS:
                surface = 'mtf_' + tf.lower()
                raw = raw_by_tf[tf]
                contract = surfaces[surface]
                require(contract['field_names'] == names and contract['field_count'] == 253,
                        'NORMALIZATION_FIELDS:' + tf)
                reference = normalizer.apply_surface_normalization(raw, contract)
                actual = backbone._normalize_input_surface(torch.from_numpy(raw), surface=surface).cpu().numpy()
                require(reference.shape == actual.shape == (len(POSITIONS), 253)
                        and np.isfinite(reference).all() and np.isfinite(actual).all(),
                        'NORMALIZED_FINITE:' + tf)
                close = np.isclose(actual, reference, rtol=RTOL, atol=ATOL)
                require(bool(close.all()), 'NORMALIZED_PARITY:' + tf + ':'
                        + names[int(np.argwhere(~close)[0, 1])])
                identity = (np.asarray(contract['binary_mask'], dtype=bool)
                            | np.asarray(contract['categorical_mask'], dtype=bool))
                require(np.array_equal(actual[:, identity], raw[:, identity])
                        and np.array_equal(reference[:, identity], raw[:, identity]),
                        'IDENTITY_MASK:' + tf)
                for name in ('mtf_level_above2_present', 'mtf_level_below_present',
                             'mtf_level_below2_present'):
                    col = field_index[name]
                    require(np.isin(raw[:, col], (0.0, 1.0)).all()
                            and np.array_equal(actual[:, col], raw[:, col]),
                            'LEVEL_PRESENCE:' + tf + ':' + name)
                for distance, present in (
                    ('mtf_level_above_dist_atr', raw[:, field_index['mtf_level_above_touch_count']] >= 1),
                    ('mtf_level_above2_dist_atr', raw[:, field_index['mtf_level_above2_present']] == 1),
                    ('mtf_level_below_dist_atr', raw[:, field_index['mtf_level_below_present']] == 1),
                    ('mtf_level_below2_dist_atr', raw[:, field_index['mtf_level_below2_present']] == 1),
                    ('mtf_level_above_pending_retest_dist_atr',
                     raw[:, field_index['mtf_level_above_pending_retest_age_bars']] >= 1),
                    ('mtf_level_below_pending_retest_dist_atr',
                     raw[:, field_index['mtf_level_below_pending_retest_age_bars']] >= 1),
                ):
                    require(np.all(raw[~present, field_index[distance]] == 0.0),
                            'ABSENT_LEVEL_STORAGE:' + tf + ':' + distance)
                rows = []
                for row, pos in enumerate(POSITIONS):
                    rows.append({
                        'parent_position': pos,
                        'cache_row': raw_result['sample_tensor_cache_parity'][str(pos)]['timeframes'][tf]['cache_row'],
                        'indicator_values': {name: {'raw': float(raw[row, field_index[name]]),
                                                    'reference': float(reference[row, field_index[name]]),
                                                    'runtime': float(actual[row, field_index[name]])}
                                             for name in FIELDS},
                        'level_witnesses': {name: {'raw': float(raw[row, field_index[name]]),
                                                   'reference': float(reference[row, field_index[name]]),
                                                   'runtime': float(actual[row, field_index[name]])}
                                            for name in WITNESSES},
                    })
                reports[tf] = {'rows': rows, 'checked_values': int(actual.size),
                               'maximum_absolute_difference': float(np.max(np.abs(actual - reference)))}
        d1_rows = reports['D1']['rows']
        require(d1_rows[0]['level_witnesses']['mtf_level_above_touch_count']['raw'] >= 1
                and d1_rows[1]['level_witnesses']['mtf_level_above_touch_count']['raw'] >= 1
                and all(row['level_witnesses']['mtf_level_above_touch_count']['raw'] == 0
                        and row['level_witnesses']['mtf_level_above_dist_atr']['raw'] == 0
                        and row['level_witnesses']['mtf_level_above2_present']['raw'] == 0
                        for row in d1_rows[2:]), 'D1_ABSENT_AND_PRESENT_WITNESSES')
    require(all(value == 0 for value in guard_counts.values()), 'NO_MODEL_WORK_GUARD')
    freeze(binding, plan, snapshots)
    for item in plan['inputs'].values():
        require(ref(item['path']) == item, 'INPUT_CHANGED:' + item['path'])
    return {'schema_version': STAGE + '_result_v1', 'claim': claim,
            'plan': binding['plan'], 'policy': binding['policy'],
            'source_head': binding['source_head'], 'sample_positions': list(POSITIONS),
            'timeframes': list(TFS), 'indicator_fields': list(FIELDS),
            'cache_file_refs': cache_file_refs,
            'view_source_compatibility': compatibility,
            'normalization_tolerance': {'rtol': RTOL, 'atol': ATOL},
            'reports': reports, 'checked_normalized_values': 8 * 253 * len(TFS),
            'model_constructions': 1, 'model_forwards': guard_counts['model_forwards'],
            'backward_calls': guard_counts['backward_calls'],
            'optimizer_steps': guard_counts['optimizer_steps'],
            'test_data_used': False, 'feature_influence_measured': False,
            'gather_parity_reused_only_for_four_indicator_fields': True,
            'independent_edge': 'UNKNOWN'}


def main():
    require(len(sys.argv) == 1 and Path.cwd() == REPO, 'CLI')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    require(ROOT.is_dir() and not any(p.is_symlink() for p in (ROOT, *ROOT.parents)), 'RUN_ROOT')
    binding = read(ref(ROOT / 'EXECUTION_BINDING.json'))
    require(set(binding) == {'plan', 'policy', 'worker', 'source_head'}
            and binding['plan']['path'] == str(PLAN_PATH)
            and binding['policy']['path'] == str(REPO / 'NEXT_RUN_POLICY.json')
            and binding['worker'] == ref(__file__), 'EXECUTION_BINDING')
    plan = read(binding['plan'])
    require(plan['schema_version'] == STAGE + '_plan_v1'
            and plan['stage'] == STAGE and plan['run_root'] == str(ROOT)
            and plan['preparation_status'] == 'PREPARED_NOT_ADMITTED'
            and plan['worker'] == binding['worker'] and plan['budget'] == BUDGET
            and plan['sample_parent_positions'] == list(POSITIONS)
            and plan['timeframes'] == list(TFS)
            and plan['indicator_fields'] == list(FIELDS)
            and plan['rtol'] == RTOL and plan['atol'] == ATOL
            and plan['source_compatibility'] == COMPATIBILITY
            and set(plan['inputs']) == {'raw_result', 'cache_manifest',
                                         'prefix_normalization', 'candidate_view',
                                         'constructor_metadata'}
            and all('TEST' not in Path(item['path']).parts for item in plan['inputs'].values()),
            'PLAN')
    policy = read(binding['policy'])
    gate = policy[STAGE]
    require(policy['training_enabled'] is False
            and gate['plan'] == binding['plan']
            and gate['execution_authorized'] is True
            and gate['data_reads_authorized'] is True
            and gate['model_constructions_authorized'] == 1
            and gate['model_forwards_authorized'] == 0
            and gate['optimizer_updates_authorized'] == 0
            and gate['claimed_invocations'] == gate['completed_invocations'] == 0
            and gate['max_invocations'] == 1
            and gate['test_authorized'] is False,
            'EXACT_AUDIT_ADMISSION')
    require(all(not (ROOT / name).exists() and not (ROOT / name).is_symlink()
                for name in ('CLAIM.json', 'RESULT.json', 'TERMINAL.json')), 'CONSUMED')
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    limits = require_capped_cpu_audit_execution()
    limits['lock'] = require_capped_lock_ancestry()
    require(all(limits[key] == BUDGET[key] for key in
                ('memory_max', 'swap', 'pids', 'numerical_threads')), 'CAPS')
    snapshots = {item['path']: regular(item['path']) for item in
                 (*plan['inputs'].values(), *plan['source_owners'].values(),
                  binding['plan'], binding['worker'], binding['policy'])}
    freeze(binding, plan, snapshots)
    claim = publish('CLAIM.json', {
        'schema_version': STAGE + '_claim_v1', 'plan': binding['plan'],
        'policy': binding['policy'], 'worker': binding['worker'],
        'source_head': binding['source_head'], 'limits': limits,
        'model_constructions': 1, 'model_forwards': 0,
        'sample_rows': len(POSITIONS), 'timeframes': list(TFS)})
    started = time.monotonic()

    def deadline(*_args):
        raise TimeoutError('DYNAMIC_M5_POSTNORM_1800_SECOND_DEADLINE')

    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.alarm(BUDGET['max_wall_seconds'])
    try:
        result = audit(plan, claim, binding, snapshots)
        result['elapsed_seconds'] = time.monotonic() - started
        result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        result_ref = publish('RESULT.json', result)
        publish('TERMINAL.json', {'schema_version': STAGE + '_terminal_v1',
                                 'status': 'COMPLETE', 'exit_code': 0, 'claim': claim,
                                 'result': result_ref, 'source_head': binding['source_head'],
                                 'test_data_used': False})
    except BaseException as exc:
        publish('TERMINAL.json', {'schema_version': STAGE + '_terminal_v1',
                                 'status': 'FAILED_CONSUMED', 'exit_code': 1,
                                 'claim': claim, 'source_head': binding['source_head'],
                                 'error': type(exc).__name__ + ':' + str(exc)[:500],
                                 'test_data_used': False})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
