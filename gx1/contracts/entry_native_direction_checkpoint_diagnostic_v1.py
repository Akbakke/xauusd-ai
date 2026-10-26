"""Exact four-state276 TRAIN diagnosis; metadata/operation authority, never fit.

No numerical imports. Pure recipe/measurement helpers do no I/O; campaign
validators read hash-bound metadata and source only. The public worker proves
the canonical guard, actual campaign/ACTIVE and fresh claim before payload.
Runtime execution still requires a separate exact admission.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath

from gx1.contracts import local_random_access_campaign_v2 as campaign

SCHEMA = 'gx1_native_direction_checkpoint_diagnostic_plan_v1'
OPERATION = 'C_saved276_checkpoint_diagnostic_v1'
STAGE = 'dynamic_m5_direction_checkpoint_diagnostic_c'
RUNTIME_ROUTE_QUALIFIED = False
PHASE = 'native_direction_checkpoint_diagnostic'
KIND = 'native_direction_checkpoint_diagnostic_window'
WINDOW_SCHEMA = 'gx1_native_direction_checkpoint_diagnostic_window_policy_v1'
POINTER_SCHEMA = 'gx1_native_direction_checkpoint_diagnostic_pointer_v1'
MODULE = 'gx1.scripts.run_unified_exit_native_candidate_window_v1'
TRANSPORT_ORIGINAL_SHA256 = {
    'scripts/gx1_capped_run.sh': 'ea3155f4112b1eaaca7d684863dff6f61a210a5d74561e7722aab346072e3234',
    'gx1/scripts/run_unified_exit_native_candidate_window_v1.py': 'e19153e356e741ef0faabe9e7a35245a8148ffb6dde6789e0b1f9305e3df5bf1',
    'gx1/contracts/local_random_access_campaign_v2.py': '85ee3011da82943dd3b0eb1b734f0111a949630bd17d187e2ef73772cebb3828',
    'scripts/collect_gx1_handover_readonly.py': '39fe0a7c090dd970aae430f9c3fd02beb16061d6e6f7a02c418672f2bd812bab',
}
TERMINAL_JSON_RESERVE_BYTES = 65536
REPO = '/home/andre2/src/GX1_CURRENT'
PROGRAM = '/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2'
BASE = '/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_PREFIX_FIT_FOLD1_20261025_276'
PROPOSAL = {'path': PROGRAM + '/C_METADATA_PROPOSAL.json',
    'sha256': 'a150effb50567e6e7cec6fbd9aa92b53f4a0b30bce020c0321ef299e481dc070'}
INITIAL_REPORT = {'path': BASE + '/RUNTIME/prefix-fit/window-0001/INITIAL_TRAIN.json',
    'sha256': '0504cb783b1054a4a19e1da84a0013abee11100038cdbcd78b6514913064c09c'}
INITIAL_PREDICTIONS = {'path': BASE + '/RUNTIME/prefix-fit/window-0001/INITIAL_TRAIN.npz',
    'sha256': 'f870c6eb0a6000b2be0197bfe650db3f35f41d125d91a90162df2d01aedf112a'}
FOLD = {'fold_id': 1, 'prefix_rows': 47602, 'oof_start_row': 47602,
    'oof_end_row': 64890, 'train_cutoff_ns': 1735689600000000000,
    'oof_cutoff_ns': 1743465600000000000}
PANELS = (('training_order_early', 0, 256),
    ('training_order_middle', 23801, 24057), ('training_order_late', 47346, 47602))
STATE_SPEC = (
    ('own0U', 0, 0, 'INITIAL_0U.pt',
     '40a9d5ad744a06512619c3072170657f6ed430a077c3187e05e817fd5b46e9f3'),
    ('u256', 4096, 256, 'w0001_u000256_p004096.pt',
     '5ce78a77aab5a7065e3ad38f71ce9b57cffcfc255c0aec64500412fd26c73f81'),
    ('u1536', 24576, 1536, 'w0001_u001536_p024576.pt',
     '7947638476783ab35757499440659f696cee380e1d16446786aaa424673b44fb'),
    ('terminal_u2976', 47602, 2976, 'w0002_TERMINAL_u002976_p047602.pt',
     'be9e9a54851085c6719f03a29d958f7c83b470236a0fc5ae5801c1bdd543af29'))
NEW_FILES = ('gx1/contracts/entry_native_direction_checkpoint_diagnostic_v1.py',
    'gx1/scripts/audit_entry_native_direction_checkpoints_v1.py',
    'tests/test_entry_native_direction_checkpoint_diagnostic_v1.py')
# Count both the caller boundary and real delegated Python function bodies.
CALL_LIMITS = {
    'checkpoint_load': 4, 'panel_forward': 192, 'gradient_forward': 2,
    'reverse': 6, 'model_build': 1, 'optimizer_build': 1,
    'physical_prepare': 1, 'prefix_prepare': 1, 'batch': 194,
    'torch_load': 4, 'numpy_load': 33, 'native_forward': 194,
    'research_forward': 194, 'trainer_forward': 194,
    'backbone_constructor': 1, 'research_constructor': 1,
    'adamw_constructor': 1, 'autograd_grad': 6, 'engine_reverse': 6,
    'gather': 194, 'axes_loader': 1, 'reuse_mapper': 1,
    'clock_loader': 1, 'cache_loader': 1, 'raw_gc_loader': 1,
    'verified_npy_loader': 30,
}
# Existing canonical native guard/window and prefix JSON/checkpoint ceilings.
RESOURCE_LIMITS = {'memory_max': 21474836480, 'swap': 536870912, 'pids': 128,
    'numerical_threads': 8, 'max_wall_seconds': 12000,
    'max_checkpoint_bytes': 1073741824, 'max_json_output_bytes': 67108864,
    'max_rchar_bytes': 137438953472}
LIMITS = ('TRAIN panels only; no OOF/TEST forwards or optimizer steps.',
    'Fixed checkpoints do not isolate the final partial-batch update.',
    'Activation geometry and gradient conflict do not establish predictive causality.',
    'Shared family encoder deltas are not per-timeframe parameter deltas.',
    'No capacity, convergence, generalization, economics or checkpoint-selection claim.')


def require(ok, reason):
    if not ok:
        raise RuntimeError('NATIVE_DIRECTION_CHECKPOINT_DIAGNOSTIC_' + reason)


def plain_ref(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'}, 'REF_KEYS')
    path, digest = value['path'], value['sha256']
    require(type(path) is str and str(PurePosixPath(path)) == path
        and PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts,
        'REF_PATH')
    require(type(digest) is str and len(digest) == 64
        and all(c in '0123456789abcdef' for c in digest), 'REF_SHA256')
    return dict(value)


def require_proposal(value):
    require(type(value) is dict and value.get('schema_version') == 'gx1_direction_C_metadata_proposal_v1',
        'PROPOSAL_SCHEMA')
    states = value['planned_saved_states']
    require(len(states) == 4, 'FOUR_FIXED_STATES')
    for item, (role, cursor, updates, filename, digest) in zip(states, STATE_SPEC):
        require((item['role'], item['consumed_parent_cursor'], item['updates']) == (role, cursor, updates)
            and item['planned_PT_ref'] == {'path': BASE + '/RUNTIME/prefix-fit/states/' + filename,
                'sha256': digest}, 'FIXED_STATE_IDENTITY')
        plain_ref(item['declaring_JSON'])
    require(value['fixed_panels']['prefix_rows'] == FOLD['prefix_rows']
        and value['fixed_panels']['seed'] == 20260911
        and value['fixed_panels']['slices'] == {name: [left, right] for name, left, right in PANELS},
        'FIXED_PRESENTATION_PANELS')
    require(value['minimal_operations'] == {
        'OOF_or_TEST_forward_calls': 0, 'autograd_reverse_calls': 6,
        'checkpoint_body_loads_total': 4, 'checkpoint_selection': False,
        'gradient_forward_calls': 2, 'model_source_or_architecture_changes': 0,
        'new_probe_fits': 0, 'optimizer_steps': 0, 'panel_batch_size': 16,
        'panel_forward_calls': 192, 'reuse_each_decoded_state_across_phases': True,
        'total_forward_calls': 194}, 'FIXED_OPERATIONS')
    return deepcopy(value)


def require_plan(value):
    """Static C recipe: current HEAD/policy are bound externally after commit."""
    fields = {'schema_version', 'operation', 'admission_stage', 'proposal',
        'source_owners', 'transport_source_transitions', 'run_root', 'runtime_root',
        'execution_binding_path', 'budget', 'operation_limits', 'initial_report',
        'initial_predictions', 'max_invocations', 'execution_authorized',
        'data_reads_authorized', 'checkpoint_reads_authorized', 'genuine_forwards_authorized',
        'gradient_reverse_authorized', 'optimizer_steps_authorized', 'test_data_used'}
    require(type(value) is dict and set(value) == fields and value['schema_version'] == SCHEMA
        and value['operation'] == OPERATION and value['admission_stage'] == STAGE, 'PLAN_SCHEMA')
    require(value['proposal'] == PROPOSAL and value['initial_report'] == INITIAL_REPORT
        and value['initial_predictions'] == INITIAL_PREDICTIONS, 'FIXED_INPUT_IDENTITIES')
    owners = value['source_owners']
    require(type(owners) is dict and all(plain_ref(ref)['path'] == REPO + '/' + key
        for key, ref in owners.items()) and all(name in owners for name in NEW_FILES), 'SOURCE_OWNERS')
    transitions = value['transport_source_transitions']
    require(type(transitions) is dict and set(transitions) == set(TRANSPORT_ORIGINAL_SHA256),
        'FOUR_TRANSPORT_TRANSITIONS')
    for name, digest in TRANSPORT_ORIGINAL_SHA256.items():
        item = transitions[name]
        require(type(item) is dict and set(item) == {'old276', 'current'}
            and item['old276'] == {'path': REPO + '/' + name, 'sha256': digest}
            and plain_ref(item['current']) == owners.get(name)
            and item['current']['sha256'] != digest, 'TRANSPORT_TRANSITION:' + name)
    root, runtime = (PurePosixPath(value[k]) for k in ('run_root', 'runtime_root'))
    require(all(str(p) == value[k] and p.parent == PurePosixPath(PROGRAM)
        and p.name.startswith('C_CHECKPOINT_DIAGNOSTIC_') for k, p in
        (('run_root', root), ('runtime_root', runtime))) and root != runtime, 'SEPARATE_NEW_ROOTS')
    require(value['execution_binding_path'] == str(runtime / 'EXECUTION_BINDING.json'), 'EXTERNAL_BINDING_PATH')
    require(type(value['max_invocations']) is int and value['max_invocations'] == 1, 'SINGLE_INVOCATION')
    require(type(value['budget']) is type(value['operation_limits']) is dict
        and value['budget'] == RESOURCE_LIMITS and value['operation_limits'] == CALL_LIMITS
        and all(type(number) is int for number in value['budget'].values())
        and all(type(number) is int for number in value['operation_limits'].values()), 'FIXED_CEILINGS')
    for key in ('execution_authorized', 'data_reads_authorized', 'checkpoint_reads_authorized',
            'genuine_forwards_authorized', 'gradient_reverse_authorized'):
        require(value[key] is True, 'ADMISSION_' + key)
    require(value['optimizer_steps_authorized'] is False and value['test_data_used'] is False,
        'NO_UPDATE_OR_TEST_AUTHORITY')
    return deepcopy(value)


def require_admission(admission, recipe_ref):
    require(type(admission) is dict and admission.get('operation') == OPERATION
        and admission.get('execution_authorized') is True
        and admission.get('recipe') == plain_ref(recipe_ref)
        and type(admission.get('max_invocations')) is int and admission['max_invocations'] == 1
        and admission.get('genuine_payload_reads_authorized') is True
        and admission.get('genuine_forwards_authorized') is True
        and admission.get('gradient_reverse_authorized') is True
        and admission.get('optimizer_steps_authorized') is False
        and admission.get('test_data_used') is False, 'ROOT_EXECUTION_ADMISSION')
    return deepcopy(admission)


def require_source_closure(recipe, original, proposal):
    """Only the four admitted transport bytes may differ from historical276."""
    previous, reused = original['source_owners'], proposal['existing_sources_reused_without_change']
    overlap = set(previous) & set(reused)
    require(len(previous) == 50 and len(reused) == 7 and len(overlap) == 6
        and all(previous[name] == reused[name] for name in overlap), 'ORIGINAL_SOURCE_INVENTORY')
    required = {**previous, **reused}
    require(len(recipe['source_owners']) == 54
        and set(recipe['source_owners']) == set(required) | set(NEW_FILES), 'EXACT_SOURCE_CLOSURE')
    for name, ref in required.items():
        expected = recipe['transport_source_transitions'][name]['current'] if name in TRANSPORT_ORIGINAL_SHA256 else ref
        require(recipe['source_owners'][name] == expected, 'UNCHANGED_SOURCE_OWNER:' + name)
        if name in TRANSPORT_ORIGINAL_SHA256:
            require(recipe['transport_source_transitions'][name]['old276'] == ref, 'ORIGINAL_TRANSPORT_OWNER:' + name)
    return deepcopy(recipe['source_owners'])



def panel_seen_status(left, right, cursor):
    require(type(left) is type(right) is type(cursor) is int and 0 <= left < right <= 47602
        and 0 <= cursor <= 47602, 'PANEL_CURSOR_DOMAIN')
    seen = max(0, min(right, cursor) - left)
    return {'presentation_slice': [left, right], 'seen_rows': seen,
        'unseen_rows': right - left - seen, 'consumed_parent_cursor': cursor}


class CallBudget:
    """The fixed C counters; caller and delegated-body counters stay separate."""
    def __init__(self):
        self.calls = {key: 0 for key in CALL_LIMITS}
        self.returns = {key: 0 for key in CALL_LIMITS}

    def start(self, key):
        require(key in CALL_LIMITS, 'UNDECLARED_CALL')
        require(self.calls[key] < CALL_LIMITS[key], 'CALL_CEILING_' + key)
        self.calls[key] += 1

    def returned(self, key):
        require(key in CALL_LIMITS and self.returns[key] < self.calls[key], 'RETURN_' + key)
        self.returns[key] += 1

    def complete(self):
        require(self.calls == self.returns == CALL_LIMITS, 'EXACT_COMPLETED_CALLS')
        return {'calls': dict(self.calls), 'returns': dict(self.returns),
            'optimizer_steps': 0, 'OOF_TEST_forwards': 0}


def _identity(value, fields, schema, seal):
    require(type(value) is dict and set(value) == fields | {seal}, 'FIELDS:' + schema)
    result = dict(value); claimed = campaign._sha(result.pop(seal), seal)
    require(result.get('schema_version') == schema and claimed == campaign.canonical_sha256(result), 'IDENTITY:' + schema)
    result[seal] = claimed
    return result


def _seal(value, seal):
    result = dict(value); result[seal] = campaign.canonical_sha256(result)
    return result


def _path(value, label):
    path = campaign._absolute(value, label)
    require(path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)), 'PATH:' + label)
    return path


def require_checkpoint_diagnostic_recipe(binding, *, source_repo, source_commit, verify_files=True):
    require(verify_files is True and Path(source_repo) == Path(REPO)
        and campaign._COMMIT.fullmatch(source_commit) is not None, 'RECIPE_SOURCE')
    ref = campaign.require_binding(binding, label='C static recipe')
    recipe = require_plan(campaign.read_bound_json(Path(ref['path']), ref['sha256']))
    proposal = require_proposal(campaign.read_bound_json(Path(PROPOSAL['path']), PROPOSAL['sha256']))
    original_ref = proposal['existing_input_refs']['recipe']
    original = campaign.read_bound_json(Path(original_ref['path']), original_ref['sha256'])
    require_source_closure(recipe, original, proposal)
    for name, owner in recipe['source_owners'].items():
        require(campaign.require_binding(owner, label='C source owner')['path'] == REPO + '/' + name, 'CURRENT_SOURCE:' + name)
    for key in ('run_root', 'runtime_root', 'execution_binding_path'):
        _path(recipe[key], key)
    return recipe


def require_checkpoint_diagnostic_window_policy(value, *, verify_files=True):
    fields = {'schema_version', 'recipe', 'invocation_number', 'max_invocation_seconds',
        'progress_path', 'campaign_cursor_path', 'output_directory', 'test_data_used'}
    window = _identity(value, fields, WINDOW_SCHEMA, 'policy_sha256')
    require(type(window['invocation_number']) is int and window['invocation_number'] == 1
        and type(window['max_invocation_seconds']) is int and window['max_invocation_seconds'] == 12000
        and window['test_data_used'] is False, 'WINDOW_SCOPE')
    campaign.require_binding(window['recipe'], label='C window recipe', verify_file=verify_files)
    output = _path(window['output_directory'], 'C output')
    require(_path(window['progress_path'], 'C progress') == output / 'PROGRESS.json'
        and _path(window['campaign_cursor_path'], 'C pointer') == output / 'DIAGNOSTIC_POINTER.json', 'WINDOW_LAYOUT')
    return window


def require_checkpoint_diagnostic_invocation_execution(result, execution, *, source_repo, source_commit):
    recipe_ref = campaign.require_binding(execution['prelaunch_manifest'], label='C recipe')
    recipe = require_checkpoint_diagnostic_recipe(recipe_ref, source_repo=source_repo, source_commit=source_commit)
    window_ref = campaign.require_binding(execution['train_session_manifest'], label='C window')
    window = require_checkpoint_diagnostic_window_policy(campaign.read_bound_json(Path(window_ref['path']), window_ref['sha256']))
    separator = result['launcher_argv'].index('--')
    require(result['launcher_argv'][separator + 1:] == [str(Path(source_repo) / '.venv/bin/python'), '-m', MODULE,
        '--window-policy', window_ref['path'], '--window-policy-file-sha256', window_ref['sha256'],
        '--progress-path', window['progress_path']], 'INVOCATION_ARGV')
    require(result['kind'] == KIND and type(result['batch_size']) is int and result['batch_size'] == 16
        and result['epoch_index'] is None and type(result['invocation_number']) is int and result['invocation_number'] == 1
        and type(result['window_index']) is int and result['window_index'] == 0
        and type(result['window_count']) is int and result['window_count'] == 1
        and type(result['optimizer_step_budget']) is int and result['optimizer_step_budget'] == 0
        and result['expected_success_outcome'] == 'COMPLETE' and result['maximum_wall_seconds'] == 13800
        and result['checkpoint'] == {'pointer_path': window['campaign_cursor_path'], 'before_mode': 'GENESIS',
            'predecessor_invocation_number': None, 'write_mode': 'ATOMIC_UPDATE'}
        and result['progress_path'] == window['progress_path'] and window['recipe'] == recipe_ref
        and window['output_directory'] == recipe['run_root']
        and execution['prelaunch_manifest_sha256'] == campaign.canonical_sha256(recipe)
        and execution['train_session_manifest_sha256'] == window['policy_sha256'], 'INVOCATION_BINDING')
    result['checkpoint_diagnostic_recipe'] = recipe_ref
    result['checkpoint_diagnostic_window'] = window


def require_checkpoint_diagnostic_campaign_plan(value, *, verify_files=True):
    fields = {'schema_version', 'decision', 'campaign_id', 'phase', 'created_utc', 'source_repo', 'source_commit',
        'runtime_root', 'gpu_uuid', 'prepared_windows_boot', 'recipe', 'invocations', 'signed_guard_sources',
        'controller_sources', 'policy', 'authority', 'test_data_used'}
    plan = _identity(value, fields, campaign.PLAN_SCHEMA, 'plan_sha256')
    require(plan['phase'] == PHASE and plan['decision'] == 'PASS_PREPARED'
        and type(plan['campaign_id']) is str and bool(plan['campaign_id']) and plan['source_repo'] == REPO
        and type(plan['source_commit']) is str and campaign._COMMIT.fullmatch(plan['source_commit']) is not None
        and type(plan['gpu_uuid']) is str and plan['gpu_uuid'].startswith('GPU-')
        and plan['test_data_used'] is False, 'CAMPAIGN_IDENTITY')
    campaign._utc(plan['created_utc'], 'C campaign creation')
    runtime = _path(plan['runtime_root'], 'C runtime')
    plan['prepared_windows_boot'] = campaign.require_boot_identity(plan['prepared_windows_boot'])
    require(plan['policy'] == campaign.campaign_hardware_policy(PHASE)
        and plan['authority'] == {'test': False, 'promotion': False, 'paper': False, 'live': False, 'cloud_spend': False}, 'CAMPAIGN_AUTHORITY')
    for key, names in (('signed_guard_sources', {'runner', 'guard', 'query', 'certificate'}),
            ('controller_sources', {'controller', 'observer', 'campaign_cli'})):
        require(type(plan[key]) is dict and set(plan[key]) == names, 'CAMPAIGN_SOURCES:' + key)
        plan[key] = {name: campaign.require_binding(ref, label=key, verify_file=verify_files) for name, ref in plan[key].items()}
    plan['recipe'] = campaign.require_binding(plan['recipe'], label='C campaign recipe', verify_file=verify_files)
    require(type(plan['invocations']) is list and len(plan['invocations']) == 1, 'ONE_CAMPAIGN_INVOCATION')
    plan['invocations'] = [campaign.require_binding(plan['invocations'][0], label='C invocation', verify_file=verify_files)]
    plan['selection_artifact_sha256'] = None
    if verify_files:
        recipe = require_checkpoint_diagnostic_recipe(plan['recipe'], source_repo=Path(REPO), source_commit=plan['source_commit'])
        invocation_ref = plan['invocations'][0]
        invocation = campaign.require_invocation(campaign.read_bound_json(Path(invocation_ref['path']), invocation_ref['sha256']),
            source_repo=Path(REPO), source_commit=plan['source_commit'], verify_files=True)
        window = invocation.get('checkpoint_diagnostic_window', {})
        require(invocation['kind'] == KIND and invocation.get('checkpoint_diagnostic_recipe') == plan['recipe']
            and recipe['runtime_root'] == str(runtime) and window['output_directory'] == recipe['run_root']
            and invocation['guard_log_path'] == str(runtime / 'logs/invocation-0001.signed-guard.log'), 'CAMPAIGN_LAYOUT')
        plan['checked_invocations'] = [invocation]
    return plan


def require_native_direction_checkpoint_diagnostic_readiness(repo, *, window_path, window_file_sha256):
    import subprocess
    repo = Path(repo)
    window = require_checkpoint_diagnostic_window_policy(campaign.read_bound_json(Path(window_path), window_file_sha256))
    policy_ref = {'path': str(repo / 'NEXT_RUN_POLICY.json'), 'sha256': campaign.file_sha256(repo / 'NEXT_RUN_POLICY.json')}
    policy = campaign.read_bound_json(Path(policy_ref['path']), policy_ref['sha256'])
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain=v1', '--untracked-files=all'], text=True)
    recipe = require_checkpoint_diagnostic_recipe(window['recipe'], source_repo=repo, source_commit=head)
    require(repo == Path(REPO) and branch == 'work/gx1-current' and not dirty
        and policy.get('schema_version') == 'gx1_next_native_run_policy_v1'
        and policy.get('canonical_source_repo') == REPO and policy.get('canonical_branch') == branch
        and policy.get('training_module') == MODULE and all(policy.get(key) is False for key in
            ('training_enabled', 'native_training_enabled', 'full_epoch_training_allowed', 'full_val_allowed')), 'READINESS_GLOBAL')
    require(window['output_directory'] == recipe['run_root'], 'READINESS_OUTPUT')
    gate = policy.get(STAGE)
    require(type(gate) is dict and gate.get('recipe') == window['recipe']
        and gate.get('run_root') == recipe['run_root'] and gate.get('runtime_root') == recipe['runtime_root']
        and type(gate.get('max_invocations')) is int and gate['max_invocations'] == 1
        and type(gate.get('claimed_invocations')) is int and gate['claimed_invocations'] == 0
        and type(gate.get('completed_invocations')) is int and gate['completed_invocations'] == 0
        and gate.get('relaunch_allowed') is False, 'READINESS_ONCE')
    for key in ('execution_authorized', 'data_reads_authorized', 'checkpoint_reads_authorized',
            'genuine_constructor_authorized', 'genuine_forwards_authorized', 'gradient_reverse_authorized'):
        require(gate.get(key) is True, 'READINESS_GATE:' + key)
    for key in ('optimizer_steps_authorized', 'training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'full_val_allowed', 'genuine_predictor_fit_authorized',
            'normalization_fit_authorized', 'synthetic_test_execution_authorized', 'native_launch_authorized',
            'test_data_used', 'exit_evaluation', 'broker_access_allowed'):
        require(gate.get(key) is False, 'READINESS_CLOSED_GATE:' + key)
    admission_ref = campaign.require_binding(gate.get('admission'), label='C runtime admission')
    require_admission(campaign.read_bound_json(Path(admission_ref['path']), admission_ref['sha256']), window['recipe'])
    campaign.require_binding(policy.get('gpu_clock_launcher'), label='existing GPU clock launcher')
    return {'decision': 'READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES', 'blocked_reasons': [],
        'canonical_source_repo': REPO, 'source_commit': head, 'policy_sha256': policy_ref['sha256'],
        'training_started': False, 'native_invocation_seconds': 12000, 'outer_guard_seconds': 13800,
        'native_window_scope': {'checkpoint_diagnostic_only': True, 'invocation_number': 1,
            'optimizer_step_ceiling': 0, 'forward_ceiling': 194, 'reverse_ceiling': 6,
            'checkpoint_load_ceiling': 4, 'full_epoch_training_allowed': False,
            'window_policy_file_sha256': window_file_sha256}}


def require_native_direction_checkpoint_diagnostic_context(policy_path, policy_file_sha256, progress_path):
    import os
    with campaign.file_verification_scope():
        window = require_checkpoint_diagnostic_window_policy(campaign.read_bound_json(Path(policy_path), policy_file_sha256))
        names = ('GX1_CAMPAIGN_PLAN_PATH', 'GX1_CAMPAIGN_PLAN_FILE_SHA256', 'GX1_CAMPAIGN_PLAN_SHA256', 'GX1_CAMPAIGN_INVOCATION_SHA256')
        require(all(os.environ.get(name) for name in names), 'CAMPAIGN_ENV')
        plan_ref = {'path': os.environ[names[0]], 'sha256': os.environ[names[1]]}
        plan = campaign.require_plan(campaign.read_bound_json(Path(plan_ref['path']), plan_ref['sha256']))
        require(plan['phase'] == PHASE and len(plan['checked_invocations']) == 1, 'CAMPAIGN_PHASE')
        invocation = plan['checked_invocations'][0]
        recipe = require_checkpoint_diagnostic_recipe(window['recipe'], source_repo=Path(REPO), source_commit=plan['source_commit'])
        marker_path = Path(plan['runtime_root']) / 'ACTIVE_INVOCATION.json'
        marker_ref = {'path': str(marker_path), 'sha256': campaign.file_sha256(marker_path)}
        marker = campaign.read_bound_json(marker_path, marker_ref['sha256'])
        require(plan['plan_sha256'] == os.environ[names[2]] and invocation['invocation_sha256'] == os.environ[names[3]]
            and invocation['checkpoint_diagnostic_window'] == window and plan['recipe'] == window['recipe']
            and str(progress_path) == window['progress_path'], 'WINDOW_CONTEXT')
        require(marker.get('schema_version') == campaign.ACTIVE_SCHEMA
            and type(marker.get('invocation_number')) is int
            and marker.get('marker_sha256') == campaign.canonical_sha256({k: v for k, v in marker.items() if k != 'marker_sha256'})
            and all(marker.get(key) == value for key, value in {'plan_sha256': plan['plan_sha256'],
                'invocation_sha256': invocation['invocation_sha256'], 'invocation_number': 1,
                'invocation_id': 'invocation-0001', 'kind': KIND, 'launcher_argv_sha256': invocation['launcher_argv_sha256'],
                'pointer_before_sha256': 'GENESIS'}.items()), 'ACTIVE_MARKER')
        require(campaign.fresh_boot(marker.get('boot'), plan['prepared_windows_boot']), 'ACTIVE_FRESH_BOOT')
        campaign._utc(marker.get('started_utc'), 'C active start')
        require(os.environ.get('GX1_CAMPAIGN_GUARD_LOG_PATH') == invocation['guard_log_path'], 'ACTIVE_GUARD_LOG')
        binding_ref = {'path': recipe['execution_binding_path'], 'sha256': campaign.file_sha256(recipe['execution_binding_path'])}
        binding = campaign.read_bound_json(Path(binding_ref['path']), binding_ref['sha256'])
        policy_ref = {'path': REPO + '/NEXT_RUN_POLICY.json', 'sha256': campaign.file_sha256(REPO + '/NEXT_RUN_POLICY.json')}
        require(set(binding) == {'plan', 'source_head', 'policy'} and binding['plan'] == plan_ref
            and binding['source_head'] == plan['source_commit'] and binding['policy'] == policy_ref, 'FROZEN_EXECUTION_BINDING')
        campaign.require_clean_source(plan)
        require_native_direction_checkpoint_diagnostic_readiness(Path(REPO), window_path=Path(policy_path), window_file_sha256=policy_file_sha256)
        policy = campaign.read_bound_json(Path(policy_ref['path']), policy_ref['sha256'])
        admission_ref = campaign.require_binding(policy[STAGE]['admission'], label='C execution admission')
        output = Path(recipe['run_root'])
        require(not output.exists() and not output.is_symlink(), 'GENESIS_NOT_FRESH')
        return {'window': window, 'plan': plan, 'invocation': invocation, 'marker': marker,
            'recipe': recipe, 'recipe_ref': window['recipe'], 'admission_ref': admission_ref, 'binding': binding,
            'metadata_refs': [plan_ref, {'path': str(Path(policy_path)), 'sha256': policy_file_sha256},
                marker_ref, binding_ref, policy_ref, admission_ref]}


def _operation_counts(calls, returns, outcome):
    require(type(calls) is type(returns) is dict and set(calls) == set(returns) == set(CALL_LIMITS), 'POINTER_COUNTER_KEYS')
    require(all(type(calls[k]) is type(returns[k]) is int and 0 <= returns[k] <= calls[k] <= limit
        for k, limit in CALL_LIMITS.items()), 'POINTER_COUNTER_BOUNDS')
    require(outcome in ('COMPLETE', 'FAILED') and (outcome != 'COMPLETE' or calls == returns == CALL_LIMITS), 'POINTER_COMPLETE')
    return {'completed_units': returns['panel_forward'] + returns['gradient_forward'],
        'forward_calls': calls['panel_forward'] + calls['gradient_forward'],
        'forward_returns': returns['panel_forward'] + returns['gradient_forward'],
        'reverse_calls': calls['reverse'], 'reverse_returns': returns['reverse'],
        'checkpoint_loads': calls['checkpoint_load'], 'checkpoint_load_returns': returns['checkpoint_load'], 'optimizer_steps': 0}


def build_checkpoint_diagnostic_pointer(*, recipe_ref, result_ref, outcome, calls, returns):
    counters = _operation_counts(calls, returns, outcome)
    return _seal({'schema_version': POINTER_SCHEMA, 'recipe': plain_ref(recipe_ref), 'result': plain_ref(result_ref),
        'outcome': outcome, 'operation_calls': dict(calls), 'operation_returns': dict(returns),
        **counters, 'test_data_used': False}, 'pointer_sha256')


def require_checkpoint_diagnostic_pointer(value, *, expected_recipe, verify_files=True):
    fields = {'schema_version', 'recipe', 'result', 'outcome', 'operation_calls', 'operation_returns',
        'completed_units', 'forward_calls', 'forward_returns', 'reverse_calls', 'reverse_returns',
        'checkpoint_loads', 'checkpoint_load_returns', 'optimizer_steps', 'test_data_used'}
    pointer = _identity(value, fields, POINTER_SCHEMA, 'pointer_sha256')
    counters = _operation_counts(pointer['operation_calls'], pointer['operation_returns'], pointer['outcome'])
    require(pointer['recipe'] == expected_recipe and pointer['test_data_used'] is False
        and all(type(pointer[k]) is int and pointer[k] == v for k, v in counters.items()), 'POINTER_ROLE')
    result_ref = campaign.require_binding(pointer['result'], label='C terminal result', verify_file=verify_files)
    if verify_files:
        result = campaign.read_bound_json(Path(result_ref['path']), result_ref['sha256'])
        require(result['recipe'] == expected_recipe and type(result['optimizer_steps']) is int
            and result['optimizer_steps'] == 0 and result['test_data_used'] is False
            and result['observed_calls'] == pointer['operation_calls'] and result['observed_returns'] == pointer['operation_returns']
            and (result['decision'] == 'COMPLETE_READONLY_TRAIN_DIAGNOSIS') == (pointer['outcome'] == 'COMPLETE'), 'POINTER_RESULT')
    return pointer


def build_checkpoint_diagnostic_progress(*, window, plan, invocation, checkpoint_pointer, pointer):
    from datetime import datetime, timezone
    fields = _operation_counts(pointer['operation_calls'], pointer['operation_returns'], pointer['outcome'])
    return _seal({'schema_version': campaign.PROGRESS_SCHEMA, 'plan_sha256': plan['plan_sha256'],
        'invocation_sha256': invocation['invocation_sha256'], 'phase': KIND, 'recipe': window['recipe'],
        **fields, 'total_units': 194, 'checkpoint_pointer': plain_ref(checkpoint_pointer), 'terminal': True,
        'outcome': pointer['outcome'], 'observed_utc': datetime.now(timezone.utc).isoformat()}, 'progress_sha256')



def build_checkpoint_diagnostic_running_progress(*, window, plan, invocation, calls, returns):
    """Observer-only RUNNING sample; terminal receipt validator rejects it."""
    from datetime import datetime, timezone
    fields = _operation_counts(calls, returns, 'FAILED')
    return _seal({'schema_version': campaign.PROGRESS_SCHEMA, 'plan_sha256': plan['plan_sha256'],
        'invocation_sha256': invocation['invocation_sha256'], 'phase': KIND, 'recipe': window['recipe'],
        **fields, 'total_units': 194, 'checkpoint_pointer': None, 'terminal': False,
        'outcome': 'RUNNING', 'observed_utc': datetime.now(timezone.utc).isoformat()}, 'progress_sha256')

def require_checkpoint_diagnostic_progress(value, *, plan_sha256, invocation, expected_selection_receipt_sha256, verify_file=True):
    fields = {'schema_version', 'plan_sha256', 'invocation_sha256', 'phase', 'recipe', 'completed_units',
        'forward_calls', 'forward_returns', 'reverse_calls', 'reverse_returns', 'checkpoint_loads',
        'checkpoint_load_returns', 'optimizer_steps', 'total_units', 'checkpoint_pointer', 'terminal', 'outcome', 'observed_utc'}
    progress = _identity(value, fields, campaign.PROGRESS_SCHEMA, 'progress_sha256')
    require(expected_selection_receipt_sha256 is None and invocation['kind'] == KIND
        and progress['plan_sha256'] == plan_sha256 and progress['invocation_sha256'] == invocation['invocation_sha256']
        and progress['phase'] == KIND and progress['recipe'] == invocation['checkpoint_diagnostic_recipe']
        and progress['terminal'] is True and progress['outcome'] in ('COMPLETE', 'FAILED'), 'PROGRESS_IDENTITY')
    campaign._utc(progress['observed_utc'], 'C progress')
    require(type(progress['total_units']) is int and progress['total_units'] == 194
        and type(progress['optimizer_steps']) is int and progress['optimizer_steps'] == 0, 'PROGRESS_ZERO_STEPS')
    for calls, returns, limit in (('forward_calls', 'forward_returns', 194), ('reverse_calls', 'reverse_returns', 6),
            ('checkpoint_loads', 'checkpoint_load_returns', 4)):
        require(type(progress[calls]) is type(progress[returns]) is int and 0 <= progress[returns] <= progress[calls] <= limit,
            'PROGRESS_COUNTER:' + calls)
        require(progress['outcome'] != 'COMPLETE' or progress[calls] == progress[returns] == limit, 'PROGRESS_COMPLETE:' + calls)
    require(type(progress['completed_units']) is int and progress['completed_units'] == progress['forward_returns'], 'PROGRESS_UNITS')
    ref = campaign.require_binding(progress['checkpoint_pointer'], label='C diagnostic pointer', verify_file=verify_file)
    require(ref['path'] == invocation['checkpoint']['pointer_path'], 'PROGRESS_POINTER_PATH')
    if verify_file:
        pointer = require_checkpoint_diagnostic_pointer(campaign.read_bound_json(Path(ref['path']), ref['sha256']),
            expected_recipe=progress['recipe'], verify_files=True)
        require(all(progress[k] == pointer[k] for k in ('outcome', 'completed_units', 'forward_calls', 'forward_returns',
            'reverse_calls', 'reverse_returns', 'checkpoint_loads', 'checkpoint_load_returns', 'optimizer_steps')), 'PROGRESS_DURABLE_POINTER')
    return progress
