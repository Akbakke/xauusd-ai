"""Exact four-state276 TRAIN diagnosis; metadata/operation authority, never fit.

No numerical imports or file reads. The public worker must prove current capped
execution and a fresh plan/admission before loading any payload or model.
Component source only: C has no admitted capped-runner/dispatcher route yet; a
separate guard-preserving integration admission is required before genuine use.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath

SCHEMA = 'gx1_native_direction_checkpoint_diagnostic_plan_v1'
OPERATION = 'C_saved276_checkpoint_diagnostic_v1'
STAGE = 'dynamic_m5_direction_checkpoint_diagnostic_c'
RUNTIME_ROUTE_QUALIFIED = False
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
    fields = {'schema_version', 'operation', 'admission_stage', 'proposal', 'source_head',
        'policy', 'source_owners', 'run_root', 'budget', 'operation_limits', 'initial_report',
        'initial_predictions', 'max_invocations', 'execution_authorized',
        'data_reads_authorized', 'checkpoint_reads_authorized', 'genuine_forwards_authorized',
        'gradient_reverse_authorized', 'optimizer_steps_authorized', 'test_data_used'}
    require(type(value) is dict and set(value) == fields and value['schema_version'] == SCHEMA
        and value['operation'] == OPERATION and value['admission_stage'] == STAGE, 'PLAN_SCHEMA')
    require(value['proposal'] == PROPOSAL and value['initial_report'] == INITIAL_REPORT
        and value['initial_predictions'] == INITIAL_PREDICTIONS, 'FIXED_INPUT_IDENTITIES')
    plain_ref(value['policy'])
    require(value['policy']['path'] == REPO + '/NEXT_RUN_POLICY.json', 'CANONICAL_POLICY')
    require(type(value['source_head']) is str and len(value['source_head']) == 40
        and all(c in '0123456789abcdef' for c in value['source_head']), 'SOURCE_HEAD')
    require(type(value['source_owners']) is dict and all(
        plain_ref(ref)['path'] == REPO + '/' + key for key, ref in value['source_owners'].items())
        and all(name in value['source_owners'] for name in NEW_FILES), 'SOURCE_OWNERS')
    root = PurePosixPath(value['run_root'])
    require(str(root) == value['run_root'] and root.parent == PurePosixPath(PROGRAM)
        and root.name.startswith('C_CHECKPOINT_DIAGNOSTIC_'), 'NEW_CHILD_RUN_ROOT')
    require(type(value['max_invocations']) is int and value['max_invocations'] == 1,
        'SINGLE_INVOCATION')
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


def require_admission(admission, plan_ref):
    require(type(admission) is dict and admission.get('operation') == OPERATION
        and admission.get('execution_authorized') is True
        and admission.get('plan') == plain_ref(plan_ref)
        and type(admission.get('max_invocations')) is int and admission['max_invocations'] == 1
        and admission.get('genuine_payload_reads_authorized') is True
        and admission.get('genuine_forwards_authorized') is True
        and admission.get('gradient_reverse_authorized') is True
        and admission.get('optimizer_steps_authorized') is False
        and admission.get('test_data_used') is False, 'ROOT_EXECUTION_ADMISSION')
    return deepcopy(admission)


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
