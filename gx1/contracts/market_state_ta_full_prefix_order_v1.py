"""One fixed 43-by-43 batch permutation against completed full-prefix002.

Only the visit order changes. Original B16 groups, tail6, both complete passes,
query policy, INITIAL and numerical/input identities remain fixed. Matching RNG
states do not mean that each row receives its historical dropout mask.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import subprocess

from gx1.contracts import market_state_ta_full_prefix_learning_v1 as full

base, campaign, anchor, REPO = full.base, full.campaign, full.anchor, full.REPO
BASE_COMMIT = 'f0e59a95225c72e34a19d65c03202723e17fb7cc'
STAGE = 'market_state_ta_full_prefix_order_20261113_001'
SCOPE = 'market_state_ta_full_prefix_order_v1'
ROOT = full.ROOT.parent / STAGE.upper()
RUNTIME_ROOT, RECIPE_PATH = ROOT / 'RUNTIME', ROOT / 'RECIPE.json'
RECIPE_SCHEMA = 'gx1_ta_full_prefix_order_recipe_v1'
RESULT_SCHEMA = 'gx1_ta_full_prefix_order_result_v1'
CURSOR_SCHEMA = 'gx1_ta_full_prefix_order_cursor_v1'
OWNER = 'gx1/contracts/market_state_ta_full_prefix_order_v1.py'
WORKER = full.WORKER
ARMS, PASSES, EVALUATION_PASSES = full.ARMS, full.PASSES, full.EVALUATION_PASSES
PREFIX_ROWS, BATCHES_PER_PASS, TRAINING_PRESENTATIONS = full.PREFIX_ROWS, full.BATCHES_PER_PASS, full.TRAINING_PRESENTATIONS
TRAIN_BATCH_INDICES, LATER_BATCH_INDICES = list(full.TRAIN_BATCH_INDICES), list(full.LATER_BATCH_INDICES)
FIXED_PASS0_QUERY_POLICY = full.FIXED_PASS0_QUERY_POLICY
MODEL_SHA256, SOURCE_MODEL_SHA256 = full.MODEL_SHA256, full.SOURCE_MODEL_SHA256
INITIAL_RNG_SHA256, VIEW_REF, LATER_REFS = full.INITIAL_RNG_SHA256, full.VIEW_REF, full.LATER_REFS
MEASUREMENT_LABELS, BUDGETS = full.MEASUREMENT_LABELS, dict(full.BUDGETS)
ORDER_CONTROL_ROOT = ROOT.parent / 'MARKET_STATE_TA_FULL_PREFIX_LEARNING_20261112_002'
ORDER_CONTROL_RESULT_REF = dict(path=str(ORDER_CONTROL_ROOT / 'RESULT.json'),
    sha256='cc5d1ba16cf8823cce63794e6210061f714c5106f04a5b6ab5a7906494cad172')
ORDER_CONTROL_RECIPE_REF = dict(path=str(ORDER_CONTROL_ROOT / 'RECIPE.json'),
    sha256='993b9f1dc7b581e700f3fb148f90028c7eb1b51fdf5e1c9d62f8b7bd7182c23f')
ORDER_CONTROL_TERMINAL_REF = dict(path=str(ORDER_CONTROL_ROOT / 'TERMINAL.json'),
    sha256='72d686c5b4da75a9a52e7c638e684d0ad357def76caa2e292becdbd053f7cba9')
ORDER_CONTROL_REFS = dict(order_control_result=ORDER_CONTROL_RESULT_REF,
    order_control_recipe=ORDER_CONTROL_RECIPE_REF, order_control_terminal=ORDER_CONTROL_TERMINAL_REF)
EXTRA_OWNERS = frozenset({OWNER})
# Existing orchestration only, plus the explicitly parameterized full contract.
# In particular the five qualified numerical owners cannot drift from full002.
ORCHESTRATION_OWNERS = full.qualified.ORCHESTRATION_OWNERS | {full.OWNER}
_read, _ref = base._read, base._ref
batch_rows, cursor_from_batch_visits = full.batch_rows, full.cursor_from_batch_visits


def _need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_FULL_PREFIX_ORDER_' + reason)


def training_batch_indices(pass_number):
    _need(type(pass_number) is int and pass_number in PASSES, 'PASS_NUMBER')
    return [43 * r + c for c in range(43) for r in range(43)]


def _expected():
    original, _ = full._expected()
    control = _read(ORDER_CONTROL_RECIPE_REF)
    metadata = {'source_owners', 'trace_worker', 'mechanical_source_commit', 'recipe_sha256'}
    _need(set(control) == set(original) | metadata
          and campaign.canonical_sha256({k: control[k] for k in original}) == campaign.canonical_sha256(original)
          and control['recipe_sha256'] == campaign.canonical_sha256({k: v for k, v in control.items() if k != 'recipe_sha256'}),
          'EXACT_FULL_CONTROL_RECIPE')
    value = copy.deepcopy(original)
    refs = {**value['input_refs'], **copy.deepcopy(ORDER_CONTROL_REFS)}
    value.update(schema_version=RECIPE_SCHEMA, admission_stage=STAGE, run_scope=SCOPE,
        runtime_root=str(RUNTIME_ROOT), execution_binding_path=str(RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        input_refs=refs, candidate_inputs=copy.deepcopy(refs),
        batch_indices_by_pass={str(p): training_batch_indices(p) for p in PASSES},
        train_batch_indices=training_batch_indices(1),
        parent_selection='all_original_mature_fold1_TRAIN_batches_twice_fixed_43x43_transpose_tail6_last')
    return value, control


def build_recipe(*, source_commit):
    value, control = _expected()
    value.update(source_owners={n: _ref(REPO / n) for n in set(control['source_owners']) | EXTRA_OWNERS},
        trace_worker=_ref(REPO / WORKER), mechanical_source_commit=source_commit)
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_recipe(value, source_repo=REPO, source_commit=source_commit)


def require_recipe(value, *, source_repo, source_commit, verify_files=True):
    expected, control = _expected()
    _need(verify_files and source_repo == REPO and isinstance(value, Mapping)
          and set(value) == set(expected) | {'source_owners', 'trace_worker', 'mechanical_source_commit', 'recipe_sha256'},
          'RECIPE_FIELDS')
    _need(campaign.canonical_sha256({k: value[k] for k in expected}) == campaign.canonical_sha256(expected)
          and value['recipe_sha256'] == campaign.canonical_sha256({k: v for k, v in value.items() if k != 'recipe_sha256'}),
          'FIXED_RECIPE')
    _require_source_bridge(value, control)
    _need(value['trace_worker'] == value['source_owners'][WORKER], 'WORKER_BINDING')
    for revision in (source_commit, value['mechanical_source_commit']):
        _need(isinstance(revision, str) and campaign._COMMIT.fullmatch(revision)
              and subprocess.run(['git', 'merge-base', '--is-ancestor', BASE_COMMIT, revision],
                  cwd=REPO, capture_output=True).returncode == 0, 'SOURCE_ANCESTRY')
    read_origin(value)
    return dict(value)


def _require_source_bridge(recipe, control):
    owners = recipe['source_owners']
    _need(set(owners) == set(control['source_owners']) | EXTRA_OWNERS, 'SOURCE_INVENTORY')
    for name, current in owners.items():
        _need(current['path'] == str(REPO / name), 'SOURCE_PATH')
        if name in control['source_owners']:
            original = control['source_owners'][name]
            _need(original['path'] == current['path'], 'CONTROL_SOURCE_PATH')
            if name not in ORCHESTRATION_OWNERS:
                _need(current == original, 'UNCHANGED_NUMERICAL_SOURCE:' + name)
        campaign.require_binding(current, label='full prefix order owner ' + name, verify_file=True)


def read_origin(recipe):
    _, control_recipe = _expected()
    _need(all(recipe['input_refs'].get(k) == v for k, v in ORDER_CONTROL_REFS.items())
          and recipe['numerical_recipe'] == control_recipe['numerical_recipe']
          and recipe['objective_recipes'] == control_recipe['objective_recipes']
          and recipe['expected_view_sha256'] == control_recipe['expected_view_sha256'], 'ORDER_CONTROL_BINDING')
    _require_source_bridge(recipe, control_recipe)
    origin = full.read_origin(recipe)
    control, terminal = _read(ORDER_CONTROL_RESULT_REF), _read(ORDER_CONTROL_TERMINAL_REF)
    _need(control['schema_version'] == full.RESULT_SCHEMA and control['outcome'] == terminal['status'] == 'COMPLETE'
          and type(terminal['exit_code']) is int and terminal['exit_code'] == 0
          and control['recipe'] == terminal['recipe'] == ORDER_CONTROL_RECIPE_REF
          and terminal['result'] == ORDER_CONTROL_RESULT_REF
          and terminal['source_unchanged'] is terminal['no_relaunch_under_this_plan'] is True
          and terminal['test_data_used'] is False
          and all(control.get(k) is True for k in ('initial_restore_exact', 'evaluation_state_unchanged',
              'source_unchanged', 'same_training_queries'))
          and control['test_data_used'] is control['canonical_resume_authority'] is False
          and control['initial_model_state_sha256'] == MODEL_SHA256
          and control['optimizer_initialization'] == 'fresh_adamw_empty_state', 'ORDER_CONTROL_COMPLETE')
    _need(all(type(control.get(k)) is int and control[k] == v for k, v in dict(
        optimizer_updates=3698, batch_visits=3698, skipped_batches=0, parent_cursor=59148,
        training_presentations=59148).items()) and set(control['arms']) == set(ARMS), 'ORDER_CONTROL_BUDGET')
    arm = control['arms']['FULL_PREFIX']
    _need(arm['initial_model_state_sha256'] == MODEL_SHA256 and arm['initial_rng_state_sha256'] == INITIAL_RNG_SHA256
          and arm['fixed_pass0_query_policy'] is True and set(arm['training_rng_by_pass']) == {'1', '2'}
          and arm['training_rng_state_sha256'] == arm['training_rng_by_pass']['2'], 'ORDER_CONTROL_INITIAL_RNG')
    for digest in arm['training_rng_by_pass'].values():
        campaign._sha(digest, 'full002 pass RNG')
    full._require_initial_parity(control['full_prefix_control_parity'])
    for panel in ('TRAIN', 'LATER'):
        label = 'INITIAL_' + panel
        ref = control['measurement_refs'][label]
        _need(ref['path'] == str(ORDER_CONTROL_ROOT / (label + '.json'))
              and control['panel_captures'][label]['path'] == str(ORDER_CONTROL_ROOT / (label + '_READOUT.npz')),
              'ORDER_CONTROL_INITIAL_PATH')
        measured = _read(ref)
        _need(measured['evaluated_model_state_sha256'] == MODEL_SHA256 and measured['optimizer_updates'] == 0
              and measured['evaluation_state_unchanged'] is True and measured['test_data_used'] is False
              and measured['capture'] == control['panel_captures'][label]
              and measured['panel_batch_indices'] == (TRAIN_BATCH_INDICES if panel == 'TRAIN' else LATER_BATCH_INDICES)
              and measured['measurement'] == control['measurements'][label]
                  == origin['full_prefix_control']['measurements'][label], 'ORDER_CONTROL_INITIAL_EXACT')
    origin['order_control'] = control
    return origin


def gate_fields():
    result = full.gate_fields()
    del result['ta_full_prefix_learning_authorized']
    result['ta_full_prefix_order_authorized'] = True
    return result


def require_gate(gate, recipe):
    _need(isinstance(gate, Mapping) and all(type(gate.get(k)) is type(v) and gate[k] == v
          for k, v in gate_fields().items()) and recipe['run_scope'] == SCOPE
          and recipe['admission_stage'] == STAGE, 'POLICY_GATE')


def preflight_read_only(recipe, *, source_commit=None):
    source_commit = source_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    require_recipe(recipe, source_repo=REPO, source_commit=source_commit)
    return dict(status='PASS_METADATA_ONLY_FULL_PREFIX_ORDER', run_scope=SCOPE,
        model_constructions=0, data_payload_reads=0, checkpoint_loads=0, optimizer_updates=0,
        unique_training_rows=29574, training_presentations=59148, passes=2, tail_rows_each_pass=6,
        reused_control_initial=True, train_normalization_unchanged=True, test_data_used=False,
        order_control=copy.deepcopy(ORDER_CONTROL_RESULT_REF))


def build_cursor(*, recipe_ref, outcome, optimizer_updates, parent_cursor, parent_order_sha256,
                 full_state, result, batch_size=16):
    return full.build_cursor(recipe_ref=recipe_ref, outcome=outcome, optimizer_updates=optimizer_updates,
        parent_cursor=parent_cursor, parent_order_sha256=parent_order_sha256, full_state=full_state,
        result=result, batch_size=batch_size, root=ROOT, recipe_path=RECIPE_PATH,
        cursor_schema=CURSOR_SCHEMA, result_schema=RESULT_SCHEMA, completion=_require_complete)


def require_cursor(value, *, expected_recipe, verify_files=True, batch_size=16):
    return full.require_cursor(value, expected_recipe=expected_recipe, verify_files=verify_files, batch_size=batch_size,
        root=ROOT, recipe_path=RECIPE_PATH, cursor_schema=CURSOR_SCHEMA, result_schema=RESULT_SCHEMA,
        completion=_require_complete)


def _require_complete(report):
    full._require_complete(report, root=ROOT, schedule=training_batch_indices, rng_stream_parity=True)
    _need(campaign.canonical_sha256(report['exposure_schedule']) == campaign.canonical_sha256(
        {str(p): training_batch_indices(p) for p in PASSES})
          and type(report['primary_endpoint_pass']) is int and report['primary_endpoint_pass'] == 2,
        'ACTUAL_ORDER_AND_PRIMARY')
    control = _read(ORDER_CONTROL_RESULT_REF)
    arm = report['arms']['FULL_PREFIX']
    _need(arm['training_rng_by_pass'] == control['arms']['FULL_PREFIX']['training_rng_by_pass']
          and arm['training_rng_state_sha256'] == arm['training_rng_by_pass']['2'], 'CONTROL_PASS_RNG_EXACT')
