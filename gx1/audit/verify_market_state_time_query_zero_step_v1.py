"""One-use TRAIN-only native time-query zero-step gradient witness.

A separately admitted guarded CUDA producer may call ``run`` once. The witness
uses the published fold-1 and fold-2 mature input normalizations, one
exact M5 issue per fold and label-blind future queries. It performs no
optimizer, checkpoint, TEST,
Entry, Exit or trading operation and grants no model-fit authority.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
from typing import Mapping

from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
from gx1.audit import fit_market_state_mature_fold1_units_v1 as fold1
from gx1.contracts.entry_native_direction_prefix_normalization_v1 import (
    require_prefix_units_normalization_bindings,
)
from gx1.audit import verify_market_state_five_endpoint_zero_step_v1 as old_zero
from gx1.contracts import market_state_time_query_binding_v1 as binding
from gx1.contracts.market_state_time_query_loss_v1 import time_query_loss_contract
from gx1.contracts.market_state_time_query_model_v1 import (
    TimeQueryMarketStateModel, time_query_model_contract,
)

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
POLICY = REPO / 'NEXT_RUN_POLICY.json'
STAGE = 'market_state_time_query_zero_step_20261107_002'
SCHEMA = 'gx1_market_state_time_query_zero_step_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
MEMORY_BYTES = 12 * 1024**3
SWAP_BYTES = 512 * 1024**2
WALL_SECONDS = 3600
MAX_JSON_BYTES = 1024 * 1024
FOLD_IDS = [1, 2]
FOLD1_ID = 1
FOLD2_ID = 2
PASS_INDEX = 0
SEED = 20261107
PARENT_ROWS = [2]
ISSUE_NS = [1714522500000000000]
EXPECTED_QUERY_K = [1, 2, 2, 5, 92]
EXPECTED_ALLOWED = [1, 1, 7, 7, 7]
EXPECTED_VIEW_SHA256_BY_FOLD = {
    '1': '7e0d13f1b4f58cc4d3dd294bf869d3195b48a69c022fe570e6b4db02f1581f15',
    '2': '0cc8bcfa2cabbff6ddb214956d7c16c8b065c0338508678698d59029abdf16ed',
}
EXECUTION = {
    'class': 'producer', 'cuda_producer': True, 'device': 'cuda',
    'module': 'gx1.scripts.validate_entry_model_native_technical_checkpoint_v1',
    'flag': '--market-state-time-query-zero-step-plan',
    'memory': '12G', 'swap': '512M', 'model_forwards': 2,
    'backward_calls': 2, 'optimizer_steps': 0,
}
LIMITS = {'memory_max': MEMORY_BYTES, 'memory_high': MEMORY_BYTES,
          'swap': SWAP_BYTES, 'max_wall_seconds': WALL_SECONDS}
MATURE_ROOT = RUNS / 'MARKET_STATE_MATURE_PREFIX_NORMALIZATION_20261107_001'
FOLD1_ROOT = RUNS / 'MARKET_STATE_MATURE_FOLD1_NORMALIZATION_20261107_001'
FIXED = {
    'fold1_plan': (fold1.PLAN_PATH,
                   '2e74f6369d3b3bf685d9cdaa606db84ed0e5cadf6f4afe1fdb59e7550ef68616'),
    'fold1_claim': (FOLD1_ROOT / 'CLAIM.json',
                    'e85eec1bc81371c7dc77fb922350010ca7a96d9baad31bf41bd9127910e124c1'),
    'fold1_result': (FOLD1_ROOT / 'RESULT.json',
                     'c6cc941203a38791c348a775e6815318fd265d659406207e59ebac374c2754fc'),
    'fold1_terminal': (FOLD1_ROOT / 'TERMINAL.json',
                       'c0aeef405973d13351f5ebb2b1451a33b8a03f72d7b24cf19f95c5a6a91c2537'),
    'fold1_qualification': (FOLD1_ROOT / 'QUALIFICATION_FOLD_1.json',
                            '0fad1ac1476205e0485207da4b5e6be6b0de5585a8cb4170f87679c63580fb94'),
    'fold1_normalization': (FOLD1_ROOT / 'PREFIX_NORMALIZATION_FOLD_1.json',
                            'e24a2b8d1ae09f6f10c2de7a7097353b52aa2f2118287535e242778272022d54'),
    'mature_plan': (mature.PLAN_PATH,
                    'b77795e606b35424d6189b7866337ca74c9f7dfc22371168452f5a6f6d665029'),
    'mature_claim': (MATURE_ROOT / 'CLAIM.json',
                     '797130de9eb86a9596a4a20a58732d4c887b2a21206eae438a6c92a6e9919d6b'),
    'mature_result': (MATURE_ROOT / 'RESULT.json',
                      '4b2772e893cdca98af91b04daa817d5b289c234f9dcd5cab5e2bac0f680c6833'),
    'mature_terminal': (MATURE_ROOT / 'TERMINAL.json',
                        'c88cca2d44189ec0284ebfea7f55e431759a43ecce49c801f59a880dff88f1bc'),
    'fold2_qualification': (MATURE_ROOT / 'QUALIFICATION_FOLD_2.json',
                            'c271f952f8befef3485de4ed5d5460791c8da2b608f684b7717cf78a64e9707b'),
    'fold2_normalization': (MATURE_ROOT / 'PREFIX_NORMALIZATION_FOLD_2.json',
                            '32f917bb9cf262f57afe2f60d58025b6679a3ffc5874dbe2d3b49382283ff9bc'),
    'constructor': old_zero.FIXED['constructor'],
    'base_view': old_zero.FIXED['old_view'],
    'candidate_tape': binding.CANDIDATE_REFS['tape'],
}
SOURCE_PATHS = mature.SOURCE_PATHS | frozenset({
    'gx1/audit/fit_market_state_mature_fold1_units_v1.py',
    'gx1/audit/verify_market_state_time_query_zero_step_v1.py',
    'gx1/audit/verify_market_state_five_endpoint_zero_step_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/entry_native_direction_research_v1.py',
    'gx1/contracts/market_state_time_query_model_v1.py',
    'gx1/contracts/market_state_time_query_loss_v1.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/market_state_time_query_binding_v1.py',
    'gx1/contracts/market_state_time_query_batch_v1.py',
    'gx1/contracts/model_state_digest_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
    'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
    'gx1/scripts/validate_entry_model_native_technical_checkpoint_v1.py',
    'scripts/gx1_guarded_trainer_exec.sh',
})
# Exact reviewed runtime drift since the mature-prefix source commit. The first
# four edits affect only the optional unobserved-units profile; fold 2 uses None.
# The fifth adds an unrelated CPU research guard. Any further drift blocks.
HISTORICAL_RUNTIME_BRIDGE = {
    'gx1/contracts/entry_gc_mtf_history_v1.py': (
        'ef64a40a7ac24fc931acc267fae298adde26f20ed91ed587fd28ffcaae71c565',
        '25881da893bd09a48b6f6ef840947af611401ac6aa3059c795a99a127293f14c'),
    'gx1/contracts/entry_model_native_input_normalization_v1.py': (
        'd61f69ef733ab756d5848867c579a0339890fc8f396f4adb41ca80545d7ac99d',
        '22a08fb03578535f39cfa959c35a77dc569fba984ea17041fce7e9dd4ee417e1'),
    'gx1/contracts/entry_native_direction_prefix_normalization_v1.py': (
        '07164df124562174c867d491e78f25e192e8a2d062056e07ab9463444fae72ac',
        '7142437d8dfdd860b89188f1bb7bc76ca75eb60a98357c62e261b140a3fbcd12'),
    'gx1/contracts/entry_native_direction_prefix_units_v1.py': (
        '37627f5321e476b00b2b5ed6c9d96b4c59f3b04124b3eb64cbedb6b3cf9a021d',
        'dd2f2458c6fb7cc0b3bddbddfd295a4ec3d20a72699149c8ea25f5c8c44445a0'),
    'gx1/contracts/gx1_capped_execution_v1.py': (
        'e8206a3d7bf6922b1c4f0fac85498c5a913270e96e94871d7f7a9c48c9611cb5',
        '60b1ccf2eda470a84ddded0d8160c628314ebb19f5766b87b70fef7c9aa3ccff'),
}
IMMUTABLE_VIEW_REF_KEYS = frozenset({
    'feature_view_design', 'input_qualification', 'raw_input_provider_plan',
    'raw_gc_layout', 'raw_gc_transition', 'time_query_candidate_tape',
    'time_query_mature_prefix_plan',
})
HISTORICAL_VIEW_REF_KEYS = frozenset({'gc_gather_source'})
CURRENT_VIEW_REF_KEYS = frozenset({
    'direction_component_source', 'native_constructor_source',
    'mtf_window_source', 'owner_profile_source',
    'time_query_model_source', 'time_query_loss_source',
    'time_query_target_source', 'time_query_target_binding_source',
    'five_endpoint_contract_source',
})
TIME_QUERY_SOURCE_PATHS = {
    'time_query_model_source': 'gx1/contracts/market_state_time_query_model_v1.py',
    'time_query_loss_source': 'gx1/contracts/market_state_time_query_loss_v1.py',
    'time_query_target_source': 'gx1/contracts/market_state_time_query_target_v1.py',
    'time_query_target_binding_source': 'gx1/contracts/market_state_time_query_binding_v1.py',
    'five_endpoint_contract_source': 'gx1/contracts/market_state_five_endpoint_contract_v1.py',
}


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('TIME_QUERY_ZERO_STEP_' + reason)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _ref(path: Path, sha256: str) -> dict:
    return {'path': str(path), 'sha256': sha256}


def _fixed_ref(name: str) -> dict:
    return _ref(*FIXED[name])


def _read_ref(ref: dict, *, limit: int = 8 * 1024**2) -> dict:
    _need(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}
          and isinstance(ref['path'], str) and isinstance(ref['sha256'], str)
          and len(ref['sha256']) == 64, 'REF_SCHEMA')
    path = Path(ref['path'])
    _need(path.is_absolute() and path.is_file() and not path.is_symlink()
          and 0 < path.stat().st_size <= limit and _sha(path) == ref['sha256'],
          'REF_HASH:' + ref['path'])
    return json.loads(path.read_text())


def _write(root: Path, name: str, value: dict, *, limit: int) -> dict:
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(len(data) <= limit and json.loads(data) == value, 'OUTPUT_SIZE_OR_LOAD')
    path = root / name
    _publish_bytes(path, data)
    return _ref(path, hashlib.sha256(data).hexdigest())


def _clean_head() -> str:
    _need(Path.cwd().resolve() == REPO
          and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
          == 'work/gx1-current'
          and subprocess.check_output(['git', 'status', '--porcelain'], text=True) == '',
          'CLEAN_SOURCE')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def _checked_plan(plan_path: Path, plan_sha256: str) -> tuple[dict, dict]:
    _need(plan_path == PLAN_PATH and plan_path.is_absolute() and plan_path.is_file()
          and type(plan_sha256) is str and len(plan_sha256) == 64
          and _sha(plan_path) == plan_sha256, 'PLAN_HASH')
    ref = _ref(plan_path, plan_sha256)
    return _read_ref(ref), ref


RUNNER_SOURCE = 'scripts/gx1_capped_run.sh'


def _reviewed_runner_sha(reviewed_runner_ref: Mapping[str, str] | None) -> str | None:
    """Accept only the current, plan-bound runner bytes for a later diagnostic."""
    if reviewed_runner_ref is None:
        return None
    path = REPO / RUNNER_SOURCE
    _need(isinstance(reviewed_runner_ref, Mapping)
          and dict(reviewed_runner_ref) == {'path': str(path),
                                           'sha256': reviewed_runner_ref.get('sha256')}
          and isinstance(reviewed_runner_ref['sha256'], str)
          and len(reviewed_runner_ref['sha256']) == 64
          and all(c in '0123456789abcdef' for c in reviewed_runner_ref['sha256'])
          and path.is_file() and not path.is_symlink()
          and _sha(path) == reviewed_runner_ref['sha256'],
          'REVIEWED_RUNNER_REF')
    return reviewed_runner_ref['sha256']


def _inspect_receipts(*, reviewed_runner_ref: Mapping[str, str] | None = None) -> dict:
    plan = _read_ref(_fixed_ref('mature_plan'))
    claim = _read_ref(_fixed_ref('mature_claim'))
    result = _read_ref(_fixed_ref('mature_result'))
    terminal = _read_ref(_fixed_ref('mature_terminal'))
    qualified = _read_ref(_fixed_ref('fold2_qualification'))
    normal = _read_ref(_fixed_ref('fold2_normalization'))
    constructor = _read_ref(_fixed_ref('constructor'))
    base_view = _read_ref(_fixed_ref('base_view'))
    mature._validate_cohort_contract(plan['cohorts'])
    fits = result.get('fits', [])
    fold2 = next((item for item in fits if item.get('fold_id') == FOLD2_ID), None)
    cohort2 = plan['cohorts'][1]
    _need(result.get('status') == 'COMPLETE_FIVE_MATURE_INPUT_QUALIFICATIONS_CONDITIONAL_FITS_NO_MODEL'
          and result.get('plan') == _fixed_ref('mature_plan')
          and result.get('claim') == _fixed_ref('mature_claim')
          and claim.get('plan') == _fixed_ref('mature_plan')
          and claim.get('normalization_only') is True
          and terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
          and terminal.get('result') == _fixed_ref('mature_result')
          and terminal.get('fit_count') == 4
          and result.get('test_data_used') is False
          and result.get('model_forwards') == result.get('optimizer_steps') == 0
          and result.get('model_training_authorized') is False
          and isinstance(fold2, dict)
          and fold2.get('output') == _fixed_ref('fold2_normalization')
          and fold2.get('qualification') == _fixed_ref('fold2_qualification')
          and cohort2['joined_fit_rows'] == 47338
          and cohort2['effective_fit_cutoff_ns'] == mature.target.FOLD_BOUNDARIES_NS[1]
              - 1440 * mature.target.MINUTE_NS
          and qualified.get('fit_allowed') is True
          and qualified.get('blockers') == []
          and qualified.get('statistics_fits') == 0
          and normal.get('prefix_qualification') == qualified
          and normal['prefix_qualification']['proof']['prefix']['rows'] == 47338
          and normal['prefix_qualification']['proof']['prefix']['cutoff_ns']
              == cohort2['effective_fit_cutoff_ns']
          and normal.get('genuine_model_fit_authorized') is False
          and normal.get('native_launch_authorized') is False,
          'MATURE_FOLD2_CHAIN')
    binding._candidate_receipts()
    binding._checked_bytes('tape', 32 * 1024**2)
    chain = mature._receipt_chain()
    runner_sha = _reviewed_runner_sha(reviewed_runner_ref)
    lineage = _historical_source_lineage(normal, result, reviewed_runner_sha=runner_sha)
    fold1_receipts = _inspect_fold1_receipts(plan, reviewed_runner_sha=runner_sha)
    return {'mature_plan': plan, 'normal': normal, 'constructor': constructor,
            'base_view': base_view, 'cohort2': cohort2, 'fold1': fold1_receipts,
            'provider_chain': chain, 'historical_lineage': lineage}


def _read_validated_fold1_receipts(mature_plan: dict) -> dict:
    """The immutable fold-1 receipt chain, independent of today's runtime."""
    plan = _read_ref(_fixed_ref('fold1_plan'))
    claim = _read_ref(_fixed_ref('fold1_claim'))
    result = _read_ref(_fixed_ref('fold1_result'))
    terminal = _read_ref(_fixed_ref('fold1_terminal'))
    qualified = _read_ref(_fixed_ref('fold1_qualification'))
    normal = _read_ref(_fixed_ref('fold1_normalization'))
    _, old_report, profile = fold1._original_receipts()
    cohort = mature_plan['cohorts'][0]
    provenance = qualified['proof']['source_bindings']
    _need(plan.get('schema_version') == fold1.SCHEMA
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('cohort') == cohort
          and plan.get('input_refs') == fold1.INPUT_REFS
          and claim.get('plan') == result.get('plan') == terminal.get('plan')
              == _fixed_ref('fold1_plan')
          and result.get('claim') == terminal.get('claim') == _fixed_ref('fold1_claim')
          and result.get('policy') == claim.get('policy')
          and result.get('source_head') == claim.get('source_head')
              == provenance.get('current_source_head')
          and result.get('status')
              == 'COMPLETE_FOLD1_MATURE_UNITS_PREFIX_NORMALIZATION_NO_MODEL'
          and result.get('input_refs') == fold1.INPUT_REFS
          and result.get('cohort') == cohort
          and result.get('old_raw_qualification')
              == fold1.INPUT_REFS['original_fold1_report']
          and result.get('new_qualification') == _fixed_ref('fold1_qualification')
          and result.get('normalization') == _fixed_ref('fold1_normalization')
          and result.get('normalization_contract_sha256') == normal['contract_sha256']
          and result.get('unit_profile_sha256') == profile['contract_sha256']
              == fold1.PROFILE_SHA256
          and result.get('fit_rows') == cohort['joined_fit_rows'] == fold1.FIT_ROWS
          and result.get('declared_unit_support_rows') == len(old_report['blockers']) == 153
          and result.get('statistics_fits') == 1
          and result.get('decoded_target_arrays') == result.get('decoded_label_arrays') == 0
          and result.get('model_forwards') == result.get('model_constructions') == 0
          and result.get('optimizer_steps') == 0
          and result.get('test_data_used') is False
          and result.get('model_training_authorized') is False
          and result.get('native_launch_authorized') is False
          and terminal.get('status') == 'COMPLETE'
          and terminal.get('exit_code') == 0
          and terminal.get('result') == _fixed_ref('fold1_result')
          and terminal.get('fit_count') == terminal.get('qualification_count') == 1
          and terminal.get('model_forwards') == terminal.get('optimizer_steps') == 0
          and terminal.get('test_data_used') is False
          and terminal.get('no_relaunch_under_this_plan') is True
          and qualified.get('fit_allowed') is True
          and qualified.get('blockers') == []
          and qualified.get('statistics_fits') == 0
          and qualified.get('unobserved_units_profile') == profile
          and qualified.get('declared_unit_support') == old_report['blockers']
          and qualified['proof']['prefix']['rows'] == fold1.FIT_ROWS
          and qualified['proof']['prefix']['cutoff_ns'] == cohort['effective_fit_cutoff_ns']
          and normal.get('prefix_qualification') == qualified
          and normal.get('unobserved_units_profile') == profile
          and normal.get('old055_or223_normalization_identity_reused') is False
          and normal.get('genuine_model_fit_authorized') is False
          and normal.get('native_launch_authorized') is False
          and provenance.get('current_plan') == _fixed_ref('fold1_plan')
          and provenance.get('current_source_owners') == plan.get('source_owners'),
          'MATURE_FOLD1_CHAIN')
    fold1._validate_new_report(qualified, old_report, profile, provenance)
    require_prefix_units_normalization_bindings(normal, profile)
    return {'normal': normal, 'qualification': qualified, 'profile': profile,
            'cohort': cohort}


def _inspect_fold1_receipts(mature_plan: dict, *, reviewed_runner_sha: str | None = None) -> dict:
    receipts = _read_validated_fold1_receipts(mature_plan)
    provenance = receipts['qualification']['proof']['source_bindings']
    receipts['lineage'] = _fold1_source_lineage(
        provenance, reviewed_runner_sha=reviewed_runner_sha)
    return receipts


def _fold1_historical_source_lineage(provenance: dict) -> dict:
    head = provenance['current_source_head']
    owners = provenance['current_source_owners']
    _need(isinstance(owners, dict) and set(owners) == fold1.SOURCE_PATHS
          and subprocess.run(['git', 'cat-file', '-e', head + '^{commit}'],
                             cwd=REPO, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0,
          'FOLD1_SOURCE_HEAD')
    historical_policy = provenance['current_policy']
    _need(historical_policy['path'] == str(POLICY)
          and hashlib.sha256(subprocess.check_output(
              ['git', 'show', head + ':NEXT_RUN_POLICY.json'], cwd=REPO,
              timeout=15)).hexdigest() == historical_policy['sha256'],
          'FOLD1_HISTORICAL_POLICY')
    for path, source_sha in owners.items():
        relative = Path(path)
        _need(not relative.is_absolute() and '..' not in relative.parts
              and len(source_sha) == 64
              and hashlib.sha256(subprocess.check_output(
                  ['git', 'show', head + ':' + path], cwd=REPO,
                  timeout=15)).hexdigest() == source_sha,
              'FOLD1_HISTORICAL_SOURCE:' + path)
    return {'source_head': head, 'historical_policy_sha256': historical_policy['sha256']}


def _fold1_source_lineage(provenance: dict, *, reviewed_runner_sha: str | None = None) -> dict:
    lineage = _fold1_historical_source_lineage(provenance)
    owners = provenance['current_source_owners']
    for path, source_sha in owners.items():
        current_sha = reviewed_runner_sha if path == RUNNER_SOURCE and reviewed_runner_sha is not None else source_sha
        _need(_sha(REPO / path) == current_sha, 'FOLD1_CURRENT_SOURCE:' + path)
    return {**lineage, 'current_source_owner_count': len(owners)}


def _historical_source_lineage(normal: dict, result: dict, *,
                               reviewed_runner_sha: str | None = None) -> dict:
    provenance = normal['prefix_qualification']['proof']['source_bindings']
    head = provenance['current_source_head']
    owners = provenance['current_source_owners']
    _need(head == result['source_head']
          and isinstance(owners, dict) and len(owners) >= 20
          and subprocess.run(['git', 'cat-file', '-e', head + '^{commit}'],
                             cwd=REPO, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0,
          'HISTORICAL_HEAD')
    historical_policy = provenance['current_policy']
    _need(historical_policy['path'] == str(POLICY)
          and hashlib.sha256(subprocess.check_output(
              ['git', 'show', head + ':NEXT_RUN_POLICY.json'], cwd=REPO,
              timeout=15)).hexdigest() == historical_policy['sha256'],
          'HISTORICAL_POLICY')
    drift = {}
    for path, old_sha in owners.items():
        relative = Path(path)
        _need(not relative.is_absolute() and '..' not in relative.parts
              and len(old_sha) == 64 and (REPO / relative).is_file(),
              'HISTORICAL_SOURCE_PATH')
        archived = subprocess.check_output(['git', 'show', head + ':' + path],
                                            cwd=REPO, timeout=15)
        _need(hashlib.sha256(archived).hexdigest() == old_sha,
              'HISTORICAL_SOURCE_BLOB:' + path)
        current_sha = _sha(REPO / path)
        if current_sha != old_sha:
            drift[path] = (old_sha, current_sha)
    expected_drift = dict(HISTORICAL_RUNTIME_BRIDGE)
    if reviewed_runner_sha is not None:
        old_runner_sha = owners.get(RUNNER_SOURCE)
        _need(isinstance(old_runner_sha, str) and len(old_runner_sha) == 64
              and old_runner_sha != reviewed_runner_sha, 'REVIEWED_RUNNER_HISTORY')
        expected_drift[RUNNER_SOURCE] = (old_runner_sha, reviewed_runner_sha)
    _need(drift == expected_drift
          and 'unobserved_units_profile' not in normal
          and 'unobserved_units_profile' not in normal['prefix_qualification'],
          'REVIEWED_RUNTIME_DRIFT')
    return {'historical_source_head': head,
            'historical_policy_sha256': historical_policy['sha256'],
            'reviewed_runtime_drift': list(sorted(drift))}


def _source_refs(plan: dict, receipts: dict, fold_id: int) -> dict:
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    _need(fold_id in FOLD_IDS, 'VIEW_FOLD')
    old = receipts['base_view']['source_bindings']
    _need(views.SOURCE_REF_KEYS <= set(old), 'BASE_VIEW_SOURCE_KEYS')
    sources = {key: old[key] for key in views.SOURCE_REF_KEYS}
    component_path = 'gx1/contracts/entry_native_direction_research_v1.py'
    constructor_path = 'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py'
    sources['direction_component_source'] = _ref(
        REPO / component_path, plan['source_owners'][component_path])
    sources['native_constructor_source'] = _ref(
        REPO / constructor_path, plan['source_owners'][constructor_path])
    normal = receipts['fold1']['normal'] if fold_id == FOLD1_ID else receipts['normal']
    qualification = ('fold1_qualification' if fold_id == FOLD1_ID
                     else 'fold2_qualification')
    mature_plan = 'fold1_plan' if fold_id == FOLD1_ID else 'mature_plan'
    provenance = normal['prefix_qualification']['proof']['source_bindings']
    sources.update(input_qualification=_fixed_ref(qualification),
                   raw_input_provider_plan=provenance['provider_plan'],
                   raw_gc_layout=provenance['boolean_raw_gc_layout'],
                   raw_gc_transition=provenance['boolean_raw_domain_transition'])
    for key, path in (('mtf_window_source', 'gx1/features/htf_features.py'),
                      ('gc_gather_source', 'gx1/contracts/entry_gc_mtf_history_v1.py'),
                      ('owner_profile_source', 'gx1/features/entry_specialist_feature_groups_v1.py')):
        sources[key] = _ref(REPO / path, provenance['current_source_owners'][path])
    for key, path in TIME_QUERY_SOURCE_PATHS.items():
        sources[key] = _ref(REPO / path, plan['source_owners'][path])
    sources['time_query_candidate_tape'] = _fixed_ref('candidate_tape')
    sources['time_query_mature_prefix_plan'] = _fixed_ref(mature_plan)
    historical = (HISTORICAL_VIEW_REF_KEYS if fold_id == FOLD2_ID else frozenset())
    current = CURRENT_VIEW_REF_KEYS | (HISTORICAL_VIEW_REF_KEYS
                                       if fold_id == FOLD1_ID else frozenset())
    _need(set(sources) == views.SOURCE_REF_KEYS | views.TIME_QUERY_SOURCE_REF_KEYS
          == IMMUTABLE_VIEW_REF_KEYS | historical | current
          and not (IMMUTABLE_VIEW_REF_KEYS & historical)
          and not (IMMUTABLE_VIEW_REF_KEYS & current)
          and not (historical & current), 'SOURCE_REF_CLASSES')
    for key, item in sources.items():
        if key in historical:
            path = 'gx1/contracts/entry_gc_mtf_history_v1.py'
            _need(item == _ref(REPO / path, HISTORICAL_RUNTIME_BRIDGE[path][0])
                  and _sha(REPO / path) == HISTORICAL_RUNTIME_BRIDGE[path][1],
                  'HISTORICAL_GC_VIEW_REF')
        elif key in IMMUTABLE_VIEW_REF_KEYS:
            _need(Path(item['path']).is_absolute()
                  and Path(item['path']).is_file()
                  and not Path(item['path']).is_symlink()
                  and _sha(Path(item['path'])) == item['sha256'],
                  'IMMUTABLE_VIEW_REF:' + key)
        else:
            _need(Path(item['path']).is_relative_to(REPO)
                  and _sha(Path(item['path'])) == item['sha256'],
                  'CURRENT_VIEW_REF:' + key)
    return sources


def _bind_view(plan: dict, receipts: dict, fold_id: int) -> dict:
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    _need(fold_id in FOLD_IDS, 'VIEW_FOLD')
    normal = receipts['fold1']['normal'] if fold_id == FOLD1_ID else receipts['normal']
    normalization_ref = ('fold1_normalization' if fold_id == FOLD1_ID
                         else 'fold2_normalization')
    profile = receipts['fold1']['profile'] if fold_id == FOLD1_ID else None
    view = views.bind_native_direction_feature_view(
        prefix_normalization=normal,
        prefix_normalization_ref=_fixed_ref(normalization_ref),
        constructor_metadata=receipts['constructor'],
        constructor_metadata_ref=_fixed_ref('constructor'),
        source_bindings=_source_refs(plan, receipts, fold_id), max_batch_rows=16,
        full_feature_profile=True, market_state_time_query=True,
        unobserved_units_profile=profile)
    key = str(fold_id)
    _need(view['contract_sha256'] == EXPECTED_VIEW_SHA256_BY_FOLD[key]
          == plan['expected_view_sha256_by_fold'][key]
          and view['research_objective']['contract_sha256']
          == time_query_model_contract()['contract_sha256']
          == plan['expected_model_contract_sha256'], 'VIEW_IDENTITY')
    return view



# The TA candidate reuses immutable fold-1 TRAIN statistics. These explicit
# runtime owners may change for the approved GC-free route; every other old
# numeric/feature owner must match the original bytes or the exact Counter pair below.
TA_PREFIX_RUNTIME_SOURCE_CHANGES = frozenset({
    'gx1/audit/qualify_market_state_mature_prefix_normalization_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/entry_exit_production_architecture_v1.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'scripts/gx1_capped_run.sh',
})
TA_SOURCE_PATHS = SOURCE_PATHS | frozenset({
    'gx1/contracts/market_state_time_query_model_v2.py',
    'gx1/contracts/market_state_time_query_loss_v2.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_native_F_batch_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/audit/verify_market_state_ta_native_input_path_v1.py',
})


TA_SIGNAL_COUNTER_SOURCE = 'gx1/contracts/entry_model_native_signal_v1.py'
TA_SIGNAL_COUNTER_SOURCE_SHA256 = (
    '521f089a9b6d98121939610ff7cc37e1093e5f651fdb2f5b94b2aab7ab2c0f83',
    '2390e52858e953886a60c414bd8fc080564beec2868984359ef4ebc4b3e2671b',
)
TA_SIGNAL_COUNTER_PARITY_REF = {
    'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_DUPLICATE_CHECK_PERF_20261108_001/RESULT.json',
    'sha256': '1532bffd0f7facd4380db3d98a9cf6aeb722d5f38d8a96f99cfb170bb131e60a',
}


def _require_ta_prefix_source_changes(old_owners: Mapping[str, str],
                                      current: Mapping[str, str]) -> set[str]:
    changed = {path for path, digest in old_owners.items() if current[path] != digest}
    numeric_changes = changed - TA_PREFIX_RUNTIME_SOURCE_CHANGES
    if numeric_changes:
        _need(numeric_changes == {TA_SIGNAL_COUNTER_SOURCE}
              and (old_owners[TA_SIGNAL_COUNTER_SOURCE], current[TA_SIGNAL_COUNTER_SOURCE])
              == TA_SIGNAL_COUNTER_SOURCE_SHA256, 'TA_UNREVIEWED_NUMERIC_SOURCE_DRIFT')
        parity = _read_ref(TA_SIGNAL_COUNTER_PARITY_REF)
        before = _read_ref(parity['before'])
        after = _read_ref(parity['after'])
        _need(parity.get('schema_version') == 'gx1_ta_duplicate_check_performance_result_v1'
              and parity.get('status') == 'PASS_METADATA_AND_FOCUSED_TA_TESTS'
              and parity.get('valid_outputs_and_error_semantics_identical') is True
              and parity.get('test_data_used') is False
              and (before.get('owner_sha256'), after.get('owner_sha256'))
              == TA_SIGNAL_COUNTER_SOURCE_SHA256
              and after.get('identical_valid_outputs_contracts_and_errors') is True
              and before.get('semantics_sha256') == after.get('semantics_sha256')
              == parity.get('semantics_sha256')
              == '4402d8dc6be0fdae3e8fadf322403f2fe40e41a1c4787ba080d7e49da68b216d',
              'TA_COUNTER_PARITY')
    return changed


def load_ta_input_receipts(*, current_source_owners: Mapping[str, str] | None = None) -> dict:
    """Read only the exact mature fold-1 chain and bind today's TA owners.

    The historical statistics retain their original Git source/policy identity.
    A supplied plan-owned map must cover every TA owner and match current bytes.
    No data array, target tape, model, normalizer fit or checkpoint is loaded.
    """
    plan = _read_ref(_fixed_ref('mature_plan'))
    mature._validate_cohort_contract(plan['cohorts'])
    receipts = _read_validated_fold1_receipts(plan)
    provenance = receipts['qualification']['proof']['source_bindings']
    lineage = _fold1_historical_source_lineage(provenance)
    current = {}
    if current_source_owners is not None:
        _need(isinstance(current_source_owners, Mapping)
              and TA_SOURCE_PATHS <= set(current_source_owners), 'TA_SOURCE_OWNER_SET')
    for relative in sorted(TA_SOURCE_PATHS):
        path = REPO / relative
        _need(path.is_file() and not path.is_symlink(), 'TA_SOURCE_PATH:' + relative)
        digest = _sha(path)
        if current_source_owners is not None:
            _need(current_source_owners[relative] == digest, 'TA_CURRENT_SOURCE:' + relative)
        current[relative] = digest
    old_owners = provenance['current_source_owners']
    changed = _require_ta_prefix_source_changes(old_owners, current)
    from gx1.contracts.entry_model_native_input_normalization_v1 import (
        project_ta_input_normalization_contract,
    )
    projected = project_ta_input_normalization_contract(
        receipts['normal']['base_normalization'],
        unobserved_units_profile=receipts['profile'])
    lineage.update(historical_source_owners=dict(old_owners),
                   current_source_owners=current, changed_runtime_owners=sorted(changed))
    receipts['lineage'] = lineage
    return {'mature_plan': plan, 'normal': receipts['normal'],
            'constructor': _read_ref(_fixed_ref('constructor')),
            'base_view': _read_ref(_fixed_ref('base_view')),
            'fold1': receipts, 'fold_ids': [FOLD1_ID],
            'provider_chain': mature._receipt_chain(),
            'historical_lineage': lineage, 'ta_source_owners': current,
            'ta_normalization': projected}


def _bind_ta_view(receipts: dict, *, max_batch_rows: int = 16,
                  dedicated_m5_side: bool = False, dedicated_non_m5_side: bool = False,
                  dedicated_tf_activity: bool = False) -> dict:
    """Bind the TA view to current sources and original fold-1 statistics."""
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    _need(receipts.get('fold_ids') == [FOLD1_ID], 'TA_FOLD1_ONLY')
    current = receipts['ta_source_owners']
    _need(set(current) == TA_SOURCE_PATHS and all(
        _sha(REPO / path) == digest for path, digest in current.items()),
        'TA_RECEIPT_SOURCE_BOOKEND')
    original = receipts['base_view']['source_bindings']
    _need(views.SOURCE_REF_KEYS <= set(original), 'BASE_VIEW_SOURCE_KEYS')
    refs = {key: original[key] for key in views.SOURCE_REF_KEYS}
    provenance = receipts['fold1']['qualification']['proof']['source_bindings']
    refs.update(input_qualification=_fixed_ref('fold1_qualification'),
                raw_input_provider_plan=provenance['provider_plan'],
                raw_gc_layout=provenance['boolean_raw_gc_layout'],
                raw_gc_transition=provenance['boolean_raw_domain_transition'],
                time_query_candidate_tape=_fixed_ref('candidate_tape'),
                time_query_mature_prefix_plan=_fixed_ref('fold1_plan'))
    paths = {key: path for key, path in TIME_QUERY_SOURCE_PATHS.items()
             if key not in ('time_query_model_source', 'time_query_loss_source')}
    paths.update(
        direction_component_source='gx1/contracts/entry_native_direction_research_v1.py',
        native_constructor_source='gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
        mtf_window_source='gx1/features/htf_features.py',
        gc_gather_source='gx1/contracts/entry_gc_mtf_history_v1.py',
        owner_profile_source='gx1/features/entry_specialist_feature_groups_v1.py',
        time_query_v2_model_source='gx1/contracts/market_state_time_query_model_v2.py',
        time_query_v2_loss_source='gx1/contracts/market_state_time_query_loss_v2.py')
    for key, path in paths.items():
        refs[key] = _ref(REPO / path, current[path])
    _need(set(refs) == views.SOURCE_REF_KEYS | views.TIME_QUERY_V2_SOURCE_REF_KEYS,
          'TA_VIEW_SOURCE_SET')
    # The candidate target tape remains a fixed ref and is verified by its own
    # target loader before training; this input-only binder reads no target data.
    for key in IMMUTABLE_VIEW_REF_KEYS - {'time_query_candidate_tape'}:
        item = refs[key]
        _need(Path(item['path']).is_file() and not Path(item['path']).is_symlink()
              and _sha(Path(item['path'])) == item['sha256'], 'TA_IMMUTABLE_VIEW_REF:' + key)
    value = views.bind_native_direction_feature_view(
        prefix_normalization=receipts['fold1']['normal'],
        prefix_normalization_ref=_fixed_ref('fold1_normalization'),
        constructor_metadata=receipts['constructor'],
        constructor_metadata_ref=_fixed_ref('constructor'), source_bindings=refs,
        max_batch_rows=max_batch_rows, full_feature_profile=True,
        full_ta_feature_profile=True, market_state_time_query_v2=True,
        dedicated_m5_side=dedicated_m5_side, dedicated_non_m5_side=dedicated_non_m5_side,
        dedicated_tf_activity=dedicated_tf_activity,
        unobserved_units_profile=receipts['fold1']['profile'])
    _need(value['prefix_normalization']['ta_base_normalization_sha256']
          == receipts['ta_normalization']['contract_sha256'], 'TA_VIEW_PROJECTION_IDENTITY')
    return value


def _qualify_plan(plan: dict, plan_ref: dict, policy: dict | None, *,
                  head: str, out_dir: Path, device: str, cap: dict | None) -> None:
    _need(isinstance(plan, dict) and plan.get('schema_version') == SCHEMA
          and plan.get('stage') == STAGE and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and isinstance(plan.get('source_base_head'), str)
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
                              plan['source_base_head'], head],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0
          and plan.get('output_root') == str(OUT_ROOT) == str(out_dir)
          and out_dir.is_absolute() and not out_dir.exists()
          and device == 'cuda'
          and plan.get('fold_ids') == FOLD_IDS and plan.get('partition') == 'fit'
          and plan.get('sample_parent_rows') == PARENT_ROWS
          and plan.get('sample_issue_ns') == ISSUE_NS
          and plan.get('pass_index') == PASS_INDEX and plan.get('seed') == SEED
          and plan.get('max_invocations') == 1
          and plan.get('max_json_bytes') == MAX_JSON_BYTES
          and plan.get('model_forwards_authorized') == 2
          and plan.get('backward_calls_authorized') == 2
          and plan.get('optimizer_steps_authorized') == 0
          and plan.get('training_authorized') is False
          and plan.get('test_data_used') is False
          and plan.get('execution') == EXECUTION
          and plan.get('limits') == LIMITS
          and plan.get('input_refs') == {key: _fixed_ref(key) for key in FIXED}
          and plan.get('expected_view_sha256_by_fold') == EXPECTED_VIEW_SHA256_BY_FOLD
          and plan.get('expected_model_contract_sha256')
              == time_query_model_contract()['contract_sha256']
          and plan.get('expected_loss_contract_sha256')
              == time_query_loss_contract()['contract_sha256'], 'PLAN_SCOPE')
    owners = plan.get('source_owners')
    _need(isinstance(owners, dict) and set(owners) == SOURCE_PATHS
          and all(isinstance(digest, str) and len(digest) == 64
                  and _sha(REPO / path) == digest for path, digest in owners.items()),
          'SOURCE_OWNERS')
    if cap is None:
        return
    _need(all(cap.get(key) == LIMITS[key] for key in ('memory_max', 'memory_high', 'swap'))
          and type(cap.get('pids')) is int and cap['pids'] <= 128
          and cap.get('class') == 'producer'
          and cap.get('execution_mode') == 'cuda_producer'
          and isinstance(cap.get('cgroup_path'), str)
          and isinstance(cap.get('lock'), dict)
          and isinstance(cap['lock'].get('scope'), str)
          and cap['lock']['scope'].startswith('/')
          and cap['lock']['scope'].endswith('.scope')
          and cap['cgroup_path'] == '/sys/fs/cgroup' + cap['lock']['scope'],
          'CAP_AND_LOCK')
    gate = policy.get(STAGE) if isinstance(policy, dict) else None
    expected = {
        'status': 'ADMITTED', 'decision': 'ADMITTED', 'plan': plan_ref,
        'execution_authorized': True, 'data_reads_authorized': True,
        'model_forwards_authorized': 2, 'backward_calls_authorized': 2,
        'optimizer_steps_authorized': 0, 'training_authorized': False,
        'test_authorized': False, 'claimed_invocations': 0,
        'completed_invocations': 0, 'max_invocations': 1,
        'output_root': str(out_dir), 'relaunch_allowed': False,
        'checkpoint_reads_authorized': False, 'broker_access_allowed': False,
        'test_data_used': False, 'memory_max': '12G', 'swap_max': '512M',
        'wall_seconds': WALL_SECONDS,
    }
    _need(isinstance(policy, dict) and policy.get('training_enabled') is False
          and policy.get('native_training_enabled') is False
          and policy.get('native_launch_authorized') is False
          and policy.get('full_val_allowed') is False
          and isinstance(gate, dict)
          and all(gate.get(key) == value for key, value in expected.items()),
          'POLICY')


def _source_bookend(plan: dict, plan_sha256: str, policy_sha256: str, head: str) -> None:
    _need(_clean_head() == head and _sha(PLAN_PATH) == plan_sha256
          and _sha(POLICY) == policy_sha256
          and all(_sha(REPO / path) == digest for path, digest in plan['source_owners'].items())
          and all(_sha(path) == digest for path, digest in FIXED.values()),
          'SOURCE_BOOKEND')


def preflight_read_only(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    """Hash/check small receipts and the 4 MiB tape; no raw feature payload/claim."""
    head = _clean_head()
    plan, ref = _checked_plan(Path(plan_path), plan_sha256)
    _qualify_plan(plan, ref, None, head=head, out_dir=Path(out_dir),
                  device='cuda', cap=None)
    receipts = _inspect_receipts()
    views = {str(fold_id): _bind_view(plan, receipts, fold_id)
             for fold_id in FOLD_IDS}
    return {'status': 'READY_NO_CLAIM_NO_RAW_FEATURE_PAYLOAD',
            'plan': ref, 'source_head': head,
            'view_sha256_by_fold': {key: view['contract_sha256']
                                    for key, view in views.items()},
            'model_contract_sha256': time_query_model_contract()['contract_sha256'],
            'loss_contract_sha256': time_query_loss_contract()['contract_sha256'],
            'source_owner_refs_checked': len(plan['source_owners']),
            'view_ref_classes': {
                'fold1': {'immutable_artifact': sorted(IMMUTABLE_VIEW_REF_KEYS),
                          'current_runtime_source': sorted(CURRENT_VIEW_REF_KEYS
                                                           | HISTORICAL_VIEW_REF_KEYS)},
                'fold2': {'immutable_artifact': sorted(IMMUTABLE_VIEW_REF_KEYS),
                          'historical_mature_source': sorted(HISTORICAL_VIEW_REF_KEYS),
                          'current_runtime_source': sorted(CURRENT_VIEW_REF_KEYS)}},
            'tape_sha256_checked': _fixed_ref('candidate_tape')['sha256'],
            'fold1_lineage': receipts['fold1']['lineage'],
            'fold2_historical_lineage': receipts['historical_lineage'],
            'claim_created': False, 'model_forwards': 0,
            'backward_calls': 0, 'optimizer_steps': 0, 'test_data_used': False}


def _preflight(plan_path: Path, plan_sha256: str, device: str,
               out_dir: Path) -> tuple[dict, dict, dict, str, dict, dict]:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_guarded_cuda_producer_execution, require_capped_lock_ancestry,
    )
    head = _clean_head()
    plan, plan_ref = _checked_plan(plan_path, plan_sha256)
    policy_ref = _ref(POLICY, _sha(POLICY))
    policy = _read_ref(policy_ref, limit=8 * 1024**2)
    cap = require_guarded_cuda_producer_execution()
    cap['lock'] = require_capped_lock_ancestry()
    _qualify_plan(plan, plan_ref, policy, head=head, out_dir=out_dir,
                  device=device, cap=cap)
    receipts = _inspect_receipts()
    for fold_id in FOLD_IDS:
        _bind_view(plan, receipts, fold_id)
    return plan, plan_ref, policy_ref, head, cap, receipts


def _gradient_summary(value: object) -> dict:
    import torch
    if value is None:
        return {'present': False, 'finite': None, 'nonzero_count': 0, 'max_abs': None}
    _need(isinstance(value, torch.Tensor), 'GRADIENT_TYPE')
    absolute = value.detach().abs()
    return {'present': True, 'finite': bool(torch.isfinite(value).all()),
            'nonzero_count': int(torch.count_nonzero(value).item()),
            'max_abs': float(absolute.max().item()) if absolute.numel() else 0.0}


def _witness(plan: dict, cap: dict, receipts: dict) -> dict:
    import numpy as np
    import torch
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.market_state_time_query_batch_v1 import (
        FEATURE_KEYS, evaluate_fit_batch,
    )
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    chain = receipts['provider_chain']
    provider_plan = chain['provider_plan']
    provider = mature._provider_module()
    info = provider.validate_metadata(provider_plan)
    prepared = mature._prepare_inputs(provider, provider_plan, info,
                                      chain['gc_layout'])
    mature._validate_axes(receipts['mature_plan']['cohorts'], prepared['axes'])
    axes = prepared['axes']
    selected = np.asarray(PARENT_ROWS, dtype=np.int64)
    issues = np.asarray(ISSUE_NS, dtype=np.int64)
    at = np.searchsorted(axes['parent_rows'], selected)
    _need(np.array_equal(axes['parent_rows'][at], selected)
          and np.array_equal(axes['decision_ns'][at], issues)
          and int(at[0]) < receipts['fold1']['cohort']['joined_fit_rows']
          < receipts['cohort2']['joined_fit_rows'],
          'EXACT_DUAL_FOLD_SAMPLE')
    for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
        axes[key].setflags(write=False)
    sequence = prepared['entry_sequence_source']
    for key in ('signal', 'times_ns', 'sample_positions'):
        sequence[key].setflags(write=False)
    for key in ('snap', 'ctx_cont', 'ctx_cat'):
        prepared['current'][key].setflags(write=False)
    bound_views = {fold_id: _bind_view(plan, receipts, fold_id)
                   for fold_id in FOLD_IDS}
    raw_by_fold = {}
    for fold_id in FOLD_IDS:
        profile = receipts['fold1']['profile'] if fold_id == FOLD1_ID else None
        raw_by_fold[fold_id] = views.gather_native_direction_feature_batch(
            bound_views[fold_id], parent_rows=axes['parent_rows'],
            bar_start_ns=axes['bar_start_ns'], decision_ns=axes['decision_ns'],
            entry_sequence_source=sequence,
            snap_x=prepared['current']['snap'],
            ctx_cont=prepared['current']['ctx_cont'],
            ctx_cat=prepared['current']['ctx_cat'],
            multi_tf_sources=prepared['cache'],
            gc_surfaces=prepared['gc']['surfaces'],
            gc_layout=prepared['gc']['layout'],
            batch_parent_rows=selected,
            unobserved_units_profile=profile)
    for raw in raw_by_fold.values():
        _need(set(raw['features']) == FEATURE_KEYS
              and raw['witness']['parent_rows'] == PARENT_ROWS
              and raw['witness']['decision_ns'] == ISSUE_NS,
              'FEATURE_ONLY_BATCH')
    mature._release_pages(provider, prepared)
    device = torch.device('cuda')
    _need(torch.cuda.is_available(), 'CUDA_AVAILABLE')
    bound = binding.load_bound_time_query_targets()
    fold_reports = {}
    for fold_id in FOLD_IDS:
        trainer._set_deterministic(SEED, device, trainer.DETERMINISTIC_FP32)
        profile = receipts['fold1']['profile'] if fold_id == FOLD1_ID else None
        normal = receipts['fold1']['normal'] if fold_id == FOLD1_ID else receipts['normal']
        view = bound_views[fold_id]
        raw = raw_by_fold[fold_id]
        model = views.build_native_direction_research_model(
            view, prefix_normalization=normal,
            constructor_metadata=receipts['constructor'], device=device,
            unobserved_units_profile=profile)
        _need(type(model) is TimeQueryMarketStateModel
              and model.feature_view_identity['contract_sha256']
                  == EXPECTED_VIEW_SHA256_BY_FOLD[str(fold_id)],
              'FRESH_MODEL:' + str(fold_id))
        model.train()
        initial_state = canonical_model_state_sha256(model.state_dict())
        tensors = {}
        for key, value in raw['features'].items():
            tensor = torch.from_numpy(value).to(device)
            if tensor.is_floating_point():
                tensor.requires_grad_(True)
            tensors[key] = tensor
        torch.cuda.reset_peak_memory_stats()
        with trainer._training_mha_fastpath_context(model.backbone), \
             trainer._training_autocast_context(), \
             trainer._training_model_finite_check_context():
            _need(not torch.backends.mha.get_fastpath_enabled(), 'CANONICAL_MHA')
            outcome = evaluate_fit_batch(model, tensors, bound, selected, issues,
                                         fold_id=fold_id, pass_index=PASS_INDEX,
                                         seed=SEED)
        _need(outcome['count'] == 2 and outcome['skip_update'] is False
              and outcome['query_k'][0].tolist() == EXPECTED_QUERY_K
              and outcome['allowed'][0].tolist() == EXPECTED_ALLOWED
              and outcome['per_tf_counts'] == {'M5': 1, 'M15': 1,
                                               'H1': 0, 'H4': 0, 'D1': 0}
              and outcome['model_contract_sha256']
                  == time_query_model_contract()['contract_sha256']
              and outcome['loss_contract_sha256']
                  == time_query_loss_contract()['contract_sha256']
              and bool(torch.isfinite(outcome['loss'])),
              'TARGET_LOSS_BINDING:' + str(fold_id))
        outcome['loss'].backward()
        gradients = {key: ({'kind': 'integer_categorical', 'applicable': False}
                           if not tensor.is_floating_point()
                           else _gradient_summary(tensor.grad))
                     for key, tensor in tensors.items()}
        head = _gradient_summary(model.query_head.weight.grad)
        projection = _gradient_summary(model.query_projection[0].weight.grad)
        backbone = _gradient_summary(model.backbone.entry_q_joint_in.weight.grad)
        _need(all(item['present'] and item['finite'] and item['nonzero_count'] > 0
                  for item in gradients.values() if item.get('applicable') is not False)
              and all(item['present'] and item['finite'] and item['nonzero_count'] > 0
                      for item in (head, projection, backbone)),
              'BACKWARD_CONSUMPTION:' + str(fold_id))
        final_state = canonical_model_state_sha256(model.state_dict())
        _need(initial_state == final_state and all(p.grad is not None
              or not p.requires_grad for p in (model.query_head.weight,
                                              model.backbone.entry_q_joint_in.weight)),
              'NO_WEIGHT_UPDATE:' + str(fold_id))
        fold_reports[str(fold_id)] = {
            'fold_id': fold_id, 'pass_index': PASS_INDEX, 'seed': SEED,
            'sample_parent_rows': PARENT_ROWS, 'sample_issue_ns': ISSUE_NS,
            'query_k': outcome['query_k'].tolist(),
            'allowed': outcome['allowed'].tolist(),
            'eligible': outcome['eligible'].tolist(),
            'mature': outcome['mature'].tolist(),
            'per_tf_counts': outcome['per_tf_counts'],
            'informative_count': outcome['count'],
            'loss': float(outcome['loss'].detach().item()),
            'view_sha256': view['contract_sha256'],
            'model_contract_sha256': outcome['model_contract_sha256'],
            'loss_contract_sha256': outcome['loss_contract_sha256'],
            'normalization': _fixed_ref('fold1_normalization' if fold_id == FOLD1_ID
                                        else 'fold2_normalization'),
            'feature_witness': raw['witness'],
            'feature_shapes': {key: list(value.shape) for key, value in tensors.items()},
            'input_gradients': gradients, 'query_head_gradient': head,
            'query_projection_gradient': projection, 'backbone_gradient': backbone,
            'initial_model_state_sha256': initial_state,
            'final_model_state_sha256': final_state,
            'cuda_peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
            'cuda_peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
            'model_forwards': 1, 'backward_calls': 1,
            'optimizer_steps': 0, 'ema_updates': 0,
            'scheduler_steps': 0, 'checkpoint_loaded': False,
            'test_data_used': False,
        }
        del outcome, gradients, tensors, model
        torch.cuda.empty_cache()
    provider.validate_metadata(provider_plan)
    return {
        'decision': 'PASS_DUAL_PROFILE_TIME_QUERY_MODEL_LOSS_BACKWARD_NO_UPDATE',
        'scope_limit': 'Two fresh TRAIN fit models on one exact row each show local coupling only; no native fit, held-out score, prediction value or trading economics.',
        'fold_ids': FOLD_IDS, 'fold_results': fold_reports,
        'candidate_tape': _fixed_ref('candidate_tape'),
        'cgroup_path': cap['cgroup_path'],
        'model_forwards': 2, 'backward_calls': 2,
        'optimizer_steps': 0, 'ema_updates': 0,
        'scheduler_steps': 0, 'checkpoint_loaded': False,
        'test_data_used': False,
    }


@contextmanager
def _bounded_worker(started: float):
    previous = signal.getsignal(signal.SIGALRM)
    _need(previous == signal.SIG_DFL
          and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
          'TIMER_OWNER')
    seconds = WALL_SECONDS - (time.monotonic() - started) - 15.0
    _need(seconds > 0, 'WALL_BUDGET')

    def expire(_signum, _frame):
        raise TimeoutError('TIME_QUERY_ZERO_STEP_WORKER_WALL_LIMIT')

    signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run(plan_path: Path, plan_sha256: str, device: str, out_dir: Path) -> dict:
    """Admitted one-use CUDA entry point; no model fit or optimizer step."""
    from gx1.contracts.immutable_event_authority_v1 import _fsync_directory
    began = time.monotonic()
    plan, plan_ref, policy_ref, head, cap, receipts = _preflight(
        Path(plan_path), plan_sha256, device, Path(out_dir))
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    claim = None
    try:
        _fsync_directory(root.parent)
        claim = _write(root, 'CLAIM.json', {
            'schema_version': 'gx1_market_state_time_query_zero_step_claim_v1',
            'plan': plan_ref, 'policy': policy_ref, 'source_head': head,
            'fold_ids': FOLD_IDS, 'sample_parent_rows': PARENT_ROWS,
            'invocation': 1, 'test_data_used': False}, limit=MAX_JSON_BYTES)
        with _bounded_worker(began):
            result = _witness(plan, cap, receipts)
        _need(result.get('fold_ids') == FOLD_IDS
              and set(result.get('fold_results', {})) == {str(item) for item in FOLD_IDS}
              and result.get('model_forwards') == result.get('backward_calls') == 2
              and result.get('optimizer_steps') == 0,
              'DUAL_WITNESS_COMPLETE')
        _source_bookend(plan, plan_sha256, policy_ref['sha256'], head)
        result_ref = _write(root, 'RESULT.json', {
            'schema_version': 'gx1_market_state_time_query_zero_step_result_v1',
            'status': 'COMPLETE', 'plan': plan_ref, 'claim': claim,
            'policy': policy_ref, 'source_head': head,
            'elapsed_seconds': time.monotonic() - began,
            **result}, limit=MAX_JSON_BYTES)
        terminal = _write(root, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_time_query_zero_step_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result_ref,
            'model_forwards': 2, 'backward_calls': 2,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True,
            'output_inventory': ['CLAIM.json', 'RESULT.json', 'TERMINAL.json'],
        }, limit=MAX_JSON_BYTES)
        return {'result': result_ref, 'terminal': terminal}
    except BaseException as exc:
        failed = _write(root, 'FAILED.json', {
            'schema_version': 'gx1_market_state_time_query_zero_step_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic() - began,
            'no_relaunch_under_this_plan': True,
        }, limit=MAX_JSON_BYTES)
        _write(root, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_time_query_zero_step_terminal_v1',
            'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'failure': failed,
            'model_forwards_upper_bound': 2, 'backward_calls_upper_bound': 2,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True,
            'output_inventory': (['CLAIM.json', 'FAILED.json', 'TERMINAL.json']
                                 if claim else ['FAILED.json', 'TERMINAL.json']),
        }, limit=MAX_JSON_BYTES)
        raise
