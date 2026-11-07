"""One admitted, frozen TRAIN fold-1 M5 k=1 diagnostic; no model fitting."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Mapping

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'market_state_fold1_paired_diagnostic_20261107_002'
SCHEMA = 'gx1_market_state_fold1_paired_diagnostic_plan_v1'
PLAN_PATH = REPO / 'configs/research/MARKET_STATE_FOLD1_PAIRED_DIAGNOSTIC_20261107_002.json'
POLICY_PATH = REPO / 'NEXT_RUN_POLICY.json'
OUT_ROOT = RUNS / 'MARKET_STATE_FOLD1_PAIRED_DIAGNOSTIC_20261107_002'
FIT_SOURCE_COMMIT = '79d4fb3de0bf9940d3392f36f5ae6834d7fb7e77'
MODEL_DIGEST = 'de8e8665a0ff31c0ec50950758597d2768d7d241d7433a735513b34164b90ac9'
SCORE_ROWS = 17753
SIDE_ROWS = 6096
BATCH_SIZE = 16
FORWARDS = 2 * ((SCORE_ROWS + BATCH_SIZE - 1) // BATCH_SIZE)
WALL_SECONDS = 3600
MAX_JSON_BYTES = 1024 * 1024
CHANGED_EXECUTION_SOURCES = frozenset({
    'gx1/audit/score_market_state_time_query_fold1_oof_v1.py',
    'gx1/audit/verify_market_state_time_query_zero_step_v1.py',
    'gx1/scripts/validate_entry_model_native_technical_checkpoint_v1.py',
    'scripts/gx1_capped_run.sh',
})
WRAPPER_SOURCE = 'gx1/audit/run_market_state_fold1_paired_diagnostic_v1.py'
FIT_ROOT = RUNS / 'MARKET_STATE_TIME_QUERY_FOLD1_FIT_20261107_002'
FOLD1_ROOT = RUNS / 'MARKET_STATE_MATURE_FOLD1_NORMALIZATION_20261107_001'
FOLD2_ROOT = RUNS / 'MARKET_STATE_MATURE_PREFIX_NORMALIZATION_20261107_001'
INPUT_REFS = {
    'fit_recipe': {'path': str(FIT_ROOT / 'RECIPE.json'),
                   'sha256': 'c1ee668a388d0be6b73ce94efc2d3827958c410bb0f51ebcb6cda9b14f0a854c'},
    'fit_result': {'path': str(FIT_ROOT / 'RUNTIME/prefix-fit/window-0001/RESULT.json'),
                   'sha256': '08172fc2c17f8eaf4f0457e38bdd3aa750bfbbf9ae719cb587aabd4d93d21254'},
    'fit_receipt': {'path': str(FIT_ROOT / 'RUNTIME/receipts/invocation-0001.json'),
                    'sha256': '8a2b638a49dfcf32558f361ee968c6c60b6d768d40d916fc87abca620b96b536'},
    'frozen_oof': {'path': str(FIT_ROOT / 'RUNTIME/prefix-fit/window-0001/OOF.json'),
                   'sha256': 'ff6e3099c5d7e59ec7f917a0a0e734b1017980b801273d4f3ecfcc444da64f11'},
    'fit_checkpoint': {'path': str(FIT_ROOT / 'RUNTIME/prefix-fit/states/FINAL_FOLD1_FIT_STATE.pt'),
                       'sha256': '0bc00ee262aedf7083d62a04b841d38f10067c621dede4fdecac3e3fb1ffccc5'},
    'fold1_qualification': {'path': str(FOLD1_ROOT / 'QUALIFICATION_FOLD_1.json'),
                            'sha256': '0fad1ac1476205e0485207da4b5e6be6b0de5585a8cb4170f87679c63580fb94'},
    'fold2_qualification': {'path': str(FOLD2_ROOT / 'QUALIFICATION_FOLD_2.json'),
                            'sha256': 'c271f952f8befef3485de4ed5d5460791c8da2b608f684b7717cf78a64e9707b'},
    'fold1_units_profile': {'path': str(FOLD1_ROOT / 'PREFIX_NORMALIZATION_FOLD_1.json'),
                            'sha256': 'e24a2b8d1ae09f6f10c2de7a7097353b52aa2f2118287535e242778272022d54'},
}


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('MARKET_STATE_FOLD1_PAIRED_DIAGNOSTIC_' + code)


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _bound(ref: Mapping[str, str], *, json_file: bool = True) -> dict | None:
    _need(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'REF_SCHEMA')
    path = Path(ref['path'])
    _need(path.is_absolute() and path.is_file() and not path.is_symlink()
          and len(ref['sha256']) == 64 and _sha(path) == ref['sha256'],
          'REF_HASH:' + str(path))
    if not json_file:
        return None
    _need(path.stat().st_size <= 8 * 1024**2, 'JSON_SIZE:' + str(path))
    result = json.loads(path.read_text(encoding='utf-8'))
    _need(isinstance(result, dict), 'JSON_OBJECT:' + str(path))
    return result


def _clean_head() -> str:
    _need(Path.cwd().resolve() == REPO
          and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
          == 'work/gx1-current'
          and subprocess.check_output(
              ['git', 'status', '--porcelain=v1', '--untracked-files=all'], text=True) == '',
          'CLEAN_SOURCE')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def _source_owners(plan: dict, recipe: dict) -> None:
    old = recipe.get('source_owners')
    current = plan.get('source_owners')
    _need(isinstance(old, dict) and len(old) == 71 and isinstance(current, dict)
          and set(current) == set(old) | {WRAPPER_SOURCE}, 'SOURCE_SET')
    for name, digest in current.items():
        _need(isinstance(name, str) and isinstance(digest, str) and len(digest) == 64
              and (REPO / name).is_file() and _sha(REPO / name) == digest,
              'CURRENT_SOURCE:' + name)
        if name in old:
            ref = old[name]
            _need(isinstance(ref, dict)
                  and ref == {'path': str(REPO / name), 'sha256': ref.get('sha256')}
                  and (name in CHANGED_EXECUTION_SOURCES or digest == ref['sha256']),
                  'FIT_SOURCE_DRIFT:' + name)


def _input_lineage(plan: dict) -> dict:
    _need(plan.get('input_refs') == INPUT_REFS, 'INPUT_REFS')
    payloads = {key: _bound(ref, json_file=(key != 'fit_checkpoint'))
                for key, ref in INPUT_REFS.items()}
    recipe, result, receipt, oof = (payloads[key] for key in
                                    ('fit_recipe', 'fit_result', 'fit_receipt', 'frozen_oof'))
    _need(result.get('outcome') == receipt.get('outcome') == 'COMPLETE'
          and result.get('source_commit') == plan['fit_source_commit']
          and result.get('full_state') == INPUT_REFS['fit_checkpoint']
          and result.get('oof_measurement') == INPUT_REFS['frozen_oof']
          and result.get('recipe') == INPUT_REFS['fit_recipe']
          and result.get('test_data_used') is False
          and receipt.get('guard_decision') == 'PASS'
          and receipt.get('invocation_number') == 1
          and receipt.get('trainer_guard_exit_code') == 0
          and receipt.get('test_data_used') is False
          and oof.get('checkpoint') == INPUT_REFS['fit_checkpoint']
          and oof.get('model_state_sha256') == plan['model_state_sha256']
          and oof.get('fold_id') == 1
          and oof.get('score_rows') == SCORE_ROWS
          and oof.get('primary_M5_k1', {}).get('certified_side', {}).get('count') == SIDE_ROWS
          and oof.get('test_data_used') is False
          and recipe.get('source_repo') == str(REPO), 'FIT_LINEAGE')
    q1 = payloads['fold1_qualification']
    q2 = payloads['fold2_qualification']
    profile = q1.get('unobserved_units_profile')
    _need(isinstance(profile, dict) and len(profile.get('fields', [])) == 51
          and payloads['fold1_units_profile'].get('unobserved_units_profile') == profile,
          'FOLD1_PROFILE')
    for surface, population_rows in (('signal', 29782), ('ctx_cont', 29574),
                                     ('gc_mtf_d1', 360)):
        spec = profile['surfaces'][surface]
        support = q1['support'][surface]
        _need(len(spec['columns']) == 51 and support['population_rows'] == population_rows,
              'FIT_SUPPORT:' + surface)
        for column in spec['columns']:
            index = column['value_index']
            name = spec['value_name_prefix'] + column['field']
            _need(support['field_names'][index] == name
                  and support['observed_counts'][index] == 0,
                  'FIT_ZERO:' + surface)
    later = q2['support']['ctx_cont']
    _need(later['population_rows'] == 47338, 'LATER_POPULATION')
    for column in profile['surfaces']['ctx_cont']['columns']:
        index = column['value_index']
        _need(later['field_names'][index] == column['field']
              and 550 <= later['observed_counts'][index] <= 7916,
              'LATER_SUPPORT')
    _source_owners(plan, recipe)
    return payloads


def _policy_gate(policy: dict, plan_ref: dict, plan: dict) -> None:
    expected = {
        'status': 'ADMITTED', 'decision': 'ADMITTED', 'plan': plan_ref,
        'output_root': str(OUT_ROOT), 'execution_authorized': True,
        'data_reads_authorized': True, 'checkpoint_reads_authorized': True,
        'model_forwards_authorized': FORWARDS, 'optimizer_steps_authorized': 0,
        'training_authorized': False, 'test_authorized': False,
        'broker_access_allowed': False, 'test_data_used': False,
        'claimed_invocations': 0, 'completed_invocations': 0,
        'max_invocations': 1, 'relaunch_allowed': False,
        'memory_max': '12G', 'swap_max': '512M', 'wall_seconds': WALL_SECONDS,
    }
    gate = policy.get(STAGE)
    _need(all(policy.get(key) is False for key in
              ('training_enabled', 'native_training_enabled',
               'native_launch_authorized', 'full_epoch_training_allowed',
               'full_val_allowed'))
          and isinstance(gate, dict)
          and all(gate.get(key) == value for key, value in expected.items())
          and plan.get('max_invocations') == 1, 'POLICY')


def _terminal(status: str, plan_ref: dict, claim_ref: dict | None,
              outcome_ref: dict, *, forwards: int) -> dict:
    _need(status in ('COMPLETE', 'FAILED')
          and (status == 'FAILED' or claim_ref is not None)
          and type(forwards) is int and 0 <= forwards <= FORWARDS, 'TERMINAL_INPUT')
    value = {
        'schema_version': 'gx1_market_state_fold1_paired_diagnostic_terminal_v1',
        'status': status, 'exit_code': 0 if status == 'COMPLETE' else 1,
        'plan': plan_ref, 'claim': claim_ref,
        'result' if status == 'COMPLETE' else 'failure': outcome_ref,
        'model_forwards' if status == 'COMPLETE' else 'model_forwards_upper_bound': forwards,
        'optimizer_steps': 0, 'test_data_used': False,
        'no_relaunch_under_this_plan': True,
        'output_inventory': (['CLAIM.json', 'RESULT.json', 'TERMINAL.json']
                             if status == 'COMPLETE'
                             else (['CLAIM.json', 'FAILED.json', 'TERMINAL.json']
                                   if claim_ref is not None else ['FAILED.json', 'TERMINAL.json'])),
    }
    return value


def _qualify(plan_path: Path, plan_sha256: str, out_dir: Path) -> tuple:
    head = _clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT and not out_dir.exists()
          and not out_dir.is_symlink() and OUT_ROOT.parent.is_dir(), 'PATH_SCOPE')
    plan_ref = {'path': str(PLAN_PATH), 'sha256': plan_sha256}
    plan = _bound(plan_ref)
    _need(set(plan) == {
        'schema_version', 'stage', 'source_base_head', 'source_owners',
        'input_refs', 'fit_source_commit', 'model_state_sha256', 'output_root',
        'partition', 'fold_id', 'score_rows', 'certified_side_rows', 'batch_size',
        'max_wall_seconds', 'max_invocations', 'model_forwards_authorized',
        'optimizer_steps_authorized', 'training_authorized', 'test_data_used',
    } and plan['schema_version'] == SCHEMA and plan['stage'] == STAGE
          and plan['source_base_head'] != head
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
                              plan['source_base_head'], head],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0
          and plan['fit_source_commit'] == FIT_SOURCE_COMMIT
          and plan['model_state_sha256'] == MODEL_DIGEST
          and plan['output_root'] == str(OUT_ROOT)
          and plan['partition'] == 'TRAIN_score_fold1'
          and plan['fold_id'] == 1 and plan['score_rows'] == SCORE_ROWS
          and plan['certified_side_rows'] == SIDE_ROWS
          and plan['batch_size'] == BATCH_SIZE
          and plan['max_wall_seconds'] == WALL_SECONDS
          and plan['max_invocations'] == 1
          and plan['model_forwards_authorized'] == FORWARDS
          and plan['optimizer_steps_authorized'] == 0
          and plan['training_authorized'] is False
          and plan['test_data_used'] is False, 'PLAN_SCOPE')
    payloads = _input_lineage(plan)
    policy_ref = {'path': str(POLICY_PATH), 'sha256': _sha(POLICY_PATH)}
    policy = _bound(policy_ref)
    _policy_gate(policy, plan_ref, plan)
    # The archived zero-step receipt chain binds the old capped-runner SHA.
    # A separately reviewed current runner is admissible only for this diagnostic.
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.audit.score_market_state_time_query_fold1_oof_v1 import VIEW_SHA256, ZERO_STEP_PLAN
    reviewed_runner = {'path': str(REPO / 'scripts/gx1_capped_run.sh'),
                       'sha256': plan['source_owners']['scripts/gx1_capped_run.sh']}
    receipts = zero._inspect_receipts(reviewed_runner_ref=reviewed_runner)
    view = zero._bind_view(zero._read_ref(ZERO_STEP_PLAN), receipts, 1)
    _need(view['contract_sha256'] == VIEW_SHA256, 'PRECLAIM_VIEW_IDENTITY')
    return plan, plan_ref, policy_ref, head, payloads


def preflight_read_only(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    """Validate actual immutable references and both report paths before CLAIM."""
    plan, plan_ref, policy_ref, head, _ = _qualify(
        Path(plan_path), plan_sha256, Path(out_dir))
    fake = {'path': str(OUT_ROOT / 'SYNTHETIC_ONLY'), 'sha256': '0' * 64}
    success = _terminal('COMPLETE', plan_ref, fake, fake, forwards=FORWARDS)
    failure = _terminal('FAILED', plan_ref, fake, fake, forwards=FORWARDS)
    _need(success['result'] == fake and success['exit_code'] == 0
          and failure['failure'] == fake and failure['exit_code'] == 1
          and not OUT_ROOT.exists(), 'REPORT_PATH_PREFLIGHT')
    return {'status': 'READY_NO_CLAIM_NO_MODEL',
            'plan': plan_ref, 'policy': policy_ref, 'source_head': head,
            'source_owner_refs_checked': len(plan['source_owners']),
            'input_refs_checked': sorted(INPUT_REFS),
            'score_rows': SCORE_ROWS, 'certified_side_rows': SIDE_ROWS,
            'model_forwards': 0, 'optimizer_steps': 0,
            'claim_created': False, 'test_data_used': False}


def _bookend(plan: dict, plan_ref: dict, policy_ref: dict, head: str) -> None:
    _need(_clean_head() == head
          and _sha(PLAN_PATH) == plan_ref['sha256']
          and _sha(POLICY_PATH) == policy_ref['sha256']
          and all(_sha(REPO / name) == digest
                  for name, digest in plan['source_owners'].items())
          and all(_sha(Path(ref['path'])) == ref['sha256']
                  for ref in INPUT_REFS.values()), 'BOOKEND')


def _write(name: str, value: dict) -> dict:
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(len(encoded) <= MAX_JSON_BYTES and json.loads(encoded) == value, 'REPORT_BYTES')
    path = OUT_ROOT / name
    _publish_bytes(path, encoded)
    return {'path': str(path), 'sha256': hashlib.sha256(encoded).hexdigest()}


@contextmanager
def _bounded(started: float):
    old_handler = signal.getsignal(signal.SIGALRM)
    _need(old_handler == signal.SIG_DFL
          and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'TIMER_OWNER')
    remaining = WALL_SECONDS - (time.monotonic() - started) - 15.0
    _need(remaining > 0, 'WALL_BUDGET')
    def expire(_signum, _frame):
        raise TimeoutError('MARKET_STATE_FOLD1_PAIRED_DIAGNOSTIC_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def run(plan_path: Path, plan_sha256: str, device: str, out_dir: Path) -> dict:
    """One capped CUDA producer call; the new output root consumes the plan."""
    import torch
    from gx1.audit.score_market_state_time_query_fold1_oof_v1 import (
        diagnose_fold1_from_checkpoint,
    )
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_lock_ancestry, require_guarded_cuda_producer_execution,
    )
    from gx1.contracts.immutable_event_authority_v1 import _fsync_directory

    started = time.monotonic()
    _need(device == 'cuda' and not torch.cuda.is_initialized(), 'EXPLICIT_CUDA')
    plan, plan_ref, policy_ref, head, _ = _qualify(
        Path(plan_path), plan_sha256, Path(out_dir))
    cap = require_guarded_cuda_producer_execution()
    lock = require_capped_lock_ancestry()
    _need(cap.get('cgroup_path') == '/sys/fs/cgroup' + lock.get('scope', '')
          and cap.get('memory_max') == 12 * 1024**3
          and cap.get('memory_high') == 12 * 1024**3
          and cap.get('swap') == 512 * 1024**2
          and os.environ.get('GX1_TRAINER_GPU_MAX_POWER_LIMIT_W') == '250'
          and os.environ.get('GX1_TRAINER_GPU_MAX_POWER_DRAW_W') == '170',
          'CAPPED_SIGNED_GUARD')
    OUT_ROOT.mkdir(exist_ok=False)
    claim = None
    try:
        _fsync_directory(OUT_ROOT.parent)
        claim = _write('CLAIM.json', {
            'schema_version': 'gx1_market_state_fold1_paired_diagnostic_claim_v1',
            'plan': plan_ref, 'policy': policy_ref, 'source_head': head,
            'execution': cap, 'lock': lock, 'invocation': 1,
            'test_data_used': False,
        })
        with _bounded(started):
            diagnostic = diagnose_fold1_from_checkpoint(
                fit_checkpoint_ref=INPUT_REFS['fit_checkpoint'],
                expected_model_state_sha256=MODEL_DIGEST,
                source_commit=FIT_SOURCE_COMMIT, execution_commit=head,
                frozen_oof_ref=INPUT_REFS['frozen_oof'],
                reviewed_runner_ref={'path': str(REPO / 'scripts/gx1_capped_run.sh'),
                                     'sha256': plan['source_owners']['scripts/gx1_capped_run.sh']},
                device=torch.device('cuda', 0),
                check_deadline=lambda: _need(
                    time.monotonic() - started < WALL_SECONDS - 15, 'WALL_LIMIT'),
            )
        _need(diagnostic.get('score_rows') == SCORE_ROWS
              and diagnostic.get('certified_side_rows') == SIDE_ROWS
              and diagnostic.get('model_forwards') == FORWARDS
              and diagnostic.get('test_data_used') is False, 'DIAGNOSTIC_COUNTS')
        _bookend(plan, plan_ref, policy_ref, head)
        result = _write('RESULT.json', {
            'schema_version': 'gx1_market_state_fold1_paired_diagnostic_result_v1',
            'status': 'COMPLETE', 'plan': plan_ref, 'claim': claim,
            'policy': policy_ref, 'source_head': head,
            'elapsed_seconds': time.monotonic() - started,
            'diagnostic': diagnostic, 'optimizer_steps': 0,
            'test_data_used': False, 'direction_value_qualified': False,
            'economic_edge_qualified': False,
        })
        terminal = _write('TERMINAL.json', _terminal(
            'COMPLETE', plan_ref, claim, result, forwards=FORWARDS))
        return {'result': result, 'terminal': terminal}
    except BaseException as exc:
        failure = _write('FAILED.json', {
            'schema_version': 'gx1_market_state_fold1_paired_diagnostic_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic() - started,
            'no_relaunch_under_this_plan': True,
        })
        _write('TERMINAL.json', _terminal(
            'FAILED', plan_ref, claim, failure, forwards=FORWARDS))
        raise
