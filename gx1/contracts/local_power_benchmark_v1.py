"""Source-bound admission and live receipt checks for the matched 160/200 W smoke comparison.

Standard-library only: usable before heavy imports. The canonical launcher and
capped guard remain execution owners; parsing never changes physical power.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import re

GPU_UUID='GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29'
FIXED={
 'schema_version':'gx1_operator_power_benchmark_scope_draft_v1',
 'gpu_uuid':GPU_UUID,'baseline_power_limit_w':160,'target_power_limit_w':200,
 'draw_stop_w':210,'core_stop_c':65,'memory_junction_stop_c':80,'vram_stop_mib':12288,
 'profile':'smoke','precision_policy':'experimental_fp32_3090_no_uninitialized_fill',
 'subsample_rows':512,'batch_size':8,'grad_accum_steps':1,'epochs':1,'max_optimizer_steps':64,
 'baseline_restore_required':True,
 'authority':{'short_power_comparison':True,'candidate_continuation':False,'test_access':False,'promotion':False,'permanent_power_change':False},
}
VARIABLE={'scope_id','operator_action_id','created_utc','expires_utc','source_commit','recipe_sha256','baseline_reference_report_sha256','run_id'}
UUID_RE=re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z')

def _utc(value):
 if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',value):raise ValueError('Canonical UTC timestamp required')
 return datetime.strptime(value,'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)

def validate_scope(value, *, now_utc, expected_source_commit, expected_recipe_sha256, expected_baseline_report_sha256):
 if not isinstance(value,dict) or set(value)!=set(FIXED)|VARIABLE:raise ValueError('Power scope fields differ from the first-point contract')
 target=value['target_power_limit_w']
 if type(target) is not int or target not in (160,200):raise ValueError('Only matched 160 W reference or 200 W treatment is allowed')
 fixed={**FIXED,'target_power_limit_w':target,'draw_stop_w':target+10}
 for key,expected in fixed.items():
  if type(value[key]) is not type(expected) or value[key]!=expected:raise ValueError('Power scope fixed value differs: '+key)
 if any(type(v) is not bool for v in value['authority'].values()):raise ValueError('Power authority fields must be explicit booleans')
 for key in ('scope_id','operator_action_id'):
  if not isinstance(value[key],str) or not UUID_RE.fullmatch(value[key]):raise ValueError('Explicit scope/operator action identity required')
 if value['scope_id']==value['operator_action_id']:raise ValueError('Scope identity is not the operator action receipt')
 for key,length,expected in (('source_commit',40,expected_source_commit),('recipe_sha256',64,expected_recipe_sha256),('baseline_reference_report_sha256',64,expected_baseline_report_sha256)):
  v=value[key]
  if not isinstance(v,str) or len(v)!=length or any(c not in '0123456789abcdef' for c in v) or v!=expected:raise ValueError('Power benchmark binding mismatch: '+key)
 if not isinstance(value['run_id'],str) or not re.fullmatch(r'[A-Z0-9_]{1,128}',value['run_id']):raise ValueError('Exact benchmark run identity required')
 if not isinstance(now_utc,datetime) or now_utc.tzinfo is None or now_utc.utcoffset()!=timedelta(0):raise ValueError('Verified current UTC time required')
 created,expires=_utc(value['created_utc']),_utc(value['expires_utc'])
 if not timedelta(0)<expires-created<=timedelta(minutes=30):raise ValueError('Power authorization must be bounded to at most 30 minutes')
 if created>now_utc+timedelta(seconds=2) or now_utc>=expires:raise ValueError('Power scope is future-dated or expired')
 return {'scope':deepcopy(value),'validation_only':True,'operator_action_verified':False,'source_and_recipe_content_verified':False,'physical_power_changed':False,'linux_guard_integrated':False,'windows_keeper_integrated':False,'required_runtime_closure':'Close authorization durably before restoring and independently verifying 160 W; expiry, failure and recovery cannot reuse high-power scope.'}


def require_exact_observed_treatment(scope, *, observed_gpu_uuid, observed_power_limit_w):
 """Treatment check only; signature/freshness belongs to existing telemetry."""
 if observed_gpu_uuid!=scope['gpu_uuid'] or isinstance(observed_power_limit_w,bool) or not isinstance(observed_power_limit_w,(int,float)) or observed_power_limit_w!=scope['target_power_limit_w']:
  raise ValueError('Observed configured watts/UUID do not match the operator treatment')


# Runtime paths are fixed by the Windows owner, never ambient environment.
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import subprocess

WINDOWS_SCOPE_ROOT = Path('/mnt/c/ProgramData/GX1/GpuPowerLimit/Benchmarks')
RECEIPT_SCHEMA = 'gx1_operator_power_benchmark_keeper_receipt_draft_v1'
RECEIPT_KEYS = frozenset(('schema_version','scope_id','operator_action_id','scope_sha256','gpu_uuid',
    'source_commit','recipe_sha256','run_id','phase','reason','expires_utc','observed_utc',
    'keeper_pid','requested_power_limit_w','baseline_restoration','training_launch_authority'))
MAX_RECEIPT_AGE_SECONDS = 15.0


def _unique_object(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('Duplicate JSON field: '+key)
        result[key]=value
    return result


def _read_json(path, expected_sha256=None):
    path=Path(path)
    if not path.is_absolute() or path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Absolute regular benchmark artifact required')
    if path.stat().st_size>4*1024*1024:raise ValueError('Benchmark metadata exceeds byte bound')
    raw=path.read_bytes()
    if expected_sha256 is not None and sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Benchmark artifact digest changed')
    value=json.loads(raw,object_pairs_hook=_unique_object,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON value: '+x)))
    if not isinstance(value,dict):raise ValueError('Benchmark artifact must be an object')
    return value


def _require_scope_path(scope_path, scope, *, scope_root=WINDOWS_SCOPE_ROOT):
    path=Path(scope_path);root=Path(scope_root)
    if not isinstance(scope.get('scope_id'),str) or not UUID_RE.fullmatch(scope['scope_id']):
        raise ValueError('Canonical operator scope ID required')
    if path!=root/scope['scope_id']/'scope.json':raise ValueError('Benchmark scope path does not match its owner/ID')
    for item in (root,path.parent,path):
        if item.is_symlink():raise ValueError('Benchmark scope cannot traverse a symlink')
    return path


def require_benchmark_recipe_geometry(scope, recipe):
    """Additional bounded experiment check; the canonical recipe owner still validates all fields."""
    if (recipe.get('profile')!='smoke' or recipe.get('run_id')!=scope['run_id']
        or recipe.get('source_commit')!=scope['source_commit']):
        raise ValueError('Power scope differs from the exact smoke recipe')
    cli=recipe['trainer_cli']
    for key,value in {'execution_tier':'canonical','device':'cuda',
        'precision_policy':FIXED['precision_policy'],'subsample_rows':512,
        'batch_size':8,'grad_accum_steps':1,'epochs':1,'num_workers':0,'train_time_window':None}.items():
        if type(cli.get(key)) is not type(value) or cli.get(key)!=value:
            raise ValueError('Power benchmark recipe geometry differs: '+key)


def require_benchmark_command(command, *, recipe_path, recipe_sha256, recipe, repo):
    args=list(command);repo=Path(repo)
    if args[:3]!=[str(repo/'.venv/bin/python'),'-m','gx1.models.entry_v10.entry_v10_ctx_train_v3']:
        raise ValueError('Power benchmark requires the exact canonical trainer module')
    def one(flag):
        if args.count(flag)!=1:raise ValueError('Power benchmark requires exactly one '+flag)
        i=args.index(flag)
        if i+1>=len(args) or args[i+1].startswith('--'):raise ValueError('Missing benchmark command value: '+flag)
        return args[i+1]
    expected={'--profile':'smoke','--execution-tier':'canonical','--device':'cuda',
        '--precision-policy':FIXED['precision_policy'],'--batch_size':'8','--epochs':'1',
        '--grad-accum-steps':'1','--subsample-rows':'512','--run-id':recipe['run_id'],
        '--recipe-audit-json':str(recipe_path),'--recipe-audit-sha256':recipe_sha256,
        '--out_bundle_dir':recipe['out_bundle_dir']}
    for argument in args[3:]:
        if not argument.startswith('--'):
            continue
        stem = argument.split('=', 1)[0]
        if '=' in argument or any(flag.startswith(stem) and flag != stem for flag in expected):
            raise ValueError('Benchmark command requires unabbreviated separate flag values')
    for flag,value in expected.items():
        if one(flag)!=value:raise ValueError('Power scope differs from actual trainer command: '+flag)
    if args.count('--train')!=1 or any(x.startswith('--candidate-') for x in args):
        raise ValueError('Power scope cannot authorize candidate continuation')


def _native_utc(raw):
    if not isinstance(raw,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{7}Z',raw):
        raise ValueError('Native UTC receipt timestamp required')
    # Windows emits seven fractional digits; CPython 3.10 accepts six.
    return datetime.fromisoformat(raw[:-2]+'+00:00')


def require_active_keeper_receipt(scope, scope_sha256, receipt, *, now_utc,
                                  expected_keeper_pid=None):
    if not isinstance(now_utc,datetime) or now_utc.tzinfo is None or now_utc.utcoffset()!=timedelta(0):
        raise ValueError('Verified current UTC time required')
    if not isinstance(receipt,dict) or set(receipt)!=RECEIPT_KEYS:
        raise ValueError('Exact keeper receipt fields required')
    fixed={'schema_version':RECEIPT_SCHEMA,'scope_id':scope['scope_id'],
        'operator_action_id':scope['operator_action_id'],'scope_sha256':scope_sha256,
        'gpu_uuid':scope['gpu_uuid'],'source_commit':scope['source_commit'],
        'recipe_sha256':scope['recipe_sha256'],'run_id':scope['run_id'],
        'phase':'active','reason':'','expires_utc':scope['expires_utc'],
        'requested_power_limit_w':scope['target_power_limit_w'],'baseline_restoration':None,'training_launch_authority':False}
    for key,value in fixed.items():
        if type(receipt[key]) is not type(value) or receipt[key]!=value:
            raise ValueError('Keeper receipt differs from active scope: '+key)
    if type(receipt['keeper_pid']) is not int or receipt['keeper_pid']<1:
        raise ValueError('Native keeper process identity required')
    if expected_keeper_pid is not None and receipt['keeper_pid']!=expected_keeper_pid:
        raise ValueError('Keeper process changed during the benchmark')
    observed=_native_utc(receipt['observed_utc'])
    age=(now_utc-observed).total_seconds()
    if not -2.0<=age<=MAX_RECEIPT_AGE_SECONDS:
        raise ValueError('Keeper receipt is stale or future-dated')
    if now_utc>=_utc(scope['expires_utc']):raise ValueError('Operator scope expired')
    return receipt


def read_active_benchmark(scope_path, scope_sha256, *, recipe_path, recipe_sha256,
                          now_utc, scope_root=WINDOWS_SCOPE_ROOT, expected_keeper_pid=None):
    scope=_read_json(scope_path,scope_sha256);path=_require_scope_path(scope_path,scope,scope_root=scope_root)
    recipe=_read_json(recipe_path,recipe_sha256)
    validate_scope(scope,now_utc=now_utc,expected_source_commit=recipe['source_commit'],
        expected_recipe_sha256=recipe_sha256,
        expected_baseline_report_sha256=scope['baseline_reference_report_sha256'])
    require_benchmark_recipe_geometry(scope,recipe)
    if (path.parent/'close.json').exists():raise ValueError('Operator scope is already closed')
    receipt=_read_json(path.parent/'receipt.json')
    require_active_keeper_receipt(scope,scope_sha256,receipt,now_utc=now_utc,
        expected_keeper_pid=expected_keeper_pid)
    previous_token=None
    for token_path in (path.parent/'consumed.json',Path(scope_root)/('operator-'+scope['operator_action_id']+'.consumed.json')):
        token=_read_json(token_path)
        expected={'schema_version':'gx1_power_benchmark_consumed_operator_token_draft_v1',
            'scope_id':scope['scope_id'],'operator_action_id':scope['operator_action_id'],
            'scope_sha256':scope_sha256,'keeper_pid':receipt['keeper_pid'],'reusable':False}
        if set(token)!=set(expected)|{'consumed_utc'}:
            raise ValueError('Consumed operator token fields differ')
        for key,value in expected.items():
            if type(token[key]) is not type(value) or token[key]!=value:
                raise ValueError('Consumed operator token differs: '+key)
        consumed=_native_utc(token['consumed_utc'])
        if not _utc(scope['created_utc'])-timedelta(seconds=2)<=consumed<=min(now_utc,_native_utc(receipt['observed_utc']))+timedelta(seconds=2):
            raise ValueError('Operator token consumption time differs from the scope/receipt')
        if previous_token is not None and token!=previous_token:
            raise ValueError('Scope and operator consumption tokens differ')
        previous_token=token
    if (path.parent/'close.json').exists():raise ValueError('Operator scope closed during inspection')
    return scope,recipe,receipt


def require_clean_benchmark_source(repo, scope):
    repo=Path(repo)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if head!=scope['source_commit'] or subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise ValueError('Power benchmark requires its exact clean committed source')


def require_prepared_benchmark_scope(scope_path, scope_sha256, *, recipe_path,
                                     recipe_sha256, now_utc, scope_root=WINDOWS_SCOPE_ROOT):
    """Dry-run metadata check; a live keeper is required separately at execution."""
    scope = _read_json(scope_path, scope_sha256)
    _require_scope_path(scope_path, scope, scope_root=scope_root)
    recipe = _read_json(recipe_path, recipe_sha256)
    validate_scope(scope, now_utc=now_utc, expected_source_commit=recipe['source_commit'],
        expected_recipe_sha256=recipe_sha256,
        expected_baseline_report_sha256=scope['baseline_reference_report_sha256'])
    require_benchmark_recipe_geometry(scope, recipe)
    return scope, recipe


def require_installed_benchmark_sources(recipe, *, install_root=WINDOWS_SCOPE_ROOT.parent):
    """Verify installed bytes; the operator run also verifies keeper startup identity."""
    paths = {'windows_power_keeper': 'GX1-ApplyGpuPowerLimit.ps1',
             'windows_power_benchmark_scope': 'GX1-PowerBenchmarkScope.ps1'}
    root = Path(install_root)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError('Canonical Windows installation root required')
    for key, filename in paths.items():
        expected = recipe['source_bindings'][key]['sha256']
        path = root / filename
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > 1024**2:
            raise ValueError('Regular bounded installed Windows owner required')
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Installed Windows source differs from recipe: ' + key)


def _uptime():
    value = float(Path('/proc/uptime').read_text().split()[0])
    if not math.isfinite(value) or value < 0:
        raise ValueError('Kernel monotonic clock unavailable')
    return value


def _parent_identity(repo):
    """Bind each invocation to the actual guard process, including PID reuse."""
    pid = os.getppid()
    process = Path('/proc') / str(pid)
    argv = process.joinpath('cmdline').read_bytes().split(b'\0')
    expected = os.fsencode(Path(repo) / 'scripts/gx1_guarded_trainer_exec.sh')
    if expected not in argv[:3]:
        raise ValueError('Power scope consumer must be a direct child of its source guard')
    fields = process.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    return {'guard_pid': pid, 'guard_start_ticks': int(fields[19])}


def _require_guard_admission(repo, scope, scope_path, scope_sha256):
    import importlib.util
    import sys
    owner_path = Path(repo) / 'gx1/contracts/gx1_capped_execution_v1.py'
    spec = importlib.util.spec_from_file_location('_gx1_benchmark_capped_owner', owner_path)
    owner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = owner
    spec.loader.exec_module(owner)
    owner.require_guarded_cuda_trainer_execution()
    expected = {
        'GX1_POWER_BENCHMARK_SCOPE_JSON': str(scope_path),
        'GX1_POWER_BENCHMARK_SCOPE_SHA256': scope_sha256,
        'GX1_TRAINER_GPU_MAX_POWER_LIMIT_W': str(scope['target_power_limit_w']),
        'GX1_TRAINER_GPU_MAX_POWER_DRAW_W': str(scope['draw_stop_w']),
        'GX1_TRAINER_GPU_MAX_CORE_TEMP_C': '65',
        'GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C': '80',
        'GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB': '12288',
        'GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS': '1',
        'GX1_TRAINER_HOST_TELEMETRY_GPU_UUID': scope['gpu_uuid'],
        'GX1_TRAINER_ATTENDED_STAGE_REQUIRED': 'false',
    }
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise ValueError('Protected benchmark environment differs: ' + key)
    return _parent_identity(repo)


def _write_once(path, value):
    """Publish complete bytes atomically without replacing an existing token."""
    import tempfile
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_owned_claim(scope_path, scope_sha256, repo):
    scope = _read_json(scope_path, scope_sha256)
    path = _require_scope_path(scope_path, scope)
    claim = _read_json(path.parent / 'linux-claim.json')
    identity = _parent_identity(repo)
    expected = {'schema_version': 'gx1_power_benchmark_linux_claim_draft_v1',
        'scope_sha256': scope_sha256, 'repo': str(repo), **identity}
    for key, value in expected.items():
        if type(claim.get(key)) is not type(value) or claim.get(key) != value:
            raise ValueError('Linux benchmark claim ownership differs: ' + key)
    return scope, claim


def _command_recipe(command):
    def one(flag):
        if command.count(flag) != 1:
            raise ValueError('Exactly one recipe flag required: ' + flag)
        index = command.index(flag)
        if index + 1 >= len(command):
            raise ValueError('Missing recipe flag value')
        return command[index + 1]
    return Path(one('--recipe-audit-json')), one('--recipe-audit-sha256')


def benchmark_cli(argv=None):
    import argparse
    import signal
    import sys
    # Bound metadata/receipt I/O without a timeout-wrapper parent PID change.
    def expired(signum, frame):
        raise TimeoutError('Power benchmark metadata operation exceeded five seconds')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(5)
    try:
        args = list(sys.argv[1:] if argv is None else argv)
        cut = args.index('--') if '--' in args else len(args)
        command = args[cut + 1:]
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('action', choices=('inspect', 'claim', 'heartbeat', 'close'))
        parser.add_argument('--scope-json', type=Path, required=True)
        parser.add_argument('--scope-sha256', required=True)
        options = parser.parse_args(args[:cut])
        repo = Path(__file__).resolve().parents[2]
        path, digest = options.scope_json, options.scope_sha256
        now = datetime.now(timezone.utc)
        if options.action in ('inspect', 'claim'):
            recipe_path, recipe_digest = _command_recipe(command)
            scope, recipe, receipt = read_active_benchmark(path, digest,
                recipe_path=recipe_path, recipe_sha256=recipe_digest, now_utc=now)
            require_benchmark_command(command, recipe_path=recipe_path,
                recipe_sha256=recipe_digest, recipe=recipe, repo=repo)
            require_clean_benchmark_source(repo, scope)
            require_installed_benchmark_sources(recipe)
            if options.action == 'claim':
                identity = _require_guard_admission(repo, scope, path, digest)
                scope, recipe, receipt = read_active_benchmark(path, digest,
                    recipe_path=recipe_path, recipe_sha256=recipe_digest,
                    now_utc=datetime.now(timezone.utc), expected_keeper_pid=receipt['keeper_pid'])
                # Calculate the deadline from a fresh clock after source checks.
                remaining = (_utc(scope['expires_utc']) - datetime.now(timezone.utc)).total_seconds()
                if not 0 < remaining <= 1800:
                    raise ValueError('Scope expired while claiming')
                claim = {'schema_version': 'gx1_power_benchmark_linux_claim_draft_v1',
                    'scope_sha256': digest, 'repo': str(repo), **identity,
                    'recipe_path': str(recipe_path), 'recipe_sha256': recipe_digest,
                    'keeper_pid': receipt['keeper_pid'],
                    'deadline_uptime': _uptime() + remaining,
                    'command_sha256': sha256(json.dumps(command).encode()).hexdigest()}
                _write_once(path.parent / 'linux-claim.json', claim)
            print(scope['target_power_limit_w'], scope['draw_stop_w'])
        else:
            if command:
                raise ValueError('Heartbeat/closure do not accept a trainer command')
            scope, claim = _read_owned_claim(path, digest, repo)
            if options.action == 'heartbeat':
                deadline = claim.get('deadline_uptime')
                if type(deadline) not in (int, float) or not math.isfinite(deadline) or _uptime() >= deadline:
                    raise ValueError('Monotonic operator scope expired')
                _, recipe, _ = read_active_benchmark(path, digest, recipe_path=claim['recipe_path'],
                    recipe_sha256=claim['recipe_sha256'], now_utc=now,
                    expected_keeper_pid=claim['keeper_pid'])
                require_installed_benchmark_sources(recipe)
            else:
                try:
                    _write_once(path.parent / 'close.json', {
                        'schema_version': 'gx1_power_benchmark_linux_close_draft_v1',
                        'scope_sha256': digest, 'reason': 'guard_exit',
                        'observed_utc': now.isoformat(), 'guard_pid': claim['guard_pid']})
                except FileExistsError:
                    pass  # Any prior closure already forbids rearming.
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, IndexError, subprocess.SubprocessError) as exc:
        print('Power benchmark rejected: ' + str(exc), file=sys.stderr)
        return 75
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    raise SystemExit(benchmark_cli())
