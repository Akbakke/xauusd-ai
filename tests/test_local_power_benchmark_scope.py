"""Scope/CLI file regressions; process and hardware admission are separate tests."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

from gx1.contracts import local_power_benchmark_v1 as owner


class PowerScopeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.now = datetime(2026, 9, 9, 0, 1, tzinfo=timezone.utc)
        self.repo = Path(owner.__file__).resolve().parents[2]

    def write(self, path, value):
        raw = (json.dumps(value, sort_keys=True) + '\n').encode()
        path.write_bytes(raw)
        return sha256(raw).hexdigest()

    def fixture(self, target=200, kind="smoke_comparison"):
        candidate = kind == "candidate_continuation"
        geometry = owner._SCOPE_KINDS[kind]
        recipe = {'profile': geometry['profile'], 'run_id': 'SYNTHETIC_POWER_TEST',
            'source_commit': 'a' * 40, 'out_bundle_dir': str(self.root / 'bundle'),
            'trainer_cli': {'execution_tier': 'canonical', 'device': 'cuda',
                'precision_policy': owner.FIXED['precision_policy'],
                'subsample_rows': geometry['subsample_rows'],
                'batch_size': 8, 'grad_accum_steps': 1,
                'epochs': geometry['epochs'],
                'num_workers': 0, 'train_time_window': None}}
        recipe_path = self.root / 'recipe.json'
        recipe_hash = self.write(recipe_path, recipe)
        candidate_gate_hash = None
        candidate_budget_hash = None
        if candidate:
            candidate_gate_hash = self.write(self.root / 'gate.json', {'kind': 'gate'})
            candidate_budget_hash = self.write(self.root / 'budget.json', {'kind': 'budget'})
        scope = {**deepcopy(owner._scope_fixed(kind, target)),
            'scope_id': '12345678-1111-2222-3333-123456789abc',
            'operator_action_id': '87654321-1111-2222-3333-123456789abc',
            'created_utc': '2026-09-09T00:00:00Z',
            'expires_utc': ('2026-09-09T01:40:00Z' if candidate
                            else '2026-09-09T00:30:00Z'),
            'source_commit': recipe['source_commit'], 'recipe_sha256': recipe_hash,
            'baseline_reference_report_sha256': 'c' * 64, 'run_id': recipe['run_id'],
            'candidate_gate_sha256': candidate_gate_hash,
            'candidate_execution_budget_sha256': candidate_budget_hash}
        directory = self.root / scope['scope_id']; directory.mkdir()
        scope_path = directory / 'scope.json'; scope_hash = self.write(scope_path, scope)
        receipt = {'schema_version': owner.RECEIPT_SCHEMA, 'scope_id': scope['scope_id'],
            'operator_action_id': scope['operator_action_id'], 'scope_sha256': scope_hash,
            'gpu_uuid': scope['gpu_uuid'], 'source_commit': scope['source_commit'],
            'recipe_sha256': recipe_hash, 'run_id': scope['run_id'], 'phase': 'active', 'reason': '',
            'expires_utc': scope['expires_utc'], 'observed_utc': '2026-09-09T00:01:00.0000000Z',
            'keeper_pid': 42, 'requested_power_limit_w': target,
            'baseline_restoration': None, 'training_launch_authority': False}
        self.write(directory / 'receipt.json', receipt)
        token = {'schema_version': 'gx1_power_benchmark_consumed_operator_token_draft_v1',
            'scope_id': scope['scope_id'], 'operator_action_id': scope['operator_action_id'],
            'scope_sha256': scope_hash, 'keeper_pid': 42, 'reusable': False,
            'consumed_utc': receipt['observed_utc']}
        self.write(directory / 'consumed.json', token)
        self.write(self.root / ('operator-' + scope['operator_action_id'] + '.consumed.json'), token)
        return scope_path, scope_hash, recipe_path, recipe_hash, recipe, receipt

    def read(self, f):
        return owner.read_active_benchmark(f[0], f[1], recipe_path=f[2], recipe_sha256=f[3],
            now_utc=self.now, scope_root=self.root, expected_keeper_pid=42)

    def test_active_reference_and_treatment_use_exact_same_geometry(self):
        f = self.fixture(160)
        self.assertEqual(self.read(f)[0]['target_power_limit_w'], 160)
        f[5]['requested_power_limit_w'] = 200
        self.write(f[0].parent / 'receipt.json', f[5])
        with self.assertRaises(ValueError): self.read(f)


    def test_candidate_continuation_binds_gate_budget_and_longer_scope(self):
        f = self.fixture(kind="candidate_continuation")
        self.assertEqual(self.read(f)[0]["authority"]["candidate_continuation"], True)
        command = [str(self.repo / ".venv/bin/python"), "-m",
                   "gx1.models.entry_v10.entry_v10_ctx_train_v3", "--train"]
        values = {
            "--profile": "candidate", "--execution-tier": "canonical",
            "--device": "cuda", "--precision-policy": owner.FIXED["precision_policy"],
            "--batch_size": "8", "--epochs": "30", "--grad-accum-steps": "1",
            "--subsample-rows": "0", "--run-id": f[4]["run_id"],
            "--recipe-audit-json": str(f[2]), "--recipe-audit-sha256": f[3],
            "--out_bundle_dir": f[4]["out_bundle_dir"],
            "--candidate-gate-json": str(self.root / "gate.json"),
            "--candidate-gate-sha256": self.read(f)[0]["candidate_gate_sha256"],
            "--candidate-execution-budget-json": str(self.root / "budget.json"),
            "--candidate-execution-budget-sha256": self.read(f)[0]["candidate_execution_budget_sha256"],
        }
        for key, value in values.items():
            command.extend((key, value))
        owner.require_benchmark_command(
            command, scope=self.read(f)[0], recipe_path=f[2], recipe_sha256=f[3],
            recipe=f[4], repo=self.repo,
        )
        changed = list(command)
        changed[changed.index("--candidate-execution-budget-sha256") + 1] = "f" * 64
        with self.assertRaises(ValueError):
            owner.require_benchmark_command(
                changed, scope=self.read(f)[0], recipe_path=f[2],
                recipe_sha256=f[3], recipe=f[4], repo=self.repo,
            )

    def test_stale_or_restarted_keeper_rejected(self):
        f = self.fixture()
        for key, value in [('keeper_pid', 43), ('observed_utc', '2026-09-09T00:00:44.0000000Z')]:
            changed = {**f[5], key: value}
            self.write(f[0].parent / 'receipt.json', changed)
            with self.assertRaises(ValueError): self.read(f)

    def test_close_intent_rejects_an_otherwise_fresh_receipt(self):
        f = self.fixture(); self.read(f)
        owner._write_once(f[0].parent / 'close.json', {'reason': 'test'})
        with self.assertRaises(ValueError): self.read(f)

    def test_duplicate_json_and_atomic_token_replacement_rejected(self):
        path = self.root / 'token.json'
        owner._write_once(path, {'value': 1})
        with self.assertRaises(FileExistsError): owner._write_once(path, {'value': 2})
        self.assertEqual(owner._read_json(path), {'value': 1})
        path.write_text('{"value":1,"value":2}')
        with self.assertRaises(ValueError): owner._read_json(path)

    def test_installed_owner_bytes_must_match_recipe(self):
        bindings = {}
        for key, filename in [('windows_power_keeper', 'GX1-ApplyGpuPowerLimit.ps1'),
                              ('windows_power_benchmark_scope', 'GX1-PowerBenchmarkScope.ps1')]:
            raw = ('# synthetic ' + key).encode()
            (self.root / filename).write_bytes(raw)
            bindings[key] = {'sha256': sha256(raw).hexdigest()}
        recipe = {'source_bindings': bindings}
        owner.require_installed_benchmark_sources(recipe, install_root=self.root)
        (self.root / 'GX1-ApplyGpuPowerLimit.ps1').write_text('# changed')
        with self.assertRaises(ValueError):
            owner.require_installed_benchmark_sources(recipe, install_root=self.root)

    def test_claim_heartbeat_close_and_single_use(self):
        f = self.fixture()
        real_active = owner.read_active_benchmark
        real_path = owner._require_scope_path
        fixed_now = self.now
        class Frozen(datetime):
            @classmethod
            def now(cls, tz=None): return cls.fromisoformat(fixed_now.isoformat())
        command = [str(self.repo / '.venv/bin/python'), '-m',
                   'gx1.models.entry_v10.entry_v10_ctx_train_v3', '--train']
        values = {'--profile':f[4]['profile'], '--execution-tier':'canonical', '--device':'cuda',
            '--precision-policy':owner.FIXED['precision_policy'], '--batch_size':'8',
            '--epochs':str(f[4]['trainer_cli']['epochs']), '--grad-accum-steps':'1',
            '--subsample-rows':str(f[4]['trainer_cli']['subsample_rows']),
            '--run-id':f[4]['run_id'], '--recipe-audit-json':str(f[2]),
            '--recipe-audit-sha256':f[3], '--out_bundle_dir':f[4]['out_bundle_dir']}
        for k, v in values.items(): command.extend((k, v))
        identity = {'guard_pid': 123, 'guard_start_ticks': 456}
        def active(p, h, **kw): return real_active(p, h, **{**kw, 'scope_root': self.root})
        def scope_path(p, s, **kw): return real_path(p, s, scope_root=self.root)
        def run(action):
            args = [action, '--scope-json', str(f[0]), '--scope-sha256', f[1]]
            if action in ('inspect', 'claim'): args += ['--', *command]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                return owner.benchmark_cli(args)
        with patch.object(owner, 'datetime', Frozen), patch.object(owner, '_uptime', return_value=100), \
             patch.object(owner, '_parent_identity', return_value=identity), \
             patch.object(owner, '_require_guard_admission', return_value=identity), \
             patch.object(owner, 'require_clean_benchmark_source'), \
             patch.object(owner, 'require_installed_benchmark_sources'), \
             patch.object(owner, '_require_scope_path', side_effect=scope_path), \
             patch.object(owner, 'read_active_benchmark', side_effect=active):
            for action, status in [('inspect', 0), ('claim', 0), ('heartbeat', 0),
                                   ('claim', 75), ('close', 0), ('heartbeat', 75), ('claim', 75)]:
                self.assertEqual(run(action), status, action)


if __name__ == '__main__':
    unittest.main()
