import hashlib

import numpy as np
import pytest

from gx1.audit import direction_gc_mtf_followup_v1 as followup


def test_guard_source_transition_allows_only_exact_readonly_directory_change():
    old = b"x\n    directories={plan['native_cache_dir'],str(Path(plan['gc_surfaces']['path']).parent)}\ny\n"
    new = old.replace(b"str(Path(plan['gc_surfaces']['path']).parent)}", b"str(Path(plan['gc_surfaces']['path']).parent),str(output)}")
    path = '/absolute/source.py'
    first = {'path': path, 'sha256': hashlib.sha256(old).hexdigest()}
    second = {'path': path, 'sha256': hashlib.sha256(new).hexdigest()}
    launch = {'source_owners': {followup.OWNER: first, 'other': {'path': '/other', 'sha256': 'a'}}}
    proof = {'schema_version': 'gx1_direction_gc_output_directory_guard_fix_v1', 'status': 'SOURCE_AND_SYNTHETIC_TESTS_ONLY',
        'source_before': first, 'source_after': second, 'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}
    got = followup.source_transition(launch, proof, old, new)
    assert got['source_owners'][followup.OWNER] == second and launch['source_owners'][followup.OWNER] == first
    with pytest.raises(RuntimeError, match='TRANSITION_HASH'):
        followup.source_transition(launch, proof, old, new + b'extra')


def test_later_fold_is_exact_and_strictly_later_than_origin():
    origin = {'oof_end_row': 64890, 'oof_cutoff_ns': 1743465600000000000, 'train_cutoff_ns': 1735689600000000000}
    later = {'fold_id': 2, 'prefix_rows': 64890, 'oof_start_row': 64890, 'oof_end_row': 82427,
        'train_cutoff_ns': 1743465600000000000, 'oof_cutoff_ns': 1751328000000000000}
    assert followup.later_fold({'folds': [{}, later]}, origin) == later
    with pytest.raises(RuntimeError, match='LATER_FOLD'):
        followup.later_fold({'folds': [{}, {**later, 'oof_end_row': 82428}]}, origin)


def test_paired_archive_roundtrips_and_rejects_wrong_shapes(tmp_path):
    n = 3; parents = np.arange(n, dtype=np.int64); decisions = parents + 1
    terminal = np.zeros((n, 60), dtype=np.int64); observed = np.ones((n, 60), dtype=np.bool_)
    sign = np.ones((n, 60), dtype=np.int8); original = np.zeros((n, 60), dtype=np.float32)
    ablated = np.ones((n, 60), dtype=np.float32); path = tmp_path / 'paired.npz'
    result = followup.save_paired(path, parents=parents, decisions=decisions, terminal=terminal,
        observed=observed, sign=sign, original=original, ablated=ablated)
    assert result['bytes'] == path.stat().st_size and result['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as archive:
        assert np.array_equal(archive['ablated_logit_up'], ablated)
    with pytest.raises(RuntimeError, match='PAIRED_ARRAYS'):
        followup.save_paired(tmp_path / 'bad.npz', parents=parents, decisions=decisions, terminal=terminal,
            observed=observed, sign=sign, original=original[:, :-1], ablated=ablated)


def test_guarded_dispatch_requires_cuda_and_exact_single_flags(monkeypatch, tmp_path):
    from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as validator
    calls = []
    monkeypatch.setattr(followup, 'run', lambda **kwargs: calls.append(kwargs) or {'status': 'SYNTHETIC'})
    args = ['--direction-gc-mtf-followup-plan', str(tmp_path / 'plan.json'),
        '--direction-gc-mtf-followup-plan-sha256', 'a' * 64, '--device', 'cuda', '--out-dir', str(tmp_path)]
    assert validator.main(args) == 0 and len(calls) == 1 and calls[0]['device'] == 'cuda'
    with pytest.raises(SystemExit):
        validator.main(args + ['--device', 'cuda'])
    with pytest.raises(SystemExit):
        validator.main(args[:4] + ['--device', 'cpu', '--out-dir', str(tmp_path)])
