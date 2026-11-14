"""One-use fold3-5 normalization receipts with mocked physical fitting."""
import json
from pathlib import Path

import pytest

from gx1.audit import qualify_market_state_ta_fivefold_span_normalization_v1 as work


@pytest.mark.parametrize('fail_fold', [None, 4])
def test_claimed_run_consumes_success_or_partial_failure(monkeypatch, tmp_path, fail_fold):
    root = tmp_path / 'RUN'
    monkeypatch.setattr(work, 'ROOT', root)
    monkeypatch.setattr(work, 'PLAN', tmp_path / 'PLAN.json')
    policy = {'path': 'policy', 'sha256': 'b' * 64}
    folds = {str(i): {'output_root': str(root / f'FOLD_{i}'),
                      'cutoff_ns': i * 10,
                      'inputs': {'parent': {'path': f'parent{i}', 'sha256': 'c' * 64}}}
             for i in work.FOLDS}
    plan = {'folds': folds}
    parents = {str(i): {'fold': i} for i in work.FOLDS}
    monkeypatch.setattr(work, 'preflight', lambda *_a, **_kw:
                        (plan, parents, 'head', policy))
    monkeypatch.setattr(work, 'bookend', lambda *_a: None)
    seen = []

    def measure(fold, *, parent, cutoff, output_root):
        i = parent['fold']
        seen.append(i)
        assert cutoff == i * 10 and output_root == root / f'FOLD_{i}'
        if i == fail_fold:
            raise RuntimeError('synthetic fold fit failure')
        return {'surface_fit_calls': 6, 'new_statistics': 12,
                'old_field_fit_calls': 0, 'normalization_qualified': True,
                'old_field_preservation': {'all_old_stat_vectors_exact': True},
                'units_contract_sha256': None}

    monkeypatch.setattr(work.owner, 'measure', measure)
    if fail_fold:
        with pytest.raises(RuntimeError, match='synthetic fold fit failure'):
            work.run(Path('plan'), 'a' * 64)
        assert seen == [3, 4]
        assert (root / 'FAILED.json').is_file()
    else:
        result = work.run(Path('plan'), 'a' * 64)
        assert result['terminal']['path'] == str(root / 'TERMINAL.json')
        assert seen == [3, 4, 5]
        assert (root / 'RESULT.json').is_file()
    terminal = json.loads((root / 'TERMINAL.json').read_text())
    assert terminal['status'] == ('FAILED_CONSUMED' if fail_fold else 'COMPLETE')
    assert terminal['no_relaunch'] is True
    assert terminal['test_data_used'] is False
    assert (root / 'CLAIM.json').is_file()
