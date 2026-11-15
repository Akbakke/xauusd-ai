"""Fixed S full-fit scope, one-use route and matched ranking math."""
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.contracts import gx1_capped_execution_v1 as guard
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_fold2_s_full_fit_v1 as scope
from gx1.scripts import run_market_state_fold2_s_full_fit_window_v1 as worker


def test_exact_one_pass_recipe_and_gate():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=scope.REPO, text=True).strip()
    recipe = scope.build_recipe(source_commit=head)
    assert recipe['run_scope'] == scope.SCOPE
    assert recipe['selected_fit_rows'] == 47338
    assert recipe['selected_score_rows'] == 4096
    assert recipe['max_forwards'] == 2959 + 512
    assert recipe['max_private_updates'] == 2959
    assert recipe['max_model_constructions'] == 1
    assert recipe['test_data_used'] is False
    assert authority.guarded_research_power_limit(recipe) == 250
    scope.require_gate({**scope.gate_fields()}, recipe)
    with pytest.raises(Exception, match='POLICY_GATE'):
        scope.require_gate({**scope.gate_fields(),
                            'optimizer_steps_authorized': 2960}, recipe)
    assert scope.batch_bounds(0) == (0, 16)
    assert scope.batch_bounds(2957) == (47312, 47328)
    assert scope.batch_bounds(2958) == (47328, 47338)
    assert scope.cursor_from_visits(2959) == 47338
    assert scope.visits_from_cursor(47338) == 2959
    with pytest.raises(Exception):
        scope.batch_bounds(2959)


def test_auc_uses_exact_ties_and_direction():
    labels = np.array([0, 1, 1, 0], dtype=np.uint8)
    assert worker._rank_auc(labels, np.array([0., 1., 1., 0.])) == 1.0
    assert worker._rank_auc(labels, np.zeros(4)) == 0.5
    assert worker._rank_auc(labels, np.array([1., 0., 0., 1.])) == 0.0


def test_router_selects_only_new_scope(monkeypatch, tmp_path):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as router
    monkeypatch.setattr(guard, 'require_guarded_cuda_trainer_execution',
                        lambda: {'synthetic_guard': True})
    monkeypatch.setattr(authority, 'require_native_direction_research_context',
                        lambda *args: ({}, {}, {}, {}, {'run_scope': scope.SCOPE}))
    seen = []
    monkeypatch.setattr(worker, 'run_fold2_s_full_fit_window',
                        lambda **kwargs: seen.append(kwargs) or {'selected': True})
    args = {'policy_path': tmp_path / 'WINDOW.json',
            'policy_file_sha256': 'a' * 64,
            'progress_path': tmp_path / 'PROGRESS.json'}
    assert router.run_native_direction_research_window(**args) == {'selected': True}
    assert seen == [args]


def test_materializer_recognizes_new_root_before_claim(monkeypatch):
    from gx1.scripts import materialize_market_state_time_query_fold1_fit_campaign_v1 as materializer
    monkeypatch.setattr(campaign, 'file_sha256', lambda path: '0' * 64)

    def reached(*args, **kwargs):
        raise RuntimeError('NEW_SCOPE_BINDING_REACHED')

    monkeypatch.setattr(campaign, 'read_bound_json', reached)
    with pytest.raises(RuntimeError, match='NEW_SCOPE_BINDING_REACHED'):
        materializer.materialize(scope.ROOT / 'PREPARATION_EXECUTION_BINDING.json')
