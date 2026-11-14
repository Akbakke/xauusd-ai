"""Read-only source and authority checks for one frozen fold 2 fit candidate."""
import copy
import subprocess

import pytest

from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_ta_fold2_m5_fit_v1 as fit


def test_frozen_recipe_rejects_sealed_schedule_mutation_and_closed_gate():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=fit.REPO,
                                   text=True).strip()
    policy_before = (fit.REPO / 'NEXT_RUN_POLICY.json').read_bytes()
    recipe = fit.build_recipe(source_commit=head)
    assert (fit.REPO / 'NEXT_RUN_POLICY.json').read_bytes() == policy_before
    assert recipe['input_refs']['frozen_design']['path'] == str(fit.DESIGN_PATH)
    assert recipe['numerical_recipe']['max_updates'] == fit.BATCH_VISITS == 2959
    assert recipe['one_window']['resume_allowed'] is False
    assert recipe['max_checkpoint_writes'] == 0
    assert recipe['test_data_used'] is False
    altered = copy.deepcopy(recipe)
    altered['numerical_recipe']['max_updates'] += 1
    altered['recipe_sha256'] = campaign.canonical_sha256({
        key: value for key, value in altered.items() if key != 'recipe_sha256'})
    with pytest.raises(ValueError, match='FOLD2_M5_FIT_EXACT_RECIPE'):
        fit.require_recipe(altered, source_repo=fit.REPO, source_commit=head)
    gate = fit.gate_fields()
    gate['full_epoch_training_allowed'] = False
    with pytest.raises(ValueError, match='FOLD2_M5_FIT_POLICY_GATE'):
        fit.require_gate(gate, recipe)


def test_campaign_terminal_receipts_count_skipped_batches_without_resume():
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority

    recipe_ref = {'path': str(fit.RECIPE_PATH), 'sha256': 'a' * 64}
    result_ref = {'path': str(fit.ROOT / 'RESULT.json'), 'sha256': 'b' * 64}
    cursor = authority.build_research_cursor(
        recipe_ref=recipe_ref, outcome='COMPLETE', optimizer_updates=2950,
        parent_cursor=fit.TRAIN_ROWS,
        parent_order_sha256=fit.TRAIN_PARENT_ORDER_SHA256,
        full_state=None, result=result_ref, batch_size=16)
    assert cursor['batch_visits'] == 2959
    assert cursor['optimizer_updates'] == 2950
    assert cursor['full_state'] is None
    assert cursor['canonical_resume_authority'] is False
    assert authority.require_research_cursor(
        cursor, expected_recipe=recipe_ref, verify_files=False,
        batch_size=16) == cursor

    window = {'recipe': recipe_ref}
    plan = {'plan_sha256': 'c' * 64}
    pointer = str(fit.RUNTIME_ROOT / 'research-cursor/RESUME_CURSOR.json')
    invocation = {'invocation_sha256': 'd' * 64, 'kind': authority.KIND,
                  'batch_size': 16, 'optimizer_step_budget': 2959,
                  'direction_research_recipe': recipe_ref,
                  'checkpoint': {'pointer_path': pointer}}
    progress = authority.build_research_progress(
        window=window, plan=plan, invocation=invocation,
        checkpoint_pointer={'path': pointer, 'sha256': 'e' * 64},
        optimizer_updates=2950, parent_cursor=fit.TRAIN_ROWS,
        parent_order_sha256=fit.TRAIN_PARENT_ORDER_SHA256,
        outcome='COMPLETE')
    assert progress['completed_units'] == progress['total_units'] == 2959
    assert progress['optimizer_updates'] == 2950
    altered = dict(progress, completed_units=2950)
    altered['progress_sha256'] = campaign.canonical_sha256({
        key: value for key, value in altered.items() if key != 'progress_sha256'})
    with pytest.raises(RuntimeError, match='FOLD2_FIT_PROGRESS'):
        authority.require_research_progress(
            altered, plan_sha256=plan['plan_sha256'], invocation=invocation,
            expected_selection_receipt_sha256=None, verify_file=False)
