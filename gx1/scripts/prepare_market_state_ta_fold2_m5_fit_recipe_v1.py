"""Publish one immutable fold 2 M5 fit candidate recipe without admission."""
from __future__ import annotations

import hashlib
import json
import subprocess

from gx1.contracts import market_state_ta_fold2_m5_fit_v1 as fit
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('FOLD2_M5_FIT_RECIPE_PREP_' + reason)


def prepare() -> dict:
    repo = fit.REPO
    _need(subprocess.check_output(['git', 'branch', '--show-current'],
          cwd=repo, text=True).strip() == 'work/gx1-current', 'BRANCH')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=repo, text=True).strip()
    _need(not subprocess.check_output(
        ['git', 'status', '--porcelain=v1', '--untracked-files=all'],
        cwd=repo, text=True), 'SOURCE_DIRTY')
    policy = json.loads((repo / 'NEXT_RUN_POLICY.json').read_text())
    _need(policy['status'] == 'TA_FOLD2_M5_003_CONSUMED_004_SOURCE_QUALIFIED_RECIPE_PENDING'
          and all(policy.get(key) is False for key in (
              'training_enabled', 'native_training_enabled',
              'full_epoch_training_allowed', 'full_val_allowed'))
          and fit.STAGE not in policy, 'CLOSED_POLICY')
    from gx1.audit.preflight_market_state_ta_fold2_m5_fit_v1 import inspect
    _need(inspect()['status'] == 'PASS_METADATA_ONLY_FIT_UNADMITTED',
          'METADATA_PREFLIGHT')
    recipe = fit.build_recipe(source_commit=head)
    _need(recipe['mechanical_source_commit'] == head
          and recipe['training_enabled'] is True
          and recipe['full_epoch_training_allowed'] is True
          and recipe['test_data_used'] is False,
          'CANDIDATE_SCOPE')
    _need(not fit.ROOT.exists() and not fit.RECIPE_PATH.exists(), 'ONE_USE_GENESIS')
    payload = (json.dumps(recipe, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(len(payload) <= 512 * 1024, 'RECIPE_BYTES')
    fit.ROOT.mkdir()
    _publish_bytes(fit.RECIPE_PATH, payload)
    binding = {'path': str(fit.RECIPE_PATH),
               'sha256': hashlib.sha256(payload).hexdigest()}
    return {'status': 'CANDIDATE_RECIPE_ONLY_NO_ADMISSION',
            'recipe': binding, 'source_commit': head,
            'training_started': False, 'optimizer_updates': 0,
            'test_data_used': False}


def main() -> None:
    print(json.dumps(prepare(), sort_keys=True, separators=(',', ':')))


if __name__ == '__main__':
    main()
