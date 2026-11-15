"""Publish the one-use Fold 2 selected-native candidate recipe without admission."""
from __future__ import annotations

import hashlib
import json
import subprocess

from gx1.contracts import market_state_fold2_selected_native_private_v1 as fold2
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes


def _need(ok, reason):
    if not ok:
        raise RuntimeError('FOLD2_SELECTED_PRIVATE_RECIPE_PREP_' + reason)


def prepare() -> dict:
    repo = fold2.old.CANONICAL_REPO
    _need(subprocess.check_output(['git', 'branch', '--show-current'],
                                  cwd=repo, text=True).strip() == 'work/gx1-current',
          'BRANCH')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=repo, text=True).strip()
    _need(not subprocess.check_output(
        ['git', 'status', '--porcelain=v1', '--untracked-files=all'],
        cwd=repo, text=True), 'SOURCE_DIRTY')
    policy = json.loads((repo / 'NEXT_RUN_POLICY.json').read_text())
    _need(policy['status'] ==
          'SELECTED_NATIVE_FOLD2_TWO_BATCH_ROUTE_CODED_CAMPAIGN_UNADMITTED'
          and all(policy.get(key) is False for key in (
              'training_enabled', 'native_training_enabled',
              'full_epoch_training_allowed', 'full_val_allowed'))
          and fold2.STAGE not in policy, 'CLOSED_POLICY')
    recipe = fold2.build_recipe(source_commit=head)
    fold2.require_recipe(recipe, source_repo=repo, source_commit=head)
    _need(recipe['fold_id'] == 2 and recipe['selected_state_endpoints'] is True
          and recipe['training_enabled'] is True
          and recipe['full_epoch_training_allowed'] is False
          and recipe['max_private_updates'] == 4
          and recipe['test_data_used'] is False,
          'CANDIDATE_SCOPE')
    _need(not fold2.ROOT.exists() and not fold2.RECIPE_PATH.exists(),
          'ONE_USE_GENESIS')
    payload = (json.dumps(recipe, sort_keys=True, indent=2,
                          allow_nan=False) + '\n').encode()
    _need(len(payload) <= 512 * 1024, 'RECIPE_BYTES')
    fold2.ROOT.mkdir()
    _publish_bytes(fold2.RECIPE_PATH, payload)
    return {'status': 'CANDIDATE_RECIPE_ONLY_NO_ADMISSION',
            'recipe': {'path': str(fold2.RECIPE_PATH),
                       'sha256': hashlib.sha256(payload).hexdigest()},
            'source_commit': head, 'training_started': False,
            'optimizer_updates': 0, 'test_data_used': False}


def main():
    print(json.dumps(prepare(), sort_keys=True, separators=(',', ':')))


if __name__ == '__main__':
    main()
