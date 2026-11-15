"""Publish one Fold 2 S-only private candidate recipe without admission."""
from __future__ import annotations

import hashlib
import json
import subprocess

from gx1.contracts import market_state_fold2_multiminute_private_campaign_v1 as scope
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes


def _need(ok: object, reason: str) -> None:
    if not ok:
        raise RuntimeError('FOLD2_S_PRIVATE_RECIPE_PREP_' + reason)


def prepare() -> dict:
    repo = scope.base.CANONICAL_REPO
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
          'FOLD2_S_25_CELL_PRIVATE_CLOCK_CORRECTED_SOURCE_READY_NO_RUN_ADMISSION_EDGE_UNQUALIFIED'
          and all(policy.get(key) is False for key in (
              'training_enabled', 'native_training_enabled',
              'full_epoch_training_allowed', 'full_val_allowed'))
          and scope.STAGE not in policy, 'CLOSED_POLICY')
    recipe = scope.build_recipe(source_commit=head)
    scope.require_recipe(recipe, source_repo=repo, source_commit=head)
    _need(recipe['fold_id'] == 2 and recipe['s_only_25_cell'] is True
          and recipe['selected_state_endpoints'] is False
          and recipe['training_enabled'] is True
          and recipe['full_epoch_training_allowed'] is False
          and recipe['max_private_updates'] == 4
          and recipe['test_data_used'] is False,
          'CANDIDATE_SCOPE')
    _need(not scope.ROOT.exists() and not scope.RECIPE_PATH.exists(),
          'ONE_USE_GENESIS')
    payload = (json.dumps(recipe, sort_keys=True, indent=2,
                          allow_nan=False) + '\n').encode()
    _need(len(payload) <= 512 * 1024, 'RECIPE_BYTES')
    scope.ROOT.mkdir()
    _publish_bytes(scope.RECIPE_PATH, payload)
    return {'status': 'CANDIDATE_RECIPE_ONLY_NO_ADMISSION',
            'recipe': {'path': str(scope.RECIPE_PATH),
                       'sha256': hashlib.sha256(payload).hexdigest()},
            'source_commit': head, 'training_started': False,
            'optimizer_updates': 0, 'test_data_used': False}


def main() -> None:
    print(json.dumps(prepare(), sort_keys=True, separators=(',', ':')))


if __name__ == '__main__':
    main()
