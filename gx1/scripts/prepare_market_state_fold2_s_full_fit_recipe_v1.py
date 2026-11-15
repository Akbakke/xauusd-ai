"""Publish the fixed S full-fit recipe after committed source qualification."""
from __future__ import annotations

import hashlib
import json
import subprocess

from gx1.contracts import market_state_fold2_s_full_fit_v1 as fit
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes


def prepare() -> dict:
    repo = fit.REPO
    branch = subprocess.check_output(['git', 'branch', '--show-current'],
                                     cwd=repo, text=True).strip()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=repo, text=True).strip()
    dirty = subprocess.check_output(
        ['git', 'status', '--porcelain=v1', '--untracked-files=all'],
        cwd=repo, text=True)
    fit.need(branch == 'work/gx1-current' and not dirty, 'CLEAN_SOURCE')
    policy = json.loads((repo / 'NEXT_RUN_POLICY.json').read_text())
    fit.need(policy['status'] == 'FOLD2_S_FULL_FIT_ROUTE_QUALIFIED_RECIPE_PENDING_EDGE_UNQUALIFIED'
             and all(policy.get(key) is False for key in (
                 'training_enabled', 'native_training_enabled',
                 'full_epoch_training_allowed', 'full_val_allowed'))
             and fit.STAGE not in policy, 'CLOSED_POLICY')
    fit.need(not fit.ROOT.exists(), 'ONE_USE_GENESIS')
    recipe = fit.build_recipe(source_commit=head)
    fit.need(recipe['max_private_updates'] == fit.BATCH_VISITS
             and recipe['max_forwards'] == fit.MAX_FORWARDS
             and recipe['full_epoch_training_allowed'] is True
             and recipe['test_data_used'] is False, 'EXACT_SCOPE')
    payload = (json.dumps(recipe, sort_keys=True, indent=2,
                          allow_nan=False) + '\n').encode()
    fit.need(len(payload) <= 512 * 1024, 'RECIPE_BYTES')
    fit.ROOT.mkdir()
    _publish_bytes(fit.RECIPE_PATH, payload)
    return {'status': 'RECIPE_ONLY_NO_ADMISSION',
            'recipe': {'path': str(fit.RECIPE_PATH),
                       'sha256': hashlib.sha256(payload).hexdigest()},
            'source_commit': head, 'training_started': False,
            'optimizer_updates': 0, 'test_data_used': False}


if __name__ == '__main__':
    print(json.dumps(prepare(), sort_keys=True, separators=(',', ':')))
