"""Fixed chronological fold 2 M5 direction-fit schedule; no run authority."""
from __future__ import annotations

from pathlib import Path

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'market_state_ta_fold2_m5_direction_fit_20261114_001'
SCOPE = 'market_state_ta_fold2_m5_direction_fit_v1'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
RUNTIME_ROOT = ROOT / 'RUNTIME'
RECIPE_PATH = ROOT / 'RECIPE.json'
TRAIN_ROWS = 47338
BATCH_SIZE = 16
BATCH_VISITS = 2959
TAIL_ROWS = 10
PANEL_ROWS = 4096
PANEL_BATCHES = 256
MAX_UPDATES = BATCH_VISITS
INITIAL_MODEL_SHA256 = '3c172867694217f240a076c34220a1b7fafb2464af5738de544cf7a17068984b'
VIEW_SHA256 = '388fadd9ce6b6d8673724f776eb0f945711ab76cc3b3650f875f3a9e5431c4ab'
PROFILE_SHA256 = '734a2bfb6b1c796205b9a77bd91989bf1f0652e6bf6594927c74ff580c634f75'
LOSS_SHA256 = 'da042c0f02d104bf5533f9d82688b986a29117436517485272de0853e7bd6150'
BOOTSTRAP_SEEDS = {'UTC_DAY': 20260911, 'UTC_MONDAY_WEEK': 20260912}
ONE_WINDOW = {'max_invocation_seconds': 12000, 'max_model_seconds': 10500,
              'wrapper_wall_seconds': 13800, 'gpu_power_cap_w': 250,
              'checkpoint_writes': 0, 'resume_allowed': False,
              'failure_consumes_one_use_plan': True}


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('FOLD2_M5_FIT_' + reason)


def batch_bounds(index: int) -> tuple[int, int]:
    """Use every TRAIN row once, in axis order; the final batch has ten rows."""
    _need(type(index) is int and 0 <= index < BATCH_VISITS, 'BATCH_INDEX')
    left = BATCH_SIZE * index
    return left, min(left + BATCH_SIZE, TRAIN_ROWS)


def require_fixed_design(hypothesis: dict) -> None:
    fit = hypothesis['fixed_fit']
    readout = hypothesis['fixed_readout']
    _need((fit['fold_id'], fit['pass_index'], fit['seed'], fit['train_rows'],
           fit['ordered_batches'], fit['batch_size'], fit['last_batch_rows'],
           fit['maximum_optimizer_updates'])
          == (2, 0, 20260911, TRAIN_ROWS, BATCH_VISITS, BATCH_SIZE, TAIL_ROWS,
              MAX_UPDATES), 'FIT_SCHEDULE')
    _need(fit['fresh_start_requires_exact_initial_model_state_sha256'] == INITIAL_MODEL_SHA256
          and fit['view_contract_sha256'] == VIEW_SHA256
          and fit['m5_equal_observation_profile_sha256'] == PROFILE_SHA256
          and fit['loss_contract_sha256'] == LOSS_SHA256
          and fit['m5_direction_only'] is True and fit['future_weight'] == 1.0,
          'FIXED_FUNCTION')
    _need(readout['bootstrap_binding']['replicates'] == 2000
          and readout['bootstrap_binding']['seeds'] == BOOTSTRAP_SEEDS
          and all(panel['selected_rows'] == PANEL_ROWS
                  and panel['batch_size'] == BATCH_SIZE
                  for panel in readout['train_and_later_panels'].values()),
          'FIXED_READOUT')
    _need(hypothesis['resource_gate']['proposed_one_window'] == ONE_WINDOW,
          'ONE_WINDOW')
    _need(batch_bounds(0) == (0, 16)
          and batch_bounds(BATCH_VISITS - 2) == (47312, 47328)
          and batch_bounds(BATCH_VISITS - 1) == (47328, TRAIN_ROWS),
          'TAIL_PARTITION')
