"""Retired historical EMA provider with M1-derived targets and dependencies.

Every retired receipt, dependency and preparation entry point refuses before
input IO. A separately qualified native M5 provider is required.
"""
from __future__ import annotations

from pathlib import Path

import torch

from gx1.audit import market_state_ta_apr_jun_campaign_inputs_v1 as legacy
from gx1.audit import qualify_ta_ema_delta_full_gather_v1 as gathered

REPO = gathered.REPO
FIT_ROWS, LATER_START, LATER_STOP = gathered.FIT_ROWS, gathered.LATER_START, gathered.LATER_STOP
LATER_ROWS = gathered.LATER_ROWS
TARGETS = Path('/home/andre2/GX1_RUNS/TA_M5_COST_PATH_TARGETS_20261117_001/FIT_TARGETS.npz')
PRIORS = Path('/home/andre2/GX1_RUNS/TA_M5_EMA_DELTA_FOLD1_PRIORS_20261118_001/ENDPOINT_PRIORS.json')
FIT_REVIEW = legacy.FIT_REVIEW
LANES = gathered.LANES


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_EMA_DELTA_CAMPAIGN_INPUT_' + reason)


def successor_input_refs():
    """Refuse the retired M1-derived direction provider before input IO."""
    raise RuntimeError('DIRECTION_M1_PROVENANCE_RETIRED_REQUIRES_NATIVE_M5')


def _receipts(inputs):
    """Refuse the retired M1-derived direction provider before input IO."""
    raise RuntimeError('DIRECTION_M1_PROVENANCE_RETIRED_REQUIRES_NATIVE_M5')


def prepare(*, device: torch.device, inputs: dict):
    """Refuse the retired M1-derived direction provider before input IO."""
    raise RuntimeError('DIRECTION_M1_PROVENANCE_RETIRED_REQUIRES_NATIVE_M5')
