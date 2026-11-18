"""Retired historical TA provider; its M1-derived inputs require replacement."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


REPO = Path('/home/andre2/src/GX1_CURRENT')
PHYSICAL = Path('/home/andre2/GX1_RUNS/TA_FOLD2_APR_JUN_INPUTS_20261117_001')
ISSUE = Path('/home/andre2/GX1_RUNS/TA_INTEGRATED_APR_JUN_ISSUE_INPUTS_20261117_002')
BASE_VIEW = REPO / 'configs/research/TA_FOLD2_APR_JUN_FEATURE_VIEW_20261117_001.json'
MODEL_VIEW = REPO / 'configs/research/TA_FOLD2_APR_JUN_INTEGRATED_VIEW_20261117_001.json'
FIT_REVIEW = REPO / 'docs/TA_INTEGRATED_FOLD2_FIT_INPUTS_REVIEW_20261116.json'
TF = ('M5', 'M15', 'H1', 'H4', 'D1')
TRAIN_ROWS = 47338
SCORE_ROWS = 17536
COLD_FIT_ROWS, COLD_SCORE_START, COLD_SCORE_ROWS = 29574, 29849, 17489
COLD_ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_COLD_FULL_LATER_INPUTS_20261117_001')
COLD_BASE_VIEW = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_OBJECTIVE_BENCHMARK_20261112_001/FEATURE_VIEW.json')
COLD_PACKET = COLD_BASE_VIEW.parent / 'PREFIX_PACKET.json'
COLD_PRIOR_ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_COLD_FOLD1_ENDPOINT_PRIORS_20261117_001')
COLD_PRIORS = COLD_PRIOR_ROOT / 'ENDPOINT_PRIORS.json'
COLD_PREFIX_SPANS = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_INPUTS_20261112_001')
COLD_VIEW_SHA256 = '7cdcc64b80ba8a1ac92aae24ac7fcd2b62252da31cf96e57ad7a86fc943c3293'


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_APR_JUN_CAMPAIGN_INPUT_' + reason)


def ref(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'FILE')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def read(binding):
    need(ref(binding['path']) == binding, 'REF')
    return json.loads(Path(binding['path']).read_text())


def axis_hash(value):
    return hashlib.sha256(value.astype('<i8', copy=False).tobytes()).hexdigest()


def _tape(binding, parents, issues, *, start=0, stop=None):
    """Refuse the retired M1-derived direction provider before input IO."""
    raise RuntimeError('DIRECTION_M1_PROVENANCE_RETIRED_REQUIRES_NATIVE_M5')


def campaign_input_refs():
    """Refuse the retired M1-derived direction provider before input IO."""
    raise RuntimeError('DIRECTION_M1_PROVENANCE_RETIRED_REQUIRES_NATIVE_M5')


def cold_campaign_input_refs():
    """Refuse the retired M1-derived direction provider before input IO."""
    raise RuntimeError('DIRECTION_M1_PROVENANCE_RETIRED_REQUIRES_NATIVE_M5')


def prepare(*, device: torch.device, cold_inputs: dict | None = None):
    """Refuse the retired M1-derived direction provider before input IO."""
    raise RuntimeError('DIRECTION_M1_PROVENANCE_RETIRED_REQUIRES_NATIVE_M5')
