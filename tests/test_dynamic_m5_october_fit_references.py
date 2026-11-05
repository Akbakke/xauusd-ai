"""Candle orientation is frozen from TRAIN pairs, including exact ties."""
import numpy as np

from gx1.audit import dynamic_mtf_december_candle as candle
from gx1.audit import dynamic_m5_october_fit_references as october


def test_independent_rank_oracle_counts_ties_without_score_selection():
    raw = np.array([0.2, 0.5, 0.5, 0.8], np.float64)
    binary = np.array([0, 1, 0, 1], np.int8)
    eligible = np.ones(4, bool)
    result = candle._rank_direction(raw, binary, eligible)
    assert result['pairs'] == 4
    assert result['twice_wins'] == october._twice_wins_independent(raw, binary, eligible) == 7
    assert result['orientation'] == 1
    reversed_raw = np.array([0.8, 0.2], np.float64)
    reversed_binary = np.array([0, 1], np.int8)
    reversed_result = candle._rank_direction(reversed_raw, reversed_binary, np.ones(2, bool))
    assert reversed_result['twice_wins'] == october._twice_wins_independent(
        reversed_raw, reversed_binary, np.ones(2, bool)) == 0
    assert reversed_result['orientation'] == -1
