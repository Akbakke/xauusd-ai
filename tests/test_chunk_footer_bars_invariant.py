#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test chunk_footer bars invariant.

Invariant: bars_total_input - bars_processed == tail_holdback_bars when status=="ok".
- Violation (gap != 0, tail_holdback_bars=0): invariant fails.
- Valid (gap=0 or gap==tail_holdback_bars): invariant passes.
"""

import pytest

from gx1.execution.chunk_footer_invariants import check_bars_invariant


def test_bars_invariant_ok_full_completion():
    """Full completion: bars_processed == bars_total_input, tail=0 -> passes."""
    assert check_bars_invariant(
        bars_total_input=70_217,
        bars_processed=70_217,
        tail_holdback_bars=0,
        status="ok",
    ) is True


def test_bars_invariant_ok_tail_holdback():
    """Tail holdback: gap == tail_holdback_bars -> passes."""
    assert check_bars_invariant(
        bars_total_input=100,
        bars_processed=50,
        tail_holdback_bars=50,
        status="ok",
    ) is True


def test_bars_invariant_violation_gap_without_tail():
    """Violation: gap != 0 and tail_holdback_bars=0 -> fails."""
    assert check_bars_invariant(
        bars_total_input=100,
        bars_processed=90,
        tail_holdback_bars=0,
        status="ok",
    ) is False


def test_bars_invariant_violation_gap_mismatch_tail():
    """Violation: gap != tail_holdback_bars -> fails."""
    assert check_bars_invariant(
        bars_total_input=100,
        bars_processed=80,
        tail_holdback_bars=10,
        status="ok",
    ) is False


def test_bars_invariant_stopped_no_enforcement():
    """Stopped status: invariant not enforced -> passes."""
    assert check_bars_invariant(
        bars_total_input=100,
        bars_processed=50,
        tail_holdback_bars=0,
        status="stopped",
    ) is True


def test_bars_invariant_failed_no_enforcement():
    """Failed status: invariant not enforced -> passes."""
    assert check_bars_invariant(
        bars_total_input=100,
        bars_processed=50,
        tail_holdback_bars=0,
        status="failed",
    ) is True
