"""Unit tests for the always-on entry-safety invariant (2026-06-03).

The -2000 incident (06-02) was a same-side pile-up (16 shorts) because the
PURE_PHASE6-gated cap was disabled. evaluate_entry_safety is the pure, always-on
replacement: no-short-in-long + no-pile-up. These tests pin the contract.
"""

from gx1.execution.v12_paper_runner import evaluate_entry_safety


def test_clean_book_allows_entry():
    assert evaluate_entry_safety("long", n_same_side=0, n_opposing=0, hard_max_same_side=2) == (True, "ok", 0)


def test_opposing_side_blocks_short_in_long():
    # one long open, want to go short -> blocked (no-short-in-long)
    ok, reason, detail = evaluate_entry_safety("short", n_same_side=0, n_opposing=1, hard_max_same_side=2)
    assert ok is False and reason == "blocked_opposing_side_open" and detail == 1


def test_opposing_side_takes_priority_over_same_side():
    # both violated -> opposing reported first (it is the harder invariant)
    ok, reason, _ = evaluate_entry_safety("long", n_same_side=5, n_opposing=2, hard_max_same_side=2)
    assert ok is False and reason == "blocked_opposing_side_open"


def test_same_side_hard_cap_blocks_pileup():
    # at the cap -> blocked (this is the actual -2000 fix)
    ok, reason, detail = evaluate_entry_safety("short", n_same_side=2, n_opposing=0, hard_max_same_side=2)
    assert ok is False and reason == "blocked_hard_same_side_cap" and detail == 2


def test_same_side_below_cap_allows():
    assert evaluate_entry_safety("short", n_same_side=1, n_opposing=0, hard_max_same_side=2)[0] is True


def test_cap_of_one_allows_first_blocks_second():
    assert evaluate_entry_safety("long", 0, 0, 1)[0] is True   # first same-side ok
    assert evaluate_entry_safety("long", 1, 0, 1)[0] is False  # second blocked


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} entry-safety invariant tests PASS")
    sys.exit(0)
