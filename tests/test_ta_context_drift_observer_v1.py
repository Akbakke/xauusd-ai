"""Selection uses retained chronological counts only; no models or data."""
import pytest
from gx1.audit.observe_market_state_ta_context_drift_v1 import select_slots


def counts(october=1459, november=1310, december=1327):
    return {"LATER/2024-"+month: [n, 0, 0, 0] for month, n in
            (("10", october), ("11", november), ("12", december))}


def test_selection_preserves_native_batches_and_excludes_month_boundaries():
    slots = select_slots(counts())
    assert [x["panel_slot"] for x in slots[:8]] == [0, 12, 25, 38, 51, 64, 77, 90]
    assert [x["panel_slot"] for x in slots[8:]] == [174, 185, 197, 208, 220, 231, 243, 255]
    for item in slots:
        left = item["panel_slot"] * 16
        lo, hi = (0, 1459) if item["month"] == "2024-10" else (2769, 4096)
        assert lo <= left < left+16 <= hi


def test_session_distribution_cannot_change_selection():
    redistributed = {k: [0, 0, sum(v), 0] for k, v in counts().items()}
    assert select_slots(redistributed) == select_slots(counts())


def test_insufficient_support_or_wrong_population_is_rejected():
    with pytest.raises(RuntimeError, match="MONTH_SUPPORT"):
        select_slots(counts(october=100, november=2669))
    with pytest.raises(RuntimeError, match="SLOT_POPULATION"):
        select_slots(counts(december=1328))
