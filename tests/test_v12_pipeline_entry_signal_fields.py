from gx1.execution.v12_pipeline import _entry_signal_fields_from_candidate


def test_entry_signal_fields_expose_margin_for_runner_sizing():
    fields = _entry_signal_fields_from_candidate(
        {
            "margin": 0.42,
            "p_hat": 0.72,
            "uncertainty_score": 0.28,
            "entropy_v1": 0.81,
        }
    )

    assert fields["margin"] == 0.42
    assert fields["margin_top1_top2"] == 0.42
    assert fields["p_hat"] == 0.72
    assert fields["uncertainty_score"] == 0.28
    assert fields["entropy_v1"] == 0.81


def test_entry_signal_fields_fallback_to_margin_top1_top2():
    fields = _entry_signal_fields_from_candidate({"margin_top1_top2": 0.35})

    assert fields["margin"] == 0.35
    assert fields["margin_top1_top2"] == 0.35
