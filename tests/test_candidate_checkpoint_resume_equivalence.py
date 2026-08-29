from __future__ import annotations

from gx1.scripts.verify_candidate_checkpoint_resume_v1 import run_equivalence


def test_candidate_resume_is_exact_across_a_fresh_python_process() -> None:
    report = run_equivalence()
    assert report["decision"] == "PASS"
    assert report["global_optimizer_steps"] == 8
    assert report["max_abs_model_weight_difference"] <= 1e-6
    assert report["max_abs_optimizer_state_difference"] <= 1e-6
    assert report["max_abs_prediction_difference"] <= 1e-6
