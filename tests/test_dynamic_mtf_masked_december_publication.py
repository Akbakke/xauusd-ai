"""The data-only publication must stop on a new empty coordinate."""
import json

import pytest

from gx1.audit import run_dynamic_mtf_train_prefix_data as runner
from gx1.audit.dynamic_mtf_gc_d1_bear_trace import BEAR_FIELDS


@pytest.mark.parametrize("stage_result", ["eligible", "additional_blocker", "inconsistent_fit"])
def test_masked_publication_keeps_every_other_field_mandatory(tmp_path, monkeypatch, stage_result):
    from gx1.audit import dynamic_mtf_december_normalization as adapter
    from gx1.contracts.entry_native_direction_prefix_normalization_v1 import WINDOWS, december_zero_support_profile

    profile = december_zero_support_profile()
    fields = ["gc_d1_observed_proxy_ob_bear_" + suffix for suffix in BEAR_FIELDS]
    geometry = {"parents": 41866, "parent_rows_sha256_le_i64": "1" * 64,
                "bar_start_ns_sha256_le_i64": "2" * 64, "issue_ns_sha256_le_i64": "3" * 64}
    stage = {"profile": {k: profile[k] for k in ("name", "contract_sha256")}, "masked_fields": fields,
             "fit_cutoff_ns": 1733011200000000000, "held_out_rows": 5736, "limits": {"max_rchar_bytes": 2**26}}
    policy = {runner.DECEMBER_STAGE: {"geometry": {"fit": geometry}}}
    blocked = stage_result == "additional_blocker"
    surfaces = ["signal", "ctx_cont", *["mtf_" + tf.lower() for tf in WINDOWS], *["gc_mtf_" + tf.lower() for tf in WINDOWS]]
    support = {}
    for surface in surfaces:
        parked = (["ctx_cont." + f for f in fields] if surface == "signal" else
                  fields if surface in ("ctx_cont", "gc_mtf_d1") else [])
        n = 0 if surface == "mtf_h1" and stage_result != "eligible" else 3
        support[surface] = {"field_names": ["ordinary", *parked], "observed_counts": [n, *([0] * len(parked))],
                            "population_rows": 3}
    report = {"fit_allowed": not blocked, "blockers": ["mtf_h1.ordinary"] if blocked else [],
              "proof": {"prefix": {"rows": 41866, "cutoff_ns": stage["fit_cutoff_ns"],
                        "parent_rows_sha256": "1" * 64, "bar_start_ns_sha256": "2" * 64,
                        "decision_ns_sha256": "3" * 64}}, "support": support}
    fitted = None if blocked else {"zero_support_profile": profile, "prefix_qualification": report,
                                   "native_launch_authorized": False, "genuine_model_fit_authorized": False}
    monkeypatch.setattr(adapter, "fit_masked_december_prefix", lambda *args: (report, fitted, {}))
    monkeypatch.setattr(runner, "_bind_masked_feature_view", lambda *_: {"synthetic_view": True})
    if stage_result == "inconsistent_fit":
        with pytest.raises(RuntimeError, match="MASKED_NORMALIZATION_ACCEPTANCE"):
            runner._masked_prefix_result(policy, stage, tmp_path, 2**20)
        assert not list(tmp_path.iterdir())
        return
    result = runner._masked_prefix_result(policy, stage, tmp_path, 2**20)
    assert result["normalization_fits"] == (0 if blocked else 1)
    assert (tmp_path / "PREFIX_NORMALIZATION.json").exists() == (not blocked)
    assert (tmp_path / "FEATURE_VIEW.json").exists() == (not blocked)
    assert json.loads((tmp_path / "QUALIFICATION.json").read_text())["blockers"] == report["blockers"]
    assert result["held_out_outcomes_used"] is False and result["reserved_december_parents"] == 5736


@pytest.mark.parametrize('masked', [False, True])
def test_window_batch_forwards_bound_observer_only_for_masked_profile(monkeypatch, masked):
    from gx1.contracts import entry_native_direction_feature_view_v1 as feature
    from gx1.scripts.run_entry_native_direction_prefix_fit_window_v1 import _batch

    tag, received = object(), []
    prepared = {'arrays':dict.fromkeys(('parent_rows', 'bar_start_ns', 'decision_ns')),
                'current':dict.fromkeys(('snap', 'ctx_cont', 'ctx_cat')),
                'gc':dict.fromkeys(('surfaces', 'layout')), 'entry_sequence_source':None, 'cache':None,
                'observer_axes':tag}

    class GatherOnly(Exception):
        pass

    def inspect(view, **kwargs):
        received.append(kwargs)
        raise GatherOnly

    monkeypatch.setattr(feature, 'gather_native_direction_feature_batch', inspect)
    with pytest.raises(GatherOnly):
        _batch(prepared, {}, {'zero_support_profile': {}} if masked else {}, [], None)
    assert ('observer_axes' in received[0]) is masked
    if masked:
        assert received[0]['observer_axes'] is tag
