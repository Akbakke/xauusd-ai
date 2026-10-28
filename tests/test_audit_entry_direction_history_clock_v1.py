import numpy as np
import pytest

from gx1.scripts import audit_entry_direction_history_clock_v1 as audit


def test_window_counts_distinguish_missing_time_from_row_count():
    t = np.array([0, 1, 2, 5, 6, 7, 8], np.int64) * audit.BAR_NS
    starts = t[[2, 3, 5]]
    g = audit.history_clock_gaps(t, starts, window_rows=3)
    np.testing.assert_array_equal(g['gap_count'], [0, 1, 0])
    np.testing.assert_array_equal(g['recent_gap_count'], [0, 1, 0])
    np.testing.assert_array_equal(g['largest_missing_minutes'], [0, 10, 0])
    np.testing.assert_array_equal(g['elapsed_minutes'], [15, 25, 15])
    out = audit.summarize(g)
    assert out['parents'] == 3 and out['with_any_gap'] == 1


def test_parent_not_found_or_source_clock_out_of_order_is_rejected():
    t = np.arange(6, dtype=np.int64) * audit.BAR_NS
    with pytest.raises(RuntimeError, match='PARENT_SOURCE_JOIN'):
        audit.history_clock_gaps(t, np.array([int(t[2] + audit.BAR_NS//2)], np.int64), window_rows=3)
    with pytest.raises(RuntimeError, match='CLOCK_DOMAIN'):
        audit.history_clock_gaps(t[::-1], np.array([t[-1]], np.int64), window_rows=3)


def test_cohorts_retain_their_own_outcome_masks_and_replay_scores():
    from gx1.contracts.entry_native_direction_context_report_v1 import _panel
    from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
    n = 6
    decision = np.int64(1735689600000000000) + np.arange(n, dtype=np.int64) * audit.BAR_NS
    sign = np.tile(np.array([[1], [-1], [1], [-1], [1], [-1]], np.int8), (1,60))
    pred = {'parent_rows': np.arange(n, dtype=np.int64), 'decision_ns': decision,
        'terminal_time_ns': decision[:, None] + np.arange(1,61, dtype=np.int64)[None,:] * audit.MINUTE_NS,
        'observed': np.ones((n,60), np.bool_), 'sign': sign,
        'logit_up': sign.astype(np.float32)}
    report = {'model_metrics': {'per_horizon': [None] * 60}}
    for j in (4,59):
        panel = _panel(pred['logit_up'][:,j], sign[:,j], pred['observed'][:,j], pred['terminal_time_ns'][:,j] < audit.LATER_CUTOFF_NS)
        report['model_metrics']['per_horizon'][j] = panel
    out = audit.cohort_scores(pred, np.array([0,0,0,1,1,1],np.int64), report)
    assert len(out) == 2 and all(x['groups'][0]['eligible'] == x['groups'][1]['eligible'] == 3 for x in out)
    assert all(x['groups'][0]['auc'] == x['groups'][1]['auc'] == 1 for x in out)
    report['model_metrics']['per_horizon'][4]['accuracy'] = 0
    with pytest.raises(RuntimeError, match='ORIGINAL_SCORE_REPLAY'):
        audit.cohort_scores(pred, np.array([0,0,0,1,1,1],np.int64), report)


def test_parquet_time_column_and_clock_math_run_under_guards(tmp_path):
    import subprocess
    import sys
    code = r'''
import sys
from pathlib import Path
import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pyarrow.dataset
from gx1.contracts import gx1_capped_execution_v1, immutable_event_authority_v1, local_random_access_campaign_v2
from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
from gx1.contracts.entry_native_direction_context_report_v1 import _panel
from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
from gx1.scripts.audit_entry_direction_history_clock_v1 import history_clock_gaps, summarize, BAR_NS
out = Path(sys.argv[1]).resolve(); path = out/'synthetic.parquet'
source = np.array([0,1,2,5,6,7],np.int64)*BAR_NS
parquet.write_table(arrow.table({'time':arrow.array(source,type=arrow.timestamp('ns',tz='UTC'))}),path)
sys.dont_write_bytecode = True
sys.addaudithook(candle.make_guard(out,[str(path)],[],synthetic=True))
sys.setprofile(candle.no_queries_or_models)
with path.open('rb') as handle: table = parquet.read_table(handle,columns=['time'],use_threads=False)
column = table['time']; times = np.ascontiguousarray(column.cast(arrow.timestamp('ns',tz=column.type.tz)).cast(arrow.int64()).to_numpy())
r = summarize(history_clock_gaps(times,source[[2,3,5]],window_rows=3))
assert r['with_any_gap'] == 1
sys.setprofile(None)
'''
    result = subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stderr
