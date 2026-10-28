import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from gx1.scripts import audit_entry_direction_gc_projection_state_v1 as audit


def test_matrix_change_distinguishes_updates_and_rank():
    first = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    last = first.clone(); last[0, 1] = 1.
    result = audit.matrix_change(first, last, shape=(2, 3), rank=True)
    assert result['changed_elements'] == 1 and result['initial_numerical_rank'] == result['trained_numerical_rank'] == 2
    assert result['delta_l2'] == 1.0
    with pytest.raises(RuntimeError, match='TENSOR'):
        audit.matrix_change(first, last, shape=(3, 2))
    last[1, 2] = float('nan')
    with pytest.raises(RuntimeError, match='TENSOR'):
        audit.matrix_change(first, last, shape=(2, 3))


def test_analyze_joins_context_gc_and_all_five_timeframes():
    family = 'smc_liquidity_encoder'; tfs = ('M5', 'M15', 'H1', 'H4', 'D1'); width = 2
    profile = {'local_specialists': [family], 'mtf_specialists': [family],
        'ctx_cont_indices_by_specialist': {family: [0, 1, 2]},
        'gc_mtf_family_indices': {tf: {family: [0, 1, 2]} for tf in tfs}}
    first = {'backbone.specialist_ctx_cont_proj.' + family + '.weight': torch.zeros((width, 3))}
    for name in ('specialist_out', 'cross_tf_out', 'family_tf_cooperation_out'):
        first['backbone.' + name + '.weight'] = torch.zeros((width, width))
    for tf in tfs:
        key = tf.lower() + '__' + family
        first['backbone.gc_mtf_family_proj.' + key + '.weight'] = torch.zeros((width, 3))
        first['backbone.gc_mtf_feature_context_gate.' + key + '.weight'] = torch.zeros((3, width))
        first['backbone.gc_mtf_feature_context_gate.' + key + '.bias'] = torch.zeros(3)
    last = {key: value.clone() for key, value in first.items()}
    last['backbone.gc_mtf_feature_context_gate.m5__' + family + '.bias'][0] = .5
    result = audit.analyze(first, last, profile, width=width)
    assert set(result['gc_projection_and_context_gates']) == set(tfs)
    assert result['gc_projection_and_context_gates']['M5'][family]['context_gate_bias']['changed_elements'] == 1
    assert result['gc_projection_and_context_gates']['H1'][family]['context_gate_bias']['changed_elements'] == 0
    missing = dict(last); del missing['backbone.cross_tf_out.weight']
    with pytest.raises(RuntimeError, match='MODEL_KEY_PARITY'):
        audit.analyze(first, missing, profile, width=width)


def test_weights_only_checkpoint_load_works_under_exact_file_guard(tmp_path):
    program = '''
import io, sys, torch
from pathlib import Path
from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
p=Path(sys.argv[1]); out=p.parent
buf=io.BytesIO(); torch.save({'x':torch.tensor([1.])},buf); buf.seek(0)
torch.load(buf,map_location='cpu',weights_only=True)
sys.addaudithook(candle.make_guard(out,[str(p)],[]))
state=torch.load(p,map_location='cpu',weights_only=True)
assert state['weight'].tolist()==[1.,2.]
'''
    with tempfile.TemporaryDirectory(dir=tmp_path.parent, prefix='gx1_projection_guard_') as directory:
        checkpoint = Path(directory) / 'synthetic.pt'
        torch.save({'weight': torch.tensor([1., 2.])}, checkpoint)
        result = subprocess.run([sys.executable, '-B', '-c', program, str(checkpoint)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_unchanged_fields_keep_bound_name_order_and_normalization_support():
    family = 'smc_liquidity_encoder'; tfs = ('M5', 'M15', 'H1', 'H4', 'D1')
    names = ['first', 'second', 'third']
    profile = {'local_specialists': [family], 'mtf_specialists': [family],
        'physical_field_names': {'ctx_cont': names}, 'ctx_cont_indices_by_specialist': {family: [0, 1, 2]},
        'gc_mtf_field_names': {tf: names for tf in tfs},
        'gc_mtf_family_indices': {tf: {family: [0, 1, 2]} for tf in tfs}}
    def surface():
        return {'field_names': names, 'binary_mask': [0, 1, 0], 'train_transformed_min': [0., 0., -1.],
            'train_transformed_max': [0., 1., 1.], 'fit_row_count': 4}
    normal = {'base_normalization': {'surfaces': {'ctx_cont': surface()}},
        'gc_mtf_history': {'normalization': {'surfaces': {'gc_mtf_' + tf.lower(): surface() for tf in tfs}}}}
    first = {'backbone.specialist_ctx_cont_proj.' + family + '.weight': torch.zeros((128, 3))}
    for tf in tfs:
        key = tf.lower() + '__' + family
        first['backbone.gc_mtf_family_proj.' + key + '.weight'] = torch.zeros((128, 3))
        first['backbone.gc_mtf_feature_context_gate.' + key + '.weight'] = torch.zeros((3, 128))
        first['backbone.gc_mtf_feature_context_gate.' + key + '.bias'] = torch.zeros(3)
    last = {key: value.clone() for key, value in first.items()}
    last['backbone.specialist_ctx_cont_proj.' + family + '.weight'][0, 1] = 1
    result = audit.map_unchanged_fields(first, last, profile, normal)
    rows = result['gc_history_unchanged_coordinates']['M5'][family]
    assert [r['name'] for r in rows] == names
    assert rows[0]['projection_column_unchanged'] and rows[0]['also_unchanged_current_context']
    assert rows[0]['train_transformed_min'] == rows[0]['train_transformed_max'] == 0
    assert rows[1]['binary'] and not rows[1]['also_unchanged_current_context']
    assert result['normalization_fit_rows']['M5'] == 4
    normal['gc_mtf_history']['normalization']['surfaces']['gc_mtf_m5']['field_names'] = list(reversed(names))
    with pytest.raises(RuntimeError, match='GC_NAME_ORDER'):
        audit.map_unchanged_fields(first, last, profile, normal)
