import pytest


def _decimal_dense_binding_case():
    from gx1.contracts import entry_native_direction_decimal_data_v1 as bind
    from gx1.scripts import materialize_entry_direction_dense_v1 as materialize
    from tests.test_materialize_entry_direction_dense_v1 import decimal_grid_fixture
    source, saved, cutoff, grid = decimal_grid_fixture()
    old = materialize.build_dense_targets(source, saved, cutoff_ns=cutoff)
    corrected = materialize.build_decimal_grid_dense_targets(source, saved, decimal_grid=grid, cutoff_ns=cutoff)
    axes = {key: saved[key] for key in bind.AXES}
    return bind, axes, saved, old, corrected


def test_decimal_dense_binding_keeps_legacy_anchor_and_separate_identity():
    bind, axes, saved, old, corrected = _decimal_dense_binding_case()
    with pytest.raises(RuntimeError, match='ANCHOR_BYTES'):
        bind.bind_dense_targets(axes=axes, legacy=saved, dense=corrected)
    joined = bind.bind_decimal_grid_dense_targets(axes=axes, legacy=saved, original_dense=old,
        corrected_dense=corrected, target_identity=bind.DECIMAL_TARGET_IDENTITY)
    assert set(joined) == {'terminal_time_ns', 'observed', 'sign', 'mid_return_bps'}
    assert joined['observed'][0, 0] and joined['sign'][0, 0] == 0 and old['sign'][0, 0] == 1
    assert not joined['observed'][0, 1] and joined['sign'][0, 1] == 0
    assert joined['sign'][0, 2] == 1 and joined['mid_return_bps'][0, 2] > 0
    assert all(not array.flags.writeable for array in joined.values())
    with pytest.raises(RuntimeError, match='DECIMAL_IDENTITY'):
        bind.bind_decimal_grid_dense_targets(axes=axes, legacy=saved, original_dense=old,
            corrected_dense=corrected, target_identity='legacy')


@pytest.mark.parametrize('key', [
    'parent_rows', 'bar_start_ns', 'decision_ns', 'dynamic_entry_m1_row',
    'terminal_time_ns', 'terminal_source_row', 'observed', 'mid_move', 'mid_return_bps', 'sign',
])
def test_decimal_dense_binding_rejects_structural_or_sign_drift(key):
    import copy
    bind, axes, saved, old, corrected = _decimal_dense_binding_case()
    wrong = copy.deepcopy(corrected)
    if key == 'observed':
        wrong[key][0, 0] = not wrong[key][0, 0]
    elif key in ('mid_move', 'mid_return_bps', 'sign'):
        wrong[key][0, 2] *= -1
    else:
        wrong[key].flat[0] += 1
    with pytest.raises(RuntimeError, match='DECIMAL_'):
        bind.bind_decimal_grid_dense_targets(axes=axes, legacy=saved, original_dense=old,
            corrected_dense=wrong, target_identity=bind.DECIMAL_TARGET_IDENTITY)


def test_decimal_dense_binding_rejects_nonfinite_and_unobserved_values():
    import copy
    bind, axes, saved, old, corrected = _decimal_dense_binding_case()
    for key in ('mid_move', 'mid_return_bps'):
        wrong = copy.deepcopy(corrected); wrong[key][0, 2] = float('nan')
        with pytest.raises(RuntimeError, match='DECIMAL_TARGET_DOMAIN'):
            bind.bind_decimal_grid_dense_targets(axes=axes, legacy=saved, original_dense=old,
                corrected_dense=wrong, target_identity=bind.DECIMAL_TARGET_IDENTITY)
    wrong = copy.deepcopy(corrected)
    wrong['sign'][0, 1] = 1; wrong['mid_move'][0, 1] = 1; wrong['mid_return_bps'][0, 1] = 1
    with pytest.raises(RuntimeError, match='DECIMAL_TARGET_DOMAIN'):
        bind.bind_decimal_grid_dense_targets(axes=axes, legacy=saved, original_dense=old,
            corrected_dense=wrong, target_identity=bind.DECIMAL_TARGET_IDENTITY)
