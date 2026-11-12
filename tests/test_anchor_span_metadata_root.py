"""Metadata redirection cannot reopen a consumed output root."""
import pytest
from gx1.contracts import market_state_ta_anchor_span_short_learning_v1 as scope


@pytest.mark.parametrize('marker', ['RESULT.json', 'RUNTIME/EXECUTION_BINDING.json'])
def test_consumed_target_is_rejected_before_metadata_reads(monkeypatch, tmp_path, marker):
    root = tmp_path/'new';root.mkdir()
    path = root/marker;path.parent.mkdir(exist_ok=True);path.write_text('{}')
    monkeypatch.setattr(scope, 'ROOT', tmp_path/'old')
    monkeypatch.setattr(scope, '_read', lambda ref: pytest.fail('read before guard'))
    with pytest.raises(RuntimeError, match='UNCLAIMED_PREPARATION'):
        scope.prepare_metadata(output_root=root)


def test_new_root_does_not_reopen_original_consumed_root(monkeypatch, tmp_path):
    old = tmp_path/'old';old.mkdir();(old/'RESULT.json').write_text('{}')
    new = tmp_path/'new';new.mkdir()
    monkeypatch.setattr(scope, 'ROOT', old)
    class StopRead(Exception):
        pass
    def stop(ref):
        assert ref == scope.PARENT_RECIPE
        raise StopRead
    monkeypatch.setattr(scope, '_read', stop)
    with pytest.raises(StopRead):
        scope.prepare_metadata(output_root=new)
    assert (old/'RESULT.json').read_text() == '{}'
    assert list(new.iterdir()) == []


@pytest.mark.parametrize('kind', ['outside', 'symlink'])
def test_output_must_be_regular_declared_sibling(monkeypatch, tmp_path, kind):
    parent = tmp_path/'runs';parent.mkdir()
    real = tmp_path/'outside';real.mkdir()
    target = real
    if kind == 'symlink':
        target = parent/'link';target.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(scope, 'ROOT', parent/'old')
    monkeypatch.setattr(scope, '_read', lambda ref: pytest.fail('read before guard'))
    with pytest.raises(RuntimeError, match='METADATA_OUTPUT_ROOT'):
        scope.prepare_metadata(output_root=target)
