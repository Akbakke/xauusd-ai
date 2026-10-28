import numpy as np
import pytest

from gx1.scripts.audit_direction_unseen_gc_support_v1 import normalized_values,summarize_group


def test_masked_normalization_preserves_binary_and_zeroes_unknown():
    values=np.array([[12.,1.],[20.,0.]],dtype=np.float32);masks=np.array([[1.,1.],[0.,1.]],dtype=np.float32)
    got=normalized_values(values,masks,[10.,0.],[2.,1.],[False,True])
    assert got[0,0]==np.float32(np.arcsinh(1.)) and got[1,0]==0. and np.array_equal(got[:,1],[1.,0.])
    summary=summarize_group(values,masks,[10.,0.],[2.,1.],[False,True],['a','b'],np.array([10,20]))
    assert summary['parents_with_any_nonzero']==1 and summary['fields'][0]['observed_rows']==1


def test_invalid_mask_and_scale_fail():
    with pytest.raises(RuntimeError,match='NORMALIZATION_DOMAIN'):
        normalized_values([[1.]],[[.5]],[0.],[1.],[False])
    with pytest.raises(RuntimeError,match='NORMALIZATION_DOMAIN'):
        normalized_values([[1.]],[[1.]],[0.],[0.],[False])


def test_invalid_binary_value_fails():
    with pytest.raises(RuntimeError,match='NORMALIZATION_DOMAIN'):
        normalized_values([[2.]],[[1.]],[0.],[1.],[True])
