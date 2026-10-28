import numpy as np
import pytest

from gx1.scripts.audit_direction_candle_spread_v1 import describe


def test_exact_spread_hurdle_counts_profit_equal_and_loss():
    b0=np.array([100.,100.,100.]);a0=np.array([100.002]*3)
    b1=np.array([100.003,100.002,100.001]);a1=np.array([100.005,100.004,100.003])
    origin=np.array([200002]*3,dtype=np.int64);delta=np.array([6,4,2],dtype=np.int64)
    result=describe(b0,a0,b1,a1,np.ones(3,np.int8),10000*delta/origin)
    assert (result['proxy_above_spread_count'],result['proxy_equal_spread_count'],result['proxy_below_spread_count'])==(1,1,1)


def test_rejects_wrong_decimal_grid_and_wrong_target():
    b0=np.array([100.]);a0=np.array([100.002]);b1=np.array([100.003]);a1=np.array([100.005])
    with pytest.raises(RuntimeError,match='THREE_DECIMAL_GRID'):
        describe(b0,a0,b1,np.array([100.0055]),np.ones(1,np.int8),np.array([0.]))
    with pytest.raises(RuntimeError,match='TARGET_PARITY'):
        describe(b0,a0,b1,a1,np.array([-1],np.int8),np.array([0.]))
