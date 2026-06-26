import numpy as np
import pandas as pd

from gx1.scripts.materialize_build_canonical_features_v1 import add_high_level_basics


def test_high_level_body_pct_is_clipped_for_degenerate_ohlc() -> None:
    n = 120
    df = pd.DataFrame(
        {
            "bid_close": np.full(n, 100.0),
            "ask_close": np.full(n, 100.1),
            "ask_high": np.full(n, 100.2),
            "bid_low": np.full(n, 99.8),
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.5),
            "low": np.full(n, 99.5),
            "close": np.full(n, 100.25),
        }
    )
    df.loc[5, ["open", "high", "low", "close"]] = [90.0, 100.0, 100.0, 110.0]

    out = add_high_level_basics(df.copy())

    assert np.isfinite(out["body_pct"]).all()
    assert out["body_pct"].between(0.0, 1.0).all()
    assert out.loc[5, "body_pct"] == 1.0
