from itertools import product

import numpy as np

from gx1.audit.dynamic_mtf_train_target_review import supported_moves


def test_witnesses_use_only_opens_closes_and_provably_ordered_extremes():
    # Either high/low ordering is compatible with each of these M1 bars.
    bars = [(5, 10, 0, 5), (4, 12, 3, 11), (14, 14, 2, 2)]
    for start in range(len(bars)):
        for end in range(start + 1, len(bars) + 1):
            window = bars[start:end]
            rise, fall = supported_moves(*(np.array([values]) for values in zip(*window)))
            possible = []
            for orders in product((False, True), repeat=len(window)):
                path = []
                for (opening, high, low, close), high_first in zip(window, orders):
                    path.extend((opening, high, low, close) if high_first else (opening, low, high, close))
                possible.append((max(price - min(path[:i + 1]) for i, price in enumerate(path)),
                                 max(max(path[:i + 1]) - price for i, price in enumerate(path))))
            assert (rise.item(), fall.item()) == tuple(map(min, zip(*possible)))
