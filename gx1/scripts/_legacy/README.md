This directory contains disabled ENTRY_V10_CTX training scripts.

- `train_entry_v10_ctx_depth_ladder.py` (legacy) is disabled; use the canonical baseline entrypoint `gx1/scripts/train_entry_v10_ctx_depth_ladder.py`.
- `build_entry_v10_ctx_training_dataset.py` (legacy) is disabled; use the canonical builder in `gx1/scripts/build_entry_v10_ctx_training_dataset.py`.

All training should go through the canonical baseline wrapper + canonical trainer. Depth-ladder/L+1 paths are not supported.
