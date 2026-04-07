# Entry/Exit Timeframe Contract

This note is the repo truth for how `seq_len` and `window_len` should be read in the canonical TRUTH pipeline.

## Canonical truth

- Entry runs on `M5` model bars.
- Exit runs on raw `M1` bars after a trade is open.
- `seq_len` for entry is not the same concept as `window_len` for exit.
- Treating `window_len` as a universal "bar count" across entry and exit is incorrect.

## Entry contract

- Policy timeframe is `M5` in [GX1_TRUTH_REPLAY_V10_CTX.yaml](/home/andre2/src/GX1_ENGINE/gx1/configs/policies/canonical_truth/GX1_TRUTH_REPLAY_V10_CTX.yaml#L16).
- Canonical truth roots split raw and model data in [canonical_truth_signal_only.json](/home/andre2/src/GX1_ENGINE/gx1/configs/canonical_truth_signal_only.json#L3).
  Raw root is M1 and model root is M5.
- Current entry bundles declare `seq_len: 30` in [bundle_metadata.json](/home/andre2/GX1_DATA/models/models/entry_v10_ctx/ENTRY_V10_CTX__RETRAIN_20260403_SWEEP_A/bundle_metadata.json#L20).
- Training dataset builder constructs entry samples as `seq = ...  # [seq_len, 7]` in [build_entry_v10_ctx_training_dataset.py](/home/andre2/src/GX1_ENGINE/gx1/scripts/build_entry_v10_ctx_training_dataset.py#L1766).
- Replay only calls `evaluate_entry(...)` on model bars in [oanda_demo_runner.py](/home/andre2/src/GX1_ENGINE/gx1/execution/oanda_demo_runner.py#L12447).

Bottom line:
- `entry seq_len=30` means `30 x M5` bars.

## Exit contract

- Canonical exit policy sets `window_len: 8` in [EXIT_TRANSFORMER_ONLY_V0.yaml](/home/andre2/src/GX1_ENGINE/gx1/configs/policies/canonical_truth/exits/EXIT_TRANSFORMER_ONLY_V0.yaml#L15).
- Current exit bundle declares `window_len: 8` and `exit_ml_io_version: EXIT_IO_V1_CTX36` in [exit_transformer_config.json](/home/andre2/GX1_DATA/models/exit_transformer_v0/EXIT_TRANSFORMER__RETRAIN_20260402_EARNED_EDGE_FLOOR_V1/exit_transformer_config.json#L2).
- Runtime hard-guards `window_len == 8` in [exit_manager.py](/home/andre2/src/GX1_ENGINE/gx1/execution/exit_manager.py#L911).
- Runtime builds the actual transformer price window from `candles.tail(window_len)` in [exit_manager.py](/home/andre2/src/GX1_ENGINE/gx1/execution/exit_manager.py#L2685).
- Sanity retrain path also hardcodes `window_len=8` in [run_truth_e2e_sanity.py](/home/andre2/src/GX1_ENGINE/gx1/scripts/run_truth_e2e_sanity.py#L2489).

Bottom line:
- `exit window_len=8` means `8 x M1` raw bars in the current canonical exit contract.

## What counts as a new contract

If we want exit to see materially more raw history before and during the trade, that is not a small config tweak on the current bundle.

That should be treated as one of these:

1. A new explicit exit contract and retrain direction.
   Example: new IO version, new model artifact, new runtime guard, new trainer default, new policy lane.
   Current next-step candidate in this repo is `EXIT_IO_V2_CTX36_M1L512`, which means `512 x M1` bars.

2. A same-contract feature extension.
   Example: keep `EXIT_IO_V1_CTX36` and `window_len=8`, but add longer-horizon context through derived features that are computed from broader M1 history and packed into the existing feature vector contract.

## Practical rule

- Changing entry `seq_len` affects the M5 entry transformer contract.
- Changing exit `window_len` affects the M1 exit transformer contract.
- Do not mix these knobs or assume they mean the same time horizon.
