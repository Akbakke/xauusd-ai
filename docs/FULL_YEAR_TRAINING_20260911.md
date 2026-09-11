# Full-year training correction

The completed V21 run covers 16,384 of the 65,295 TRAIN Entry pairs in
2025-06-01 through 2026-05-31. Its 1,024 steps are a sampled technical smoke;
they do not establish completion of the requested full-year epoch.

Full epochs now select exactly one population-length interval from the existing
outcome-blind sampler stream. Partial sampler chunks are included at both ends,
so every Entry pair occurs exactly once per full epoch. The original four
transitions per Entry, both-side supervision, anchor bytes, features, target
construction and normalization remain unchanged. Full Entry coverage does not
mean enumerating every possible lifecycle transition.

The production dataset adapter exposes this full-population mode and its exact
entry order. The schedule is suitable for all subsequent full epochs and does
not repeat the tail of a chunk merely to fill a batch. At batch 16, the real
year needs 4,081 steps, including a final batch of 15 pairs.

The first 16,384 entries of full epoch zero are the original sampler's chunk
zero. Reusing their completed checkpoint requires an exact prefix witness and
strict preservation of model, target, optimizer, scheduler, EMA and RNG state.
The frozen adce execution checkout and its checkpoint remain unchanged.

Integration still required before launch: bind the full-year session and
checkpoint continuation to the runner/campaign, use full-cohort counts in
progress/final authority, and evaluate the completed year. Then implement the
30-epoch ceiling and VAL early stopping with a correctly coupled Entry/Exit
net-Bps metric. The prepared V23 VAL is not a completed-year evaluation and
must not be launched as one. TEST remains sealed.

Focused scheduler/adapter regressions cover exact legacy-stream preservation,
complete coverage across partial chunk boundaries, production row ordering,
and returning to legacy mode. These tests do not claim completed training.
