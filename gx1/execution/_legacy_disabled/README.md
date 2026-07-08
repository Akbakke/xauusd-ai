# gx1/execution/_legacy_disabled/ — parked dead execution modules

Cleanup-wave continuation 2026-07-08 (user vedtak 2026-07-08 "Samtidig rydd legacy";
follows the 2026-07-07 park wave, see
`GX1_DATA/reports/cand4_julyext_evidence_20260705/cleanup_wave_20260707/PARK_AND_POINTER_FIX_REPORT.md`).

These modules were reported DEAD (zero references) in the 2026-07-07 orphan scan but were
NOT parked then because `gx1/execution/` is protected core. After the smart-chain contract
flip (commit d98bc61e, 2026-07-08) legacy entry serving is formally superseded; each file
below was individually RE-verified 2026-07-08 with a fresh `scan_orphans_v1.py` run
(corpus: git-tracked repo incl. `.claude/` + `Makefile`, `~/.config/systemd/user/`,
`~/.claude/settings.json` + hooks, `/home/andre2/CLAUDE.md`, `GX1_DATA/config/`,
cand4 evidence scripts, crontab=none) plus manual dotted-path (`execution.<stem>`) grep:
ZERO live code references, zero doc references, zero test references, zero dynamic-import
hits, not exported by `gx1/execution/__init__.py`.

Moves are `git mv` — fully reversible. Do NOT import from this directory (AGENTS.md:
no shims importing from legacy).

## Files

- `fast_path_verification.py` — replay-mode fast-path env/flag guard for the pre-V12
  TRUTH-era replay runner. Consumer chain (`replay_chunk.py`) removed 2026-05-24
  (f4c2a0f9); the v12 phase6/paper chain never used it. Zero refs since.
- `killchain_export.py` — deterministic KILLCHAIN_EXPORT.json entry-funnel telemetry
  snapshot for the TRUTH-era replay runner. Consumers removed 2026-05-24 (f4c2a0f9).
  No successor (v12 journals telemetry inline).
- `regime_histogram.py` — per-session regime-distribution tracking for replay
  calibration coverage (TRUTH era, c2d1fdbb 2026-01-26). Replay runner removed
  2026-05-24; superseded by regime handling in the v12 chain (BASE34 REGIME_V4 cols).
- `replay_features.py` — replay↔live tag-parity helpers (session/vol/trend) for the old
  replay mode. Superseded by the prebuilt BASE34/cv3 parity chain (train==serve via
  prebuilt features); consumers removed 2026-05-24.
- `runtime_mode.py` — replay-vs-live fail-fast/tolerant policy helper for the old replay
  runner (TRUTH era). Consumers removed 2026-05-24; v12 uses env flags
  (GX1_PURE_PHASE6 etc.) directly. Zero refs.
- `tick_watcher.py` — callback-based live tick monitor for TP/SL/BE triggers. Only
  historical consumer is the archived
  `gx1/legacy/_legacy_disabled/oanda_demo_runner__forbidden_exit_modes__archived.py`;
  the v12 exit chain decides per M1 bar (V3 exit transformer + Exit-IQL), no tick
  watcher. Dead since the demo-runner was archived.
- `exit_critic_controller.py` — ExitCritic V1 runtime integration for the old
  `exit_manager.py`. Exit decisions are now V3 exit transformer + Exit-IQL via
  `v12_pipeline.make_exit_decision`; ExitCritic V1 was never part of the v12 chain.
  Zero refs (last touched 07a35471, 2026-02-27).
- `chunk_footer_invariants.py` — pure invariant check (bars_total_input − bars_processed
  == holdback) for replay chunk footers. The replay-chunk chain was removed 2026-05-24
  (f4c2a0f9). Zero refs.
- `chunk_footer_writer.py` — atomic dumb writer of `chunk_footer.json` for
  `replay_chunk.py`; same removed 2026-05-24 replay-chunk chain. Zero refs.

## Explicitly NOT parked (still in gx1/execution/)

- `broker_client.py` — scan LIVE only via an `_archive_artifacts` script; on the
  serving-wave hold list (avvent), decide in a deliberate protected-core wave.
- `oanda_backfill.py` — has a COLLECTING test (`tests/test_backfill_paging_reaches_target.py`),
  NOT dead; also on the serving-wave hold list.
