# Canonical GPU telemetry: retired bridge record

Status: **the Windows host-telemetry bridge is not a canonical CUDA
prerequisite.** As of 2026-08-28, offline research uses the pinned native WSL
`nvidia-smi` path owned by `scripts/gx1_capped_run.sh` and
`scripts/gx1_guarded_trainer_exec.sh`.

The authoritative canonical CUDA guard samples once per second and terminates
the trainer process group when any of these facts is exceeded or unavailable:

- core temperature above 70 C;
- actual GPU draw above 251 W;
- resident GPU memory above 12 GiB;
- configured power limit above 390 W; or
- numeric core/power/residency telemetry unavailable.

The current RTX 3090 reports a configured 390 W limit. That is permitted; it
is not an authorization to draw 390 W continuously. The independently sampled
**actual draw** boundary is 251 W in canonical mode. This is a one-watt
measurement tolerance over the reviewed 250 W target, added after a safe
250.48 W sample otherwise stopped a bounded smoke. WSL currently reports
`temperature.memory=N/A`. The guard records this as `memory_observed=false`;
it does not invent a junction temperature and retains the other three hard
limits.

The 2026-08-28 batch-32 canonical V46 attempt demonstrates the intended
behavior: its full data preflight completed, the first CUDA forward peaked at
65 C and 8,873 MiB, then the former 250 W guard stopped it at 250.48 W. No optimizer step,
bundle, validation, TEST access, edge claim, demo or live action resulted.

## Historical bridge material

`scripts/gx1_host_telemetry_bridge_query.sh` and the Windows installer scripts
remain historical/optional host-sensor tooling. They are not invoked by the
canonical runner, cannot supply a file-based substitute for native telemetry,
and must not be used to weaken a GPU guard. The earlier requirements for a
signed bridge response, a physical 250 W driver cap and numeric VRAM-junction
telemetry are retired design history, not current execution constraints.

Any future change to the active safety thresholds requires an explicit operator
decision, a reviewed source commit, a fresh source-bound recipe and a bounded
measurement. It does not grant candidate, OOS, demo, paper or live authority.
