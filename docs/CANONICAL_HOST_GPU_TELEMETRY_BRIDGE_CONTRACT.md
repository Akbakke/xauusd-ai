# Canonical GPU telemetry: retired bridge record

Status: **the Windows host-telemetry bridge is not a canonical CUDA
prerequisite.** As of 2026-08-28, offline research uses the pinned native WSL
`nvidia-smi` path owned by `scripts/gx1_capped_run.sh` and
`scripts/gx1_guarded_trainer_exec.sh`.

The authoritative canonical CUDA guard samples once per second and terminates
the trainer process group when any of these facts is exceeded or unavailable:

- core temperature above 70 C;
- actual GPU draw above 220 W;
- resident GPU memory above 12 GiB;
- configured power limit above 390 W; or
- numeric core/power/residency telemetry unavailable.

The current RTX 3090 reports a configured 390 W limit. That is permitted; it
is not an authorization to draw 390 W continuously. The independently sampled
**actual draw** boundary is 220 W in canonical mode. A physical 220 W driver
limit was requested through pinned WSL `nvidia-smi` and rejected with
`Insufficient Permissions`; therefore this boundary is a one-second process
stop, not a hardware throttle. The first permitted continuation is exactly one
fresh batch-8 V46 canonical smoke. The configured 390 W driver ceiling remains
never a workload authorization. WSL currently reports
`temperature.memory=N/A`. The guard records this as `memory_observed=false`;
it does not invent a junction temperature and retains the other three hard
limits.

The 2026-08-28 batch-32 canonical V46 attempts demonstrate the intended
behavior. After the 250/251 W draw stops, the historical 300 W guard allowed a
10,000-row recipe to reach 71 C / 263.77 W / 8,951 MiB and a
1,000-row recipe to reach 71 C / 261.33 W / 8,951 MiB. Both process groups
were stopped safely for core temperature; no optimizer step, bundle,
validation, TEST access, edge claim, demo or live action resulted. The
subsequent batch-8/32-row smoke completed four CUDA optimizer steps, validation
and active episode movement proof under the same guard (65 C / 211.77 W /
8,751 MiB), then its bundle loader imposed a candidate-only Exit gate on
smoke. Commit `31f376ca` scopes the gate by profile. The local CUDA route
permits exactly one fresh recipe-bound/dry-run-checked repeat, then
returns to hold.

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
