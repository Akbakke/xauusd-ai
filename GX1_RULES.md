# GX1 scope freeze

This is the only active GX1 scope.

## Allowed

```text
immutable XAUUSD snapshot
        -> one shared featurebase
        -> Entry M5 head: LONG / SHORT / FLAT
        -> Exit M1 head: HOLD / EXIT_NOW
        -> offline train / OOS / replay evidence
```

- Entry uses M5. Exit uses M1.
- Both consume the same eight causal specialist families and the same field
  ownership, formula, normalization and lineage contracts. Resolution is the
  only intentional difference.
- A completed featurebase may be reused only when its source manifests,
  formula inventory, schema and hashes match exactly.
- A new tail may be appended only after exact overlap and source-hash proof.
- The model is the only direction authority. No live rule, threshold,
  overlay, fallback, synthetic decision or duplicate feature owner is allowed.

## Forbidden

- No live, paper, demo or broker operation.
- No daemon, polling loop, watchdog, live-tail admission or launcher work.
- No continual adaptation, drift handling, online weight update or promotion.
- No full-history recompute for a tail update when an exact immutable cache or
  overlap-verified append is available.
- No new architecture, feature family, compatibility lane, versioned copy or
  operational route unless this scope is deliberately changed first.

If the source, cache, overlap, schema or model evidence is invalid, stop
closed. Do not simplify the evidence by guessing, filling, clipping or routing
around the failure. No trained edge is claimed until untouched OOS evidence
proves it.

All other Markdown, handover text and historical live/adaptation material is
reference only and cannot expand this scope.

## Takeover protocol

Any person taking over GX1 must begin with the executable handover, not with
an old run directory or a historical Markdown claim:

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
git status --short --untracked-files=all
```

Then read `AGENTS.md`, `SYSTEM_MAP.md`,
`HANDOVER_XAU_DIRECTION_REPAIR_20260714.md` and
`PROJECT_STATE_xau_direction_launch.json` in that order. The handover's
`worktree_fingerprint` is source identity; a clean-looking path count is not.
If a heavy process is active, do not start another one or kill the protected
collector/dashboard/notifier processes. Every heavy command must enter through
`scripts/gx1_capped_run.sh` with `MemoryMax/High=10G`, swap `512M`, two CPU
affined cores and one numerical thread. A cap kill, missing output or incomplete hash
chain is terminal evidence, not a reason to reuse partial files.
On WSL, requests above 4G also fail closed until the active VM memory and swap
totals match `/mnt/c/Users/Andre/.wslconfig`; editing the file without a WSL
restart does not count as protection.

The current offline V8/V13 evidence anchors are the dataset directory,
`UNIFIED_EXIT_LIFECYCLE_MANIFEST.json`, the five-timeframe V4 manifest and the
explicit train-recipe audit printed by the handover. These artifacts permit
the bounded smoke/candidate evidence path only; they do not admit a model,
OOS edge or launch. A new maintainer must never select an artifact through
`latest`, glob order, mtime or an old model name. If the current handover and
machine-readable launch state disagree, stop closed and repair the authority
before continuing.
