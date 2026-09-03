# GX1 cleanup candidates — 2026-09-03

This is a deletion plan, not a deletion authorisation. No file was removed
while producing it. Preserve the V9 evidence and use the retention contract,
not bulk deletion commands.

## Safe to regenerate locally

- Python `__pycache__/`, `.pytest_cache/`, and `.ruff_cache/` are rebuildable
  caches. They are not source artifacts and may be cleared only from the
  selected worktree after confirming that the path is not a nested registered
  worktree.
- Do not remove `.venv/` as routine cleanup: it is the verified runtime for
  `scripts/gx1_handover.sh` and tests. Recreate it deliberately only when its
  dependency contract is being refreshed.
- Do not remove `.claude/worktrees/...`: it is a separate worktree and needs an
  explicit ownership/reachability review before any prune.

## Code that looks retired but is **not safe to delete today**

| Area | Why it stays for now | Required proof before removal |
| --- | --- | --- |
| `gx1/execution/v12_ctx_augment_live.py`, `v12_model_native_state_live.py`, `v12_state_from_prebuilt.py`, `v12_canonical_incremental.py` | Current offline feature, state, pair and lifecycle contracts import them. | Move each offline owner to an explicitly named replacement; pass import, parity, cache and lifecycle tests without the module. |
| `gx1/execution/v12_trade_state.py` | Current Exit lifecycle and trade-journal code import it. | Replace/rebind those consumers and verify full Exit trajectory contracts. |
| `gx1/execution/oanda_client.py`, `oanda_credentials.py` | Historical-data backfill and tape provenance use them; they are not merely live execution. | Separate archival ingestion from credentialed client code, then prove no dataset/rebuild path imports the client. |
| `gx1/contracts/live_tail_publication_v1.py` | The launch-transaction contract imports it even though live operation is blocked. | Remove the live-tail launch branch at the architectural level and update the transaction contract/tests together. |
| `gx1/execution/v12_pipeline.py`, `v12_smart_entry_live.py`, `v12_paper_runner.py`, `v12_oanda_data_collector.py`, `scripts/gx1_dashboard.py` | These are forbidden operational routes, but source and AST tests still intentionally enforce their fail-closed behaviour. | First decide to retire all serve/paper/collector capability, remove the entrypoints and their tests/contracts in one reviewed change, then prove no launch command remains. |

The `v12` prefix is historical naming, not proof of dead code. Repository-wide
imports and tests currently make a direct deletion unsafe.

## Retention-protected data and history

Never treat the following as cache:

- V9 recipe, session pointer, published bundle, epoch seal, validation reports
  and source-rebind dry-run recipe;
- V8 incident evidence and its host-hang record;
- V46 manifests, split/normalization/MTF evidence and sealed TEST boundary;
- historical Markdown audits that explain why retired paths are retired.

Generated research artifacts must be retired through the project retention
owner after their hashes, consumers and replacement evidence are recorded.

## Recommended cleanup order

1. Clear only regenerable caches in the currently selected worktree when disk
   space is needed.
2. Make a separate architectural decision: retain the future serve path, or
   permanently remove it. Do not leave a half-deleted paper/live stack.
3. If permanent removal is chosen, begin with explicit entrypoints and service
   launch wiring; then eliminate consumers and tests; finally delete the
   implementation modules once repository-wide reference checks are empty.
4. Run `git diff --check`, `bash scripts/gx1_handover.sh --check`, targeted
   import/contract tests, and a complete reference search before committing.

No old model/run code is currently classified as deletion-safe merely because
it has an old version number. The verified cleanup win today is documentation
consolidation plus optional local cache cleanup, not destructive source edits.
