# Audit: References to `build_entry_v10_ctx_training_dataset.py`

**Scope:** Repo-wide (gx1/, scripts/, configs/, docs/, README*, .md, .sh, .yaml/.yml, .json, .toml, .ini, .cfg, .py, CI, Makefile, pre-commit).  
**Goal:** Find all direct and indirect references to the base script; propose replacements (signal_only vs legacy) and a safe-deletion checklist.

---

## Section A: Findings

| # | File path | Line | Matched text | Type |
|---|-----------|------|--------------|------|
| 1 | `gx1/scripts/build_entry_v10_ctx_training_dataset.py` | 609 | `# script is at: gx1/scripts/build_entry_v10_ctx_training_dataset.py` | comment (self-reference) |
| 2 | `gx1/tests/test_session_histogram_logging.py` | 5 | `from gx1.scripts.build_entry_v10_ctx_training_dataset import compute_session_histogram` | import |
| 3 | `gx1/tests/test_entry_v10_ctx_dataset_contract.py` | 135 | `from gx1.scripts.build_entry_v10_ctx_training_dataset import build_dataset` | import |
| 4 | `gx1/tests/test_entry_v10_ctx_dataset_contract.py` | 144 | `from gx1.scripts.build_entry_v10_ctx_training_dataset import write_manifest` | import |
| 5 | `docs/PREBUILT_FILE_INVENTORY.md` | 80 | `- \`build_entry_v10_ctx_training_dataset.py\`` | docs (inventory) |
| 6 | `docs/FULLYEAR_2025_TRAINING_PIPELINE_FIX.md` | 18 | `**File:** \`gx1/scripts/build_entry_v10_ctx_training_dataset.py\`` | docs |
| 7 | `docs/FULLYEAR_2025_TRAINING_PIPELINE_FIX.md` | 36 | `**File:** \`gx1/scripts/build_entry_v10_ctx_training_dataset.py\`` | docs |
| 8 | `docs/FULLYEAR_2025_TRAINING_PIPELINE_FIX.md` | 51 | `python gx1/scripts/build_entry_v10_ctx_training_dataset.py \` | docs (CLI) |
| 9 | `docs/FULLYEAR_2025_TRAINING_PIPELINE_FIX.md` | 134 | `python gx1/scripts/build_entry_v10_ctx_training_dataset.py \` | docs (CLI) |
| 10 | `docs/FULLYEAR_2025_TRAINING_PIPELINE_FIX.md` | 211 | `1. \`gx1/scripts/build_entry_v10_ctx_training_dataset.py\` (NEW)` | docs |
| 11 | `docs/FULLYEAR_2025_ROBUST_PIPELINE.md` | 16 | `**File:** \`gx1/scripts/build_entry_v10_ctx_training_dataset.py\`` | docs |
| 12 | `docs/FULLYEAR_2025_ROBUST_PIPELINE.md` | 32 | `python gx1/scripts/build_entry_v10_ctx_training_dataset.py \` | docs (CLI) |
| 13 | `docs/FULLYEAR_2025_ROBUST_PIPELINE.md` | 46 | `python gx1/scripts/build_entry_v10_ctx_training_dataset.py \` | docs (CLI) |
| 14 | `docs/FULLYEAR_2025_ROBUST_PIPELINE.md` | 58 | `**File:** \`gx1/scripts/build_entry_v10_ctx_training_dataset.py\`` | docs |
| 15 | `docs/FULLYEAR_2025_ROBUST_PIPELINE.md` | 118 | `**File:** \`gx1/scripts/build_entry_v10_ctx_training_dataset.py\`` | docs |
| 16 | `docs/FULLYEAR_2025_ROBUST_PIPELINE.md` | 207 | `2. \`gx1/scripts/build_entry_v10_ctx_training_dataset.py\` (UPDATED)` | docs |
| 17 | `docs/FULLYEAR_2025_ROBUST_PIPELINE.md` | 228 | `python gx1/scripts/build_entry_v10_ctx_training_dataset.py \` | docs (CLI) |

**Not referencing the base script (no change):**
- `gx1/scripts/build_entry_v10_ctx_training_dataset_legacy.py` — mentions `build_entry_v10_ctx_training_dataset_signal_only.py` and `build_entry_v10_ctx_training_dataset_legacy` only.
- `scripts/run_entry_v10_offline_pipeline.sh` / `run_entry_v10_1_offline_pipeline.sh` — use `gx1.rl.entry_v10.build_entry_v10_dataset` (different module).
- `.github/workflows/`, `git-hooks/pre-commit`, `Makefile` — no references found.

---

## Section A2: Evidence-based table (rg-verified; current state)

**Commands run:** `rg "build_entry_v10_ctx_training_dataset\.py" .` | `rg "build_entry_v10_ctx_training_dataset" .` | `rg "python -m gx1\.scripts\.build_entry_v10_ctx_training_dataset" .` | `rg "gx1\.scripts\.build_entry_v10_ctx_training_dataset" .`

| File | Line | Matched text | Type |
|------|------|--------------|------|
| `gx1/scripts/build_entry_v10_ctx_training_dataset.py` | 9 | `"[DEPRECATED] build_entry_v10_ctx_training_dataset.py is removed. ..."` | comment (stub message) |
| `docs/FULLYEAR_2025_TRAINING_PIPELINE_FIX.md` | 6 | Canonical note; mentions old name as "deprecated and stubbed" | doc (canonical note) |
| `docs/FULLYEAR_2025_ROBUST_PIPELINE.md` | 6 | Canonical note; mentions old name as "deprecated and stubbed" | doc (canonical note) |
| `docs/AUDIT_build_entry_v10_ctx_training_dataset_references.md` | (many) | Historical findings and before/after | audit (history only) |

**Outside audit + stub:** No runbook/CLI uses the old script name. No imports from `gx1.scripts.build_entry_v10_ctx_training_dataset` (tests use `gx1.datasets.entry_v10_ctx_legacy`). No `python -m gx1.scripts.build_entry_v10_ctx_training_dataset` (without suffix). No dynamic dispatch to the old name.

---

## Section B: Proposed replacements (by file)

### 1. `gx1/scripts/build_entry_v10_ctx_training_dataset.py` (line 609)

- **Before:** `# script is at: gx1/scripts/build_entry_v10_ctx_training_dataset.py`
- **After:** `# script is at: gx1/scripts/build_entry_v10_ctx_training_dataset_legacy.py` (or remove comment when deleting this file)
- **Note:** Self-reference; only relevant if file is kept. If file is removed, delete the comment.

### 2. `gx1/tests/test_session_histogram_logging.py` (line 5)

- **Before:** `from gx1.scripts.build_entry_v10_ctx_training_dataset import compute_session_histogram`
- **After:** `from gx1.scripts.build_entry_v10_ctx_training_dataset_legacy import compute_session_histogram`
- **Reason:** `compute_session_histogram` exists in the legacy script; signal_only does not expose it.

### 3. `gx1/tests/test_entry_v10_ctx_dataset_contract.py` (lines 135, 144)

- **Before (135):** `from gx1.scripts.build_entry_v10_ctx_training_dataset import build_dataset`
- **After (135):** `from gx1.scripts.build_entry_v10_ctx_training_dataset_legacy import build_dataset`

- **Before (144):** `from gx1.scripts.build_entry_v10_ctx_training_dataset import write_manifest`
- **After (144):** `from gx1.scripts.build_entry_v10_ctx_training_dataset_legacy import write_manifest`

- **Reason:** Contract test targets `build_dataset` and `write_manifest`; both exist in legacy. signal_only does not expose them.

### 4. `docs/PREBUILT_FILE_INVENTORY.md` (line 80)

- **Before:** `- \`build_entry_v10_ctx_training_dataset.py\``
- **After:** Either list all three scripts, or replace with the canonical one(s) used for prebuilt builds, e.g.:
  - `- \`build_entry_v10_ctx_training_dataset_signal_only.py\` (signal-only dataset build)`
  - `- \`build_entry_v10_ctx_training_dataset_legacy.py\` (legacy/full with calibration)`
- **Note:** REVIEW_REQUIRED: depends on which script(s) the inventory is intended to document.

### 5. `docs/FULLYEAR_2025_TRAINING_PIPELINE_FIX.md` (lines 18, 36, 51, 134, 211)

- **Before:** All occurrences of `gx1/scripts/build_entry_v10_ctx_training_dataset.py` or the same in bold/file refs.
- **After:** REVIEW_REQUIRED. Suggested convention:
  - If the step is “build training dataset only” (no calibration): use `build_entry_v10_ctx_training_dataset_signal_only.py` and `python -m gx1.scripts.build_entry_v10_ctx_training_dataset_signal_only`.
  - If the step is “full build with calibration / v9 runtime”: use `build_entry_v10_ctx_training_dataset_legacy.py` and `python -m gx1.scripts.build_entry_v10_ctx_training_dataset_legacy`.
- **Reason:** Doc describes pipeline steps; need to match each step to signal_only vs legacy.

### 6. `docs/FULLYEAR_2025_ROBUST_PIPELINE.md` (lines 16, 32, 46, 58, 118, 207, 228)

- **Before:** All occurrences of `gx1/scripts/build_entry_v10_ctx_training_dataset.py` or same in bold/file refs.
- **After:** Same as Section B.5 — REVIEW_REQUIRED; choose signal_only vs legacy per step, then replace path and CLI accordingly.

---

## Section C: Ambiguities / needs review

1. **Docs (FULLYEAR_2025_*.md):** Each pipeline step that “builds the dataset” must be classified:
   - **Signal-only:** dataset build only (no calibration, no v9 runtime) → `build_entry_v10_ctx_training_dataset_signal_only` and `-m gx1.scripts.build_entry_v10_ctx_training_dataset_signal_only`.
   - **Legacy:** full pipeline with calibration / runtime / multi-step → `build_entry_v10_ctx_training_dataset_legacy` and `-m gx1.scripts.build_entry_v10_ctx_training_dataset_legacy`.
   - Recommendation: Read each doc section; if it mentions calibration, thresholds, or “full” pipeline, use legacy; if it only builds the parquet dataset for training, use signal_only.

2. **PREBUILT_FILE_INVENTORY.md:** Clarify whether the inventory should list the base script, the signal_only script, the legacy script, or all. Then apply Section B.4 accordingly.

3. **No dynamic dispatch found:** Grep for `subprocess`/`importlib`/string-based module names did not reveal any call that invokes `build_entry_v10_ctx_training_dataset` by string. No change needed for dynamic calls.

---

## Section D: Safe deletion checklist + GO/NO-GO

**GO/NO-GO (must all be GO):**
- [x] **0** refs to old name outside audit doc and stub (runbooks/CLI use only _legacy / _signal_only).
- [x] **0** imports from `gx1.scripts.build_entry_v10_ctx_training_dataset` (tests use `gx1.datasets.entry_v10_ctx_legacy`).
- [x] **0** CLI invocations of base script in scripts, CI, Makefile, hooks.
- [ ] **Legacy/signal_only --help:** `python -m gx1.scripts.build_entry_v10_ctx_training_dataset_legacy --help` and `..._signal_only --help` succeed.
- [ ] **Base script fails:** `python -m gx1.scripts.build_entry_v10_ctx_training_dataset` exits with DEPRECATED RuntimeError.
- [ ] **pytest:** `pytest gx1/tests/test_session_histogram_logging.py gx1/tests/test_entry_v10_ctx_dataset_contract.py -v` pass.

**Historical (done):**
- [x] Stub applied; docs use canonical _legacy; PREBUILT lists _signal_only + _legacy.
- [x] Runbooks/docs updated so that “dataset build” steps use either signal_only or legacy explicitly; no remaining “build_entry_v10_ctx_training_dataset” without suffix.

**Final verification:** See [REPORT_build_entry_v10_ctx_routing_verification.md](REPORT_build_entry_v10_ctx_routing_verification.md).

---

## Patch summary / Files changed

| File | Change |
|------|--------|
| `gx1/scripts/build_entry_v10_ctx_training_dataset.py` | Replaced with fail-fast stub (RuntimeError). |
| `gx1/datasets/__init__.py` | New; package init. |
| `gx1/datasets/entry_v10_ctx_legacy.py` | New; re-exports from legacy script. |
| `gx1/tests/test_session_histogram_logging.py` | Import from `gx1.datasets.entry_v10_ctx_legacy`; call with `df=df` (legacy API is keyword-only). |
| `gx1/tests/test_entry_v10_ctx_dataset_contract.py` | Imports from `gx1.datasets.entry_v10_ctx_legacy`. |
| `docs/PREBUILT_FILE_INVENTORY.md` | Base script replaced with _signal_only + _legacy. |
| `docs/FULLYEAR_2025_*.md` | Canonical note (2 lines); all refs to _legacy. |
| `docs/AUDIT_...references.md` | Section A2 evidence; Section D GO/NO-GO; patch summary. |

---

## Summary

- **Code (tests):** Tests now import from `gx1.datasets.entry_v10_ctx_legacy` (canonical module; re-exports from legacy script). No "scripts as API".
- **Module:** `gx1/datasets/entry_v10_ctx_legacy.py` re-exports `compute_session_histogram`, `build_dataset`, `write_manifest` from the legacy script.
- **Docs:** PREBUILT_FILE_INVENTORY lists _signal_only and _legacy. FULLYEAR_2025_* docs use _legacy and canonical note at top.
- **Base script:** Replaced with fail-fast stub (RuntimeError). Run or import of the old name fails with clear message.
- **No** references in CI, Makefile, or pre-commit; no string-based/dispatch references found.
