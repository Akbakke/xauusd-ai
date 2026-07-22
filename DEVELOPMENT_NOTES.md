# Development notes

Use `/home/andre2/src/GX1_ENGINE/.venv/bin/python`.

Before editing, read `AGENTS.md`, `SYSTEM_MAP.md` and the current handover.
Prefer the existing owner over a parallel script. Preserve unrelated changes
in the dirty worktree.

Use `rg`/`rg --files` for source searches. Do not recursively walk `.venv`,
`.git` or `/home/andre2/GX1_DATA`; they are not source and dominate disk I/O.

Minimum verification for a bounded change:

```bash
.venv/bin/python -m py_compile <changed-python-files>
.venv/bin/python -m pytest -q <focused-tests>
.venv/bin/python -m pytest --collect-only -q
git diff --check
```

Also scan for deleted filenames, retired contract modes, fallback wording,
mutable artifact selection and obsolete CLI arguments in active code.

Contract-source verification should inspect parsed imports and executable use
from the canonical owner. A raw text search for duplicated mode/dimension/field
literals is not a valid wiring check and can reject correctly centralized
consumers.

Do not run a dataset rebuild, trainer, large replay or live launcher as a test.
Entry rebuild/training require their immutable prerequisites and one shared
`--run-id`; it is lineage rather than manual approval. Live launch and
destructive data work keep their separate authorization contracts.

Every heavy GX1 job must use the capped runner, explicit RAM/swap limits and
the one host-wide heavy-job lock. Never start another heavy job merely to test
a wrapper. Destructive `GX1_DATA` work must use the sole evidence-retention
cleanup owner with exact leaf targets, immutable inventory, separate approval
and execution evidence.
