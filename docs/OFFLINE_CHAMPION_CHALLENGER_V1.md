# Offline champion/challenger v1

This is a review-only contract for comparing two already completed
walk-forward, out-of-sample candidate results. It is not reinforcement
learning, not online learning and not a promotion mechanism.

The contract deliberately enforces all of the following:

- both results bind immutable files by SHA-256;
- both candidates use the same unseen evaluation window, feature contract,
  target contract, raw-Q decision contract and cost model;
- each candidate's training window ends at or before that unseen window;
- the sealed TEST set is forbidden as a repeatable comparison surface;
- the comparison only reports metric deltas for human review;
- activation, promotion, online weight updates and background scheduling are
all `false`.

> Runtime-state rule: this review does not own a candidate checkpoint. Read
> `bash scripts/gx1_handover.sh` for the runtime-verified recipe, source
> closure, session contract and active-state identity.

> **Current status, 2026-08-30:** no eligible champion or challenger exists.
> The technical VAL decision journal is intentionally excluded: it is a
> one-smoke-checkpoint label/plumbing diagnostic, not a completed rolling-OOS
> candidate result. The current 20-minute candidate guard cannot reach first
> VAL and therefore cannot supply either event. It has now reached checkpoint
> 640 through a fresh-process resume, which changes neither restriction. This
> contract remains inactive until two independently materialised, like-for-like
> rolling-OOS candidate result events exist.

The report compares net PnL, win rate, maximum drawdown **loss** (a positive
loss magnitude), MAE, MFE and the share of trades where MAE came before MFE.
It only records deltas; it never chooses a winner automatically.

When the first honest rolling-OOS reports exist, materialize one comparison:

```bash
scripts/gx1_capped_run.sh --class audit --mem 4G --swap 512M -- \
  .venv/bin/python -m gx1.scripts.materialize_entry_offline_challenger_v1 \
  --champion-result-json /absolute/champion-result.json \
  --challenger-result-json /absolute/challenger-result.json \
  --out-dir /absolute/review-events
```

The command cannot load a model or training dataset. It only reads the two
bound result events and writes one immutable review record.
