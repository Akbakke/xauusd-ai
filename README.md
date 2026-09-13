<!-- GX1_DOCUMENT_CLASS: CANONICAL | repository entry point -->
# GX1 XAUUSD lifecycle-v2

Offline research: one shared Entry/Exit model, complete contracted features,
learned cooperation across families and timeframes, and cost-adjusted net Bps.
The one-year TRAIN smoke and full June VAL are complete. The first full
five-year TRAIN epoch completed on September 13, 2026; its batch-16 VAL was stopped on the user’s instruction. A batch-128 successor
is prepared from that completed TRAIN checkpoint; use the live handover command
to observe its subsequent start.
No profitable/admitted model or live readiness is claimed.

Start with [CURRENT_HANDOVER.md](CURRENT_HANDOVER.md),
[GX1_ARBEIDSMAAL.md](GX1_ARBEIDSMAAL.md), and [SYSTEM_MAP.md](SYSTEM_MAP.md).
[CURRENT_NATIVE_RUN.json](CURRENT_NATIVE_RUN.json) binds the frozen active
source and runtime. This documentation branch is not a replacement training
source. Read current status on the training host:

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
```

The commands are read-only and never load model checkpoints or start training.
The Windows task owns sequential TRAIN/VAL and pause/reboot/resume. Monitor
about every 15 minutes, silently on healthy progress. TEST remains sealed.
Use [DOC_INDEX.md](DOC_INDEX.md) for historical evidence and current documents.
Git preserves source/configuration/docs; dataset and checkpoint backups are
separate, as described in the handover.
