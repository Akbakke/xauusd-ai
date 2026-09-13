<!-- GX1_DOCUMENT_CLASS: CANONICAL | short current status -->
# GX1 status

First full five-year TRAIN epoch completed at 2026-09-13 04:35 UTC:
313,399 Entry rows and 19,588 optimizer steps, checkpoint 309, phase validation.
A batch-128 June-VAL successor is prepared after the user-requested restart.
The completed TRAIN state is preserved; June VAL starts with fresh accumulators. Read live process/cursor observations with
`bash scripts/gx1_handover.sh`; use `CURRENT_HANDOVER.md` for exact bindings,
prior smoke results and unresolved quality limits.

The controller is authorized to continue sequentially to at most 30 epochs,
with June VAL after each and patience 5. TEST remains sealed. No positive Bps,
accepted model, completed main VAL or live readiness is claimed here.
