# XAUUSD OANDA backfill boundary

The retained backfill implementations are
`gx1/scripts/backfill_xauusd_m5_bidask_2020_2025.py` and
`gx1/scripts/backfill_xauusd_m5_from_oanda.py`. They write large external data
and are not diagnostic commands.

Do not run either script during ordinary source cleanup or testing. An
authorized backfill must provide explicit immutable output/checkpoint paths
under `/home/andre2/GX1_DATA`, valid OANDA credentials, a bounded UTC range and
the relevant `--vedtak`. Inspect running collectors first so two writers never
touch the same tape.

The CLI contract is fail-closed for both retained writers: `--vedtak` is
mandatory and is validated before environment loading, credential access,
network access, checkpoint/cache creation or output writes. Bounded invocations
therefore start as follows (the paths and timestamps remain event-specific):

```bash
.venv/bin/python gx1/scripts/backfill_xauusd_m5_bidask_2020_2025.py \
  --vedtak XAU_OANDA_BACKFILL_2020_2024_V1 \
  --granularity M5 \
  --start 2020-01-01T00:00:00Z \
  --end 2025-01-01T00:00:00Z \
  --out /home/andre2/GX1_DATA/events/XAU_OANDA_BACKFILL_2020_2024_V1/XAUUSD_M5.parquet \
  --checkpoint-dir /home/andre2/GX1_DATA/events/XAU_OANDA_BACKFILL_2020_2024_V1/checkpoints

.venv/bin/python gx1/scripts/backfill_xauusd_m5_from_oanda.py \
  --vedtak XAU_OANDA_REPAIR_2025_V1 \
  --repair-mode \
  --raw-in /home/andre2/GX1_DATA/events/XAU_OANDA_REPAIR_2025_V1/source.parquet \
  --raw-out /home/andre2/GX1_DATA/events/XAU_OANDA_REPAIR_2025_V1/repaired.parquet \
  --start-ts 2025-01-01T00:00:00Z \
  --end-ts 2025-02-01T00:00:00Z
```

Before admission, validate the schema in `DATA_OANDA_SCHEMA_SSOT.md`, complete-
bar semantics, monotonic unique UTC timestamps, expected grid/gaps, overlap
equality, row count and content hash. A backfill artifact is not automatically
an accepted Entry dataset.
