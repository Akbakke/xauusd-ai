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

Before admission, validate the schema in `DATA_OANDA_SCHEMA_SSOT.md`, complete-
bar semantics, monotonic unique UTC timestamps, expected grid/gaps, overlap
equality, row count and content hash. A backfill artifact is not automatically
an accepted Entry dataset.
