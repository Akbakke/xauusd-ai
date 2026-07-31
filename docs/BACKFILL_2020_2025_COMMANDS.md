# XAUUSD OANDA source-production boundary

Updated 2026-07-31.

The active immutable native M1/M5 owner is
`gx1/scripts/backfill_xauusd_m5_from_oanda.py`. Its historical filename is not
its contract: it supports strict `M1` or `M5` immutable bootstrap and
schema-v4 successor publication.

This owner produces bounded immutable snapshots only. It is not a daemon.
Successor mode reuses verified parent history and fetches only one bounded
overlap chunk plus the new tail; its existence does not itself create the
canonical pair publications or two-event live admission.

Do not run it during ordinary source cleanup or testing. A real run accesses
credentials/network and writes large external evidence. It requires an
explicit vedtak, bounded left-closed/right-open UTC interval and fresh
non-existing output root.

Example of one pair-compatible successor request. Both timeframes retain the
same parent vedtak/start and advance to the same exclusive end; their parent
roots and manifest hashes remain timeframe-specific:

```bash
scripts/entry_next_edge_control.sh model-native-native-m1-source \
  --publication-mode successor \
  --vedtak PARENT_VEDTAK_ID \
  --end-utc 2026-08-01T00:00:00Z \
  --parent-root /home/andre2/GX1_DATA/events/PARENT_EVENT_ID/M1 \
  --expected-parent-manifest-sha256 <m1-parent-lowercase-sha256> \
  --out-root /home/andre2/GX1_DATA/events/CHILD_EVENT_ID/M1

scripts/entry_next_edge_control.sh model-native-native-m5-source \
  --publication-mode successor \
  --vedtak PARENT_VEDTAK_ID \
  --end-utc 2026-08-01T00:00:00Z \
  --parent-root /home/andre2/GX1_DATA/events/PARENT_EVENT_ID/M5 \
  --expected-parent-manifest-sha256 <m5-parent-lowercase-sha256> \
  --out-root /home/andre2/GX1_DATA/events/CHILD_EVENT_ID/M5
```

The dates, IDs, roots and hashes above are illustrative contract shapes, not
permission to execute. A bootstrap has a different required argument set; use
the control surface help rather than deleting successor requirements. Inspect
running collectors/heavy GX1 processes first.

Policy is code-owned:

- M1 requests use fixed three-day chunks;
- M5 requests use fixed 15-day chunks;
- both cap theoretical request slots at 4,320;
- only literal complete OANDA MBA candles are admitted;
- exact 14-column Arrow order, UTC grid, positive finite geometry and
  non-negative integer volume are required;
- retained normalized response chunks must rederive the published parquets
  byte-for-byte;
- successor mode requires exact parent-root/manifest CAS, unchanged
  timeframe/vedtak/OANDA environment, strict end advancement and byte-exact
  completed-row overlap;
- publication is fsynced atomic no-replace.

The retained
`gx1/scripts/backfill_xauusd_m5_bidask_2020_2025.py` is a historical diagnostic
writer. It is not the current native-source authority and must not be used to
repair, merge into or bootstrap canonical production.

A native source artifact does not automatically authorize Entry. The
native→canonical pair, V4 cache, source cascade, split manifests and all later
audits must bind its exact manifest and bytes.
