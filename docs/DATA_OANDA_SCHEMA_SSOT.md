# OANDA XAUUSD tape schema

## Instrument and grids

- OANDA instrument: `XAU_USD`.
- Prices: mid, bid and ask (`MBA`).
- Raw path grid: M1, complete candles only.
- Entry model grid: M5, derived or fetched with the same complete-bar semantics.
- Timestamps: unique, strictly increasing UTC `DatetimeIndex` values.

M1 timestamps align to one-minute boundaries; M5 timestamps align to
five-minute boundaries. Weekend/market closures are explicit gaps, not rows to
forward-fill.

## Required columns

In exact logical groups:

- mid: `open`, `high`, `low`, `close`;
- activity: `volume`;
- bid: `bid_open`, `bid_high`, `bid_low`, `bid_close`;
- ask: `ask_open`, `ask_high`, `ask_low`, `ask_close`.

All price values must be finite and satisfy OHLC consistency. Ask must not be
silently replaced by bid or mid. Volume may be zero but must be finite.

## Ingestion invariants

- use half-open request ranges and retain only `complete=true` candles;
- normalize timestamps to UTC without shifting bar meaning;
- reject duplicate, non-monotonic, malformed or partial bars;
- detect and classify gaps rather than filling them;
- record exact source range, rows, schema and content hash in a manifest;
- never merge two tapes without proving overlap equality and schema identity.

M15/H1/H4/D1 inputs must close before or at the M5 decision cutoff. Exit may
use the M1 path; Entry must not see future M1 information inside its M5 bar.
