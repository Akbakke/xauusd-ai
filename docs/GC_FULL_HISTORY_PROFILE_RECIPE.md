# Complete GC history for dynamic profiles — 18

Observed blocker: 017 qualifies OFI/BBO in 83,776 M5 windows but yields zero
complete session prefixes. Its reused 005 ladders were collected only on native
spot-eligible M1 coordinates. At the GC 18:00 New York anchor, 004 contains only
18:04 in each of 558 sessions, not 18:00–18:03. None of 523 native anchor M5 bars
can qualify from that subset. This is a history-selection limitation, separate
from actual GC quality flags (some 18:04 rows also fail).

019 corrects the 11 roll-minute differences caught before raw replay in 018.
One new local-only full-clock build, no API/fit/TEST. Reuse the 004 causal roll
schedule, making its choice available at each M1 START. Prove that every original
native-eligible contract and all eleven trade metrics are unchanged. Extend
eligibility to the GC source clock solely for profile history, keep EXACTLY the
same source.row_masks trade-quality predicate, including repeated T signatures,
gap flags, bad receive times, snapshots, invalid trades and positive trusted
trades per minute. No deduplication, zero-trade inference or quality relaxation.

Stream through the existing source parser and TradeLadderCollector, save the
complete selected M1 inputs and real-trade ladders before level reduction. The
existing dynamic_profiles_v2 owner then consumes a complete UTC M5 grid rather
than the spot-only grid. Use the same 18:00 NY anchor, exact VWAP/POC/70% VA,
quality/roll resets and rolling 15/60-minute windows. Zero-trade minutes still
reject strict history, including closures; historical holiday truth is not
inferred from absence. Session-complete means complete admitted history TO THE
DECISION, not an assertion that a whole finished exchange day is clean.

Project to the original 141,806 native M5 decisions. Retain original 017 F fields
as a separate bound source. Every previously admitted rolling profile must stay
admitted and retain identical VWAP/POC/VAH/VAL. Recheck full-grid prefixes,
exact volume and native valid closes, typed Parquet readback and source hashes.
The first study day remains unselected; no missing-contract fallback or contract
mixing. Published decision availability remains strictly before native TEST.

P0 preparation uses F from 017 and L from 018 under explicit clock equality and
finite joint masks. No production change, model fit or automatic P0 admission.
