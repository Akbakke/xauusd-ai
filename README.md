# GX1 XAUUSD — start her

<!-- GX1_CURRENT_RESTART_POINTER -->
## Gjeldende inngang

Kjør read-only handover og les deretter docs/RESTART_POINT_20261005.md
samt CURRENT_RESTART_POINT.json. INDEX_FEATURE_SOURCE_REVIEW_001 er fullført;
samplerbenchmark-planlegging er neste steg.

Eneste kodebase: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`. Én agent om gangen.
Kjør `bash scripts/gx1_handover.sh --check` for fersk, lesende status (starter aldri trening).

1. [GX1_RULES.md](GX1_RULES.md) — bindende regler for alle agenter.
2. [AGENTS.md](AGENTS.md) — arbeidsmåte.
3. [CURRENT_HANDOVER.md](CURRENT_HANDOVER.md) — gjeldende status.
4. [VEIEN_VIDERE.md](VEIEN_VIDERE.md) — eksakt neste steg og åpne operatørvedtak.
5. [GX1_ARBEIDSMAAL.md](GX1_ARBEIDSMAAL.md) — mål og vedtak.
6. [DOC_INDEX.md](DOC_INDEX.md) — evidens og historikk.

`NEXT_RUN_POLICY.json` er maskinlesbart tillatt kjøreomfang; `training_enabled=false` stenger
ny trening. Korrekt læring, generalisering og positiv kostnadsjustert økonomi er ikke
dokumentert. Teknisk PASS er ikke handelsfordel.
