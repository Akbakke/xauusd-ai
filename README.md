# GX1 XAUUSD — start her

Kjør `./handover.sh --check` for fersk status. `--verbose` viser status og handover.
Dette er lesende observasjon og starter aldri trening.

1. [CURRENT_HANDOVER.md](CURRENT_HANDOVER.md): aktiv jobb, kode, checkpoint og avgrensning.
2. [VEIEN_VIDERE.md](VEIEN_VIDERE.md): neste handling, sluttkontroll og korrekt baseline.
3. [GX1_ARBEIDSMAAL.md](GX1_ARBEIDSMAAL.md) og [AGENTS.md](AGENTS.md): mål og arbeidsregler.
4. NEXT_RUN_POLICY.json og docs/LEARNING_GATE_20260916.md: faktisk tillatt omfang.

Eneste treningsrepo er `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Mac-mappen er overlevering. Siste native256-prøve er fullført; neste arbeid er
den separat bundne signaldiagnosen etter fullført måling av modusavvik.
Læringsporten er ikke bestått. Bruk vanlig Linux-inngang
`bash scripts/gx1_handover.sh --check`. Ekstern OPERATOR_HANDOVER er arkiv.

`current_work` gjelder dagens jobb. Øvrige gamle checkpoint-/VAL-felt i scriptets
JSON er fullført historikk. Ingen historiske planer er startinstrukser.
`training_enabled=false` og fravær av aktivt kjøreunntak stenger ny trening.
Prosesser og receipts avgjør faktisk kjøretilstand. Den brukte planen må ikke relanseres.

Korrekt læring, generalisering og positiv kostnadsjustert økonomi er fortsatt
ikke dokumentert. Teknisk PASS er ikke handelsfordel.
