# GX1 XAUUSD — start her

Kjør `./handover.sh --check` for fersk status. `--verbose` viser status og handover.
Dette er lesende observasjon og starter aldri trening.

1. [CURRENT_HANDOVER.md](CURRENT_HANDOVER.md): aktiv jobb, kode, checkpoint og avgrensning.
2. [VEIEN_VIDERE.md](VEIEN_VIDERE.md): neste handling, sluttkontroll og korrekt baseline.
3. [GX1_ARBEIDSMAAL.md](GX1_ARBEIDSMAAL.md) og [AGENTS.md](AGENTS.md): mål og arbeidsregler.
4. NEXT_RUN_POLICY.json og docs/LEARNING_GATE_20260916.md: faktisk tillatt omfang.

Eneste treningsrepo er `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Mac-mappen er overlevering. Entry fuse256 er ferdig analysert: begge Entry-sider har bedre samlet verdiestimering enn tidligere modeller og TRAIN-konstanter, men fortsatt FLAT256/256 og uendret sidefast Exit. Samlet læringsport er ikke bestått. Brukt scope er stengt; ingen ny kjøring er bundet.

`current_work` viser siste fullførte prøve, sluttkvittering, finalmåling,
`latest_model_correction` og `handover_resume_point`.
Disse angir ferdige kontroller og nøyaktig uferdig neste steg. Øvrige gamle checkpoint-/VAL-felt i scriptets
JSON er fullført historikk. Ingen historiske planer er startinstrukser.
`training_enabled=false` og fravær av aktivt kjøreunntak stenger ny trening.
Prosesser og receipts avgjør faktisk kjøretilstand. Den brukte planen må ikke relanseres.

Korrekt læring, generalisering og positiv kostnadsjustert økonomi er fortsatt
ikke dokumentert. Teknisk PASS er ikke handelsfordel.
