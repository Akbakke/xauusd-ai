# GX1 — overlevering 18. september 2026

Stoppunkt: modellrettelsen er implementert og kontrollert. Ny startmåling
mangler. Ingen native jobb kjører; Windows-task er Disabled. Læringsporten
er ikke bestått. Ingen ny kjøreplan er bundet.

Kode: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Data: `/home/andre2/GX1_DATA`. Mac-mappen er overleveringskopi.
Start med `./handover.sh --check` på Mac eller
`bash scripts/gx1_handover.sh --check` i Linux. `current_work` viser nåstatus,
rettelse og stoppunkt; øvrige gamle checkpoint-/VAL-felt er historikk.
`working_head` og `working_tree_clean` må leses fra fersk scriptutgang.

Siste læringsprøve (residualnormalisering, kilde 88310075) ble avvist:
Entry FLAT 256/256; Exit fast valg per side. Påfølgende diagnose (bff19fbe)
målte stor nesten felles hovedbane: main-fuse L2 1,05→117,71. Variasjonen
etter felles Entry-normalisering var 9,51 ganger mindre enn initialt,
Entry-hidden 7,07 ganger mindre. Rå MTF/context varierer fortsatt.
Dette lokaliserer et problem, men beviser ikke at neste rettelse gir læring.

Rettelsen er én parameterfri LayerNorm på utgangen av hovedencoder.
ONLINE-funksjonen endres. Prefix-læreren kopieres eksplisitt uten den nye
normaliseringen, slik at original lærerfunksjon bevares. Faktisk produksjons-
initialisering bekrefter identiske vekter og RNG; komplette syntetiske Entry-/
Exit-tester bekrefter lærerparitet. Ingen ny native startmåling eller trening.
Gamle ONLINE-startprediksjoner kan IKKE brukes som ny modellbaseline.

Viktig korrigering av arbeidsnotatet: TRAIN-only-admission og binding mot
foreldet modellkilde ble ikke endret. `TRAIN_ONLY_TESTS.log` finnes ikke.
Dette er konkret neste arbeid, ikke beståtte kontroller. Se
[VEIEN_VIDERE.md](VEIEN_VIDERE.md) og
[rettelse og bevis](docs/MAIN_ENCODER_NORMALIZATION_HANDOVER_20260918.md).

Bevar original initialisering, avsluttede checkpoints, targets, alle 200 features,
åtte familier/tidsrammer, kausalitet og kostnader. Én agent/én tung jobb.
Ingen full epoch/VAL, CONTROL/TEST, live/paper, spending eller brede søk.
`training_enabled=false`; dette dokumentet gir ingen launch-tillatelse.
Stående offentlig push gjelder ferdig kode/docs/stier/aggregater, aldri rådata,
modellvekter eller hemmeligheter. Historiske operatører skal ikke relanseres.
