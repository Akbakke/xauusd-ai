# Veien videre — 26.09.2026

Rekkefølgen under er bindende. Hvert steg avsluttes med fokuserte tester, `git diff --check`
og oppdatert handover i samme commit (GX1_RULES.md regel 12). Én tung jobb om gangen via
`scripts/gx1_capped_run.sh`.

## Operatørvedtak som gjenstår

1. **Guard-referansen** blokkerer alle commits til `.claude/settings.reference.json` er lik live
   `~/.claude/settings.json` (feltet `model`). Kun operatøren endrer dette.
2. **Lengre gullhistorikk** (XAU_USD, samme instrument, f.eks. D1/H4 fra ~2005 via den
   eksisterende OANDA-backfill-produsenten med manifest) — nødvendig for å lære noe annet enn
   «vær long» på ukeshorisont.
3. **GPU-kjernestopp på native rute:** 85 °C i dag vs. 70 °C vedtatt 20.–21.09 for grenens rute.

## Kodesteg

1. **M1 — konsolidering** (denne bølgen): commit, arkiv-tag, rot-loader, relativ
   `core.hooksPath`. Se [docs/CONSOLIDATION_20260926.md](docs/CONSOLIDATION_20260926.md).
2. **M4 — forskningsinstrument for uker** (før datasett): `--decision-clock {M5,H1,H4,D1}`,
   statistikk på ikke-overlappende perioder med parvis differanse mot alltid-LONG og en kausal
   konstant valgt på fit-perioden, vern mot sirkulær-null ved ≤ 512 rader, `model_kind` i
   metadata. HAC/sirkulær-null brukes ikke som PASS på lange horisonter.
3. **M2 — featureflaten v36** (241, per-TF 190) med tilpasningene i konsolideringsrapporten;
   eierne eksekveres for å bekrefte dimensjonene; fokuserte tester.
4. **Forhåndsregistrert ukesmåling** med instrumentet: beslutning på H4/D1-slutt, horisonter
   1/2/4 uker, ridge og HGB med konstant-alternativ, fire årsholdouts, mot alltid-LONG; på de
   tidlig kalibrerte v36-dataene. Første, enklere måling er gjort
   ([docs/DIRECTION_TIMESCALE_20260926.md](docs/DIRECTION_TIMESCALE_20260926.md) §6).
5. **Målkontrakt for ukeshorisont**: nytt, eksplisitt kontraktvalg (ikke en stille økning av
   96-barers-taket), der Entry-verdien kommer fra direkte utførbare utfall og FLAT = 0 etter
   netto kost. Beslutningsklokke og horisont fra steg 4.
6. **Rebuild**: squeeze-refit (v3, lukket-bar-vindu) → `scripts/run_seq513_rebuild_chain_v1.sh`
   med ny run-id → post-rebuild-audits → lifecycle-v2-laget (ENTRY_WINDOW, normalisering,
   bindinger, random-access-indeks, recipes). Gjenbrukbart: tapene, M1-child-views, stengning,
   kostpolicy og økonomi (etter egne hash-bindinger).
7. **M3 — trenerfeil** før neste native trening (gate-entropi fail-open, active-head-diagnostikk,
   gamma-metadata).
8. **Native trening** først når steg 4 viser noe utover drift, og innenfor en ny, bundet
   NEXT_RUN_POLICY.

## Ikke gjør

Ikke relanser selector-, direct-outcome- eller 512-planene. Ikke tren på 95-minuttersmålet
igjen. Ikke gjør terskel-, horisont- eller featuresøk uten forhåndsregistrering. Ikke rydd
artefakter fra den arkiverte grenen før en arkivautoritet dekker dem (regel 9).
