# Veien videre — 26.09.2026

Rekkefølgen under er bindende. Hvert steg avsluttes med fokuserte tester, `git diff --check`
og oppdatert handover i samme commit (GX1_RULES.md regel 12). Én tung jobb om gangen via
`scripts/gx1_capped_run.sh`.

## Operatørvedtak som gjenstår

Ingen. **Vedtatt 26.09: hent fra 2005.** Native M5 + M1 XAU_USD fra OANDA, 2005-01-01 →
2026-07-01 (TEST-grensen), via den eksisterende produsenten med manifest; parvedtaket
`OANDA_PAIR_PRETEST_2005_20260927` (samme id på M1 og M5, som parprodusenten krever) er bundet i
`gx1/contracts/oanda_history_ingest_approval_v1.py`. **Vedtatt 27.09 (B, erstatter A):** kjeden kjøres
uendret i full modus; 2006-tapene forlenges i successor-modus til 2026-09-01, og juli–august 2026
blir forseglet TEST (samme ordning som V46). A ble forlatt fordi en pretest-modus i kjeden krever
endrede valideringer. Grunn: 2019–26 mangler de fallende
gullmarkedene (2008, 2011–15, 2016, 2018). **Hentet 27.09**: M5 og M1 fra 2006-03-19 (OANDAs
første bar), overlappet mot 2019-tapen er rad-identisk bortsett fra et fylt hull 2024-05-20; se
[docs/HISTORY_2005_INTAKE_20260926.md](docs/HISTORY_2005_INTAKE_20260926.md).

Vedtatt 26.09: guard-testen ignorerer preferanser (`model`, `theme`); GPU-kjernestopp 85 °C,
nedtrekk til 220 W ved 80 °C.

## Nå (27.09): modellfrie grunnlinjer før mer bygging

Etter gjennomgangen ([docs/FEATURE_SURFACE_SWING_REVIEW_20260927.md](docs/FEATURE_SURFACE_SWING_REVIEW_20260927.md))
måles først, forhåndsregistrert, om enkle regler gir retningsgevinst etter kost — scalp på M5
(operatørens førstevalg) og swing på D1 — på 2009-tapen, TRAIN-perioden 2011-06 → 2025-05
([docs/MODEL_FREE_BASELINES_PREREG_20260927.md](docs/MODEL_FREE_BASELINES_PREREG_20260927.md)).
seq513-bootstrapen (squeeze → C0 → par → kjede) står på pause; 2009-tapene, direkte kilder og lineage
(`HISTORY2009_BOOTSTRAP_20260927`) er klare. **Resultat 27.09: NO-GO på alle 62 celler**
([docs/MODEL_FREE_BASELINES_RESULT_20260927.md](docs/MODEL_FREE_BASELINES_RESULT_20260927.md)):
scalp-regler har null brutto retning etter spread (kost ~6 bps/rundtur); trendfiltre på D1 gir bare
risikoreduksjon i bjørnemarkedet (beste t 1,87). Makrohendelser (FOMC/NFP/KPI) testet samme dag: **NO-GO 0/18**
([docs/MACRO_EVENT_BASELINES_RESULT_20260927.md](docs/MACRO_EVENT_BASELINES_RESULT_20260927.md)).
**Bølge 1 (operatørvedtak 27.09):** fem intradag-mekanismer forhåndsregistrert i
[docs/INTRADAY_MECHANISMS_PREREG_20260927.md](docs/INTRADAY_MECHANISMS_PREREG_20260927.md) — rundtall,
oppsettene som speilede par over 2011–25, COMEX-momentum, LBMA-auksjonen og sesjoner/ORB på lokal klokke;
beslutning på policyens low-slippage (1 bps per utførelse). Bølge 2 (OANDAs ordre-/posisjonsbok) krever
operatørvedtak; bølge 3 (maskinlæring) bare innenfor en populasjon som blir GO/LOVENDE og bekreftes på VAL.

## Kodesteg

1. ~~M1 — konsolidering~~ **ferdig 26.09** (809ba049, 755dd3aa). Se
   [docs/CONSOLIDATION_20260926.md](docs/CONSOLIDATION_20260926.md).
2. ~~M4 — forskningsinstrument for uker~~ **ferdig 26.09**: `--decision-clock {M5,H1,H4,D1}`,
   statistikk på ikke-overlappende perioder med parvis differanse mot alltid-LONG og en kausal
   konstant valgt på fit-perioden, vern mot sirkulær-null ved ≤ 512 rader, `model_kind` i
   metadata. HAC/sirkulær-null brukes ikke som PASS på lange horisonter.
3. ~~M2 — featureflaten v36~~ **ferdig 26.09**: signal 241, per-TF 190 (eierne eksekvert).
   Gamle V9-/lifecycle-v2-artefakter (v34) kan ikke lastes på HEAD; evaluer dem på 7c9421a5.
4. ~~Forhåndsregistrert ukesmåling~~ **ferdig 26.09: NO-GO**
   ([resultat](docs/WEEKLY_DIRECTION_RESULT_20260926.md)): beslutning på H4/D1-slutt, horisonter
   1/2/4 uker, ridge og HGB med konstant-alternativ, mot alltid-LONG, på de tidlig kalibrerte
   v36-dataene. Kjørt på tre gyldige årsholdouts 2023-06..2026-05 med strengere GO-regel (3 av 3);
   fold 0 stoppet på kronologivakten (avviket står i resultatet). 0 av 48 celler slo alltid-LONG.
5. ~~Hent 2005–~~ **ferdig 27.09** (fra 2006-03-19), deretter rebuild på 2006-tapene med
   tidligst mulig TRAIN-start (lengste lookback avgjør), deretter samme forhåndsregistrerte
   måling på data med fallende markeder. Først ved GO/LOVENDE: **målkontrakt for
   ukeshorisont** — nytt, eksplisitt kontraktvalg (ikke en stille økning av 96-barers-taket),
   der Entry-verdien kommer fra direkte utførbare utfall og FLAT = 0 etter netto kost;
   beslutningsklokke og horisont tas fra den målingen.
6. **Rebuild** (kildeidentitetsporten er portert 26.09 og CURRENT har eget Python-miljø; porten
   blokkerer til `gx1/monitoring/` med foreldreløs bytekode er fjernet, se Opprydding):
   squeeze-refit (v4, lukket-bar-vindu) → `scripts/run_seq513_rebuild_chain_v1.sh`
   med ny run-id → post-rebuild-audits → lifecycle-v2-laget (ENTRY_WINDOW, normalisering,
   bindinger, random-access-indeks, recipes). Gjenbrukbart: tapene, M1-child-views, stengning,
   kostpolicy og økonomi (etter egne hash-bindinger).
7. **M3 — trenerfeil** før neste native trening (gate-entropi fail-open, active-head-diagnostikk,
   gamma-metadata).
8. **Native trening** først når målingen i steg 5 viser noe utover drift, og innenfor en ny,
   bundet NEXT_RUN_POLICY.

## Opprydding

Gjort 26.09 (se konsolideringsrapporten). Gjenstår: triage av 17 eksisterende testfeil; fjerning av
`gx1/monitoring/` (bare bytekode for en slettet modul), worktree-ene V22/V30/V31/V33/V37/V38/V39/V41
og backup-tarballen (operatørens kommando; V30 trengs ikke etter at CURRENT fikk eget miljø);
EXIT_LIFECYCLE_V2 og V40 når kostbevisene og handover-sjekken er flyttet hit.

## Ikke gjør

Ikke relanser selector-, direct-outcome- eller 512-planene. Ikke tren på 95-minuttersmålet
igjen. Ikke gjør terskel-, horisont- eller featuresøk uten forhåndsregistrering. Ikke rydd
artefakter fra den arkiverte grenen før en arkivautoritet dekker dem (regel 9).
