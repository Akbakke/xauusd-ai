# Veien videre — 26.09.2026

Rekkefølgen under er bindende. Hvert steg avsluttes med fokuserte tester, `git diff --check`
og oppdatert handover i samme commit (GX1_RULES.md regel 12). Én tung jobb om gangen via
`scripts/gx1_capped_run.sh`.

## Operatørvedtak som gjenstår

1. **Lengre gullhistorikk** (XAU_USD, samme instrument): native M5 + M1 fra OANDA via den
   eksisterende backfill-produsenten med manifest, så alle tidsrammer (M5/M15/H1/H4/D1) bygges
   av samme eiere. Uten nedmarkeder i dataene (2021–26 er ett oksemarked) kan modellen ikke lære
   å gå short i et fallende marked.

Vedtatt 26.09: guard-testen ignorerer preferanser (`model`, `theme`); GPU-kjernestopp 85 °C,
nedtrekk til 220 W ved 80 °C.

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
5. **Lengre historikk først** (operatørvedtak, se øverst), deretter samme forhåndsregistrerte
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
