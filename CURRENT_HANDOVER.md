# Gjeldende status — 26.09.2026: én kodebase, retning på uker

**Les først:** [GX1_RULES.md](GX1_RULES.md) (bindende regler), [AGENTS.md](AGENTS.md)
(arbeidsmåte), [GX1_ARBEIDSMAAL.md](GX1_ARBEIDSMAAL.md) (mål og vedtak) og
[VEIEN_VIDERE.md](VEIEN_VIDERE.md) (eksakt neste steg). `bash scripts/gx1_handover.sh --check`
overstyrer prosa.

## Konsolidering (operatørvedtak 26.09)

Det fantes to spor: fra 19.09 ble det arbeidet i `/home/andre2/src/GX1_ENGINE` på en foreldet
08.09-base fordi rot-loaderen pekte dit. Nå er dette eneste kodebase; den arkiverte grenen er
tagget `archive/gx1-engine-audit-v9-20260926` og slått inn selektivt. Detaljer, hva som er tatt
og ikke tatt, og hvorfor: [docs/CONSOLIDATION_20260926.md](docs/CONSOLIDATION_20260926.md).
Rot-loaderen importerer nå denne kodebasens `CLAUDE.md`, som importerer `GX1_RULES.md` og
`AGENTS.md` — samme regler for Claude og Codex. Én agent om gangen.

## Hva vi vet

- **Retningen ligger på uker–måneder, ikke på 95 minutter.** Trenden er ~1,6 % av bevegelsen
  per 95-minutters vindu; bekreftede M1-svingninger fortsetter med 49–51 %; retningsmålet hadde
  et hardt tak på 8 timer. På ukeshorisont tjente alltid-LONG etter all kost i 3 av 4 år, og
  ingenting (380 D1/H4-felt, trendregler) slo den — i 2021–26 er det lærbare driften. Se
  [docs/DIRECTION_TIMESCALE_20260926.md](docs/DIRECTION_TIMESCALE_20260926.md).
- **Den direkte M1-hypotesen** (25.–26.09) bruker samme etiketter som knee-målet, og vent-målet
  velger side i ettertid (+13,7 bps skjevhet); den kjøres ikke videre.
- **Native lifecycle-v2-modellen:** Entry FLAT256/256 (alle side-Q negative etter kost), Exit
  slår umiddelbar lukking på gjenbrukt TRAIN, samlet læringsport ikke bestått
  ([docs/ENTRY_SELECTOR_CACHE_FIT_20260924.md](docs/ENTRY_SELECTOR_CACHE_FIT_20260924.md),
  [docs/ENTRY_EXIT_LINKAGE_AND_COST_20260919.md](docs/ENTRY_EXIT_LINKAGE_AND_COST_20260919.md)).
- **Retningsforskningen 23.–24.09** (walk-forward, mønstre, kryss-aktiva, HGB, direkte
  klassifikasjon) er slått inn under `docs/`, med Codex' forbehold 24.09; ingen robust
  retning på 95 min–8 t ble funnet.
- **Ukesretning, forhåndsregistrert (26.09): NO-GO.** 0 av 48 celler slår alltid-LONG i noe år
  2023–26; modellene kollapser til «vær long» eller taper når de avviker. 2021–26 er ett
  oksemarked — lengre historikk med fallende markeder er den avgjørende forutsetningen;
  operatøren vedtok 26.09 å hente fra 2005; native M5 + M1 fra 2006-03-19 er hentet 27.09
  ([docs/HISTORY_2005_INTAKE_20260926.md](docs/HISTORY_2005_INTAKE_20260926.md))
  ([docs/WEEKLY_DIRECTION_RESULT_20260926.md](docs/WEEKLY_DIRECTION_RESULT_20260926.md)).

## Tilstand

Ingen jobb kjører. `training_enabled=false`; full epoch, full VAL, CONTROL, TEST, live og
papirhandel er stengt. TEST er forseglet. Kildekoden har nå den reparerte featureflaten v36
(signal 241, per-TF 190); datasettet må bygges på nytt. Eksisterende V9-/lifecycle-v2-artefakter
og sjekkpunkter hører til v34-flaten og evalueres bare på commit 7c9421a5 eller eldre.
Walk-forward-instrumentet støtter ukeshorisont (beslutningsklokke, ikke-overlappende statistikk,
kostpolicy, tidlig kalibrerte inputs). CURRENT har eget Python-miljø (`.venv`, identisk pakkesett,
avhengighetssjekken består), og `gx1_handover.sh` skriver kildeidentiteten som tunge ruter krever
(`source_identity_gate`).
