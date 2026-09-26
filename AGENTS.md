# Arbeidsregler for GX1 — oppdatert 2026-09-26

Gjelder alle agenter (Codex leser denne fila; Claude får den via `CLAUDE.md`). De bindende
prosjektreglene står i [GX1_RULES.md](GX1_RULES.md) — les dem først. Krev målbar læring før
mer omfattende trening. Teknisk PASS er ikke bevis på bedre handelsbeslutninger eller
positiv kostnadsjustert netto bps.

- **Én kodebase:** bruk bare `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
  Andre worktrees og grener er git-lagring og historikk, aldri arbeidssteder. Den arkiverte
  `audit/v9-premiere-20260905` (worktree `/home/andre2/src/GX1_ENGINE`) er slått inn her og
  tagget `archive/gx1-engine-audit-v9-20260926`; ikke commit der. Mac-mappen er en
  overleveringskopi.
- **Én agent om gangen.** Claude og Codex jobber aldri parallelt eller på hvert sitt spor.
  Start med `git branch --show-current` (må være `work/gx1-current`) og `git log -5`; bygg på
  den forrige agentens commits.
- Start med `bash scripts/gx1_handover.sh --check`. Les deretter CURRENT_HANDOVER.md,
  GX1_ARBEIDSMAAL.md, VEIEN_VIDERE.md og NEXT_RUN_POLICY.json. Ikke relanser en aktiv
  eller fullført plan.
- CURRENT_HANDOVER.md er eneste gjeldende fortelling. Prosesser, checkpoints og receipts må
  bekrefte nåstatus. COMPLETED_RUN.json og filer merket historikk er bevis, aldri
  startinstrukser.
- Én tung jobb samtidig. Gjenbruk ferdige analyser, cachede inputs, targets, outputs og
  beståtte tester. Ingen minuttvis modellpolling. Kontroller stabil langkjøring omtrent
  hver time; automatiske vakter håndterer hyppig maskinvarekontroll.
- Endre modell-/treningskode bare for en konkret, observert blokkering eller et vedtatt
  designskifte. Forklar blokkeringen og minste rettelse først. Ingen forebyggende
  refaktorering, nye rammeverk, brede regel-/terskel-/modell-/tapsvektsøk eller gjentatte
  fullsuiter.
- Tung trening bare via eksisterende native campaign gjennom `scripts/gx1_capped_run.sh` og
  etablerte maskinvarevakter; profil og tillatt omfang står i NEXT_RUN_POLICY.json.
  `training_enabled=false` stenger ny trening. Ingen full epoch eller full VAL mens
  læringsporten er uavklart.
- Bevar alle features, åtte familier, tidsrammer og kausalitet. Ingen fast tapsgrense eller
  maksimal holdetid; en beregningshorisont er ikke en handelsregel.
- Ved endret ONLINE-funksjon må ny initialbaseline måles; lik vekthash er ikke
  funksjonsparitet.
- Skill lærerens verdiestimat fra fasit i observerte markedsutfall. Alltid FLAT er ikke
  dokumentert selektivitet; alltid HOLD er ikke dokumentert tålmodighet. TRAIN-tilpasning,
  senere generalisering og samlet økonomi rapporteres hver for seg. På lange horisonter er
  alltid-LONG / kjøp-og-hold valgt før perioden referansen, ikke myntkast.
- TEST forblir forseglet. Ingen live-/papirhandel eller spending. Medregn alle valgte handler
  og åpne posisjoner ved økonomivurdering. Juni 2026 er gjenbrukt utviklings-VAL.
- Bevar fullførte resultater, originale checkpoints og aktiv kjøring. Ikke endre frosset kilde
  under kjøring.
- Stående autorisasjon gjelder nødvendig arbeid innen oppgaven; ikke spør om samme godkjenning
  igjen. Oppdater overlevering ved vesentlige endringer i samme bølge som koden.
  Brukeren ga 2026-09-17 stående godkjenning for push av ferdig GX1-kode, dokumentasjon,
  interne artefaktstier og aggregerte bevis til Akbakke/xauusd-ai, `work/gx1-current`.
  Rådata, modellvekter og hemmeligheter publiseres aldri. Force-push krever eget vedtak.
- Diskopprydding: bare via retention-eieren (GX1_RULES.md regel 9), med etterprøvbar logg.
- Skill lokale driftsgrenser fra produsentspesifikasjoner.

Ingen antagelser der tilstanden kan måles. Når en nødvendig rettelse er kontrollert, gå videre
mot målet uten å utvide jobben.
