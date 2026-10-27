# Arbeidsregler for GX1

Les [GX1_RULES.md](GX1_RULES.md) helt før arbeidet starter. Den er bindende for
Claude og Codex. En teknisk PASS er ikke bedre handelsbeslutninger eller profitt.

- Bruk bare /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
  Andre worktrees/grener er lagring, aldri arbeidssteder.
- Flere agenter er vedtatt18.10: avgrens oppgaver og fileierskap.
  Ingen parallelle tunge jobber eller
  kildeendringer under en fryst kjøring. Start med git branch --show-current
  og git log -5; bygg på eksisterende arbeid.
- Kjør bash scripts/gx1_handover.sh --check. Les kun siste CURRENT_HANDOVER.md
  og aktuelle status-/scopefelt i NEXT_RUN_POLICY.json ved overtakelse.
  Eldre kvitteringer åpnes bare ved konkret avvik eller nødvendig ny gatekontroll.
  CURRENT_HANDOVER samler gjort,
  ikke gjort, hindring og neste steg; NEXT_RUN_POLICY er eneste arbeidsstatus.
  Les øvrige metode-/designdokumenter bare når den aktuelle oppgaven trenger dem.
  Ingen daglige historikkvedlegg eller gjentatt gjennomgang av uendrede steg.
  Prosessobservasjon og eksakte terminalkvitteringer overstyrer prosa.
- Ikke relanser fullførte, claimede eller avbrutte engangsplaner. En ny plan er
  ikke autorisert av en gammel godkjenning. training_enabled=false stenger trening.
- Én tung jobb innen CURRENT, alltid gjennom scripts/gx1_capped_run.sh og
  eksisterende maskinvarevakter. Separate prosjekter har separate prosjektlåser;
  dette svekker ingen kapasitets-, minne-, CPU-, effekt- eller temperaturgrense.
- Gjenbruk ferdige inputs, targets, outputs og beståtte kontroller. Stabil
  langkjøring kontrolleres omtrent hver15.–30.minutt, gjerne sjeldnere ved
  stabil drift; ingen minuttvis modellpolling.
- Modell-/treningskode endres bare for en konkret observert blokkering eller
  vedtatt designendring. Begrunn minste rettelse før implementasjon. Ingen
  forebyggende refaktorering, brede søk, nye rammeverk eller gjentatte fullsuiter.
- Følg det varige M5/MTF/26måneders-MBP-målet i GX1_RULES.md. Revider ALLE
  genuine indikatorer og alle tidsrammer til faktisk modellkonsum; åtte
  spesialistfamilier og niende D1/makro-kontekst beholdes. Meld og rett målte
  mangler; en kort tidligere test er ingen generell dom over prosjektet.
- Bevar genuine features, tidsrammer, kausalitet og alle vakter.
  Ingen fast tapsgrense eller maksimal holdetid. En beregningshorisont er ikke
  en handelsregel. Endret ONLINE-funksjon krever fersk initialbaseline.
- Skill TRAIN-fit, senere generalisering, referanseverdi og realisert økonomi.
  Alltid FLAT/HOLD beviser ikke selektivitet/tålmodighet. På lange horisonter
  brukes forhåndsvalgt alltid-LONG/kjøp-og-hold, ikke myntkast, som referanse.
- TEST er forseglet for all tilpasning og alt utvalg. Bare forward-journalen
  (GX1_RULES.md, vedtak 11.10 kveld) scorer forseglet periode, én gang per måned,
  uten refit. Ingen live/paper, broker, ordre, spending eller promotion.
  Juni 2026 er gjenbrukt utviklings-VAL. Økonomi omfatter alle valgte handler
  og åpne posisjoner, ikke bare lukkede vinnere.
- Ikke endre fryst kilde under kjøring. Bevar originalkvitteringer og checkpoints.
  DATA/RUNS ryddes bare med retention-eieren etter GX1_RULES regel 9.
- Fjern bevist frakoblet kode og foreldede repo-filer etter referanse-, import-,
  test- og eierskapskontroll. Git er gjenopprettingskilden, ikke nye historikkmapper.
- Oppdater overlevering i samme bølge som koden. Avslutt med fokuserte tester,
  syntaks, stale-path-scan, git diff --check og ærlige ubeviste grenser.
- Stående brukerautorisasjon fra 17.09.2026 tillater push av ferdig kode,
  dokumentasjon, interne artefaktstier og aggregerte bevis til Akbakke/xauusd-ai,
  work/gx1-current. Aldri rådata, vekter eller hemmeligheter; aldri force-push
  uten eget vedtak. Ikke spør på nytt om allerede gitt relevant autorisasjon.

Ingen antagelser der tilstanden kan måles. Gjenværende læringsmetode:
[docs/NATIVE_LEARNING.md](docs/NATIVE_LEARNING.md). Aktuell oppryddingsgrense
står i [CURRENT_HANDOVER.md](CURRENT_HANDOVER.md); historikken finnes i Git.
