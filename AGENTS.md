# Arbeidsregler for GX1 — oppdatert 2026-09-16

Krev målbar læring før mer omfattende trening. Teknisk PASS er ikke bevis på
bedre handelsbeslutninger eller positiv kostnadsjustert netto Bps.

- Bruk bare /home/andre2/src/GX1_CURRENT, branch work/gx1-current. Mac-mappen er
  en overleveringskopi. Historiske kildekopier er avhengigheter og dokumentasjon,
  aldri alternative oppstartsveier.
- Start med ./handover.sh --check på Mac, eller bash scripts/gx1_handover.sh --check
  i Linux-repositoriet. Les CURRENT_HANDOVER.md, GX1_ARBEIDSMAAL.md,
  NEXT_RUN_POLICY.json og docs/LEARNING_GATE_20260916.md.
- CURRENT_HANDOVER.md er eneste gjeldende fortelling. RUNNING_NATIVE_CALIBRATION.json
  beskriver siste arbeid; prosesser, checkpoints og receipts må bekrefte nåstatus.
  COMPLETED_RUN.json og filer merket historikk er bevis, aldri startinstrukser.
- Én agent og én tung jobb samtidig. Gjenbruk ferdige analyser, cachede inputs,
  targets, outputs og beståtte tester. Ingen minuttvis modellpolling eller arbeid
  for å fylle ventetiden. Kontroller stabil langkjøring omtrent hver time;
  automatiske sikkerhetsvakter håndterer hyppig maskinvarekontroll.
- Endre modell-/treningskode bare for en konkret, observert blokkering. Forklar
  blokkeringen og minste rettelse først. Ingen forebyggende refaktorering, nye
  rammeverk, brede regel-/terskel-/modell-/tapsvektsøk eller gjentatte fullsuiter.
- Bruk bare eksisterende native campaign via gx1_capped_run.sh og etablerte
  maskinvarevakter. TRAIN16, VAL256, åtte CPU-arbeidere, tre timers VAL-vinduer,
  FP32/TF32 av. Ingen separate VAL-kjørere eller historiske smoker som fallback.
- Ingen full epoch eller full VAL mens læringsporten er uavklart. En gjennomført
  teknisk kontroll skal ikke automatisk gjentas eller utløse større trening.
  Gjeldende tillatt omfang står i NEXT_RUN_POLICY.json; training_enabled er false.
- Bevar alle 200 features, åtte familier, tidsrammer og kausalitet. «Ingen fast
  grense»: ingen fast tapsgrense eller maksimal holdetid. Femstegs targetberegning
  er en læringsberegning med bootstrap, ikke en handelsregel om holdetid.
- Skill lærerens verdiestimat fra fasit i observerte markedsutfall. Alltid FLAT
  er ikke dokumentert selektivitet; alltid HOLD er ikke dokumentert tålmodighet.
  TRAIN-tilpasning, senere VAL-kvalitet og samlet økonomi rapporteres hver for seg.
- TEST forblir forseglet. Ingen live-/papirhandel eller spending. Medregn alle
  valgte handler og åpne posisjoner ved økonomivurdering; lukkede vinnere alene
  er aldri samlet lønnsomhet. Juni 2026 er gjenbrukt utviklings-VAL.
- Bevar fullførte resultater, originale checkpoints og aktiv kjøring. Ikke endre
  frosset kilde under kjøring eller starte om for dokumentasjon/opprydding.
- Stående autorisasjon gjelder nødvendig arbeid innen oppgaven. Ikke spør om
  samme godkjenning igjen. Oppdater overlevering ved vesentlige endringer og
  commit/push ferdig arbeid når den aktive kildebindingen tillater det.
- Brukeren har bestilt diskopprydding. Slett bare dokumentert overflødige filer;
  kontroller aktive referanser og bevar modell-/data-/runtime-avhengigheter,
  unike resultater og checkpoints. Sletting skal ha en kort etterprøvbar logg.
- Skill lokale driftsgrenser fra produsentspesifikasjoner. En overskredet lokal
  temperaturgrense er ikke alene bevis på overoppheting.

Ingen antagelser der tilstanden kan måles. Bruk tokens konservativt. Når en
nødvendig rettelse er kontrollert, gå videre mot læringsmålet uten å utvide jobben.
