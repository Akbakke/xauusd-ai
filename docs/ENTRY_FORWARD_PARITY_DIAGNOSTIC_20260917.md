# Én måling av Entry-forwardens numeriske samsvar

Signaldiagnosen på3ee12e57 stoppet17:27:26UTC /19:27:26Oslo17.september på
ENTRY_SIGNAL_CACHED_PREDICTION_CHANGED. Native prosess er terminal, guard
child_status1 og Windows-tasken er deaktivert med siste resultat1. Ingen
læringsrapport eller terminal native receipt ble skrevet. Ingen optimizersteg;
original checkpoint og pointer er rehashet og bevart.

Målekoden logget ikke variant eller avviksstørrelse før stopp. Ingen av disse
kan antas. Logg/plan/cache og feilforsøk bevares. Avviksloggingen er rettet.

Minste avgjørende måling er nå separat bundet: samme cachede TRAIN16, både
startmodell og sluttmodell, hver i inferensmodus og gradientaktivert eval.
Fire forwards totalt, ingen backward eller optimizer. Begge modeller inngår
fordi forrige logg ikke identifiserte hvilken som feilet. Ingen nye data/mål.

Rapporter hver modells nye inferens og gradientforward mot dens lagrede
native inferens og mot hverandre: maksavvik, RMS, per side og handlingsbytter.
Bevar eksisterende0,0001Bps-grense; rapporter avvik også når den ikke består.
En fullført rapport er ikke et numerisk PASS eller tillatelse til å øke
toleransen. Modell-/treningsmatematikk og læringskonklusjon er uendret.

Native campaign, kildefrys og vakter beholdes. Ingen Exit-forward, CONTROL,
VAL, TEST, søk eller automatisk ekstra kjøring. Se gjeldende policy og
BASE/NATIVE_ENTRY_FORWARD_PARITY_20260917/PLAN.json. Planen er ikke startbevis;
PREPARATION_RESULT.json, faktiske prosesser og receipt avgjør status.

## Klar overlevering — ikke startet

`BASE` er `/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912`.
`A` er `BASE/NATIVE_ENTRY_FORWARD_PARITY_20260917`.
Planens filhash er `dc07584410072b56b35248683f8935e2ab237e3cbb70845d2a4777489db23b87`.
Recipe, campaign og PREPARATION_RESULT finnes ennå ikke. Windows-tasken står
deaktivert på den feilede signalplanen. Et runtime-navn i status er planlagt,
ikke bevis på start eller på aktuell Windows-boot.

1. Kjør vanlig handover, kontroller ingen native prosess, ren og pushet
   `work/gx1-current`, deaktivert task og at den nye planen ikke allerede er brukt.
2. Fra `/home/andre2/src/GX1_CURRENT`, sett `GX1_PARITY_DIR` til den absolutte
   A-stien og kjør `bash scripts/gx1_capped_run.sh --class audit -- .venv/bin/python
   "$GX1_PARITY_DIR/OPERATOR_HANDOVER/PREPARE.py"` som én shell-kommando.
   Operatøren bruker aktuell boot og bundet policy; den lager recipe/campaign
   uten modellforward. Gjenbruk eksisterende PREPARATION_RESULT hvis forberedelse
   allerede finnes; ikke overskriv delvis eller fullført arbeid.
3. Kontroller PREPARATION_RESULT, recipe- og campaign-filhash. Lag en ny kopi av
   `OPERATOR_HANDOVER/ACTIVATE_TEMPLATE.ps1` og erstatt bare `__NEW_SHA__` med
   campaign-planens filhash og `__SOURCE_COMMIT__` med dens kildecommit.
   Uutfylte maler avvises før taskendringer. Kjør den utfylte kopien på Windows
   via eksisterende SSH/PowerShell-inngang. Den verifiserer tidligere plan,
   uendrede kontrollere og taskinnstillinger, tar backup og starter én gang.
4. Bevar kildefrys. Kontroller controller/prosess etter start; planlagt reboot
   kan midlertidig bryte SSH og er ikke grunn til relansering. Bruk
   `_native_processes` i handover-innsamleren; en grep som leter etter filsti kan
   overse `python -m`-prosessen. Følg normalt opp etter 15–30 minutter.
5. Ved terminaltilstand: deaktiver tasken, verifiser guard/receipt og uendret
   originalcheckpoint. Les A/entry_gradient_diagnostic/RESULT.json. Rapporter
   initial og final hver for seg, bevar eventuelle feil og steng brukt scope.

Inferens som matcher cache mens gradientforward avviker støtter hypotesen om
ulike numeriske kodeveier. Hvis også ny inferens avviker, er den forklaringen
utilstrekkelig. Ingen terskelendring eller ny trening følger automatisk.
Formålet er å gjøre den opprinnelige Entry-signaldiagnosen tolkbar, ikke å
utvide prosjektet med nye modellforsøk.

Cache-/scope-admission og tre målrettede diagnostikktester består. Bevar
`ADMISSION_CHECK.py`, admission-logg og testlogg i A; ikke gjenta beståtte
modellkontroller. OPERATORS_MANIFEST.json binder de varige operatørmalene.
