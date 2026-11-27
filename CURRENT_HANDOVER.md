# CURRENT_HANDOVER — GX1

## Autoritet og arbeidssted

- Arbeid bare i `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
- Les `GX1_RULES.md` helt; uendret SHA256
  `2c7f0fc9e564d5986db59157dee6cc947bdfd3260f742bacc464853a28683a58`.
- `NEXT_RUN_POLICY.json` er eneste arbeidsstatus. Originale terminaler,
  prosessobservasjon og hashbundne kvitteringer overstyrer prosa.
- Overtakelse: branch/log, `bash scripts/gx1_handover.sh --check`, dette
  dokumentet og aktuelle policyfelt. Åpne eldre bevis bare ved konkret behov.
- Root eier policy, launch og samlet kontroll; andre agenter har avgrenset eierskap.
- Historikk ligger i Git og originalkvitteringene, ikke i denne overleveringen.
  Siste fulle historikkversjon: `c6a8c6672:CURRENT_HANDOVER.md`.

## Gjeldende status og neste steg

Les disse policyfeltene for deres respektive scope:

| Felt | Siste avsluttede tilstand |
|---|---|
| `native_host_maintenance_logging_20261127_001` | Logging utrullet og verifisert; delmålet fullført. |
| `native_v3_bounded_development_test_20261127_001` | FIT_001 og FIT_002 er `FAILED_CONSUMED`; kun delresultater. |
| `prelearning_validity_remediation_20261125_001` | OANDA-/historisk scopereview ferdig; ingen ny fit-admission. |
| `native_twenty_step_qualification_20261123_001` | Fagstatus 13 PASS / 3 FAIL / 4 UNKNOWN; samlet ikke kvalifisert. |

Eldre «neste steg» i underliggende prelearning-/20-stegsfelt er historikk,
ikke launchautoritet over nyeste top-level status og de to nyeste scopefeltene.
Kriterier: `docs/NATIVE_QUALIFICATION_20_STEPS.json`.

Bruker27.11 ber om nytt mål og videre arbeid mot retningsedge med alle
indikatorer og samspillet mellom dem. Nytt appmål er registrert aktivt.
Root kvalifiserer én ny FIT_003 som neste avgrensede læringsmilepæl under
stående integratoradgang16.11; gammel FIT_001/FIT_002 relanseres aldri.
Autorisasjon og presis tolkning er bundet i
`fit003_continuation_authorization` i utviklingstestfeltet.

FIT_003 namespaceendring er kvalifisert:74 fokuserte tester, begge kildebroer,
72 uendrede baselineeiere og uavhengig diff-review PASS. Ingen modelljobb.
Alle seks modell-/launchflagg er foreløpig false. Ingen ny kjøring er
innrømmet før kilde-/input-/hostkontroll og metadata-preflight består.
FIT_003 beholder20727rader, alle5TF/familier, fullt koblet None, fersk
factoryINITIAL, B96/E256,2fastepass/maks432oppdateringer og barePASS2-kandidat.
Kohort/warm/cold gjenbrukes; ingen endring av modell, tap, vekter eller terskler.
Samme låste INITIAL og COHORT-prior måles med rad-/målbarvekting på alleTF.
Dette er TRAIN-optimeringsdiagnostikk, ingen senere generalisering eller edge.

Neste arbeid: kvalifiser og innrøm kun den nye eksakte engangsplanen med
fersk signert host-/idle-status og fryst kilde/policy. Ved stopp leses
journalens eksakte komponenter. Ingen automatisk retry/serie/hovedtrening.

## Vedlikeholdslogging — ferdig

Kjørende SYSTEM-task `GX1HostTelemetryBridge` bruker servicekode SHA256
`94cb46fc591aa20eef5d81811cb5e155bcc312c4213939303a41db9c6a0231a9`.
Journalen er append-only; gamle poster roteres, trunkeres eller slettes
ikke automatisk. Fil:

`C:\ProgramData\GX1\HostTelemetryBridgeV4\GX1-HostTelemetryBridgeMaintenance.jsonl`

Vitnet bruker samme snapshot som det signerte flagget: prosessnavn/PID/parent,
kjent starttid, setup/OOBE/reboot/CBS/WU, UTC, nonce og monotonic.
Første sample, tilstandsendringer og alle aktive forespørsler lagres;
uendret idle dedupliseres. Fsync/strict readback skjer før 200; feil gir 503.
Vedlikeholdspredikat, signert 14-feltsrespons, sertifikat, runner, config,
taskdefinisjon og alle sikkerhetsgrenser er bevart. Ingen helinstaller kjørt.

18 unike fokuserte tester, faktisk PowerShell-syntetikk, uavhengig review,
utrulling og faktisk signert sensorkontroll består. Originalkode/logg og
rollback er bevart. Første kalde forespørsel timeoutet på uendret 2 s;
etter readonly kontroll bestod neste forespørsel. Faktisk idle-vitne og
deduplisering er lest tilbake. Aktiv vedlikeholdshendelse i produksjon er
ennå ikke observert i ny journal; positive/negative baner er syntetisk testet.

Flagget betyr tilstedeværelse av TiWorker/MoUsoCoreWorker/SetupHost/SetupPrep/
WindowsUpdateBox/wuauclt, også en idle worker, eller setup/OOBE/reboot-in-progress.
Det beviser ikke aktiv installasjon. Eksakte utløsere ved de gamle stoppene
forblir UNKNOWN; historikken omskrives ikke.

## Siste V3-utviklingstest — avbrutt

Kohorten har 20 727 TRAIN-rader fra 1 377 kildeklokkebøtter, valgt før målinnsyn.
Alle fem TF og genuine familier beholdes; ingen etikettkvoter/topup.
Whole-TRAIN factoryprior/normalisering beholdes. COHORT-prior er kun scoring.
COHORT/WARM/COLD er COMPLETE med seks bookends PASS og alle ti cold-bevis
bitidentiske. Tekniske replayvekter er aldri fit-initializer.

FIT_002-planen er kun referanse for bundne inputs/recipe, ikke en launchordre:
`configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_DEVELOPMENT_20261127_001_FIT_002.json`.
Den hadde None/default, fersk factoryINITIAL, B96/E256, seed 20260911,
to faste pass og maksimalt 432 oppdateringer. Bare PASS2 var kandidat.

FIT_001 stoppet på vedlikehold før første oppdatering. FIT_002 stoppet
17:44:41 UTC 27.11 etter 468 s på samme flagg. Øvrige ressursgrenser ble ikke
brutt. Begge originale RESULT/TERMINAL er FAILED_CONSUMED med bookends PASS.

FIT_002 bevarer INITIAL_SCORE/LOGITS, nøyaktig lik factory-kontrollen og
`NOT_CANDIDATE_FIXED_PASS2`, samt hash-/readbackbekreftet CURSOR001.json/.pt.
Checkpointet beviser én varig oppdatering. Totalt utført er UNKNOWN, begrenset
til 1..216 av kontrollflyt og manglende PASS1-artefakter; ikke kall dette
nøyaktig én oppdatering. PASS1/PASS2/FINAL mangler. Ingen endelig læringsdom,
senere generalisering eller økonomisk edge er påvist.

## Kvalifisert datagrunnlag og funksjon

- Native OANDA practice/XAU_USD M5 MBA finnes fra 2009 til juni 2026.
  Alle kildeår/TF er skannet. Separat rettet provider-observasjonsreferanse
  har 542 753 høyere-TF-barer og 7 055 789 eksakte feltsammenligninger uten avvik.
- 1 410 restaurerte M5-observasjoner er utenfor vedtatt PAIR_WEEKCLOSED-domene.
  Referansen er ikke innført som modellinput eller ny featurehistorikk.
- 354 550 TRAIN-rader (2021–2025) har null tapte/ekstra rader fra lagret råkilde til
  canonical/CAPTURE. Alle 17 870 ukentlig-grid-fravær er allerede fravær i råsvar;
  ingen priser skal oppdiktes eller gap automatisk klassifiseres som stengt.
- V3 har 1 772 410 modne, kjente aktivitetsceller og 340 umodne haler
  (M5/M15/H1/H4/D1: 1/3/12/48/276). Tvetydig side blir aldri FLAT.
  Originale nominal-/kompletthets-/sourcegap-/kausale ATR-masker beholdes.
- Full fersk V3 INITIAL på 354 550 rader og egen C2-kald replay er kvalifisert.
  Låst COHORT-scoring skiller factoryreferanse, INITIAL og COHORT-prior;
  alle fem TF vurderes med rad-/målbarvekting, kalibrering og 25 tidsblokker.
- Åtte genuine spesialistfamilier og D1/makro samarbeider med M5 som hovedklokke
  og M15/H1/H4/D1-kontekst. M30 trenger egen kontrakt-/nyttevurdering.
- De 18 interne recurrence-/squeeze-prefixfunnene ble avkreftet før faktisk
  konsumhistorikk. Ingen ny inputbit eller gjentatt full replay er begrunnet.

## Fortsatt ubevist

13/3/4-statusen er ikke oppgradert av teknisk logging, dataparitet eller replay.
Historiske V1/V2 FAIL er ikke gjeldende V3-nevnere eller nye V3-læringsresultater.

- 03/07: historisk samme-feed markedskalender/quote-tilgjengelighet og relevant
  source-known/asof er uavklart. Fravær alene beviser ikke markedsstenging.
- 09: relevant publisering/revisjonsvintage for makro er fortsatt uavklart.
  44 av 61 felt er OANDA-prisderivater fra 11 instrumenter; øvrige 17 er
  COT4/Treasury9/VIX2/GLD2. Ingen aktuell OANDA44-lookahead er målt.
  May18 Treasury er kjent 2467 s før første faktiske input; Apr1 før faktisk
  input 2022-04-03T22:05Z er UNKNOWN. Deklarert 2022-04-03T04:00Z er et separat,
  sterkere klokkekrav; begge publiseringspåstander er ubevist.
  Ingen automatisk arkiv-/HTTP-retry.
- 11/12: provider-HLC-observasjon er ikke ubetinget markedskompletthet eller
  representativitet. Nominal klokke og observerte providerbarer er ulike bevis.
- 15/19: V3-læring og senere generalisering er ikke kvalifisert. TRAIN-år med
  whole-TRAIN-statistikk er ikke kronologisk OOF eller uavhengig senere evaluering.

Full kalender/vintage er ikke et universelt hinder for alle lokale tekniske
kontroller. En avgrenset fit trenger egen eksakt admission; læring og senere
generalisering må måles som utfall, ikke antas bevist før første fit.

## Varige kjøre- og oppryddingsgrenser

- Det varige M5/MTF/26-måneders-MBP-målet i `GX1_RULES.md` består.
  Målet er nyttige tosidige prognoser og senere LONG/SHORT med liten MAE før MFE;
  GC/orderbok er utsatt, ikke slettet. Metode: `docs/NATIVE_LEARNING.md`.
  Én tung CURRENT-jobb via `scripts/gx1_capped_run.sh`, lås og eksisterende
  hardware-/hostvakter; frosne kilder/policy under kjøring. Stabile lange
  kjøringer kontrolleres omtrent hvert 15.–30. minutt, ikke minuttvis.
- Ingen TEST-tilpasning, M1-retningsinput/fasit/normalisering/utvalg, GC,
  Entry/Exit, broker/live/paper/ordre/spending/promotion i denne bølgen.
  TEST er forseglet; juni 2026 er gjenbrukt utviklings-VAL.
- Ingen fast tapsgrense/maksimal holdetid eller fjerning av genuine familier/TF.
  Teknisk PASS, alltid FLAT/HOLD og TRAIN-score beviser ikke handelsnytte.
- Alle originale MAIN-/recovery-checkpoints, kvitteringer, inputs og backuper
  beholdes. Intet nytt slettevedtak foreligger.
  `full_project_retention_cleanup_20261120_001` er forbrukt med resterende 0.
  DATA/RUNS ryddes bare av `gx1.scripts.cleanup_gx1_evidence_v1` etter regel 9,
  med fersk dry-run, kanonisk prosjektlås, verifisert eierskap/referanseomfang
  og eksakt godkjent inventar.
- Denne dokumentforkortingen endrer ingen kode, policy, data eller vakter.

## Bevis som åpnes ved konkret behov

`E` nedenfor betyr:
`/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001`.
Eksakte SHA-bindinger og øvrige originalreferanser ligger i policy.
Metodegrenser: `docs/NATIVE_PRELEARNING_CALENDAR_EVIDENCE_20261125_001.json`,
`docs/NATIVE_PRELEARNING_MACRO_EVIDENCE_20261125_001.json` og
`docs/NATIVE_PRELEARNING_SOURCE_CLOCK_BOUNDARY_20261125_001.json`.

| Under E | Beviser |
|---|---|
| `HOST_MAINTENANCE_LOGGING_ACCEPTANCE_20261127_001.json` | Utrulling, faktisk signert sensorrespons og idle-vitne. |
| `HOST_MAINTENANCE_LOGGING_FINAL_BACKUP_20261127_001.json` | Separat Mac-disk/readback for original, ny kode og logger. |
| `V3_DEVELOPMENT_FIT002_STOP_ACCEPTANCE_20261127_001.json` | Avbrutt fit og bevarte delresultater/oppdateringsgrenser. |
| `V3_DEVELOPMENT_FIT002_FINAL_BACKUP_20261127_001.json` | Separat sikring av delcheckpoint og originalkvitteringer. |
| `ALL_TF_PROVIDER_HLC_TARGET_ACCEPTANCE_20261125_001.json` | Gjeldende V3-fasit/prior/dekning. |
| `ALL_TF_V3_INITIAL_ACCEPTANCE_20261125_001.json` | Full fersk factoryINITIAL. |
| `ALL_TF_V3_REPLAY_ACCEPTANCE_20261125_001.json` | Egen C2-replay på opprinnelig V3-funksjon. |
| `ALL_TF_OBSERVED_BAR_REPAIR_ACCEPTANCE_20261125_001.json` | Separat reparert observasjonsreferanse, ikke nye inputs. |
| `OANDA_PROVIDER_DATA_AND_TEST_SCOPE_REVIEW_20261126_001.json` | Kilde-/kalender-/makrogrenser og teknisk testscope. |

Gjeldende delresultater og selected-cohort-replay ligger i
`/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_DEVELOPMENT_20261127_001_`
med suffiks `cohort`, `warm`, `cold`, `fit` og `fit_002`.
Loggingens originale/utrullede kode og kvitteringer ligger i
`/home/andre2/GX1_RUNS/HOST_MAINTENANCE_LOGGING_20261127_001`.
