# GX1 — overlevering 17. september 2026

## Les dette først

**Den kausale Entry-prøven er ferdig. Neste oppgave er én paret analyse av
lagrede resultater. Ikke start ny trening eller relanser den brukte planen.**

Native-kjøringen startet 14:45:41.962204 UTC / 16:45:41 Oslo og sluttet
15:59:49.236680 UTC / 17:59:49 Oslo 17. september. Checkpoint5 har 256
optimizersteg, offset256, epoch0. Native prosess er borte. GuardPASS,
trainer0/observer0; topp 59 °C kjerne, 66 °C minne, 156,89 W og 8014 MiB VRAM.
Windows-tasken er deaktivert med kvittering 16:11:52 UTC / 18:11:52 Oslo.

Slutt-ONLINE er målt på TRAIN256, 256 Exit-ankere og 1024 samplede Exit-states.
Resultatet bekrefter eksakt bevart korrigert fasit, frossen lærer og bare TRAIN.
**Dette er teknisk fullføring. Læringsgevinst, generalisering og profitt er
ennå ikke vurdert eller dokumentert av denne prøven.**

Kjør `./handover.sh --check` på Mac eller `bash scripts/gx1_handover.sh --check`
i Linux for fersk status. `--verbose` viser også denne overleveringen.
Ingen handover-kommando starter trening. Gjeldende policy har stengt det brukte
unntaket; tillatt neste arbeid er analyse av eksisterende TRAIN-artefakter.

## Autoritativ kilde og siste fullførte kjøring

- Kode: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
- Treningscommit (historisk kildebinding): `955abf191a3960025e297f331f0c33df90e731c4`, pushet.
- Data: `/home/andre2/GX1_DATA`. Mac-mappen er overlevering, ikke treningsrepo.
- Run: `NATIVE_CAUSAL_ENTRY_FIXED256_20260917`.
- Artefaktmappe: `/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_CAUSAL_ENTRY_FIXED256_20260917`.
- Runtime: `/home/andre2/GX1_RUNS/NATIVE_CAUSAL_ENTRY_FIXED256_20260917_BOOT456`.
- Faktisk boot er **457**; `BOOT456` i mappenavnet er booten planen ble forberedt på.
- Native invocation startet 14:45:41.962204 UTC / 16:45:41 Oslo.
- Session: artefaktmappen + `/.gx1-candidate-training-session.CANDIDATE_BUNDLE`.
- Checkpoint: session + `/CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json`.
- Sluttmåling: session + `/final_online_measurement/RESULT.json` og `TRAIN_OBSERVATION.json`.
- Terminalkvittering: runtime + `/receipts/invocation-0001.json`.
- Windows-task: `GX1NativeLearningCalibration`, deaktivert. Ingen automatisk utvidelse.

`NEXT_RUN_POLICY.json` har `training_enabled=false` og ingen aktiv
`chronological_learning_run`. Den eksakte launch-policyen er bevart i
artefaktmappens `COMPLETED_NEXT_RUN_POLICY.json`. Receipt-feltet RESUMABLE og
checkpointets complete=false beskriver en ufullført full-epoch-session;
den bestilte 256-prøven og sluttmålingen er ferdige. Dette gir ikke gjenopptakstillatelse.

## Hva vi vet — og ikke vet

To observerte feil er rettet. Først hindret en detach Entry-Q-feilen i å trene
upstream Entry-representasjon. Å åpne forbindelsen bedret TRAIN-verdiestimatene,
men retningsvalg slo ikke konstantbaseline, og Exit ble litt svakere.

Deretter fant vi etterpåklokskap i Entry-fasiten: positive framtidige HOLD-utfall
ble beholdt, negative ble klippet til null. Det løftet TRAIN256-fasiten kunstig
med 8,6852 Bps i snitt. Rettelsen på `ad464322` bruker den allerede fastsatte
kausale referansepolicyens forventning, `(119/120)*Q_HOLD`, pluss kanonisk første
likvidasjonsverdi. Negative utfall beholdes. Dette er policy-evaluering, ikke
fasit for optimal handel. Exit-mål, kostnader, gamma og bootstrap er uendret.

Kanonisk økonomi gjenskapte alle 512 gamle sidemål eksakt. En tydelig merket
avledet TRAIN-baseline bruker korrekt Entry-fasit og bevarte native prediksjoner.
Den opprinnelige målingen er ikke omdøpt. 33 tester for fasitrettelsen og 13 for
baselinebindingen består. Ny faktisk læring med korrekt fasit er ennå ikke vurdert.

## Hva prøven skal avgjøre

Samme lagrede ferske starttilstand, samme 4096 Entries i samme rekkefølge,
TRAIN16, 256 oppdateringer, frossen lærer og slutt-ONLINE. Mål TRAIN256,
256 Exit-ankere og 1024 samplede Exit-states. Ingen CONTROL-forward eller TEST.
Sammenlign mot lagret initialmodell, connected256 og TRAIN-konstanter med
identisk korrekt Entry-fasit og uendrede Exit-mål. Alle ni måneder og begge
retninger skal med. Konstant nivåflytting, all-FLAT/all-HOLD eller bare bedre
hjelpeprognoser består ikke læringsporten. Dette er TRAIN med fitted-overlapp;
generalisering og lønnsomhet er ikke dokumentert.

**Fortsett etter `VEIEN_VIDERE.md`.** Der står eksakt sluttkontroll, baseline-
referanser, hva som skal måles og hva som ikke skal gjentas.

## Handover og bevis

Alle operative innganger peker nå på samme fullførte jobb og neste analyse:
CURRENT_HANDOVER.md, CURRENT_GX1_STATUS.md, VEIEN_VIDERE.md, README.md,
GX1_ARBEIDSMAAL.md, RUNNING_NATIVE_CALIBRATION.json og NEXT_RUN_POLICY.json.
Mac- og Linux-script er kontrollert. Innsamleren velger en eventuell aktiv
session fra gjeldende policy og faller aldri tilbake til forrige checkpoint.
Tre målrettede tester består; ingen full modelltestserie er gjentatt.

Maskinbevis: `handover_snapshot/CAUSAL_ENTRY_FIXED256_COMPLETION_20260917.json`.
Historikk før oppryddingen er bevart i aktiv artefaktmappes
`OPERATOR_HANDOVER/BEFORE_CANONICAL_REFRESH` og Mac `handover_snapshot/`.
Den eksterne OPERATOR_HANDOVER-kopien som ble brukt under kildefrys er nå
arkiv. Bruk de vanlige kanoniske handover-inngangene. Ingen gamle planer er
startinstrukser. Stående godkjenning for ferdig kode/dokumentasjon/aggregater
på offentlig work/gx1-current gjelder; ikke spør på nytt.
