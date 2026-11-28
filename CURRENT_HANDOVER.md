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

NEXT_RUN_POLICY.json er eneste arbeidsstatus. Felt001
native_v3_observed_train_continuation_20261128_001 er faktisk COMPLETE/stengt/
forbrukt0/relaunch=false. Prosessen er faktisk avsluttet, indre/ytre exit0,
alle6bookends PASS. Alle seks globale flagg er false. Edge-målet er aktivt og
uoppfylt. Ingen senereVAL/Control/TEST/ordre/paper/live er kjørt.

Fryst kilde58504b7bb, plan-SHA
c4e9b8ff8f760b062d773b08a585caeff88cec551ced3f6e57704da61c50f41f.
Ekte opprinnelig FINAL432 ble full10-validert før modell/AdamW/fullRNG-copy.
216nøyaktige andrepass-takes216..431 gikk til låst terminal648, hele20727TRAIN/
1377kildedøgn2021–25, alle5TF/8familier+D1makro/None. Ingen teknisk C433/C434
som initializer, ingen ny feature/target/prior/norm-fit eller hyperparameter-
endring.297forwards=216TRAIN+81fullFINAL-EVAL,216backward/update/observasjoner,
1checkpointload/restore,2atomiske checkpointwrites/readbacks,1logitarkiv,
0autograddiags/skips. Fullførte på435,35s. C433 er mekanisk; bare FINAL648 er
kandidat. Nye ekte FINAL648-referanser ligger i policy/RESULT/FINAL648.json:
checkpoint-SHA c21159dcade8f0fe36eddd4d0d53c62d1fa07be2985a19b18b7b0d4232b8d150,
receipt-SHA f3534bdeeb460a56f04b5c8fa7b7142a56e43341ef6075b653fd30558fc8d0b3.

Uendret numerisk skåreier bruker PASS2/gammel spec, eksplisitt ytre FINAL648-
navnerom/fasealias. Ingen gammel PASS2-kvittering er omskrevet. Baseline432/
factoryINITIAL/COHORTprior er gjenbrukt. Actualreview-SHA
9351f5f72573d3ec5b223ff2f26f9debac91aded6477f556d35b51c50b4f6a08.
Alle5TF fikk bedre NLL/Brier under både rad- og målbarvekting og bedre rad-AUC.

| TF | BASELINE432 → FINAL648 AUC | FINAL648 Brier minus COHORT-prior, rad | År-PASS rad/målbar |
|---|---:|---:|---:|
| M5 |0.519403 →0.528606|+0.000338720|1/5 ·1/5|
| M15 |0.518758 →0.527050|-0.000523637|5/5 ·5/5|
| H1 |0.524846 →0.541390|-0.000684374|4/5 ·4/5|
| H4 |0.524267 →0.539902|-0.001246586|5/5 ·5/5|
| D1 |0.614417 →0.673314|-0.022056640|5/5 ·5/5|

M5 feiler fortsatt prior/factory; samlet proper/stabilitet/TRAINKVAL FAIL.
År-PASS20/25 under begge vektinger, mot18/25rad og17/25målbar ved432.
M5s2023NLL/Brier forverres svakt (+0,0000714049/+0,0000355959); andre
år/TF-proper forbedres. Alle5TFsAUC forbedres i hvert år. M5EVAL-
pUP48,4560→48,1775%, fasit50,8831%; middelfeil−2,4271→−2,7057pp. M5Brier
bedres0,0000597014 samtidig som kvadrert middelfeil øker0,0001429706;
algebraisk rest bedres0,0002026721. Klasseseparasjon øker, sannsynlighetssenter
faller. Dette isolerer ingen representasjonsårsak eller historisk feil.
TRAIN er ikke senere generalisering, lønnsomhet eller bevist indikatorverdi.
Målbar-AUC er fortsatt NOT_IMPLEMENTED; D1s13968rader er875målbarer.

Alle1080nativeF32gradientkontroller PASS; maxfeil3,17e-9, høyest1,48% av
F32-grensen. Global klipping8/216, ingen biasgradientsignflip. M5nativebias
flyttes bare−0,00000205636, m1+.000880403→+.000575018; observerte AdamW-
steg er konsistente, ingen sign-/bias-/tapspatch er begrunnet. Utviklende
TRAIN-batchsnitt er ikke frosset FINAL648-modusgap. Endpointassessment-SHA
b7a5138b2450ec2c81e4887b7771851a53ad91f2d12aeaede96dfcf36547b783.

Ytre signert vakt:278målinger/child_status0, ingen stop/fatal/maintenance;
peakGPU61C/minne68C/186,83W/9772MiB, CPU43C/35,55W, hostledig minst54,93GB/
commitmargin49,62GB. Fysisk GPUgrense200W. Boot4 og sammeWSL-ID som CLAIM,
ingen queryfeil, prosessPID1388327/start48897827 borte. Vaktene ble ikke endret.

Før NN bestod60fokuserte kontroller og86-eier/fersk modell-/inputidentitet
med0metadata-modell/optimizer/PT/NPZ/CUDA-forsøk. Feil i ny testimport/fixture
ble rettet; original001/002/003 bevart. Første ubrukte hardware-preflight
stoppet på0telemetrimålinger; fersk signert normal host og egen002-preflight
bestod før admission. Backupkoblingens feltnavn ble rettet før admission;
samme immutable arkiv/kvitteringer gjenbrukt. Ingen vakt ble svekket. SOURCE:
146filer, fsync/F_FULLFSYNC/allemedlemmer verifisert på separat Macdisk.
FINAL-backup er fysisk fsync/F_FULLFSYNC/allemedlemmer-verifisert på Mac:
181filer/229070801bytes, arkiv-SHA
eed13afe13b138db115dc3c24ef317508c5c4f334e33984153cc1112ee21427c.
Manifest/kvittering ligger i policy; avsluttende binding/overlevering bevares i
Git. OriginalFINAL432/tekniske referanser gjenbruker verifiserte backuper.

Neste felt002 native_v3_observed_train_continuation_20261128_002 er kun
DESIGNED_SOURCE_PREPARATION_ONLY_CLOSED, execution=false/training=false/
remaining0/relaunch=false. Design-SHA
f52d056a1715f13180b6aee375ae17fd45eb992644bf447f38a0239ebf34bf30.
Etter faktisk all5-forbedring er én separat forhåndsdefinert fortsettelse
foreslått: ekte FINAL648 til låst864, samme originale takes216..431 én gang,
216updates+81fullEVAL/297fw, uendrede funksjoner/inputs/recipe/kapper. Ingen
automatisk serie eller ny NN-admisjon. Nytt current648-checkpoint trenger
sin egen ekte648-trajektorie/full10restore; aldri feil gammel432-lineage.
Neste arbeid: minste nye eksekverings-/lineagebinding med gjenbrukte beståtte
matematiske eiere/kontroller, fersk source/host/preflight/SOURCEbackup og fysisk
FINAL648-backup før ny eksakt snapshot-først admission. TEST fortsatt forseglet.
Original mislykket COLD/alle forbrukte tidligere roller forblir urørt/stengt0.

### Bevarte forutsetninger og læringsbaseline

Original REFERENCE/COLD-kvalifisering brukte011ab4ad2: REFERENCE gjorde2tekniske
steg fra FINAL432 til C433/C434 og bestod; original COLD gjorde1steg/all10paritet,
men beholdes FAILED_CONSUMED på boots/host_maintenance_active. Etterfølgende
boot-/hostobservasjon reparerer ikke den gamle feilen. Egen ny recovery ovenfor
bestod senere. Alle tekniske vekter er aldri kandidat eller initializer.

Observatørens fire detach-/skalarkopipunkter gjør ingen ekstra forward/backward/
update/zero_grad/RNG-/modusoperasjon. Standardbanen projiseres eksakt til gammel
core. Rapporter ligger utenfor legacy steps/scores. I det bevarte tekniske steget
var globalnorm0,726546<clip1, alle5native biasgradienter innen F32-grensene og
pre-/postklipp like. M5g=-0,005005045 reduserte positivt AdamW-m1 fra0,001463901
til0,000817007; bias sank0,0000105165. Dette er konsistent med momentum i dette
steget, ikke historisk årsak, læringskvalitet eller universelt fravær av feil.

To endpointforklaringer er allerede kvantifisert: frossen TRAIN/EVAL-modus flyttet
M5s pUP-middel bare+0,037710pp; batch-/nevnervekting flyttet effektiv UP-andel
høyst0,004853pp, mot M5-middelfeil−2,427107pp. Direkte native bias stod for0,584%
av whole-batch→issue-miks logitendring;99,416% er samlet representasjons-/
projeksjonsrest og F32-avrunding, uten isolert årsak. Ingen bias-/dropout-/
tapsnormaliseringspatch er begrunnet. Bevisene og gamle feil er bundet i policy/Git.

Forrige læringsbaseline var låst issue-miks PASS2, fryst kilde7471ddf48:
20727TRAIN-issues/1377kildeklokkedøgn2021–2025, B96/E256, seed20260911,
2pass à216updates. Alle8familier+D1/makro og M5/M15/H1/H4/D1, fullt koblet
None-policy og egne kausale vinduer beholdes. Fersk FIT fullførte432updates/
597forwards/812,48s,0checkpointloads/restores/skips og seks bookends. Opprinnelig
factory/fullRNG/INITIAL/fasit er kvalifisert og gjenbrukt, aldri tekniske vekter.

| TF | Låst PASS2 side-AUC | Brier minus COHORT-prior, rad | År-PASS rad/målbar |
|---|---:|---:|---:|
| M5 |0.519403|+0.000398422|1/5 · 1/5|
| M15 |0.518758|-0.000249777|4/5 · 4/5|
| H1 |0.524846|-0.000190210|3/5 · 3/5|
| H4 |0.524267|-0.000416408|5/5 · 4/5|
| D1 |0.614417|-0.008619807|5/5 · 5/5|

Proper-score består4/5TF under begge vektinger; M5 feiler. Samlet proper/stabilitet
FAIL, learning_qualified=false; år18/25rad og17/25målbar. Aktivitet/rank består
alle fem uten bevist læring/generalisering. D1s13968rader er875målbarer. M5DOWN-
argmax12721/12739 er betinget side, ikke full treklassehandling/handelsvalg.
PASS1→PASS2: M5-AUC.509672→.519403 mens Brier forverres0,000156777 og
UP-middelfeil−1,758→−2,427pp; D1-AUC.561885→.614417. Flere oppdateringer må
måles som utfall, ikke antas å gi fremgang. All TRAIN-statistikk er gjenbrukt,
ikke kronologisk OOF eller uavhengig senere evaluering. Målbar-AUC er NOT_IMPLEMENTED.

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

## Bevis som åpnes ved konkret behov

E betyr /home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001.
Policy binder eksakte SHA, originale kvitteringer og egne snapshots.

- V3_OBSERVED_TRAIN_CONTINUATION_DESIGN_20261128_001.json: neste stengte216steg,
  ekte FINAL432→låst648, alle5TF og eksplisitt numerisk fasealias.
- V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_ACTUAL_REVIEW_20261128_001.json:
  ny faktisk COMPLETE/all6/all10/ytreexit0, originalfeil fortsatt bevart.
- V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_SOURCE_QUALIFICATION_20261128_002.json:
  66fokuserte kontroller,83eiere, fersk inputidentitet og uendret numerisk funksjon.
- V3_ACTUAL_STEP_OBSERVER_ACTUAL_REVIEW_20261128_001.json: faktiske2+1roller,
  all10paritet og alle5TF-observasjoner under original COLD-fail.
- V3_ACTUAL_STEP_OBSERVER_POST_FAILURE_HOST_OBSERVATION_20261128_001.json og
  COLD_RECOVERY_DESIGN_20261128_001.json: senere boot/hostfakta, neste stengte omfang.
- V3_ACTUAL_STEP_OBSERVER_SOURCE_QUALIFICATION_20261128_002.json og
  QUALIFICATION_PROTOCOL_20261128_001.json:77tester/kildebroer/ferskinputkontrakt.
- V3_ACTUAL_STEP_OBSERVER_SOURCE/REFERENCE/FINAL_BACKUP_20261128_001.json:
  fysiske separate Mac-backuper, inkludert alle faktiske tekniske checkpoints.
- V3_ACTUAL_STEP_OBSERVER_DESIGN_20261128_001.json: observeromfang; ingen fitadmisjon.
- V3_FROZEN_BATCH_RESIDUAL_REVIEW_20261128_002.json: ny faktisk måling, alle
  fem TF/to pass, ærlige grenser og nøyaktige input-/output-/terminalbindinger.
- V3_FROZEN_BATCH_RESIDUAL_PLAN/DESIGN_20261128_002.json: eget clock-only-scope;
  001 og V3_FROZEN_BATCH_RESIDUAL_ACCEPTANCE_20261128_001.json er forbrukte feil.
- V3_FROZEN_BATCH_RESIDUAL_FINAL_BACKUP_20261128_001.json og manifest: begge
  CPU-roller, kilder, tester, kvitteringer og aggregerte resultater på separat disk.
- V3_TRAIN_EVAL_GAP_MEASUREMENT_REVIEW_20261128_001.json og SERIALIZATION_FIX:
  fullført original405forwardmåling og separat CPU-sammendrag, originalNNfeilbevart.
- V3_DIRECTION_ISSUE_MIX_TRAIN_REVIEW_20261128_001.json og
  V3_ISSUE_MIX_PASS_DIAGNOSTIC_20261128_001.json: låste kandidater/diagnostikk.
- V3_ISSUE_MIX_READOUT_ATTRIBUTION_20261128_001.json: autentisert ny arm mot
  tidligere JSON-referanser; eldre checkpoints ble ikke lastet om.
- V3_DIRECTION_ISSUE_MIX_REFERENCE/COLD/FIT_ACCEPTANCE_20261128_001.json og
  SOURCE/REFERENCE/QUALIFIED/FINAL_BACKUP: tidligere fullførte rollekvitteringer.

Aktuelle nye roots:
/home/andre2/GX1_RUNS/NATIVE_V3_FROZEN_BATCH_RESIDUAL_AUDIT_20261128_001 (FAILED)
og samme prefiks med002 (COMPLETE). Original issue-miks FINAL ligger i
DIRECTION_ISSUE_MIX_20261128_001_fit-rooten bundet i plan/policy.
Originalkvitteringer overstyrer prosa.
