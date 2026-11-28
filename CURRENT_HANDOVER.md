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

NEXT_RUN_POLICY.json er eneste arbeidsstatus. Siste faktiske jobb er frozen864/
1080block001, CLOSED_CONSUMED_COMPLETE/remaining0/relaunchfalse, execution/
trainingfalse. Alle6globale flagg varfalse ved admission/CLAIM/lukking.
Edge-målet er aktivt/uoppfylt; ingen ny NN, trening eller patch er innrømmet.
Neste felt native_v3_m5_shared_objective_gradient_design_20261128_001 er bare
DESIGN_PREPARATION_ONLY_CLOSED/executionfalse/trainingfalse/remaining0.

Bruker presiserte at M5 skal bruke øvrigeTF som hjelp. Verifisert faktiskforward:
familie-/TF-attention, kryss-TF-blanding og felles/global D1makro-kontekst inngår
FØR M5head. Det er én felles MTFmodell med egne outputheads, ikke uavhengige
TFmodeller. Dagens native tap summerer kjent aktivitet+sertifisert side på alle
5TF og deler på samlet kjent aktivitetsantall; M5 har ingen særskilt prioritet.
CurrentNone betyr felles backbonegradienter. Kobling er ikke dokumentertM5nytte;
andreTFs resultatforbedring skal aldri brukes som bevis for M5edge.

Frossen2x2 brukte kilde66793b162, plan-SHA
177189f40629861e4555981a77ff42809564e44fba4aab913d6cb4779fe0a8f0.
Faktisk408.28s/indre-ytreexit0:4EVAL/no_grad×81=324forwards,0updates/backward/
autograd/checkpointwrites/fits. Alle26tellere eksakt:2CPloads+full10prechecks
FØRcopy,3strikte restoreringer,2kompositter,4logitwrites/readbacks,2savedloads;
1model/optimizer/cache/target. Hele20727TRAIN/1377døgn2021–25, alle5TF/
8familier+D1makro/None bevart. SELF864/SELF1080 kompletteF32archiveSHA, råreview
OG81batchhashes er eksakt lagredeoriginaler FØRbeggekryss. Ingen kandidat valgt.

Statepartisjon er794disjunkte persistente keys:769backbone/24readout/1identisk
future_state_model_sha256.60nonpersistentbackbonebuffere er separat bundet til
konstruktørhash. Alle4før/etter-proof==forventetfull10;625navngitte modi,
optimizer/fullRNG/gradNone/nonpersistenthash uendret før nestecopy. Kryss endrer
bare model_state; øvrige9bevis er ekte1080. STATEPASS og final_restore_all_checks_pass
true; sistefull10 er eksakt ekte1080. Ingen restore skjuler feilet passering.
UavhengigJSONkontroll gjenoppbygger ikkeCP/kompositt-tensorer; siste modus-/NP-/
gradtest er kollektiv boolsk attest, mens perpass-snapshots/full10 er eksplisitte.

Kontrollerte row-ΔBrier (før→etter:backbone864→1080 med R864/R1080;
readout864→1080 med B864/B1080). Komplette beggevektinger/år/aktiviteter i policy.

| TF | Backbone, R864 / R1080 | Readout, B864 / B1080 | Interaksjon |
|---|---:|---:|---:|
| M5 |+0.000190703 / +0.000253402|-0.000155656 / -0.000092957|+0.000062699|
| M15|-0.000087314 / -0.000021320|-0.000082615 / -0.000016622|+0.000065994|
| H1 |-0.000186463 / -0.000250927|-0.000303592 / -0.000368056|-0.000064464|
| H4 |-0.001316397 / -0.001686410|+0.000040040 / -0.000329974|-0.000370014|
| D1 |-0.086266362 / -0.099431535|+0.001155176 / -0.012009996|-0.013165172|

M5backboneeffekt senker pUP-mean0.467560pp/0.632428pp ved beggefrosnereadouts,
og økerNLL0.000383089/0.000508086. Readouteffekt forbedrerM5NLL/Brier ved begge
backbones, men pUP-mean skifter fortegn:+0.048872pp/-0.115996pp. Interaksjon
pUP=-0.164868pp/NLL=+0.000124997/Brier=+0.000062699/AUC=+0.001872201.
M5pUP faller i ALLE5år ved beggebackbonekontraster. M5proper blir verre2021–24
medR864 ogalle5år medR1080. Readout bedreralleM5år medB864, men forverrer24/25
medB1080. Alle4armer harpooledM5FAIL/andre4TFPASS. Andre4TF haralleårPASS,
selv om betingede proper-regresjoner finnes. M5s2021/23/24FAIL består;2025FAIL
finnesbareSELF1080. M5aktivitet bedres ved beggebackboneskift (ΔBrier
-0.000694889/-0.000916743), samtidig med verre M5side. Dette avgrenser neste
spørsmål om dagens felles læringsmål; det beviser ingen optimizerhistorisk årsak.

Alle2400betingede skalardeltaer/600interaksjonsblader er kontrollert;540kjente/
60UNKNOWN interaksjoner. De60 er manglende redundant equal-target modelmean-
error; separat sidekalibreringsmean er kjent. NLL/Brier er kjente; målbarAUC er
NOT_IMPLEMENTED. Truth/support/utvalg/priors/spec identiske. Kryss måler
funksjonsavhengighet av samtilpassede blokker, ikke enkeltindikatornytte,
optimalvekter eller ny prediktor. EkteFINAL1080 forblir TRAINFAIL; ingen senere
generalisering eller økonomisk edge er målt. Originale masker/UNKNOWN beholdes.

Alle6bookends PASS;265signerte vaktmålinger/ingenstop/fatal/maintenance;
peakGPU60C/minne66C/182.24W/2012MiB, CPU43C/34.692W. Hostledig minst54.259GB/
commitmargin56.906GB/fysisk200W, alle20G/512M/CPU/VRAM/tidsvakter uendret.
Boot4/sammeWSL/ingenqueryfeil; PID1471978/start49897120 er borte.
98eiere/106fokuserte tester(23+83) og obligatoriskEntry-kontrakt PASS. Gamle95
ikke-guard-kilder er byte-uendret; guard/shell-tillegg reverseres eksakt til0c06.
Freshmetadata/egne2lineager/identitet/0modeller/optimizer/PT/NPZ/CUDAinit PASS,
inkl.prosjektimport. Begge9foreldrebindinger dekket av fysisk rehashedeidentiske
Macarkiver. Første isolerte hostrequest traff2s førCLAIM; ny samme2småling og
faktiskCLI/hardware-preflight bestod uten vaktendring. SOURCEbackup149filer er
fysiskF_FULLFSYNC/allemedlemmer-verifisert på separatMac. FINALbackup187filer/
10685300arkivbytes er fysisk F_FULLFSYNC/allemedlemmer-verifisert, inkl. alle4
logit-/score-sett, tilstandsbevis, metodeforberedelse og kilder. Arkiv-SHA
6f590bc594b3b9dcc965fcf6f80763d75da09c52873c6bd67ffce1a41312e8d9.
Kvittering/manifest i dette og neste policyfelt; sluttbinding/overlevering i Git.
Ingen nyevekter bleprodusert; begge originaleCPbackuper gjenbrukes. Alle gamle
feil/CP/backuper bevares.

Neste: bare metode-/designklargjøring for current1080 frosne gradienter. Gjenbruk
native_provider_hlc_learning_v1.direction_gradient_* og kvalifiserte timing- /
støttekvitteringer. Skill M5side/M5aktivitet/8øvrigeTFtasks og mål ALL-TENjoint
mot M5side, ikke bareparvise summer. Lås eksakte paneler/dekning/mode/RNG/
forward-autogradbudsjett og minne/runtime FØR ny NN. Eksisterendeprobe er lokal
EVAL-gradient; det er ikke automatisk faktiskeTRAIN/dropout/AdamWoppdateringer.
Behold alleinput/familier/TF/None, alle statevakter/UNKNOWN. Ingen kortprobe blir
en generelldom. Ingen loss-/bias-/backbonepatch, nyfit eller relansering av gammel
gradientplan er autorisert. Egenkvalifisering/plan/host/preflight/SOURCEbackup/
snapshotførst-admisjon kreves; alle6globale flagg skalfalse ved eventuellprobe.
Endretadoptertfunksjon kreveregenfreshINITIAL og lærings-/generaliseringstest.

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

- V3_FIXED_CONTINUATION_ACTUAL_REVIEW/ENDPOINT_ASSESSMENT_20261128_004.json:
  faktisk864→1080/all23tellere/all6/all10, alleTF/år/vektinger og samletFAIL.
- V3_FIXED_CONTINUATION_SAVED_SCORE_MOMENTS_20261128_004.json:
  beskrivende all5TF row-Brieridentitet; ingen ekstraNN/fit eller kausalpåstand.
- V3_FIXED_CONTINUATION_SOURCE/FINAL_BACKUP_20261128_004.json:
 153/186fysiskverifiserte medlemmer; ekte1080 råvektbackup er utenforGit.
- V3_FROZEN_BLOCK_ATTRIBUTION_ACTUAL_REVIEW/ASSESSMENT_20261128_001.json:
  faktisk324fw/0updates/beggeSELFpariteter/all26/all6/all10 og allebetingedeeffekter.
- V3_FROZEN_BLOCK_ATTRIBUTION_SOURCE/FINAL_BACKUP_20261128_001.json:
  149/187fysiskverifiserte filer, alle4outputs/tilstand/metodeforberedelse.
- V3_M5_SHARED_OBJECTIVE_GRADIENT_DESIGN_PREPARATION_20261128_001.json:
  bare neste metode-/designklargjøring; ingen nyNN/gradientobservasjon innrømmet.
- V3_FROZEN_864_1080_BLOCK_ATTRIBUTION_DESIGN_20261128_001.json:
  nå COMPLETE/consumed0; original forhåndsdeklarert4armdesign beholdes.

- V3_FROZEN_FINAL864_MODE_GAP_ACTUAL_REVIEW/ASSESSMENT_20261128_001.json:
  faktisk405forwards/0updates, all6/all10, alle5TF/år/4draws og ærlige grenser.
- V3_FROZEN_FINAL864_MODE_GAP_SOURCE_QUALIFICATION_20261128_001.json:
  77kontroller/92eiere/nullmetadataNN/egen864tilstand/alle6globale flagg stengt.
- V3_FROZEN_FINAL864_MODE_GAP_SOURCE/FINAL_BACKUP_20261128_001.json:
  separate fysiske Mac-backuper; aktuell status og manifeste i policy.
- V3_OBSERVED_TRAIN_CONTINUATION_DESIGN_20261128_003.json: forbrukt avbrutt
  ekte864→låst1080-scope; ingen relansering.
- V3_FIXED_CONTINUATION_FAILURE_ASSESSMENT/HOST_MAINTENANCE_20261128_003.json:
  faktisk avbrudd/eksakt vedlikeholdsårsak/bevarte artefakter/UNKNOWN-grenser.
- V3_FIXED_CONTINUATION_SOURCE_QUALIFICATION_20261128_003.json:
  68kontroller/95eiere/nullmetadataNN/egen864lineage/eksakt1080-schedule.
- V3_OBSERVED_TRAIN_CONTINUATION_ACTUAL_REVIEW/ENDPOINT_ASSESSMENT_20261128_002:
  faktisk648→864, all5pooledfremgang men større middelfeil og samletFAIL.
- V3_OBSERVED_TRAIN_CONTINUATION_FINAL_BACKUP_20261128_002.json: verifisert
  ekteFINAL864/kvittering/plan/score/logits; originalvekter gjenbrukes derfra.
- Historiske bevis åpnes bare ved konkret behov via policy/Git. Alle gamle
  feil, checkpoints og backuper bevares; ingen ny historikkmappe eller rydding.

Aktuelle roots:
/home/andre2/GX1_RUNS/NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_002 er COMPLETE.
/home/andre2/GX1_RUNS/NATIVE_V3_FROZEN_FINAL864_MODE_GAP_20261128_001 er COMPLETE.
/home/andre2/GX1_RUNS/NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_003 er FAILED_CONSUMED.
/home/andre2/GX1_RUNS/NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_004 er COMPLETE.
/home/andre2/GX1_RUNS/NATIVE_V3_FROZEN_864_1080_BLOCK_ATTRIBUTION_20261128_001 er COMPLETE.
Neste gradientdesign har ingen NN-root/CLAIM/admisjon.
Originalkvitteringer/prosessobservasjon overstyrer prosa.
