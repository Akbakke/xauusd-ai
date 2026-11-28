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

NEXT_RUN_POLICY.json er eneste arbeidsstatus. Siste faktiske NN er separat004,
CLOSED_CONSUMED_COMPLETE/remaining0/relaunchfalse, execution/trainingfalse;
alle6globale flagg er false og prosess er borte. Edge-målet er aktivt/uoppfylt.
Gjeldende neste felt er native_v3_frozen_864_1080_block_attribution_20261128_001,
bare SOURCE_PREPARATION_ONLY_CLOSED/executionfalse/trainingfalse/remaining0.
Ingen ny NN eller treningsforlengelse er innrømmet.

004 brukte fryst kildeac0796e24, plan-SHA
20ce68f848c0e48a5c81d3e1684ee2ab4dadd7c13c439ef904615c83d5085734.
Faktisk438.91s/indre-ytreexit0:297forwards=216TRAIN+81fullFINAL-EVAL,
216optimizerupdates og216eksterne observasjoner. Alle23tellerverdier eksakt;
originaltakes216..431 ble brukt én gang/hele20727TRAIN/1377kildedøgn2021–25,
alle5TF/8familier+D1makro/None. Ekte864/egen864lineage/full10/RNG/AdamW ble
validert før copy, deretter egen1080trajektorie. MekaniskC865 og bareFINAL1080
ble skrevet/fysisk reåpnet. Alle6bookends/full10start/full10slutt PASS.
OBSERVATIONS har eksakt864..1079; alle1080nativeF32grad-/klippegrenser PASS,
42klippede steg, ingen signflip/ikke-endelig verdi eller målt numerisk feil.
Maksavrundingsavvik2.794e-9, verst1.424% av egenF32grense. Ingen kodepatch.

Faste row-resultater864→1080 (komplette beggevektinger/år/kalibrering i policy):

| TF | Rad-AUC før→etter | ΔNLL | ΔBrier | Brier minus COHORTprior | Middelfeil etter, pp |
|---|---:|---:|---:|---:|---:|
| M5 |0.538210→0.544450|+0.000196017|+0.000097746|+0.000302648|-3.63100|
| M15|0.533312→0.538425|-0.000208778|-0.000103936|-0.000896964|-1.43089|
| H1 |0.549848→0.558974|-0.001118144|-0.000554519|-0.001723437|-2.20068|
| H4 |0.552826→0.569917|-0.003385598|-0.001646370|-0.003826939|+0.31994|
| D1 |0.775208→0.936813|-0.246532900|-0.098276359|-0.149822408|-4.03796|

Alle5pooled og alle25år-radAUC forbedres, men M5properFAIL under beggevektinger;
samletproper/stabilitet/fiksertTRAIN-sammenligningFAIL. År-PASS faller22→21/25:
M5 bare2022PASS;2021/23/24/25FAIL,2025forverres. Alle4andreTF har5/5PASS, men
M15s NLL/Brier blir verre i2024/25. M5NLL/Brier blir verre i2021/23/24/25.
M5s pUP faller0.47835671→0.47252115 mot observert0.50883115. Større meanbias²
(+0.000389724799) overstiger bedre Var(p)-2Cov(p,y)(-0.000291978507), slik at
ΔBrier=+0.000097746292. Dette er beskrivende eksakt momentidentitet fra lagrede
skalartall, ikke full kalibreringsdekomponering/årsaksbevis eller ny modell.

M5s observerte additive headbias øker+0.000108655542. Med alle andre ledd frosset
kan denne isolert bare øke sigmoidp med høyst0.00271639pp (reell aritmetikk,
ikke målt kontrafaktisk/F32feilgrense). Hele modellens pUP faller0.583556pp;
readoutvekter og backboneegenskaper gjenstår som mulige bidrag, ikke bevist feil.
COHORTprior/factoryINITIAL/actual864 er ulike referanser. Numeriskeierens
learning_qualified=false er hardkodet; faktiskeM5prior-/properFAIL er separat.
D1s13968rader har875unike mål og er avhengige. Dette er gjenbruktTRAIN, ingen
senere generalisering, enkeltfamilie-nytte, handelsbeslutning eller økonomisk edge.

283signerte vaktmålinger, exit0/ingenstop/fatal/maintenance; peakGPU61C/minne68C/
188.27W/9772MiB, CPU44C/34.389W. Hostledig minst54.355GB/commitmargin49.049GB;
fysisk200W og alle20G/512M/CPU-/VRAM-/tidsvakter uendret. Boot4/sammeWSL,
ingen spørrefeil; PID1451748/start49633928 er borte.003 forblir FAILED_CONSUMED0:
MoUso/TiWorker-vedlikeholdsavbrudd er bevart; mekanisk865 ble aldri initializer.

004kvalifisering gjenbrukte68beståtte16+52XML/log-bevis ved eksaktidentiske95kilder;
fersk modell/input/egen864/schedule og0metadataNN/PT/NPZ/CUDA-forsøk PASS.
Faktisk eksaktCLI/hardware-preflightPASS; SOURCEbackup153filer er fysisk
F_FULLFSYNC/allemedlemmer-verifisert på separatMacdisk. Begge eldre864/avbrutt003-
arkiver er rehashet/identiske. Én variabelkollisjon i midlertidig pakkehjelper
ble målt/rettet førNN; feilarkivet/loggen beholdes, korrektSOURCE er verifisert.
Ingen numerisk/prosjektkilde endret. FINALbackup186filer/182726262arkivbytes er
fysisk F_FULLFSYNC/allemedlemmer-verifisert på separatMacdisk, inkl. begge nye
checkpoints/fullbevis/alleoutputs/neste stengte design. Arkiv-SHA
5a385f016a97eed0bfa7a42603ca2a1b1cf2d58d642b36dc8457567410236e0d.
Kvittering/manifest er i004 og neste policyfelt. Overlevering/sluttbinding i Git.

Neste stengte design spør om M5s målte mean-skift avhenger av backbone,
readoutvekter eller samspillet: SELF864,SELF1080 og begge krysskombinasjoner,
4låsteEVAL/no_grad×81=324forwards/0updates. Begge SELF må først matche lagrede
F32outputs OG numeriske reviews eksakt. Behold alle input/TF/familier/masker/
priors, fullstendig eksplisitt state-partisjon og mutation-/full10-vakter.
Rapportér begge betingede blokkeffekter og2x2interaksjon på alleTF/år/vektinger;
kryss er diagnostiske blokk-kombinasjoner, ikke nye kandidater eller grunnlag
for patch/feature-årsak/optimizerhistorie. Latente baser kan være samtilpasset.
Bare kildeklargjøring er autorisert etter verifisert1080backup: gjenbruk eksisterende
EVAL/scoring/state/attribution-eiere, minste nye binding og fokuserte kontroller.
Egen kilde/plan/host/preflight/SOURCEbackup/snapshot-først engangsadmisjon kreves;
alle6globale treningsflagg skal være false også i eventuellfrossendiagnostikk.
Ingen mer trening/laterVAL/Control/TEST/M1/GC/Entry/Exit/broker/ordre/promotion.

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
- V3_FROZEN_864_1080_BLOCK_ATTRIBUTION_DESIGN_20261128_001.json:
  bare neste stengte kildeomfang; firelåsteEVAL-armer/324fw/0updates.

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
Foreslått frozen864/1080block-root finnes ikke; ingen CLAIM/admisjon.
Originalkvitteringer/prosessobservasjon overstyrer prosa.
