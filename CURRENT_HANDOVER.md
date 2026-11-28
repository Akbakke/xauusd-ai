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

NEXT_RUN_POLICY.json er eneste arbeidsstatus. Gjeldende fullførte felt
native_v3_frozen_final864_mode_gap_20261128_001 er faktisk COMPLETE/stengt/
forbrukt0/relaunch=false. Indre/ytre exit0; prosess borte. Alle seks globale
flagg var false ved admission/CLAIM/hele kjøringen og er fortsatt false.
Edge-målet er aktivt/uoppfylt. Ingen ny fit eller predictor er laget; ingen
senere VAL/Control/TEST/ordre/paper/live er kjørt.

Fryst kilde8d0b80e2d, plan-SHA
0739ca250b1629a5dac9400aff0a4ce1ee16626a23e1a6ccd4863e1ebd9cd388.
Gjenopptok bare ekte FINAL864 (produsent002/be53..., egen864-trajektorie/
full10/modell/AdamW/fullRNG) og gjenbrukte uendret train_eval_gap_passes/
saved_outputs/score/summary. Hele20727 TRAIN-rader/1377kildedøgn2021–25,
alle5TF/8familier+D1makro/None-policy og alle masker/priors/norm bevart.

Faktisk370.78s:405forwards=1EVAL+4sekvensielleTRAIN/no_grad à81B256-batcher.
0optimizersteps/backward/autograd/observer/checkpointwrites/prior-/norm-fits.
1model/optimizer/cache/target/checkpointload/restore,5logitwrites/readbacks og
1lagretEVAL-logitload. EVALs komplette F32-outputs OG numeriske review er eksakt
lik opprinnelig lagret FINAL864. EVALRNG var uendret; deretter4forhåndslåste
sekvensielleRNG-avanser uten reset mellom draws. Ingen beste draw/TF/batch valgt.
Alle8ikke-modus/ikke-RNG semantiske bevis, tomme gradienter og modulregister
var uendret før cleanup etter hver passering. Bare named-modi/fullRNG ble
restaurert ETTERalle4draws; alle10sluttbevis er eksakt ekte FINAL864. Ingen
vekter/buffere/optimizer ble lastet om for å skjule mutasjon. Original CP urørt.

Actualreview-SHA
a367dec9c6c043ab4404f46e63af4aa4214f557648ad0be194d4c2922ba8872a.
Alle741sammendragsfelt er kontrollert mot4scoreJSONer og gjennomsnitt/SD/SE;
0UNKNOWNsammendragsmetrikker, men målbar-AUC er fortsatt NOT_IMPLEMENTED.
Støtte/fasit/utvalg/priors/spec/UNKNOWN-masker er identiske, aldri nye labels.

| TF | Middelfeil pp, EVAL → TRAIN-snitt, rad | EVAL → TRAIN-snitt, målbar | TRAIN-snitt minus EVAL NLL, rad | Brier, rad |
|---|---:|---:|---:|---:|
| M5 |-3.04744 →-2.99825|-3.04744 →-2.99825|-0.000027180|-0.000013499|
| M15 |-0.61993 →-0.57144|-0.49831 →-0.44870|+0.000073154|+0.000036425|
| H1 |-1.99734 →-1.96334|-1.95010 →-1.91587|+0.000059102|+0.000029227|
| H4 |+0.48946 →+0.48293|+0.87750 →+0.87410|+0.000126340|+0.000061657|
| D1 |+4.37500 →+4.65885|+4.44462 →+4.73175|+0.006046328|+0.002726255|

M5s samlede modusforskjell flytter pUP bare+0.04920pp (drawspenn+0.04562 til
+0.05181pp); omtrent-2.99825pp gjenstår. D1s middelfeil og proper-score blir
verre i TRAIN. Alle5TFs gjennomsnittlige rad-AUC faller. Aktivitets-NLL/Brier
under begge vektinger og aktivitets-AUC forverres i ALLE4draws på ALLE5TF.
Alle4draws beholder pooled M5properFAIL/øvrige4PASS, samletproper/stabilitetFAIL.
EVALår22/25PASS under begge vektinger; draw01 har21/25 (M5s2025flipperFAIL),
øvrige3draws22/25. PersistenteFAIL er M5s2021/2023/2024. M5 underpredikerer
UP i alle5år også i TRAIN. Komplette draw/TF/år/vekting-resultater er bevart.

Dette er beskrivende B256/no_grad TOTAL modusforskjell, inkl. dropout og
EVAL-only TFmerge. Målingen isolerer ingen komponent eller historisk B96-
optimaliseringsårsak. Ingen score av gjennomsnittsprediksjon, ensemble,
konfidens/signifikans/presisjon eller ny kandidat. Frossen EVAL864s TRAINKVAL
forblir FAIL og senere generalisering/økonomisk edge er umålt. Assessment-SHA
77e69b366ec5fab0381fef407b4510382e2cdaabfb3de051c2d8568ec626b2a7:
modusmålingen fjerner ikke M5feilen og begrunner ingen dropout-/bias-/tapspatch.

Ytre signerte vakt:240målinger/child_status0, ingen stop/fatal/maintenance;
peakGPU60C/minne70C/189.33W/1978MiB, CPU41C/30.49W. Hostledig minst54.22GB,
commitmargin56.87GB; fysisk GPUgrense200W i alle målinger. Boot4/sammeWSL som
CLAIM og ingen spørrefeil; PID1425432/start49281816 er faktisk borte. Alle6
bookends og alle10tilstandsbevis PASS. Ingen vakt/kappe/timeout er endret.

Før NN:77fokuserte kontroller PASS (14adapter/63guard),92eiere/fersk modell- og
inputidentitet med0metadata-modell/optimizer/PT/NPZ/CUDA-forsøk; faktisk eksakt
CLI-binding og kanonisk hardware-preflight bestod. SOURCEbackup127filer er
fysisk fsync/F_FULLFSYNC/allemedlemmer-verifisert på separat Macdisk. Ekte864-
backup er rehashet/identisk og alle9foreldrebindinger validert uten ny vektkopi.
FINAL-backup er fysisk fsync/F_FULLFSYNC/allemedlemmer-verifisert på Mac:
161filer/35724222bytes, arkiv-SHA
53dbe83d451218956403febb60e49f3fd4bd913f8738e581a116482fb84e8811.
Dette dekker kilder/alle5score-/logitsett/råbevis/neste003design. Kvittering og
manifest ligger i policy; avsluttende overlevering/binding bevares i Git.
EkteFINAL864-vekter gjenbrukes fra verifisert eldre backup, uten ny CPwrite eller
vektkopi. Alle gamle backuper/checkpoints/feil beholdes.

Neste felt native_v3_observed_train_continuation_20261128_003 er kun
DESIGNED_SOURCE_PREPARATION_ONLY_CLOSED, execution=false/training=false/
remaining0/relaunch=false. Design-SHA
6d68837c399baa06732f281cd7b7ad20aaed069cafa842cf1413538eb74dd70b.
Én separat forhåndslåst uendret fortsettelse foreslås: ekte FINAL864→FINAL1080,
216originaltakes216..431 én gang/hele20727TRAIN/alle5TF/8familier+makro/None,
297forwards=216TRAIN+81fullFINAL-EVAL,216eksterne observasjoner, mekaniskC865/
kunFINAL1080kandidat. Bruk egen864lineage/full10/RNG/AdamW, behold864prefix,
lag ny1080trajektorie. Gjenbruk BASELINE864/factoryINITIAL/COHORTprior og samme
PASS2-numeriskeier under eksplisitt faktisk BASELINE864→FINAL1080-alias.
Rapportér alleTF/år/vektinger/proper/rank/kalibrering/støtte/stabilitet, også
forverring/UNKNOWN. Ingen baselineNN/inputbuild/ny prior/norm/ekstraautograd.

Eksisterende adaptere hardkoder forbrukte endepunkter. Ny minimal eier
native_future_state_fixed_continuation_v1 skal binde kun lineage/cursor/fase/
output til autentisert enkelt-runID/design/policy/plan, slik at samme faste
216steg kan gjenbrukes uten ny adapterkopi per passering. Kanonisk config/root/
stage må avledes og matche eksakt runID. Ingen vilkårlige stier, epokevalg,
adaptivsløyfe eller automatisk nestejobb. Bare gjeldende003scope er definert;
hver kjøring trenger sin egen forhåndsdeklarasjon/kvalifisering/plan/backup/
ferske snapshot-først engangsadmisjon. Gamle adaptere og alle numeriske eiere
skal forbli byte-uendret. Dette er bare kildeklargjøring, ingen ny fitadmisjon.
Samme20G/512M1770/1800/8tråder/CPU0..18/pids128/12GiBVRAM og alle vakter.
Ingen TEST/laterVAL/Control/M1/GC/Entry/Exit/broker/ordre/spending/promotion.

### Bevarte forutsetninger

FactoryINITIAL/hele-TRAIN-prior og COHORT-scoring-prior er ulike frosne eiere og
bevares; ingen nye fits/baselineforwards. Original issue-miks432 ligger i sin
gamle fit-root; egne648-/864lineager/kvitteringer må aldri erstattes med432.
Tekniske C433/C434/C649 er aldri kandidat eller initializer. Original COLD er
FAILED_CONSUMED/0; egen observer-recovery er COMPLETE/0. Alle bevis ligger i
policy/originalkvitteringer/Git, uten ny historikkmappe.

Observatørens fire detach-/skalarkopipunkter endrer ingen numerisk operasjon.
Endpoint-/batch-/gradientkontroller har ikke begrunnet bias-, dropout- eller
tapsnormaliseringspatch. Målingen ved432 fant bare+0.037710pp M5-modusgap; den separate fullførte864-
målingen ovenfor fant+0.049198pp. Begge er vektspesifikke totalmodusmålinger. Gjenbruk
beståtte uendrede kontroller, og rapporter ærlige bevisgrenser.

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

- V3_FROZEN_FINAL864_MODE_GAP_ACTUAL_REVIEW/ASSESSMENT_20261128_001.json:
  faktisk405forwards/0updates, all6/all10, alle5TF/år/4draws og ærlige grenser.
- V3_FROZEN_FINAL864_MODE_GAP_SOURCE_QUALIFICATION_20261128_001.json:
  77kontroller/92eiere/nullmetadataNN/egen864tilstand/alle6globale flagg stengt.
- V3_FROZEN_FINAL864_MODE_GAP_SOURCE/FINAL_BACKUP_20261128_001.json:
  separate fysiske Mac-backuper; aktuell status og manifeste i policy.
- V3_OBSERVED_TRAIN_CONTINUATION_DESIGN_20261128_003.json: neste stengte
  kildeomfang, kun ekte864→låst1080 og minimal gjenbrukbar engangseksekvering.
- V3_OBSERVED_TRAIN_CONTINUATION_ACTUAL_REVIEW/ENDPOINT_ASSESSMENT_20261128_002:
  faktisk648→864, all5pooledfremgang men større middelfeil og samletFAIL.
- V3_OBSERVED_TRAIN_CONTINUATION_FINAL_BACKUP_20261128_002.json: verifisert
  ekteFINAL864/kvittering/plan/score/logits; originalvekter gjenbrukes derfra.
- Historiske bevis åpnes bare ved konkret behov via policy/Git. Alle gamle
  feil, checkpoints og backuper bevares; ingen ny historikkmappe eller rydding.

Aktuelle roots:
/home/andre2/GX1_RUNS/NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_002 er COMPLETE.
/home/andre2/GX1_RUNS/NATIVE_V3_FROZEN_FINAL864_MODE_GAP_20261128_001 er COMPLETE.
Foreslått NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_003-root skal ikke finnes
før egen faktisk CLAIM. Originalkvitteringer/prosessobservasjon overstyrer prosa.
