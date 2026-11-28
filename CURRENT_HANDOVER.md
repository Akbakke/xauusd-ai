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

NEXT_RUN_POLICY.json er eneste arbeidsstatus. Felt002
native_v3_observed_train_continuation_20261128_002 er faktisk COMPLETE,
stengt/forbrukt0/relaunch=false. Prosessen er borte; indre/ytre exit0 og alle
seks bookends PASS. Alle seks globale flagg er false. Edge-målet er aktivt og
uoppfylt. Ingen senere VAL/Control/TEST eller ordre/paper/live er kjørt.

Fryst kilde543e0ae33, plan-SHA
be53cb01fce7c80f8ba120c88de2f36d299112f20a573e85b85cda82d8db6875.
Ekte FINAL648 ble validert med sin egen648-trajektorie/produsent/full10-bevis
før modell/AdamW/fullRNG-copy. Deretter nøyaktig216 originale takes216..431
mot låst FINAL864: hele20727 TRAIN-rader/1377kildedøgn2021–25, alle5TF,
åtte genuine familier+D1/makro, None-policy. Ingen funksjons-/input-/target-/
prior-/norm-/hyperparameterendring. Gamle001-adapter/tester og alle matematiske
og hardware-eiere er byte-uendret; foreldreplaner er hashreferanser i ny plan.

Faktisk437,24s:297forwards=216TRAIN+81fullFINAL-EVAL,216backward/updates/
observasjoner,1checkpointload/restore,2atomiske checkpointwrites/readbacks,
1logitarkiv.0autograddiags/skips/prior-/norm-fits. C649 er mekanisk; bare
FINAL864 er kandidat. Ny receipt-SHA
06b48e4fa08306f1979d201c334573adaf80f9f6d9f4b05dfd220c15f69a605d;
checkpoint-SHA
26bbcc969348592b5cdea4ee85fe5629d9ed6250bd1aa9415e570fdc46250368.
Egen864-trajektorie og alle10-bevis ligger i FINAL864.json/RESULT/policy.
Faktisk fase er FINAL864; uendret numerisk skåreier bruker eksplisitt PASS2-
alias/gammel spec i eget FINAL864-navnerom. Gamle kvitteringer er urørt.

Actualreview-SHA
a9e04e62111ace56874b586791413d0c85943eb9b57197786cd3e475a0844790.
Alle fem fikk bedre samlet NLL/Brier under rad- og målbarvekting og bedre
rad-AUC mot648. Samtidig forverres absolutt UP-middelfeil på ALLE5 under begge
vektinger. Sluttresultatet er derfor fortsatt samlet proper/stabilitet/
TRAINKVAL FAIL; M5 feiler begge låste priorreferanser. Ingen generalisering
eller økonomisk edge er dokumentert.

| TF | AUC648 →864 | Brier minus COHORT-prior, rad | Middelfeil648 →864, pp | År-PASS rad/målbar |
|---|---:|---:|---:|---:|
| M5 |0.528606 →0.538210|+0.000204902|-2.70565 →-3.04744|2/5 ·2/5|
| M15 |0.527050 →0.533312|-0.000793028|-0.32204 →-0.61993|5/5 ·5/5|
| H1 |0.541390 →0.549848|-0.001168918|-1.69425 →-1.99734|5/5 ·5/5|
| H4 |0.539902 →0.552826|-0.002180568|-0.14811 →+0.48946|5/5 ·5/5|
| D1 |0.673314 →0.775208|-0.051546050|+0.23248 →+4.37500|5/5 ·5/5|

År-PASS22/25 under begge vektinger, mot20 ved648. Rest-FAIL er bare M5
2021/2023/2024. M5s2023NLL/Brier forverres svakt (+0.0000028357/
+0.0000011552); øvrige år/TF-proper forbedres. H1sAUC faller i2021/2025
(-0.00029945/-0.00144630); øvrige23AUC-celler forbedres. Ingen UNKNOWN blant
primær-/årverdicts. Støtte/fasit/utvalg/spec/priors er identiske, ingen kjente
aktivitets-/sideceller utenfor oppgitte år. D1s13968rader er875målbarer;
målbar-AUC er fortsatt NOT_IMPLEMENTED. Dette er gjenbrukt TRAIN, aldri OOF.

Alle1080nativeF32gradientkontroller PASS, maxfeil2.98e-9 og høyest1.385% av
F32-grensen. Klipping12/216, ingen individuell gradientsignflip. M5bias flytter
-0.000118230, mens frossen EVALpUP48.17746→47.83567%, fasit50.88311%.
Utviklende TRAIN-batchsnitt50.90531% er IKKE måling av frosset864-modusgap.
D1s summerte biasgradient skifter fortegn ved positiv, stegvist forskjellig
klippskalering; ingen individuell signflip/AdamW-feil er målt. Ingen bias-,
dropout-, taps- eller modellpatch er begrunnet. Endpointassessment-SHA
1a18bbd1ff7021e57b2569ca525b75f25679d444033f6e519fce0b2151a9fa51.

Ytre signerte vakt:281målinger/child_status0, ingen stop/fatal/maintenance.
PeakGPU61C/minne68C/186.12W/9772MiB, CPU43C/32.47W; hostledig minst54.52GB,
commitmargin49.07GB. Alle fysiske GPU-effektgrenser200W. Boot4/sammeWSL-ID som
CLAIM og ingen spørrefeil; PID1406995/start49081960 er borte. Vaktene er uendret.

Før NN:64nye kontroller PASS (15adapter/49guard),89eiere/fersk modell- og
inputidentitet med0metadata-modell/optimizer/PT/NPZ/CUDA-forsøk. Egne648-
linje-/864planfelt og faktisk CLI-binding verifisert. Første separate signerte
hostforespørsel traff2s-grensen før trainerpreflight/CLAIM; egen ny forespørsel
svarte normalt, kanonisk første trainerpreflight bestod. Ingen timeout/vakt ble
endret. SOURCEbackup154filer er fysisk fsync/F_FULLFSYNC/allemedlemmer-verifisert
på separat Macdisk. Ekte648-backup er rehashet/identisk og alle9foreldrebindinger
verifisert, uten ny vektkopi. FINAL864-backup er fysisk fsync/F_FULLFSYNC/allemedlemmer-verifisert på Mac:
187filer/238265329bytes, arkiv-SHA
84ad81944413645e405fdb55b70c04a51d7cd7bbcabd6cb597baabf710d1a4bf.
Manifest/kvittering er bundet i policy; avsluttende overlevering/binding bevares i
Git. Originale432-/648-backuper gjenbrukes.

Neste felt native_v3_frozen_final864_mode_gap_20261128_001 er kun
DESIGNED_SOURCE_PREPARATION_ONLY_CLOSED, execution=false/training=false/
remaining0/relaunch=false. Design-SHA
3b49a1de3bb2bc257e157d9fd8de137b57121d94fe30b76c75edac0b7519c097.
Ny faktisk bekymring er voksende middelfeil ved864; gammel432-modusmåling
avgjør ikke nytt checkpoint. Før mer fit skal samme frosne ekte FINAL864
måles med1full EVAL-paritet mot lagrede864F32-logits +4forhåndslåste sekvensielle
TRAIN/no_grad-draws:405forwards,0updates/backward/autograd/CPwrites, ingen ny
kandidat. Gjenbruk uendrede train_eval_gap_passes/saved_outputs/score/summary.
Ikke kall gamle432-spesifikke final_metadata/current_input_identity/physical;
gjenbruk bare de oppgitte cursor-nøytrale eierne med egen864-metadata. Interne
prior/spec-contract_sha256 bindes til dekodet kontrakt, aldri rawfilens SHA.
Bare opprinnelige named-module-modi/fullRNG kan gjenopprettes etter ALLE4draws;
alle8øvrige semantiske bevis må være uendret før cleanup og alle10 etterpå.
Ingen vekter/buffere/optimizer-reload kan skjule mutasjon. Rapportér alle5TF/
år/vektinger og alle4draws med beskrivende metrikkgjennomsnitt/SD/SE; ingen
beste draw, ensemble, konfidens-/presisjons- eller årsakspåstand. Målingen er
TOTAL modusforskjell, inkl. dropout og EVAL-only TFmerge, ikke isolert årsak.
Alle6globale flagg skal forbli false også ved eventuell0updateNN-admisjon.
Neste: minste864-lineage-/eksekveringsbinding og source/host/runtime/preflight/
SOURCEbackup + fysiskFINAL864backup før separat snapshot-først engangsadmisjon.
Ingen ny fit/automatisk serie/laterVAL/Control/TEST/M1/GC/Entry/Exit/ordre/
spending/promotion. Gamle forbrukte roller, feil og checkpoints bevares.

### Bevarte forutsetninger

FactoryINITIAL/hele-TRAIN-prior og COHORT-scoring-prior er ulike frosne eiere og
bevares; ingen nye fits/baselineforwards. Original issue-miks432 ligger i sin
gamle fit-root; egne648-/864lineager/kvitteringer må aldri erstattes med432.
Tekniske C433/C434/C649 er aldri kandidat eller initializer. Original COLD er
FAILED_CONSUMED/0; egen observer-recovery er COMPLETE/0. Alle bevis ligger i
policy/originalkvitteringer/Git, uten ny historikkmappe.

Observatørens fire detach-/skalarkopipunkter endrer ingen numerisk operasjon.
Endpoint-/batch-/gradientkontroller har ikke begrunnet bias-, dropout- eller
tapsnormaliseringspatch. Målingen ved432 fant bare+0.037710pp M5-modusgap;
dette er vektspesifikt og avgjør ikke foreslått frosset864-måling. Gjenbruk
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

- V3_FROZEN_FINAL864_MODE_GAP_DESIGN_20261128_001.json: gjeldende stengte
  kildeomfang; ekte frosset864,405forwards/0updates, alle5TF/alle4draws.
- V3_OBSERVED_TRAIN_CONTINUATION_DESIGN_20261128_002.json: fullført648→864,
  opprinnelig forhåndslåst216updateomfang og fasealias.
- V3_OBSERVED_TRAIN_CONTINUATION_SOURCE_QUALIFICATION_20261128_002.json:
  64fokuserte kontroller/89eiere/fersk identitet/egen648lineage/nullmetadataNN.
- V3_OBSERVED_TRAIN_CONTINUATION_ACTUAL_REVIEW/ENDPOINT_ASSESSMENT_20261128_002:
  faktisk648→864, all5pooledfremgang men all5større middelfeil og samlet FAIL.
- V3_OBSERVED_TRAIN_CONTINUATION_FINAL_BACKUP_20261128_002.json:
  ekte FINAL864, egen kvittering/plan/score/logits og råbevis; fysisk status i policy.
- Historiske432-modus-/residual-/observer- og inputbevis åpnes bare ved konkret
  behov via policy/Git. Originale feil, checkpoints og fysiske backuper beholdes.

Aktuell produsentroot:
/home/andre2/GX1_RUNS/NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_002 er COMPLETE.
Foreslått NATIVE_V3_FROZEN_FINAL864_MODE_GAP_20261128_001-root skal ikke finnes
før egen faktisk CLAIM. Originalkvitteringer og prosessobservasjon overstyrer
prosa; alle tidligere checkpoints bevares.
