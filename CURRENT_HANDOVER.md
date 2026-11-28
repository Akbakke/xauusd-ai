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

NEXT_RUN_POLICY.json er eneste arbeidsstatus. Aktuelt felt
native_v3_actual_step_observer_20261128_001 har REFERENCE faktisk COMPLETE og
COLD faktisk FAILED_CONSUMED, begge stengt/forbrukt0/relaunch=false. COLDs
oppdatering/observasjon og alle10C2-bevis ble fullført, men boot-bookend feilet
og ytre maskinvarevakt stoppet på host_maintenance_active (exit75). Hele
kvalifiseringen er derfor IKKE PASS. Ingen originalkvittering omskrives.
Alle seks globale trenings-/evalueringsflagg er false, active_lane=null.
Ingen ny NN-/fit-/kandidat-/servingadmisjon. Edge-målet er aktivt/uoppfylt;
låst issue-miks PASS2 feiler fortsatt M5/samlet TRAIN-kvalitet.

Neste felt native_v3_actual_step_observer_cold_recovery_20261128_001 er kun
stengt kildeklargjøring, execution=false/training=false/remaining=0.
V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_DESIGN_20261128_001.json binder minste
separate enstegs gjenopptakelse fra samme originale REFERENCE-C433 mot C434,
med uendret matematikk/inputs/observer, eksakt gammel teknisk trajektorie og
fersk host-/boot-/maintenancefri admission. REFERENCE og beståtte kildebevis
skal gjenbrukes; original COLD-plan relanseres aldri. Ingen hovedfit/ekstraepoch.

### Faktisk observatørkvalifisering og bevarte funn

Fryst kilde gjennom begge roller:011ab4ad26fec29e1c365c6c3f2dbf52158f8b20.
Core endres bare med optionalobserver/fire callbackpunkter. Eksakt reversering
matcher hele gamle core byte-/AST-messig; standardbanen og alle gamle vakter/
kaps er bevart. Observatøren gjør kun detach/skalar-kopier ved eksisterende
forward-loss/backward/clip/step, uten ekstra NN/backward/update/zero_grad,
modus-/RNG-operasjon eller endret taps-/optimizer-matematikk. Rapportene er
utenfor checkpoint steps/scores. Native F32-reduksjonsgrense brukes på faktiske
TRAIN-output, ikke F64-endepunktstoleranse eller lagrede EVAL-sannsynligheter.

77forskjellige fokuserte kontroller består (7leaf/70guard); bare én berørt test
ble kjørt om etter en tegnsettingsadvarsel. Obligatoriske native kontrakttester
bestod ved commit.81kildeeiere/fersk fysisk modellkontrakt og inputidentitet er
faktisk kontrollert; ingen modell/CUDA/cache-/checkpointdekoding i metadata-
kvalifiseringen. Tidlig M1-firewall gjelder ny adapter. Ny434stegs teknisk
trajektorie bevarer nøyaktig original432prefix +takes0/1 på indices432/433;
alle take-/størrelses-/hash-/updatevektorer og oppdateringsantall er434.

REFERENCE lastet autentisk original FINAL432 med fullRNG/AdamW, gjorde2steg
uten observatør og publiserte ekte egne C433/C434.56,87s,2forwards/2updates,
0observerrapporter/ekstra autograd og alle seks bookends PASS. Egen SOURCE- og
REFERENCE-backup er fysisk fsync/F_FULLFSYNC/alle-medlems-readback-verifisert.

Separat COLD lastet bare egen REFERENCE-C433, gjorde ett steg med observatør
og publiserte C434+énrapport; alle10bevis matchet referansens C434 nøyaktig,
inkludert modell/AdamW/fullRNG/steps/scores/trajektorie/cursor. Faktiske budsjetter
matcher én NN/backward/update/checkpointload/restore/write;0ekstra autograd/
scoreforwards/normaliserings-/priorfits. Alle3tekniske vekter er aldri kandidat
eller initializer. All8familier+D1/makro, alle5TF og ekte kausale vinduer beholdes.

COLD beholdes FAILED_CONSUMED:49,75s,5/6bookends PASS, boots=false;
observatør/PÅ-AV-paritet er observert, men samlet operasjonell kvalifisering
feiler. Boot-bookend fanger unntak som false, så den opprinnelige kvitteringen
skiller ikke endret boot fra mislykket identitetsforespørsel. Ytre vakt logget
maintenance_active=false ved start og true07:12:56Z før stop/fatal.37målinger:
peakGPU47C/minne48C/151,2W/9772MiB, CPU39C/31,93W; ingen annen målt grensebrist.
Ingen tidsgrense/predikat/vakt er endret. Begge faktiske prosesser er avsluttet.

Én etterfølgende CPU-observasjon07:16:26UTC viste fysisk BootID4 og samme
WSL-ID som CLAIM, signert maintenance_active=false og begge rebootpendingfalse.
Dette er en senere observasjon, ikke reparasjon/retroaktiv PASS eller ny NN-admisjon.

I det ene bevarte steget er A_b=480 og global gradientnorm0,726546<clip1.
Alle5native F32-biasgradienter matcher analytisk beregning innen deklarert
F32-grense; pre-/postklipp er like. Dette viser ingen klippeeffekt i dette steget.

| TF | Faktisk biasgradient | AdamW-m1 før→etter | Faktisk biasendring |
|---|---:|---:|---:|
| M5 |-0.005005045|+0.001463901→+0.000817007|-0.0000105165|
| M15 |+0.014887785|+0.002275258→+0.003536510|-0.0000443757|
| H1 |-0.004835799|-0.000725070→-0.001136143|+0.0000143498|
| H4 |-0.010804160|+0.000090685→-0.000998799|+0.0000128448|
| D1 |+0.013537055|+0.000203346→+0.001536717|-0.0000186563|

M5s negative någradient reduserte positivt arvet moment uten å snu fortegnet;
biasnedgang er konsistent med AdamW-momentum, ikke målt fortegnsfeil.
Maksimal analytisk/native differanse1,0555e-9 er under M5s F32-grense2,3062e-7.
Dette ene tekniske steget forklarer ikke historiske432steg eller kvalifiserer
læring, prediksjon, senere generalisering eller økonomisk edge. Ingen patch i
modell/tap/optimizer/læringsrate/ruting/epoker følger av det.

SOURCE-backup:122filer/3 372 844bytes; REFERENCE-backup:150filer/180 621 598bytes.
FINAL-backup er fysisk verifisert:176filer/269 547 851bytes, arkiv-SHA
ecd7c8ab393da05eb8503116204d731c154777121d73af026718bdabe09908e8.
Den binder også mislykket COLD med C434/observasjon, originalkvitteringer,
etterfølgende CPU-observasjon og neste stengte recoverydesign. Eksakte manifeste/
kvitteringer ligger i policy; avsluttende kvitteringsbinding/overlevering bevares i Git. Originale FINAL-vekter gjenbruker allerede verifisert
backup. Ingen rådata eller vekter skal i Git. Ingen ny sletteautorisasjon.

### Ny faktisk CPU-måling av batchvekting

Design002 retter bare en målt indeksblokkering i001. Forløperen stoppet etter
én logit- og én målarkivlesing fordi en metadataavledet sammenhengende
radakse ikke matchet originalens SHA. Differansen95 mellom to populasjons-
størrelser var ikke en kilderadoffset. Ingen residualanalyse fra001 godkjennes;
originale helper-/testfiler, CLAIM/RESULT/TERMINAL og feilkvittering er bevart.

002 leste faktisk bar_start_ns fra allerede bundet FEATURE_INPUTS.npz:
1215513int64/9 724 104bytes. Eksisterende provider eier issue_ns=bar_start_ns+
BAR_NS og source_row=arange(len(issue_ns)). Eksakt tidskobling til hele TRAIN
matchet original normaliseringsakse, utvalgte source/positions og alle432
STREAM_BINDING-koblinger til source_rows, issue_ns og maskert fasit.
Ingen gjettet offset eller numerisk featurematrise ble brukt.

Faktisk jobb fullførte på6,44s med alle åtte bookends og tre uavhengig
observerte NPZ-åpninger, én per deklarert arkiv: logits, targets og klokkefelt.
17felt/61 005 110ndarraybytes ble dekodet. Én åpning av featurearkivets klokke
rapporteres eksplisitt;0numeriske featurematrisefelt/fulle providerlastinger.
0modeller/optimizer/checkpoint/autograd/CUDA/NN/updates/nye prediksjoner/priorfits.
Originale mål/masker/sertifikat, COHORT-populasjoner og F32-priorlogit med
Laplace1 beholdes; factoryprioren er en separat whole-TRAIN-referanse.

Faktisk tap er(sum activityBCE+sum kjent sideBCE)/A_b, der A_b er alle fem TFs
kjente aktivitetsceller. Ingen ekstra TF-vekt, aktivitetssannsynlighet eller
1/5-faktor. Rå lokal side-biasderivert er sum(side_mask*(p-y))/A_b før klipping
og AdamW. Alle432takes og fem TF er med,216batcher per pass. Hvert pass har
eksakt103610aktivitetsceller og gjenskaper hele utvalgets residual/derivert
innen forhåndsdeklarert FP64-reduksjonsgrense; alle heltallsstøtter er eksakte.

| TF | Effektiv UP-andel minus radandel, pp pass1 | pp pass2 |
|---|---:|---:|
| M5 |-0.000118834|-0.004852117|
| M15 |-0.000918503|-0.001857099|
| H1 |-0.003318605|+0.005071017|
| H4 |-0.005430882|-0.002752352|
| D1 |-0.002845232|+0.000399319|

M5s rad-UP-andel er50,883115%; frossent sannsynlighetsmiddel48,456008%.
Den maksimale målte endringen i effektiv UP-andel er bare0,004853pp, mot
middelfeil−2,427107pp. Rå hele-utvalgsderivert−0,0029841630993707384
gjenskapes i begge pass. Batchens prediksjonsavvik og fasitsammensetning er
separat dekomponert rundt COHORT-prioren; liten Laplace-rest er ikke kalt bias.
Batch-/nevnervektingen lukker ikke M5-mangelen. Ingen tapsnormaliseringspatch.

Dette er lagrede B256 EVAL-outputs gruppert i de originale B96-takes, ikke
skiftende TRAIN/dropout-gradienter, full Jacobian, klipping eller rekonstruerte
AdamW-momenter. Et endepunkts fortegn er ikke en historisk optimizerforklaring.
Numerisk residual-/dekoder-/tapsformel er AST-identisk med001;22 beståtte
kontroller gjenbrukes. Kun12 berørte klokke-/koblings-/dekoderkontroller ble
kjørt for002, alle PASS. Ingen uendret fullsuite eller fit ble gjentatt.

### Bevarte læringsresultater og øvrige avgrensninger

Siste læringskandidat er låst issue-miks PASS2, fryst kilde7471ddf48:
20727TRAIN-issues/1377kildeklokkedøgn2021–2025, B96/E256, seed20260911 og
2pass à216updates. Alle åtte familier+D1/makro og M5/M15/H1/H4/D1 beholdes
med fullt koblet None-policy. Hvert issue beholder egne genuine kausale vinduer;
input/prior/normalisering/fasit/masker/tap/optimizer og vakter er uendret.
REFERENCE/COLD fullførte2+1tekniske updates/fullproof10, ikke kandidater.
Fersk FIT fullførte432updates/597forwards/812,48s,0checkpointloads/restores/skips
og seks bookends. OriginalINITIAL/factory/full RNG/fasit er gjenbrukt eksakt.

| TF | Låst PASS2 side-AUC | Brier minus COHORT-prior, rad | År-PASS rad/målbar |
|---|---:|---:|---:|
| M5 |0.519403|+0.000398422|1/5 · 1/5|
| M15 |0.518758|-0.000249777|4/5 · 4/5|
| H1 |0.524846|-0.000190210|3/5 · 3/5|
| H4 |0.524267|-0.000416408|5/5 · 4/5|
| D1 |0.614417|-0.008619807|5/5 · 5/5|

Proper-score består4/5TF under begge vektinger; M5 feiler. Samlet proper-score/
tidsblokkstabilitet er FAIL, learning_qualified=false; år18/25rad og17/25målbar.
Aktivitet/rankguard består alle fem, uten kvalifisert læring/generalisering.
D1s13968kjente rader er875forskjellige målbarer. M5DOWN-argmax12721/12739 er
betinget side, ikke full treklassehandling eller dokumentert handelsvalg.
PASS1→PASS2: M5-AUC.509672→.519403 mens Brier forverres0,000156777 og
UP-middelfeil−1,758→−2,427pp; D1-AUC.561885→.614417 samtidig. Dette beviser
ikke at flere epoker gir samlet fremgang eller at en bestemt ruting er feil.

Tidligere autentisert readout-attribusjon: M5s middel-logit whole-batch→issue-
miks−0,130168, native bias0,584% og samlet logit-minus-bias-rest99,416%.
Resten omfatter representasjon/projeksjon og F32-avrunding uten isolert årsak.
Direkte bias med uendret rest kan flytte p høyst0,019pp mot observert−3,252pp.
Modus-NN fullførte405forwards/alle fem skår og tilstandsbevis; feilet først ved
NumPy-float64-JSON. Én native float-cast er rettet/testet; separat CPU-plan
sammenstilte eksisterende skår med0nye NN. Originalfeilen bevares.
EVAL-logits matcher PASS2 byte-eksakt. Fire sekvensielle B256/no_grad TRAIN-
draws flytter M5s UP-middel bare+0,037710pp; feil−2,389396pp består i alle fire.
Samlet frossen modusforskjell inkluderer dropout og eval-only TF-gruppering;
dette isolerer ikke dropout eller den historiske B96/backward-banen.

Originale FINAL-vekter og tidligere kilde-/målebølger har separate verifiserte
Mac-backuper. Denne CPU-bølgens FINAL-arkiv er fysisk fsync/F_FULLFSYNC/
alle-medlems-readback-verifisert:144filer/5 944 375bytes, arkiv-SHA
624c7dea44f860b9ec1e6b415ea14ac91891863b5a555e23d5e504f3bf989afb.
Manifest/kvittering er bundet i begge CPU-policyfelter. Mislykket001, fullført002,
begge frosne kilder/tester, rapporter og neste stengte observatørdesign beholdes.
Avsluttende kvitteringsbinding/overlevering bevares i Git. Ingen rådata/vekter i Git.

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
