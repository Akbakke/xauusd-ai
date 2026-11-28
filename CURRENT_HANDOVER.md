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

NEXT_RUN_POLICY.json er eneste arbeidsstatus. Sist fullførte felt er
native_all_tf_provider_hlc_v3_direction_issue_mix_20261128_001; neste
kildeforberedelse eies av native_all_tf_provider_hlc_v3_train_eval_gap_20261128_001.
REFERENCE, COLD og FIT er faktisk COMPLETE og lukket/forbrukt med remaining=0;
alle seks trenings-/evalueringsflagg er false og active_lane=null. Produsentene
er avsluttet. Ingen modelljobb eller ny fitserie er innrømmet. Appmålet mot
retningsedge med alle genuine indikatorer og deres samspill er aktivt/uoppfylt.

### Fullført kontroll av batchsammensetning

Felles fryst kilde gjennom alle tre roller var 7471ddf48; protokoll-SHA er
fe51cc828772f9050b59aef287814b2e0aaa2f1401acefa8b5bdedd5069c74ad.
To private PCG64-permutasjoner blander bare hele prognoseeksemplers medlemskap.
Indeksene sorteres innen hver batch for eksisterende packet-API. Alle20727
issues er med én gang per pass; 87-raderssteget ligger fortsatt på globale
nullindekser111/428. Hvert issue beholder sine egne genuine kausale vinduer.
Det finnes ingen aktiv carry mellom issues. Modelldata, prior, normalisering,
mål, masker, tapsformel, optimizer, klipping og vakter er uendret.

Faktisk STREAM_BINDING binder alle432 takes til source_rows, issue_ns og
maskert fasit. REFERENCE utførte ferske to steg og publiserte ekte C1/C2.
Separat COLD lastet bare egen REFERENCE-C1, utførte ett steg og matchet C2
på alle ti tilstandsbevis, med ærlig egen eksekveringsprodusent. Disse2+1
tekniske oppdateringene er ikke kandidat eller initializer.

FIT startet fra fersk factory/full RNG, med0 checkpointloads/restores.
Fersk modell/RNG og faktisk truth-SHA matchet samme originale INITIAL;
lagrede INITIAL-outputs ble gjenbrukt med0 nye INITIAL-forwards.
Den fullførte432 nye oppdateringer/597 forwards på812,48s, exit0, med0skips,
én featurecache-/targetlasting og0 norm-/priorfits. CURSOR001 og FINAL432
med egne ordrebevis er varige. Alle seks bookends består i hver rolle.
Egne immutable policy-snapshots og separate SOURCE/REFERENCE/QUALIFIED/FINAL-
backuper er fysisk fsync/readback-verifisert på Mac. FINAL inneholder216 filer,
inkludert alle faktiske checkpoints, CPU-bevis og neste stengte kildedesign.
Eksakte arkiv-/manifestbindinger og kvittering ligger i policy.

Studien har20727 TRAIN-issues fra1377 kildeklokkedøgn2021–2025, B96/E256,
seed20260911 og to pass à216 steg. Alle åtte genuine familier og D1/makro
beholdes med M5/M15/H1/H4/D1 og fullt koblet None-policy. Tidligere originale
stopp/ukjent fysisk arbeid1..216 beholdes i sine lukkede felt; ingen gammel
engangsplan ble relansert eller historisk kvittering omskrevet.

### Låst PASS2 — fortsatt TRAIN FAIL

Bare forhåndslåst PASS2 er kandidat; INITIAL/PASS1 er diagnostikk. Side NLL/Brier
slår COHORT-prioren på2/5TF kronologisk,4/5 ved whole-batch-stokking og4/5 med
issue-miks under begge vektinger. Nå består M15/H1/H4/D1; M5 feiler. Samlet
proper-score og tidsblokkstabilitet er FAIL, learning_qualified=false.
Aktivitet og pooled rank-guard består alle fem; dette er ikke alene edge.

| TF | Side-AUC, issue-miks | Side Brier minus prior, radvektet | År-PASS rad/målbar |
|---|---:|---:|---:|
| M5 |0.519403|+0.000398422|1/5 · 1/5|
| M15 |0.518758|-0.000249777|4/5 · 4/5|
| H1 |0.524846|-0.000190210|3/5 · 3/5|
| H4 |0.524267|-0.000416408|5/5 · 4/5|
| D1 |0.614417|-0.008619807|5/5 · 5/5|

Årskontrollen har18/25 rad-PASS og17/25 målbar-PASS, mot14/25 og16/25
under begge vektinger i de eldre kontrollene. M5 feiler2021/2023/2024/2025,
M15 feiler2021, H1 feiler2024/2025 og H4 feiler2024 bare ved målbarvekting.
D1 består begge proper-scorene i alle fem år;13968 kjente rader representerer
bare875 forskjellige målbarer. Alle25 års-AUC-er er over0,5 på TRAIN,
uten at dette er uavhengig eller senere generalisering.

M5s kjente side har UP-sannsynlighetsfeil-2,4271prosentpoeng og betinget
DOWN-argmax12721/12739. M15/H1/H4/D1 har DOWN424/4/0/4765 av henholdsvis
11936/10980/11590/13968 kjente sider. Dette er conditional-side-prognoser,
ikke full treklassehandling, klassekvote eller dokumentert handelsvalg.
M5s Brier-differanse+0,000398422 består algebraisk av middel-feil-kvadrat
0,000589085 og sentrert rest-minus-prior-0,000190663. Ingen kalibrator ble fittet.

PASS1→PASS2 er diagnostikk, ikke kandidatutvalg. I issue-miks forbedres M5-AUC
0,509672→0,519403, mens side-Brier forverres0,000156777 og UP-middelfeilen
går-1,758→-2,427prosentpoeng. D1-AUC forbedres0,561885→0,614417 samtidig.
Dette beviser ikke at flere epoker vil gi samlet fremgang på alle fem.

### Fullført ny CPU-attribusjon og neste kildeforberedelse

Ny CPU-analyse består med1 ny FINAL-dekoding,1logitarkiv,1targetarkiv/16felt,
79 kildeeiere og alle fem bookends. Ingen modell, optimizer, NN, autograd,
oppdatering, featurecache eller CUDA ble kjørt. Alle10 semantiske checkpoint-
bevis og alle10 sidevekt-/bias-momentslots ved432 er autentisert. Gamle armer
ble bare lest som bundne JSON-rapporter, ikke dekodet eller kjørt om.

M5s kjente middel-logit endres-0,130168 mellom whole-batch og issue-miks:
native bias-0,000760 er0,584% av additiv logitendring, resten99,416% er
logit-minus-bias-rest med representasjon/projeksjon og F32-avrunding samlet.
Med uendret rest kan direkte biasendring flytte sannsynlighet høyst0,019pp,
mot observert middelendring-3,252pp. Dette er ikke additive sannsynlighets-
årsaker eller bevis for en bestemt representasjons-, taps- eller rutingfeil.
En enkel biaspatch følger ikke av målingen.

Neste kildeforberedelse eies av
native_all_tf_provider_hlc_v3_train_eval_gap_20261128_001.
V3_TRAIN_EVAL_GAP_DESIGN_20261128_003.json binder én måling på samme frosne
FINAL: én full B256 EVAL-paritet mot lagrede logits og fire forhåndslåste,
sekvensielle no_grad TRAIN-moduspass på alle20727issues/alle fem TF/år.
Dette er405forwards og0updates, ikke en ny læringskandidat. Fire draws er et
fast diagnosebudsjett, ikke kvalifisert presisjon eller optimalitet.
Rapporter hver draws score og mean/SD/SE av scorene; ingen gjennomsnitts-
logit/-sannsynlighet skåres som ny prediktor. Dette er B256/no_grad, ikke
replay av opprinnelig B96/backward-trening. Før opprydding må alle vekter,
buffere, optimizer, steps/scores og tomme gradienter være uendret; bare
intensjonelt endret modulmodus/full RNG gjenopprettes før fullproof10.
Design003 presiserer faktisk tilgjengelige skår før kjøring: rad-AUC og
rad-/målbarvektet NLL/Brier/sannsynlighetsmiddel. Målbarvektet AUC er
NOT_IMPLEMENTED; ingen ny numerisk eier eller kvalitetsregel innføres.
Uutførte001/002 beholdes som opprinnelige design; den nye diagnostikken
er kildekvalifisert, uten at modelljobb er startet eller forbrukt.

Faktisk arkitektur bruker dropout0,05 under train(); scoring bruker eval().
Eval har også gruppering av like TF-lengder. Foreslått måling gjelder derfor
samlet modusforskjell, ikke isolert dropoutårsak. Ingen faktisk modusgap er
målt. Vekter, optimizer, modellmodus/full RNG og alle10 bevis må bevares ved
slutt; EVAL må først matche de lagrede F32-outputene eksakt. Alle draws og
all støtte/dispersion rapporteres; ingen beste draw, terskel eller kalibrator.

Kildeadapter og vakter er kvalifisert med80 guard-/kildebrokontroller og36
forskjellige leaf-kontroller. Feil forventet unntak i én test ble rettet og
bare berørt test kjørt om; fersk fysisk metadataidentitet har tre egne
kontroller. Ekte output-allowlist er testet før NN. Ingen faktisk modusgap
er målt. Ny eksakt read-only NN-plan, aktuell runtime/host-admisjon,
backup og fryst kilde kreves fortsatt. Alle seks globale
flagg er false. Ingen dropout-, læringsrate-, ruting-, taps-, epoke- eller
servingendring følger av dette. En eventuell modusforskjell er ikke bevist
historisk årsak, bedre optimalisering, generalisering eller økonomisk edge.
Eksisterende tre FINAL-paneler gir heller ingen universell rutingfeil.

Kildekvalifisering bestod62 guard-/kildebrotester og31 leaf-kontroller;
obligatoriske kontrakttester, syntaks, shellsyntaks og diffkontroll består.
Eksakt additiv reversering bevarer gamle numeriske eiere og alle vakter.
Fokuser på aktuelle bevis; ikke gjenta uendrede fullsuiter.

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

- V3_DIRECTION_ISSUE_MIX_TRAIN_REVIEW_20261128_001.json: tre låste PASS2-armer,
  alle fem TF, begge vektinger, år og ærlige ubeviste grenser.
- V3_DIRECTION_ISSUE_MIX_REFERENCE/COLD/FIT_ACCEPTANCE_20261128_001.json:
  faktiske roller, fullproof10 og seks bookends; bruk konkrete filnavn fra policy.
- V3_DIRECTION_ISSUE_MIX_SOURCE_QUALIFICATION_20261128_001.json:
  kildebroer,93 fokuserte tester og faktisk metadata før NN.
- V3_DIRECTION_ISSUE_MIX_DESIGN_20261128_001.json: original énvariabelhypotese.
- V3_DIRECTION_ISSUE_MIX_SOURCE/REFERENCE/QUALIFIED/FINAL_BACKUP_20261128_001.json:
  separate diskbackuper med egne manifeste og fysisk verifiserte kvitteringer.
- V3_ISSUE_MIX_READOUT_ATTRIBUTION_20261128_001.json: faktisk ny arm, original
  fasit og gamle JSON-sammenligninger; ingen nye prediksjoner.
- V3_ISSUE_MIX_PASS_DIAGNOSTIC_20261128_001.json: alle tre arms PASS1→PASS2;
  PASS1 er aldri kandidat.
- V3_ISSUE_MIX_TRAIN_EVAL_SOURCE_REVIEW_20261128_001.json og
  V3_TRAIN_EVAL_GAP_DESIGN_20261128_003.json: kildefakta og neste stengte
  kjøringsdesign; ingen målt modusgap eller ny treningsadmisjon.

Aktuelle roots har prefiks
/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_ISSUE_MIX_20261128_001_
og suffiks reference/cold/fit. Eldre whole-batch- og kronologiske kontrollroots
bindes i den nye FIT-planen og TRAIN-review. Originalkvitteringer overstyrer prosa.
