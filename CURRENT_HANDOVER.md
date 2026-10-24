# CURRENT HANDOVER — XAUUSD, retning først

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`,
`gx1-3090-lan`. Les GX1_RULES helt ved første arbeid; start overtakelse med
`bash scripts/gx1_handover.sh --check`. NEXT_RUN_POLICY er eneste arbeidsstatus.
Root eier integrasjon, policy og launch. Prosess og terminal overstyrer prosa.
Historikken finnes i Git og originale eksterne kvitteringer; åpne den bare ved
konkret avvik eller nødvendig ny gate. Gjenbruk beståtte kontroller.

## Mål og aktuell status

Bruker spisset24.10 til **retning først**: etterprøvbare OPP/NED-prognoser.
Trades, bevegelsesstørrelse/MFE/MAE-optimering, risiko og Exit utsettes.
Målet er aktivt og ikke oppnådd. TRAIN-fit, senere overføring, generalisering
og økonomisk edge er forskjellige bevis. Teknisk PASS er ikke gode prognoser.

252 er COMPLETE/consumed og **forkastet som retningskandidat**. Uavhengig
actualreport-peer er PASS. Alle globale/lokale training/data/fit/executiongater
er lukket igjen. Ingen ny kode, fit, native/full_epoch/fullVAL/TEST/Exit eller
trades er innrømmet. Ingen gammel engangsplan relanseres eller parameterjusteres.

## Siste faktiske retningsmåling

252 testet én fast ExtraTrees med alle3345 kvalifiserte snapshotfelt,
32trær/dybde8/minleaf256/sqrt/éntråd/seed20260911 og alle19parametre låst i251.
Fem ekspanderendeTRAIN-prefixer, identiske245renprisfasiter og frosset249R1.
Ingen nye normaliseringer, klassevekter, HPO, terskler eller kalibreringsfit.

| Horisont | TRAIN-fit, fem prefixer | Senere OOF | AlltidOPP | Balansert OOF |
|---|---:|---:|---:|---:|
|1min|55.81–56.89%|50.4972%|50.3099%|50.5338%|
|5min|54.41–56.82%|50.7493%|51.0086%|50.1333%|
|15min|55.93–60.11%|51.6819%|51.6899%|50.6187%|

Alle15TRAIN-paneler forbedrer logloss og Brier mot egen modneprefixprior.
Senere er fellesnegativefolds mot begge tap/begge referanser3/3/2, underkravet4.
Alle24pooled dag/uke95%-CI krysser0. Proper-pointdiff motprior er små negative
alleh; motR1 negative1/5, positive15. Hardaccuracy feiler motalltidOPP5/15.
Alle trehorizon-followups og samletfollowup er false. R1 blir ingen autovinner.

Dette er et målt gap fra historisktilpasning til senere kvartaler. Støytilpasning,
ustabile relasjoner og representasjonsgrenser er ikke skilt fra hverandre.
Forsøket endrer både modellklasse og inputbredde motR1; ingen kausal
featureattribusjon eller generell dom over indikatorer/native/temporale ruter.
OPP-sannsynlighet har OOFstd1,34/1,20/1,73prosentpoeng, tett ved klasseprior.
Ingen genuinep=0/1/.5 eller loglossklipp påvirket resultatet.

Utføring:385,293sworker/387,050sytre, fryst a2fdd86b/policy. Seks score/datagrupper
og én separat syntetiskforest bestod før genuinepayload. Genuine15forestfits/
480treefits/30forestpopulasjoner/960treepredict;synthetic1/32/2/64 holdes separat.
Alle480genuine og32synthetictreeinnganger er F32/C og deler aktivfit-matrise.
Femfeatureproofs kom før0/3/6/9/12genuinefits.15egne modeller har verifisert
pickle-state og readonlystat uten ekstra genuineinferens;4,821MBmodellfiler og
14,236MBOOF bevares utenforGit.24hjelpere er AST-identiske med249.

Faktisk cap4GiB/512MiB/64pids/énnumerisktråd og prosjektlås er kvittert.
Prosess-RSSpeak2235478016B, rchar1826822048B, scratchpeak1582171620B.
Prosess-RSS er ikke cgrouppeak; endelig cgrouppeak ble ikke fanget.
0nativeF/U/backward/constructor/CONTROL/TEST/quote/normfit/R1refit.
Actualpeer avstemte33paneler,231confusions,51kontraster/204CI og modellstat.
Peer åpnet ikke modell-/NPZ-/inputkropper eller bootstrapet råresultater igjen.

Eksakte terminal-/resultat-/actualpeer-/decisionrefs er i policy252 og:
`/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_NONLINEAR_REFERENCE_20261024_252`.
Originale receipts,15forskningsmodeller og OOF bevares; ingen launch/promotion.

## Datagrunnlag som skal gjenbrukes

245 har kvalifisert ren F64-midtpris ved D=bar_start+300s og T=D+1/5/15min,
fra eksakte M1-openklokker. Full9-array198kostnads-/klokkeparitet er bit-eksakt;
originale targets/kostnader er bevart.118249TRAINforeldre og88400OOFforeldre.
Alle foreldre, nullbevegelser, manglende endpoints og umodne rader beholdes i
nevnerne. Fit:observed & sign!=0 & D<cutoff & terminal<cutoff; score tilsvarende
terminal<egetholdoutend. OPP er betinget på observert ikke-null-bevegelse.

Samme fem cutoffs2024-10-01/2025-01-01/2025-04-01/2025-07-01/2025-10-01;
holdoutender neste kvartal, siste2026-01-01. Prefix/holdout-rader:
29849/17753,47602/17288,64890/17537,82427/18072,100499/17750.
Dag/Mandaguke-bootstrap2000/seed20260911/min8blokker er gjenbruktutvikling,
nominale eksplorative intervaller, ikke urørt generalisering.

206train_raw, lagret204SOURCE_MAP,209layoutbro og HELE210kontrakt gjenbrukes.
Ingen historisk kontraktgjenbygging eller relativtransform/anchor påbaseline.
Fem eksisterende210baseline-prefixnormaliseringer; ingen nyfit.236helperens
blokkviseencoder leverer3345kolonner. Hverfolds fullcol1ret_1 F32hash er lik
249kvalifisert66flates kol65. Én scratchfold/én direkte modenF32fit-kopi;
rawmap/encodedmap lukkes førfit og fit_X slippes førto heleprediksjonspopulasjoner.
Dette er et current-snapshotforsøk; parkertchart og fulltemporalsekvens er umålt.

239renpris-/temporal-/sannsynlighetseiere er uendret,50syntetiske cases varPASS.
252 bruker råETp forBrier/reliability/p>.5; midlertidigF64eps kunlogloss.
Saved249R1 bevarer239stabil logitloss og z>0; eksakt14-arrayUnicode/paritet og
18FOLD/pooled-R1scorepaneler ble avstemt førførstegenuinefit. IngenR1prefixrefit.

247S_PRICE66 og249R1_M5_BOOK5 er complete/consumed, peerPASS og forkastet;
236lineærfamilieprobe gav ingen godkjent videreføring.246klokkediagnose er
complete/peerPASS:206asofD mot223s siste nativeprisbar forklarer tidligere
fullTF-bookmisforhold. M5current/predecessor kvalifisert i249; hel86pakke er
fortsatt ikke kvalifisert,239strictgate er uendret. Disse jobbene gjentas ikke.

## Neste konkrete spørsmål

Kun lesende kildefeasibility: kartlegg hvilken kausal hendelsesrekkefølge som
allerede finnes i snapshotfeltene, og hvilken som bare finnes i eksisterende
historikkruter. Definer minste faste state-versus-history retningskontrast med
identiske245kohorter og frosset252/R1-kontroll. Hypotesen om representasjon er
åpen, ikke etablert årsak. Ingen dupliserteindikatorer eller valg fraOOF.
Ny kode/fit krever konkret design, kilde-/eier-/ressurskvalifikasjon og ny
engangsadmisjon. Ingen tilfeldig modell-/vektsveip eller gjentatt gammeltest.

## Varige grenser

NativeTransformer, åtte spesialister og niendeD1/makro, genuinefeatures/TF og
055/057/111 beholdes.19.10kompaktprofil har7aktivefamilier;chart er eksplisitt
parkert. NativeGC-rutene nårencoder/head/backward/AdamW; konsum beviser ikke
prediktivverdi.252erstatter eller fjerner ingen nativefeature/ONLINE-funksjon.
Siste nativegenerasjon bevarer223inputs/227fresh/229initial/230stateaudit/
231privatequalification/232epoke/233analyse.232 gav ikke stabil retningsgevinst;
alle12dag/ukeCONTROL-directionCI krysset0, alle512actions varFLAT. Originale
kvitteringer/checkpoints bevares. ONLINE-endring krever ny initialbaseline.

Fysiskgrunnlaget er23gjenbrukteutviklingsmåneder:TRAIN mai2024–desember2025,
CONTROL januar–mars2026.26måneder er målet; april–juni er ikke fysiskbundet.
Juni er gjenbruktutviklingsVAL. TEST frajuli er forseglet for alltilpasning/utvalg;
bare vedtattmånedligforwardjournal scorer utenrefit. IngenExit/live/paper/broker/
ordre/spending/promotion. Beregningshorisonter er ingen taps-/holderegel.
Langstøtte bruker alltidLONG/kjøp-og-hold; økonomi må inkludere åpneposisjoner.

Én tungCURRENT-jobb via gx1_capped_run.sh med lås/hardwarevakter, fryst kilde og
policy tilterminal. Stabil langkjøring kontrolleres15–30min eller sjeldnere;
ingenminuttmodellpolling/fyllarbeid. IngenDATA/RUNS-rydding uten retention-eier.
Oppdater policy/handover i sammebølge; fokuserte kontroller/AST/stale-path/
gitdiff--check og ærlige grenser. Pushrett tilwork/gx1-current gjelder kode/docs/
interne refs/aggregater, aldri rådata/vekter/hemmeligheter ellerforcepush.
