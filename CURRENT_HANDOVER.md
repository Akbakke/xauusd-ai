# CURRENT HANDOVER — XAUUSD M5 Entry

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current` på
`gx1-3090-lan`. GX1_RULES er bindende; NEXT_RUN_POLICY eier arbeidsstatus.
Prosesser og terminalkvitteringer overstyrer prosa. Én integrator, én capped
CURRENT-jobb og fryst kilde/policy gjennom hver kjøring. Originalmålet står i
[docs/ORIGINAL_PROJECT_GOAL.md](docs/ORIGINAL_PROJECT_GOAL.md).

## Aktivt mål og neste steg

Målet er bedre M5 Entry LONG/SHORT/FLAT med MBP-1, SMC og lukket M15/H1/H4/D1:
retning, størrelse, liten MAE før første MFE, kostnad, kvalitet og dekning.
Brukeren prioriterte22.10 en solid featuregrunnmur og deretter generell robusthet:
ingen bittesmå tilpasninger eller overtilpassing; undersøk også motsatt retning
ved mye MAE. MAE alene beviser ikke feil retning eller invers edge.

184s avgrensede MFE-endring er implementert og185 kildekvalifisert:133 fokuserte
CPU-tester,0feil/hopp,6,863s, audit4G/512M/én tråd/pids64. Kilde-/matematikkpeer,
AST/compile og diffkontroll bestod. Native183roller/182unike kildefiler er bundet;
kun tre nativefiler er endret. Alle øvrige nativehashes er bevart.
Dette er teknisk bevis.184 er ikke trent.188 har nå målt en fersk initial på
genuine inputs;189s savedstate-audit er bestått.173 gjelder forrige170funksjon.172s
kildekvalifikator er tilpasset185 i187; metadatakontroll og CPU-konstruksjon bestod.

186s brede periode-/reverseringskontroll er COMPLETE/consumed og actualpeer-
kontrollert: fire lagrede167/177 OBS én gang hver,0F/R/U/fit,22,488s ytre tid.
Samme entry og opprinnelige horisontsett/masker/kostnader er bevart; FLAT/ties
beholdes. Ingen ny input-/Parquet-/PT-lesing eller prediksjon. Mainresultatet er
bevart. En for stor kvartalseksport feilet; korrigert kompakt eksport bestod.
Digest dekker fem brede kvartalsbånd;1/5/15 finnes i ALL/originalRESULTS, ikke
kvartalsdigest. Ingen estimator eller originalanalyse er endret.

Historisk167s faktisk valgte signaler, terminalrespons etter deklarerte kostnader:

| Periode | Valgte | Original bps | Motsatt bps | Ukeblokkintervall for forskjell |
|---|---:|---:|---:|---:|
|TRAIN|58|−6,983|−5,115|[−2,183;8,072]|
|Q1CONTROL|154|−6,091|−6,263|[−1,952;1,714]|

Forskjellen er motsatt minus original; begge sider har negativt gjennomsnitt.
CONTROL preMAE øker7,361→7,587bps og MFE faller3,185→3,002bps ved inversjon.
TRAIN-kvartalenes inversjonseffekt skifter fortegn.177s faktiske valg er fortsatt
alle512FLAT; tvungen retningsdiagnostikk er ingen valgt handel eller entrygevinst.
Ingen universell snuregel eller invers edge er kvalifisert. MAE alene avgjør
ikke motsatt sides verdi. Dette er prisbanemål, ingen realisertP&L.

22Oct-kostnadsavstemmingen188/COST_AND_DIRECTION_STATUS_REVIEW.json bekrefter:
bid/ask-spread ligger allerede i quoteavkastningen; ekstra4bps er to deklarerte
slippageledd à2bps, commission0. Dette er konservativ, ikke-fittet preregistrering;
historisk quote-til-fill-clock/slippagetruth er ukjent. Inversjonen bruker fysisk
motsatt side og legger ikke på en ny kostnad. Felles4bps kanselleres i den separate
retningskontrasten; svak kort retningslæring kan ikke forklares med dette alene.
Nyeste177 har diagnostisk fortegnstreff49,2/59,4/49,2% ved1/5/15 på fastCONTROL;
dette er ikke valgte handler eller kvalifisert edge. Originalresultatene bevares.

TRAIN256 dekker193dager/84uker fordelt25/34/36/42/32/42/45rader i syv kvartaler;
fem av syv kvartaler har under8støttede uker for167s valgte signaler. CONTROL256
dekker68dager/14uker i Q1. Sparse prober er ikke alle118249TRAIN-rader eller
urørt generalisering. Ingen favorittmåned/-minutt eller ny terskel velges.

186/DECISION.json beholder184 som én falsifiserbar M5/MFE-målhypotese, uten
påstått retningsgevinst.187s CPU-fresh-konstruksjon er COMPLETE/consumed:
118,744s ytre/117,917s worker,0modellforwards/0updates/0refit/0Exit. Faktisk
preclaim-metadatakvalifikator bestod mot195filsnapshot;186kvalifiserte unike
kilde-/testeiere,183native roller/182filer og bare tre endrede roller er bundet.
ONLINE=EMA=TARGET har modellhash2eaf88…, optimizer er tom, TARGET fryst og
ingen checkpointvekter er arvet.11inputrefs/profil/normalisering/111RMS er bevart;
kilde/policy var fryst og prosjektlåsen er verifisert fri etter terminalen.
Dette er en fersk CPU-starttilstand;188 har deretter målt native initial32F/0U.
ActualCPU-peer bestod uten uavklarte funn. Original187kvitteringer/state bevares; ingen relansering.

188 er COMPLETE/consumed: én sealer/materialisering og én faktisk native initial,
256TRAIN+256Q1CONTROL,16+16ONLINEforwards,0TARGETforwards/0optimizersteg. Faktiske
tellere er produsentrapportert i loggen; root/peer har ikke gjenlest OBS. Canonical
receipt er RESUMABLE/guardPASS/exit0, planlagt pause ved optimizer_step_ceiling0,
ingen hovedepoke eller ferdig modellbundle. InitialRESULT a76b95e8… har ONLINE=
TARGET2eaf88… og samme187init/11inputs/111/183roller/182filer/digest ed1c6f… .
Kilde e97fc3ff/policy f654eade var fryst gjennom kjøringen. Rootterminal afe658eb…
og actualpeer b1fc7e6b… bekrefter initialmetadata;189 har bestått savedstate-audit.

FysiskWindowsboot527/22Oct18:47:22.5Z er bekreftet etter526. Maskinen var ved
fersk lategrense uten observerte prosjektjobber/låser; første NOTidle-inventar og
egencollectorFDgap er bevart, ingen absolutt OS-zero-IO-påstand. Eksakt ny task
var ferdig Ready/LastResult0 og er nå Disabled/restart0; keeper160W/SHA62b53… er
bevart. Signert guard hadde peak56Ccore/64Cminne/129,52W/738MiB og exit0.
Prosjektlåsen er målt fri. Ingen188relaunch eller ny omstart/native er autorisert.

189 er COMPLETE/consumed: én CPU-savedstate-audit,4G/512M/éntråd/pids64,
3,385s, to187/188PT-arkiver,0modellkonstruksjon/F/R/U/fit, ingen marked/OBS/111.
Worker er working173audit etter bare stage-/kildedigestbytte. Firefeltbinding
55ba6f5c… frøs HEAD dcb3fb11/policy519669e4. Terminal f89dff5c…/exit0 og actualpeer
ca247b85… har ingen blokkere; proof de43c53d… og audit cee9fc3f… er immutable.
Saved ONLINE=TARGET=EMA2eaf88…, optimizer/EMA/scheduler og CPU/Python/NumPyRNG
stemmer eksakt mot187fresh. Dispatcher er borte og prosjektlåsen er målt fri.

Dette gjelder durable PREmeasurement tilstand; CUDA-RNG kun schema. Postmålt
RNG-digestlikhet er ubevist. Eksisterende technical/main-dispatch gjenoppretter
saved187freshstate i ny prosess;188s levende postmålingstilstand brukes ikke.
190s canonical materialisering har utført FullOBS-konsumentvalideringen; ingen
ekstra native-rekonstruksjon eller ny originalOBS-audit trengs.

190 er COMPLETE/consumed: canonical materialisering og faktisk privat teknisk
kvalifikasjon bestod. RESULT8e887e2b…/actualpeer2c4ce00b…/rootterminal03567770…
binder samme187fresh/188initial/189audit/185kilde,183roller og11inputs/111. Fire
fysiske private oppdateringer er2referanse+1avbrutt+1gjenopptatt; logicalcursor2.
Alle8resumehasher stemmer. Fire parityforwards/16TRAIN+16CONTROL/full480 har
maksavvik0bps i alle16outputpaneler; masker/actionties/modus/RNG er eksakte.
Dette beviser mekanikk, ingen Entry-/læringsgevinst. Private vekter bevares som
audit og brukes aldri i main. FullOBS-konsumentvalidering er utført i190.

Faktisk fysiskWindowsboot528/22Oct21:00:42.5Z etter527 er bekreftet, guardPASS og
canonicalreceipt553af7ed… er RESUMABLE/exit0 ved planlagt2stegspause. Kilde35cd3727/
policy31109f91 var fryst. Controller7764 er avsluttet; eksakt190task er Disabled/
last0/restart0, keeper160W/SHA62b53… uendret. CanonicalUID1000-prosjektlås er målt
fri22:17UTC, ingen CURRENT-Pythonjobb. Ingen190relaunch eller privatevektovertakelse.

191 er en NY, separat og betinget admitted matchet hel TRAIN-epoke for184hypotesen,
under brukerens aktive kontinuerlige mål; ingen gammel engangsautoritet gjenbrukes.
PLAN8d9af59e…/DESIGN205808b7…/EVALUATION434fc973…/PREPARATION4d9bca4d… og
preexecpeer28157042… er immutable. Sealer90decbac… er working177 etter23literal-,
firepeernesting- og én185→184metadatajoinendring; controller8b263087… bare stagebytte.
Roottransform-/AST/compile og avgrenset peer bestod uten kildeblokkere.185s133
fokuserte tester gjenbrukes; ingen ny baseline, fullsuite eller modellhypotese.

Ny191admission tillater én capped metadatasealer (900s) og én canonical CPUaudit-
materialisering (7199s+10TERM),4G/512M/éntråd/pids64, før native. Eksakt20felt mainmap
er installert; privattechnical- og initial-modes er fjernet. Globaltraining/full_epoch
true gjelder bare191. Før utførelse kreves clean postcommit og faktiske bindinger.
191PLAN.execution_ready er betinget metadataklarhet, ikke egen kjøretillatelse.

Native tillates kun etter faktiske late Windows/WSL-idle/source/prosjektlåser,
eksakttask, canonicalintent/confirm og NY fysiskboot etter528. Før-boot528 er bundet
som191/PREPARED_WINDOWS_BOOT.json2dc63219…; dette er ingen ny191nativeboot. Bevar
160Wkeeper og alle eksisterende signerte ressurs-/temperatur-/effektvakter.
191starter fra saved187fresh, aldri190private/177trente vekter;188owninitial og189audit
gjenbrukes. Budsjettet er nøyaktig177:118249TRAIN,7391updates,B16/siste9,seed20260911,
12000s native/13800s outer,32sluttONLINEforwards på samme256TRAIN+256Q1CONTROL/480.
Data, features, arkitektur, optimizer/LR/EMA,111RMS, targets/masker/kost/score bevares.

Vurder mot egen188initial og matchet177 etter preregistrert191EVALUATION. Rapportér
også186s faste kalenderkvartaler/brede bånd og faktisk dag-/ukestøtte; sparsomt
under8blokker er kun deskriptivt. Ingen favorittmåned/-minutt, terskelvalg, kostfit,
HPO eller automatisk ekstraepoke. Retning, størrelse, MAE før MFE, kostnad og
LONG/SHORT/FLAT-kvalitet/dekning vurderes separat. Lavere pooledtap eller alltidFLAT
fullfører ikke målet.23gjenbrukte måneder/faste512prober er ikke26måneders OOF,
urørt generalisering eller realisert økonomisk edge. Exit/TEST/broker er stengt.

## Siste trente modellresultat og tapshypotese

177s ene matchede epoke er consumed:118249unike TRAIN-rader/7391oppdateringer/
32sluttforwards.178s lagrede sammenligning er canonical-validert/consumed.
LONG0/SHORT0/FLAT256 på både TRAIN og CONTROL; alltidFLAT er ingen selektivitet.
Begge siders predikerte beste horisont er1 på alle512sluttrader.

CONTROL nettkontrast-MSE (bps²), samme rader/masker:

| Minutt | Egen173initial | 167final | 177final |
|---|---:|---:|---:|
|1|57,459|572,260|61,678|
|5|228,358|686,241|227,369|
|15|772,101|1358,138|774,629|

Alle6primærpaneler/TRAIN+CONTROL forbedres mot167, men5av6 forverres mot
173initial. CONTROL5min har liten bedring med usikkerhet over/under0.
CONTROLh1 predstd2,461 motobservasjon7,504bps, corr0,0578. Overamplituden fra167
er dempet; bedre kvalifisert retning/Entry, generalisering og edge er ubevist.

181 viste at GELU-decoderen tillater tilstand×horisont-respons, med alle480 egne
sideceller/masker/RMS.182 gjenbrukte179s taskgradienter: samlet gradient er lokalt
samrettet med terminalen i alle7 undersøkte, overlappende grupper. Ingen tvungen
horisontlikhet, axisfeil eller begrunnelse for klonet terminaldecoder er påvist.
Dette er lokale vitner, ingen AdamW-/7391stegsårsaksanalyse.

183s lagrede TRAIN256-analyse er COMPLETE/consumed,0F/R/U,0,674s ytre tid.
MFE pooled enhetsfeil1,061→0,519 skjuler tidlig1–15-feil1,360→1,877, mens tidlig
rå-MSE bedres56,173→51,891bps². PreMAE bedres både tidlig1,221→0,684 og senere
1,035→0,459. Tidlig MFE har allerede92,65% av idealisert outputgradient-L2²;
50:50-gruppemiddel gir99,988% og12,73× output-L2 ved samme prediksjoner.
Dette beviser ikke svak tidlig parametergradient eller forventet læringsgevinst.

184 er bare MFE-only-prioritet: snitt av ikke-tomme1–15-/16–480-gruppemidler.
Hver gruppe pooler inklusive+closed LONG/SHORT SSE/ekteN med original111RMS;
ingen FLAT-path. Én gruppe får fullt middel, ingen gruppe utelater MFE/+s.
Terminal170, MAE/forecast,111unitkontrakt/-primitiv, rå/pooled diagnostikk,
inputs/targets/kostnader/score og modellarkitektur er bevart. Ny MFEoptimizer-
evidens navngis separat; kontraktskjema16 krever ny baseline før nativearbeid.
15/50:50 er en eksplisitt M5-designkonvensjon, aldri målt optimum eller Exitregel.

## Featuregrunnmur og gjenstående bevis

175 avstemte aktive kilder/input/numerikk/kausalitet. Aktiv profil er
`compact_mbp_smc_ema_atr_clock_macro_entry_v1`:7lokale/5MTF-familier etter19Oct-
unntaket; hele bibliotekets8spesialister og9.D1/makro beholdes. Alle695aktiveMTF-
felt139×5TF er numerisk kvalifisert; aktive GC-verdier/masker hadde0F32avvik.
SMC, EMA/trend, swing, støtte/motstand, volumprofil og MBP-1 beholdes kausalt.
Sjeldne/manglende observasjoner forblir ukjente; de fylles ikke med syntetikk.
MBP-1 er aggregert BBO/ordretall/OFI, ikke ordreidentitet/kø/skjult dybde.

180 målte trent177s lokale inputavhengighet:1F/5VJP/0updates,16genuine TRAIN,
alle84640paritetsceller delta0bps og identiske action-/horisontties.3317kanoniske
koordinater er regnskapsført. Alle32aktive av48deklarerte familie/TF-laner og
alle8flytende inputflater har lokal ikke-null respons; makro61 beholdes som
overlapp. Dette beviser konsum, ikke featureviktighet eller optimal utnyttelse.
Parkerte felt er ikke generelt numerisk-/nytte-kvalifisert eller forkastet.
Nesten parallelle1/5/15-inputgradienter er ikke alene en arkitekturfeil.
Kronologisk OOF-merverdi og generell feature-/modellnytte gjenstår.

## Varige grenser

Aktive beslutningsinputs: TRAIN[2024-05-01,2026-01-01), Q1CONTROL[2026-01-01,
2026-04-01), altså23måneder. April–juni er senere gjenbrukt utvikling som venter
separat fysisk binding; råkildemålet er26måneder.256-raders prober er ikke hele
TRAIN118249. MTF-cache har lengre historikk. TEST fra juli er forseglet for
utvalg/tilpasning; bare vedtatt månedlig forwardjournal scorer den uten refit.
Ingen Exit/broker/live/paper/ordre/spending/promotion.

Entry beslutter på lukketM5.1–480min er prognosedomene, aldri fasttap eller
maksimal holdetid. Rapportér teknikk, TRAIN-læring, kronologisk gjenbrukt utvikling,
generalisering og økonomi separat. Lange støttemål har alltidLONG/kjøp-og-hold
som forhåndsvalgt referanse. Ingen ekstraepoch/HPO eller relansering av consumed
engangsplaner. Native initialmåling krever ny fysiskboot; CPU-konstruksjon har
ikke eget bootpredikat.191 er betinget admitted; native krever alle ferske runtimegater.

Originalkvitteringer/checkpoints bevares; DATA/RUNS-rydding krever retention-eier.
Historikk ligger i Git/originalkvitteringer og åpnes bare ved konkret avvik/gate.
Metoden er [docs/NATIVE_LEARNING.md](docs/NATIVE_LEARNING.md).
