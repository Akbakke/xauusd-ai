# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_NON_M5_MATCHED_SHORT_LEARNING_ADMITTED_RESULT_PENDING.
Målet 11.11 er aktivt: avklar målte årsaker til svak retningsverdi i indikatorer,
sammensetting og læring. [Samlet status](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte runder er feilsøking, ikke kapasitetsdom. Full kvalifisert TRAIN-historikk
og 30–40 epoker er senere skalering, ikke et krav ved hver rettelse.
Alle tidligere modellplaner og siste lesende initialparitetskontroll er konsumert.
Én ny eksakt nonM5kort sammenligning er innrømmet etter verifisert boot641; ennå ikke startet.
Training/native_training er true kun for den nye256update-planen; generell native_launch,
full_epoch/full_val er fortsatt false. Begge gamle
c171 Windowsoppgaver er Disabled. GC er på pause; TEST, Entry/Exit, broker,
handel, spending og promotion er stengt. Sluttcheckpoint er evidens, ikke promotion.

## Indikatorer og faktisk læring

Åtte TA-familier og M5/M15/H1/H4/D1 når faktisk modellkonsum, med makrokontekst.
Numerisk dekning og ikke-konstante TRAIN-støtter er bekreftet; dette beviser ikke
prediktiv familienytte. Alle 11 temporal-/akse-encodere og 228 undersøkte tensorer
hadde vektendring og Adam-andremomenter. Mean+last kansellerer ikke systematisk;
tidligere fast MTF-intervensjon ga ingen bedre rangering. Trendlinjeslots mangler
entydig samme-linje-identitet/ankere/endringsmarkør; prediktiv kostnad er UNKNOWN.
Alle genuine features og tidsrammer er bevart. En eksplisitt opt-in sidehead-rettelse
er mekanisk kontrollert; tapsformler og vekter beholdes. Standardmodellen er uendret.

Siste native trening er fortsatt short002: 256 updates, 4096 kronologiske TRAIN-rader,
0 skips, fersk tom AdamW og samme c171-initial/originaltap. Kilde 9de7e20b.
TRAIN-perioden er 1. mai 2024 00:05 til 21. mai 2024 20:35 UTC.
Den opprinnelige før/etter-målingen hadde bare 256 rader, hvorav 48 var nyoppdatert.
Dens svake resultater kunne ikke gi en generell dom over hele brukt-prefix-fit.
[Opprinnelig kontroll](docs/TA_M5_C171_SHORT_LEARNING_REVIEW.json).

## Fullført kontroll av hele den brukte perioden

[Resultat og metode](docs/TA_C171_FULL_USED_PREFIX_REVIEW.json).
ROOT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_FULL_USED_PREFIX_20261111_001.
Kilde 480addcaa0f267f3023385f753dc61e3a2055655 uendret under kjøringen.
RESULT 23afd4e6, TERMINAL ccf476bf, controller b3172bbf/exit0.
647,48 s worker / 650,43 s controller; én modell, 512 forwards, to checkpointlesninger,
én eksplisitt import av finalvekter; 0 optimizer/autograd/updates/checkpointskriv.
Alle 4096 rader per fase matcher TRACE sine 10 faste akse/query/maskefelt og fire
støttecounts byteeksakt. TRACE sine pre-update-logits er ikke brukt som fast baseline.
Kontrollbatchene 0/123/246 matcher originale panelposisjoner 0/1/2 i alle 12 felt og tap
for begge vektsett. Modell/buffers/RNG/modus/gradflagg/identiteter/hooks er passive.

TRAIN_FUSION.npz 9fbfbc66 inneholder originalfeltene + fase, 8192 evalueringsrader og
4096 unike rader; 1 699 064 byte under 2 MiB. Ingen head-/boundary-hooks ble brukt.
68 fokustester, metadata-preflight, faktisk cap-selektor og uavhengig kildereview bestod.
Vakter beholdt: 12 GiB/512 MiB, CPU0–7/pids64, 8 numeriske tråder, 250 W limit /
170 W draw / 65 C / 80 C, 900 s worker / 1200 s guard. Ingen numeriske eiere ble endret.

## Hva hele prefixet viser

Begge M5-utganger forbedres litt på sine egne brukte TRAIN-rader:

| Mål | Før | Etter |
| --- | ---: | ---: |
| S/M5 AUC | 0,546185 | 0,551466 |
| F/M5 AUC | 0,546125 | 0,553079 |
| S/M5 Brier | 0,247999 | 0,247924 |
| F/M5 Brier | 0,248788 | 0,248193 |
| Originalt samlet tap | 1,878003 | 1,892871 |

AUC er rangering, ikke prosent riktige handler. Kjent retning finnes for 1378 S-rader
og 1644 F-rader; observasjonene er avhengige TRAIN-data. Begge M5 log-tap forbedres også.
Gjennomsnittlig P(UP) øker likevel 2,398/2,007 prosentpoeng; dette er ikke lik bedre edge.

Det samlede originaltapet øker 0,0148681. Beholdt tapsoppdeling viser bidrag
+0,00757676 fra aktivitet og +0,00729134 fra retning. F/M15 og F/H1 retning samt
S/D1 aktivitet blir dårligere, mens F/D1 og F/H4 retning blir bedre. Dette er målte
oppgave-/tidsrammeavveininger, ikke bevist gradientkonflikt eller grunn til å fjerne D1.
Fire faste kronologiske kvartaler har totalendring +0,045023, +0,077455, −0,006415,
−0,056590: tidlig halvdel blir dårligere, sen halvdel bedre. En effekt av nylig eksponering
er mulig, men unik årsak, glemsel og regimeskifte er ikke bevist.

FULL_USED_PREFIX_EFFECT_REVIEW a2c88641 er kontrollert uavhengig av 473e6967:
954 tallfelt og 206 eksakte felt, ingen avvik utover 4,55e−13 i summering. Rangbasert
AUC bekrefter den første parvise beregningen. FULL_USED_PREFIX_LOSS_ACCOUNTING
68ccd418 gjenskaper de opprinnelige batch-komponentene med maks FP64/FP32-avvik
4,23e−7 og samlet endringsrest 8,16e−9. Dette er regnskap, ikke kausal oppdateringsandel.
To lokale CPU-reviewfeil før scoring (importsti og prioroppslag) er bevart; bare konkret
import-/referanseoppslag ble rettet. Ingen modellkjøring ble gjentatt.

Tidligere paret source/tail-måling viste betinget rangfall ved sourcebytte med finaltail
fast og stort middelskift fra kontekst. [Avgrensningen](docs/TA_C171_PAIRED_JOINT_BOUNDARY_REVIEW.json)
består; den er ikke bevis for generell svekkelse av fullprefix-læring eller informasjonstap.

## Kontrollert treningsmodus og målt samvariasjon mellom horisonter

[Kontrollen](docs/TA_C171_MATCHED_TRAIN_MODE_REVIEW.json) er COMPLETE/CONSUMED.
TRACE_ZERO_UPDATE_MODE_REVIEW 11734549 viste råforskjeller før første update med
samme vekter og rader: S0,114820/F0,035089. Original dropout0,05 er aktiv; dette er
modusforskjell, ingen påvist dropoutfeil. Tidligere paret eval-måling står uendret.

ROOT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_MATCHED_TRAIN_MODE_20261111_001.
Kilde a6ece68d uendret; RESULT 3f637f8c, TERMINAL 236bdfb7, controller c812725d/exit0.
676,92 s worker / 680,52 s controller, 512 forwards og 0 backward/optimizer/updates.
Faktiske INITIAL/FINAL-vekter, original module_training/gradflagg og train/enable_grad;
grafen kastes etter hver batch. Begge faser starter original INITIAL-RNG 67daad3d.
INITIAL batch0 matcher TRACE i alle12felt/tap. Alle256parede RNG-grenser er bitlike,
og begge faser ender på original FINAL-RNG b06f3644. Alle øvrige passivitetskrav består.
79 fokustester, preflight, cap-selektor og uavhengig kildereview bestod. Samme12GiB,
CPU0–7, effekt-/temperaturvakter og900sworker/1200sguard; ingen numeriske eiere endret.
Originale historiske perbatch-masker finnes ikke. Dette er én paret dropout-realisasjon.

MATCHED_MODE_TRAJECTORY_REVIEW 2804f842: originalt gjennomsnittstap INITIAL1,877814,
TRACE1,883131 og FINAL1,891130. Svekkelsen består med lik treningsmodus. De tidlige
kvartalene blir dårligere; de sene bedre. Første kvartal er også dårligere allerede ved
TRACE pre-update enn med faste initialvekter: ikke uniform læring etterfulgt av glemsel.
INDEPENDENT_MODE_TRAJECTORY_CHECK f08a779f bekrefter alle3×256tap bitlikt og179
flyttallsfelt innen2,22e−16. Ingen ny modellkjøring var nødvendig for dette regnskapet.

F/M15 gjennomsnittlig P(UP) øker til70,144% mot faktisk52,083%; F/H1 til71,109% mot
53,969%. Deres Brier/log-tap blir dårligere. F/H4 og F/D1 forbedres i samme korte TRAIN.
SHARED_SIDE_PRESSURE_REVIEW de17a2dc: 99,971% av kvadrert F-endringsenergi på de
fire ikke-M5-horisontene er felles endring innen hver rad; S99,924%. Dette gjelder
endringer i logits, ikke identiske prognoser eller bevist informasjonstap.
Analytisk derivert av det faktiske TRACE-tapet viser motsatt fortegn mellom horisonter
mot samme side-bias i103/256 F-batcher og129/256 S-batcher. F/H4 og F/D1 trekker samlet
oppover på dette koordinatet, M15/H1 nedover; D1 trekker ikke alltid oppover, heller ikke
i siste kvartal. Sentraldifferanse bekrefter derivatet innen1,27e−11. Dette er ett
bias-koordinat, ikke full parametergradient, Adam-attribusjon eller vedtatt tapsendring.

## Lokalisert endring før sluttlaget

[All-TF faktorregnskap](docs/TA_C171_ALL_TF_TAIL_FACTOR_REVIEW.json) er COMPLETE.
ALL_TF_TAIL_FACTOR_REVIEW f46e9509, uavhengig kontroll7e240502. Kun én beholdt NPZ og
to CPU-checkpointlesninger, 0modeller/nativeforwards/autograd/updates; audit4GiB/512MiB.
Alle8kombinasjoner av faktisk beholdt joint_hidden(H), queryprojeksjon(Q) og sluttlag(R)
ble beregnet. Alle5TF, beggeS/F og begge råkanaler er med. Seks rekkefølger og alle
par-/treveisinteraksjoner er rapportert; originale partisjoner256/48/208rader beholdt.
Endepunktsrest maks1,01e−6 i rålogits; identitetsrest under9e−16. Uavhengig NumPy/SciPy
kontroll bekrefter5246flyttallsfelt og666eksaktefelt; faktor-/scoreavvik maks7,55e−15.

På dette eval-panelet øker F/M15–D1 gjennomsnittlig0,958606–0,994707logits. Over alle
seks rekkefølger bidrar H0,524105–0,703829, Q0,1854–0,3402 og R0,084487–0,126114.
Intervallene er rekkefølgeavhengige og ytterpunkter skal ikke summeres. Felles F-sidebias
endrer seg bare+0,002662: en ren biasforklaring av den store forskyvningen avkreftes her.
Det meste av endelig endring ligger før sluttlaget. Liten direkte R-endring frikjenner
ikke delt readout som strukturell kobling mellom oppgavene under læring.

Dette er256avhengige TRAIN-rader, bare48nyoppdaterte. De208andre er ikke heldout.
Tre enkeltklasseceller har korrekt null AUC. Ingen kausalandel for hele4096prefixet,
fullkapasitet, senere generalisering eller økonomisk edge er fastslått.

## Målt tidsrammekondisjonering og minste valgte forsøk

[Beholdt følsomhetskontroll](docs/TA_C171_DECODER_CONDITIONING_REVIEW.json) er COMPLETE.
DECODER_CONDITIONING_REVIEW e8c16451: én NPZ/to CPU-checkpoints, ingen native modell,
forwards, autograd eller updates; audit4GiB/512MiB. Originalt16B16-panel, begge endepunkter,
S/F, alle5TF og begge råkanaler. Analytisk Linear/GELU-Jacobian bekreftes med bundet
sentraldifferanse i to retninger; maksavvik3,17e−10. Originale endepunktsrester og tidligere
finite H-effekter stemmer. Nullvektorer, normer, parvise cosinus og støtter er rapportert.

Felles nonM5-sidefølsomhet er S99,804%→99,840%, F99,760%→99,832% i hele panelet;
alle tre partisjoner overstiger99,7%. Rangeringene mellom nonM5horisontene er også nesten
identiske. Like querykoordinater gir like nonM5utganger ved konstruksjon; denne kontrollen
må ikke brukes som ekstra prediktivt bevis. M5 har separate hoder og andre retninger.
Ingen kapasitetsdom, unikt treningsårsakbevis eller full4096kausalandel følger av dette.

[Valgt opt-in rettelse](docs/TA_NON_M5_SIDE_DECODER_REPAIR.json): fire uavhengige
nonM5-sidehoder perS/F, klonet fra dagens delte siderad med eksisterende RNG-bevarende
helper.16tensorer/1032parametre. M5, aktivitet, query/backbone, alle features/TF og
originalt tap beholdes. StandardFalse bevarer modell-/tapskontrakter og vekter;
dagens view-hash binder de nye kildebytene.
Smal weights-only-migrasjon bevarer M5, modus, gradflagg og normalisering; ingen
optimizer-/cursorhistorie arves. Nye modeller avvises av gamle numeriske oppskrifter før AdamW-konstruksjon.

167fokuserte CPU-tester bestod, exit0, kilde uendret under testen. Ny opt-in kloner
uavhengige16tensorer uten RNG-forskyvning; gammel default bevarer kontrakter/statekeys/
output. M5 og aktivitet er bitlike i decoderfixtures; nonM5startparitet innen1e−6.
PerTFside-perturbasjon og backward treffer bare valgt readout; sharedrepresentasjon får
fortsatt gradient. Originalt masket/vektet tap og rålogitgradient er identiske.
Migrering bevarer source, M5, modus, gradflagg og normalisering; feil avvises før mutasjon.
Gamle statefixture-tester bruker syntetisk optimizerhistorie; ingen markedstrening kjørt.
ROOT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_NON_M5_SIDE_MECHANICAL_20261111_001.
SOURCE_BINDING9b4514d0, FOCUSED_TESTS.log0edd42ee; MECHANICAL_REVIEW har samlet kvittering.

## Faktisk initialparitet for uavhengige sidehoder er verifisert

[Resultat og grenser](docs/TA_NON_M5_INITIAL_PARITY_REVIEW.json) er COMPLETE/CONSUMED.
ROOT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_NON_M5_INITIAL_PARITY_20261111_001.
Kilde d30163ec834ae505a074c8407079a2024e521f64 uendret under kjøring.
PLAN e828f7dd, PREFLIGHT dc719d25, RESULT a4a248d5, TERMINAL d5c1223d,
controller c456075b/exit0.127,77s worker/130,48s controller.211fokuserte tester bestod,
og repoets obligatoriske kontrakttester bestod før commit. Uavhengig kildereview er gjort.

Én faktisk c171 INITIAL-lesning, to modeller,32forwards,16klonede tensorer/1032parametre.
Ingen optimizer, backward, RNG-restaurering, updates eller nye checkpoints/kandidatvekter.
Originalt16B16-panel:256unike TRAIN-rader,512evalueringsrader; bare48tidligere nyoppdatert.
Fase0 gjenskaper originale12felt og16tap byteeksakt. Fase1 bevarer alle akser/targets,
M5 og aktivitet byteeksakt. Øvrige sideutganger avviker maksimalt2,384185791e-7 forS/F,
innen forhåndsbundet atol2e-6/rtol1e-6. Alle16tap er også byteeksakt like, avvik0,0.
Ny initial modellhash2a3847d0adf85d8aade4a2ceed31cab39188ba99f14cb09773a6a79e873e0e1a.

Modell/buffere/RNG/modus/gradflagg/identiteter/hooks er passive. TRAIN_FUSION.npz
6ed85a58 er109304byte. INDEPENDENT_INITIAL_PARITY_REVIEW a6b53635 bekrefter de
beholdte arrayene og tapsskalarene uavhengig med to NPZ-lesninger,0modeller/checkpoints.
Tellere/passivitet er kontrollert fra forseglet produsentkvittering, ikke et nytt liveobjekt.
Vakter uendret:12GiB/512MiB, CPU0–7/pids64,8tråder,250Wlimit/170Wdraw/65C/80C,
900sworker/1200sguard. Et feil prefix-view-tillegg ble rettet før kjøring; originalporten er uendret.

Dette kvalifiserer native oppstart på det lille panelet, ikke fullpopulasjon eller læring.
Den uavhengige headstrukturen er fortsatt opt-in; standardmodellen er uendret.
Læringseffekt, senere generalisering og økonomisk edge er fortsatt ubevist.

## Neste avklaring

[Den nye korte sammenligningen](docs/TA_NON_M5_SHORT_LEARNING_REVIEW.json) er innrømmet én gang
etter kildekvalifisering og trygg boot640→641.348fokuserte tester består; kilde uendret under test.
Eksakt oppskrift,92kildeeiere, faktisk feature-/normaliseringsidentitet og capens
3600/3900sekunders mapping er kontrollert. Kanonisk modul/argument-oppstart stopper
som forventet i audit-vakten før claim/modell; signert native miljø kontrolleres ved launch.
En feil forventet feiltekst i smoke-review er rettet uten prosjekt-/vaktendring; bevart kvittering.

ROOT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_NON_M5_SHORT_LEARNING_20261112_001.
RECIPE0dfe1a15, PREFLIGHTbd03d7bd, FOCUSED_TESTS.log950fcd2c.
Samme4096kronologiske TRAIN-rader/256updates som short002, alle features/targets/tapsvekter
og AdamW-innstillinger beholdt.16klonede nye side-tensorer/1032parametre, fersk tom AdamW,
original INITIAL-RNG eksplisitt importert og original FINAL-RNG kreves. Før første update
må alle4096INITIAL-rader gjenskape den beholdte fullprefixreferansen; alle4096måles igjen
etter læring.256initial+256train+256final=768forwards,2modellkonstruksjoner.

Dette bruker uendret kanonisk trainer fra short002:20GiB/512MiB,CPU0–18/pids128,8tråder,
GPU-minne12288MiB,250Wlimit/310Wdraw/85Ccore;1800smodell/3600svindu/3900sytrevakt.
Det er en annen eksisterende profil enn de lesende produsentkontrollene over; ingen vakt er økt.

Global training/native_training er true kun for denne eksakte engangsplanen; generell
native_launch/full_epoch/full_val er false. Trygg fysisk boot641 er verifisert, med alle92
kildeeiere og artefakter intakte.11gamle active-markører tilhører tidligere boot504–639;
de er bevart, ingen jobb var aktiv. Prepared boot640 beholdes. En ad-hoc preflight-fixture
manglet kanonisk newline; rettet med eksisterende serialisering før plan/claim, ingen kildeendring.
Materialiser nå fra ren committed kilde/policy gjennom eksisterende capped materializer;
krev byteidentisk WINDOW mot CANONICAL_PREFLIGHT_WINDOW. Installer deretter ny
GX1NonM5ShortLearning001 og start én gang på fersk boot641 gjennom signert kontroll.
Ingen gammel plan relanseres.
Analyser beggeS/F og alle5TF på hele bruktperioden: originaltap/komponenter,AUC,Brier,logtap,
P(UP),kronologiske kvartaler og felles nonM5endring. Rapporter også forverringer.
Dette er fortsatt bruktTRAIN, ikke senere generalisering eller økonomisk edge.
Full historikk og30–40epoker er senere skalering.
