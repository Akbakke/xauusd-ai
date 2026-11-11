# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_MATCHED_MODE_SHARED_HORIZON_PRESSURE_MEASURED.
Målet 11.11 er aktivt: avklar målte årsaker til svak retningsverdi i indikatorer,
sammensetting og læring. [Samlet status](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte runder er feilsøking, ikke kapasitetsdom. Full kvalifisert TRAIN-historikk
og 30–40 epoker er senere skalering, ikke et krav ved hver rettelse.
Tidligere modellplaner og siste lesende kontroll er konsumert. Ingen ny native
modellkjøring er innrømmet. Neste arbeid bruker allerede beholdte mellomresultater.
Training/native_training/native_launch/full_epoch/full_val er false. Begge gamle
c171 Windowsoppgaver er Disabled. GC er på pause; TEST, Entry/Exit, broker,
handel, spending og promotion er stengt. Sluttcheckpoint er evidens, ikke promotion.

## Indikatorer og faktisk læring

Åtte TA-familier og M5/M15/H1/H4/D1 når faktisk modellkonsum, med makrokontekst.
Numerisk dekning og ikke-konstante TRAIN-støtter er bekreftet; dette beviser ikke
prediktiv familienytte. Alle 11 temporal-/akse-encodere og 228 undersøkte tensorer
hadde vektendring og Adam-andremomenter. Mean+last kansellerer ikke systematisk;
tidligere fast MTF-intervensjon ga ingen bedre rangering. Trendlinjeslots mangler
entydig samme-linje-identitet/ankere/endringsmarkør; prediktiv kostnad er UNKNOWN.
Alle genuine features og tidsrammer er bevart. Ingen modell-/tapsendring er valgt.

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

## Neste avklaring

Gjenbruk beholdte joint_hidden/source og faktiske INITIAL/FINAL-dekodertensorer fra
[paret boundary-kontroll](docs/TA_C171_PAIRED_JOINT_BOUNDARY_REVIEW.json).
Utvid eksisterende endelig tail-replay til alle5tidsrammer for S/F og originale
ALL16/UPDATED3/OTHER13. Skill sluttlag, queryprojection og input; kontroller endepunktsrest,
begge rekkefølger og interaksjoner. Ingen ny native modell/fit/gradientprobe er innrømmet.
Panelet har256rader og bare48fra brukt-prefixet: ingen full4096kausalandel eller
kapasitetsdom. Sluttlag-/tapsendring velges først når denne lokaliseringen er kjent.
Ingen ny handelsedge, senere generalisering eller kapasitet ved full trening er påvist.
