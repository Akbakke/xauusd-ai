# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_HEAD_INPUT_COMPLETE_PROJECTION_REPLAY_001_ADMITTED.
Edge-målet er uoppnådd. Alle treningsflagg er false. Ingen GPU-jobb kjører.
Én CPU-only replay er admittert: MARKET_STATE_TA_HEAD_PROJECTION_REPLAY_20261113_001,
PLAN74516173,REPLAY.py09d5889b.2smålagredearkiver,0modeller/checkpoints/encoder/updates;
64FP32Linear+GELU-batcher,16FP64affinegrids,30celler.120s/4G/512M/1tråd.
Alle8hidden/projeksjon/hode-endepunktskombinasjoner,6ordnede stier og symmetriskmiddel;
begge nativepostGELU oglogitdiagonaler må bestå eksisterende1e-6/2e-6-toleranse først.
Metadata-preflight og uavhengig kildekontrollPASS. Neste: én replay, alle resultater, lukk.
Ingen unik historisk årsaksandel eller bedreaccuracy/edge følger av algebraen.

GPU-observasjon MARKET_STATE_TA_HEAD_INPUT_OBSERVATION_20261113_001 er COMPLETE/CONSUMED:
kilde f8a43930,RESULT697b3a53,TERMINALd94ac5d0,VALIDATION5b7989c4.89,90s,
32forwards/2checkpoints/32gathers/12sidecars/0updates. Alle32baseliner ogM5inputbitlike;
modell/inputs/RNG/kilde uendret.20fokuserte tester kvalifisert;20testenes originale kvitteringer bevart.
256klokkevalgte rader,128oktober/128desember. Alle5TF/S/F faktiske hodeinput er nå lagret,
sammen med skjult vektor førquery og smådekoderparametre for viderealgebra utennyGPU.
Desember F/M15-H1-H4-D1 Δlogit1,205/1,313/1,360/1,399; symmetriskrepresentasjon1,128/
1,227/1,268/1,308, hodevekter0,075/0,084/0,091/0,090 ogbias0,001812. Representasjonstermen
dominerer ogsåbeggeordnederegnskap; interaksjon0,120–0,134 gjørandelene rekkefølgeavhengige.
Gammelt hode pånyrepresentasjon girPUP0,735–0,773; nytt hode pågammelrepresentasjon0,514–0,535.
Maksaffinereplay4,93e-7,radclosure1,33e-15. Regnskapetblanderforeløpigbackbone ogqueryprojeksjon;
den admitterte CPUreplayen skillerdisse medalleredeobserverte arrays.
TidligeregenerellM5attribution bef35→1542(RESULT3545cbf2)gjenbrukes;ingen nygenerellM5-probe.

Forrige kontekstprobe er COMPLETE/CONSUMED: docs/TA_CONTEXT_DRIFT_REVIEW.json,
RESULT8f1bf7cf.32bitlikebaseliner,128forwards/0updates. Avgrensning av sekscurrentD1/makrofelt
reduserte non-M5 desemberUP med≈1pp, men månedsskjevheten bare0,192–0,229pp.
Det gjelder denne TRAIN-bound-avgrensningen; ikke generell irrelevans avD1/makro/MTF.

CPU-observasjon: MARKET_STATE_TA_QUERY_INPUT_DRIFT_20261113_001, kildea427db8f,
RESULT2c87d24c,VALIDATION0625774c.256,56s under4G/512M;3lagrede prediksjonsarkiver,
12sidecars,512native featuregathers/8192issue-rader;0modeller/checkpoints/forwards/
updates/nye mål/normaliseringsfit. Alle aktive TA-felt/tidsvinduer/måneder beholdt;
syntetisk accumulatorPASS, månedlig reaggregasjon innen1,12e-16. Ingen modellkodeendring.

M15 S-TRAIN blandet k1/2/3. INITIAL AUC0,765923/0,574648/0,486006; PASS2
0,680804/0,549059/0,491303, kjentstøtte492/515/490. Den native15min-oppgaven er altså
svak allerede påTRAIN; blandetTRAIN≈0,574 mot senere≈0,448 er ikke rent generaliseringsgap.
Kilden bruker ekte M1-oppdatert lokaltilstand og eksplisittTF/k; bareATR holdesfra nativebar.
Ingen fortegn-/klokkefeil funnet;118706M15issues hadde0bit-/M1-radavvik i beholdtparitet.
Gjenbrukt QUERY_REFERENCES RESULT1ef8e55e, samme tape/fullakse:17 753senereissues,
hardpersistens79,0773/57,6298/48,7318%ved5/10/15min på5267/4430/4061begge-kjentepar.
Dette dokumenterer horisontavhengig oppgavevanskelighet. Ulike masker hindrer direkte
modell-mot-persistenspåstand på dagens4096undergrupper. Ingen ny bred tape-jobb nødvendig.

NATIVE INPUTDRIFT: current ctx d1_ema_stack_aligned_v2=+1 gjennomalle29574fitbeslutninger,
normalisert0/ingenvarians. Oktober/november fortsattkonstant; desember endretpå22,91%av
1327panelrader. D1-historikkvinduene inneholder eldre tilstandsvariasjon; modellen har ikke
bare sett én historiskD1tilstand. December EUR_USD_rv_20D ogcurrent D1trendage ligger
100%utenforfrossetfitintervall; XPT-vol75,96% og H4divergenceage88,40%avkonsumertvindu.
Asinhbevarerverdiene utenklipping. Dette er målte kontekst-/fordelingsgap, ikke påvist
normaliseringsfeil eller kausalbevis forUP-drift. Aliasfelter er ikke uavhengige indikatorer.
NativeF TRAIN→senere-gapet består; flere repetisjoner av sammetreningsperiode tilfører
ikke nye aktuelle regimer. Fullkvalifisert historikk/30–40epokers kapasitet er fortsattuprøvd.
Forrige komplette accuracy/ordre-rapport: docs/TA_ORDER_ACCURACY_REVIEW.json.

Native ordre001: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FULL_PREFIX_ORDER_20261113_001.
Kildea31e2448,boot649,RESULTa3ff7bdb,COMPLETION_VERIFICATION160cb198.3698updates,
59148presentasjoner,29574unike mai–september2024-TRAIN-rader i hvert av2pass,0skip.
8075,16s; terminalexit0/vaktPASS; Windows Disabled/LastResult0; ingen aktivinvokasjon.
INITIAL og begge pass-RNG matcher beholdt fullprefix002. Kun batchvisitrekkefølge
43×43-transponert, intakte B16-batcher/tail6 sist; dropout-til-rad/optimizerbane følger med.

Avlesning: /home/andre2/GX1_RUNS/MARKET_STATE_TA_ORDER_ACCURACY_READOUT_20261113_001.
Kilde62a79a05,RESULTb972ad67,REVIEW_VALIDATIONb7c00014.8arkiver,13,04s,
0modeller/checkpoints/forwards/updates.15målrettede syntetiske tester PASS.
Original faktoriseringsfunksjon og kvalifisert scorer/bootstrap gjenbrukt; publiserte
proper scores replayet innen1,12e-15;270matriser kontrollregnet og agentkontroll PASS.
Pass2 primær; pass1 sekundær;70dager/14uker,2000parede trekk hver, punktvise intervaller.

ACC-AVKLARING: Tidligere≈0,51 var AUC, ikke treffprosent; known_side_accuracy var
alltid-LONG-referanse. Ingen regnefeil funnet. Ny faktisk M5F-sideaccuracy865/1689=
51,2137%; balansert50,3929%;AUC0,507776; alltidUP51,9242%. Modellen svarerUP71,3440%,
trefferUP71,7218% ogDOWN29,0640%. Uke-AUC95%[0,48444;0,53239]; sterk skjult rangering
bak0,5-terskelen er ikke påvist. Alle4096 M5F-argmaxprediksjoner erFLAT både INITIAL,
gammelCONTROL og nyePASS1/2:58,7646%klasseaccuracy er nøyaktig alltidFLAT, balansert1/3.
Dette er diagnostisk klasseavlesning, ikke en operativ handelsbeslutning eller tålmodighet.
Fasit2407FLAT,877UP,812DOWN; kjent-side-nevner er ikke valgbar Entry-dekning.
Faste sannsynlighetsgrupper vises ALLE: [.2,.4)224rader/100kjent side/57%treff;
[.4,.6)2553/1038/49,42%;[.6,.8)1319/551/53,54%;yttergruppene tomme. Ingen utvalgsregel.

Ny ordre gir lavere senere Brier-punktestimat i alle10S/F-celler enn gammelordre, men
9/10er fortsatt dårligere enn INITIAL. Kun F/M15Brier og F/M15,H1log skiller seg fra
kontroll i begge blokkeringsintervaller. Ingen AUC-/accuracy-/balansert-endring gjør det.
TRAIN F-AUC M5/M15/H1/H4/D1:0,5652/0,6165/0,7118/0,8028/0,8855, alle overkontroll.
Senere F-AUC:0,5078/0,5050/0,4804/0,5073/0,5499. Et reelt deskriptivt generaliseringsgap;
årsaken er ikke isolert. S/M5 har svak overtilfeldig rangering; S/M15AUC0,4480under0,5
med begge punktvise intervaller. D1S0,6364 og D1F0,5499 har brede intervaller rundt0,5.
Alle10celler/allefaser/TRAIN/LATER og eksplisitte nevnere finnes i rapporten. Ingen
modell-,feature-,mål- eller tapsendring; kun manglende metricavlesning er rettet.

Tidligere fullpopulasjonsordre ER testet: FULL_REVERSE20261110_002 RESULTbe7a403c/
TERMINAL40a284e6, alle29574én gang, eldre ae772-INITIAL. TRAIN-tap2,021683→1,883423;
senere192 M5F Brier0,248827→0,243230, men M5S0,235061→0,238067. Samlet gateFAIL.
Eldre16×16-paring RESULTb11a1932 brukte4096rader/256updates og ga også M5-avveiinger.
Disse gjenbrukes; nytt forsøk er ingen første test av ordre eller generell interleavinggevinst.
CurrentCOVERAGE RESULTd5ba8fa9 endret innhold/repetisjon/ordre samlet. Første4096
spenner allerede over hele mai–september; ingen utvidet kalenderperiode foreslås.

Siste moduskontroll20261113_001 er COMPLETE/CONSUMED: RESULT6275bbfe,
TERMINAL2cb8e04a, MODE_EFFECT_REVIEW873044be;224forwards/0updates.
FINAL non-M5F train-mode meanP er bare0,160–0,178prosentpoeng under eval;
INITIAL→FINAL-driften består med+21,04–22,22prosentpoeng på256TRAINrader.
Ren modusforskjell forklarer ikke den store skjevheten på dette utvalget.
Gradientobservasjon003 RESULTd5efba0d/TERMINAL81432923 viste at FINAL totalgradient
og lagret siste Adam-retning allerede peker ned for non-M5F. Historisk læringsbane
forblir uisolert; ingen generell vedvarende oppgavekonflikt ble påvist her.
Begge observasjoner, tidligere001/002 og numerikkontroll001 er konsumert.
Ingen modell-/feature-/tapsrettelse eller ny edge er etablert av disse observasjonene.

## Nytt resultat

Samlet kilde: docs/TA_FULL_PREFIX_LEARNING_REVIEW.json.
Native root: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FULL_PREFIX_LEARNING_20261112_002.
Frossen kilde2f825e27; fysisk boot648; RECIPE993b9f1d; RESULTcc5d1ba1;
TERMINAL72d686c5; COMPLETION_VERIFICATION2630485c. Windows Disabled/LastResult0.
To komplette kronologiske pass av29574TRAIN-foreldre fra mai–september2024, inkludert tail6.
3698updates/59148presentasjoner/0skip;5234forwards;1modell/optimizer/checkpointdekoding;
0checkpointskrivinger,1privat evidence-kapsel. Ingen kanonisk resume eller promotion.
Samme bef35-INITIAL/kontrollmodell;4096TRAIN/LATER var bitlike kontrollen før updates.
Måling0/1/2, pass2 primær. Native8283,25s; vakt exit0, peak61C/62C/168,26W/2207MiB.

Readout root: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FULL_PREFIX_READOUT_20261113_001.
Frossen kilde0d36ca3f; RESULTf1b0321e, TERMINAL440db398, REVIEW_VALIDATION6bf3c18f.
6arkiver,0modeller/checkpoints/forwards/updates,8,125s. Kvalifisert scorer/bootstrap gjenbrukt.
Alle publiserte proper scores replayet innen1,12e-15; månedsaggregering innen2,23e-16.
70dager/14uker,2000parede trekk per gruppering; punktvise utviklingsintervaller.

TRAIN-panelets totaltap1,864995→1,960094→1,909092. Senere native-tap2,350915→2,639348→2,645637,
mot prior2,316114. Alle10senere AUC-endringsintervaller inkluderer0 i begge grupperinger.
Alle8ikke-D1 S/F-celler har dårligere Brier/log i begge intervaller. D1-endringene er usikre.
M5F AUC0,511290→0,509071; meanPup0,515273→0,600113 mot fasitUP0,519242.
M15/H1/H4/D1 F meanPup blir0,806739/0,816133/0,818379/0,814294, mot fasit0,491060/0,5/
0,539045/0,575089. S/M5 ogS/D1 rangerer over konstantprior på dette gjenbrukte panelet;
S/M15 ligger under. Alle5F-rangeringer mot konstantprior er usikre. Dette er ikke økonomi.

Månedstabellene støtter ikke generell siste-måned-glemsel som tilstrekkelig forklaring:
F/H1 forverres mai–juli og bedres august–september; F/M15/H4 forverres også i september;
F/D1 TRAIN-Brier bedres i alle5måneder. UP-nivået øker bredt, også på senere desemberdata.
TARGET_WEIGHT_REUSE_REVIEW3ed18d86 gjenbruker eksisterende BATCH_COUNTSf51405c6 som JSON.
18inputreferanser,8target-/loss-eiere, numerisk recipe, queryseed og begge1849batchpass matcher.
Ingen nye arrays/modeller. Fullpopulasjonens konstante F-UP-optima er0,511126/0,531247/
0,537731/0,568059/0,641793. Siste256 er UP-rike, men siste16har23kjente F/D1, alleDOWN.
Verken samlet klassevekting eller «de aller siste etikettene var UP» forklarer alene≈0,81.
Kumulativ ordre-, representasjons- og Adam-effekt er fortsatt uisolert.

## Grenser og gjenbruk

Alle genuine features/8familier/5TF/D1 beholdes. TRAIN sampled-S og LATER native-S er ulike
oppgaver; sammenlign endringer innen hvert panel. Kjent side i fasit er ingen valgbar Entry-
dekning. AUC er rangering, ikke treffrate eller økonomi. Alltid-LONG native-F-referansen er
publisert som kjent-side-treffandel; ingen avkastning er målt. Ingen uavhengig generalisering,
Entry-kvalitet eller økonomisk edge er etablert. To pass på fem måneder dømmer ikke full
kvalifisert historikk og30–40epoker. INITIAL-backbone arver2tidligere fulle pass over resetgrenser;
hoder har annen historie. Alternative FINAL-armer er ikke arvet. Combined-variant er ikke promotert.
Feilet klargjøring001 skal ikke relanseres. Tidligere generelt interleaving-forsøk gjenbrukes;
det ga kalibrerings-/M5-avveiing og er ikke grunnlag for enda en generell ordretest.
TEST forseglet; GC på pause; ingen broker/handel/spending/promotion. Originalkvitteringer bevares.
Opprydding: ingen DATA/RUNS-sletting eller generell repo-opprydding i denne bølgen.
