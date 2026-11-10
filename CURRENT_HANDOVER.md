# CURRENT_HANDOVER — GX1

Arbeidssted: `/home/andre2/src/GX1_CURRENT`, gren `work/gx1-current`. Les
[GX1_RULES.md](GX1_RULES.md) før arbeid og kjør `bash scripts/gx1_handover.sh --check`.
[NEXT_RUN_POLICY.json](NEXT_RUN_POLICY.json) er eneste maskinlesbare arbeidsstatus;
terminalkvitteringer og faktiske prosesser overstyrer tekst her. Eldre arbeid,
feil og kvitteringer finnes i Git og på sine opprinnelige artefaktstier. Åpne dem
bare ved et konkret avvik eller en ny nødvendig gatekontroll.

## Mål og gjeldende port

Lær løpende framtidig OPP/NED/FLATT markedstilstand **før** Entry-plassering.
M5 er prognosetakten; kausalt lukkede M15/H1/H4/D1 og eksisterende genuine
indikatorhistorikk er input. Alle fem TF har egne framtidige mål. Nåværende
forskningsfokus er SMC og teknisk analyse etter brukerens vedtak 08.11; GC-planen
er utsatt. Entry, Exit, ordre, broker, handel, spending og forseglet TEST er stengt.

Policystatus: TA_M5_PAIRED_OBJECTIVE_TRADEOFF_MEASURED_CAUSE_UNRESOLVED.

[Oppdeling av tapsfunksjonen](docs/TA_M5_PAIRED_OBJECTIVE_DECOMP_REVIEW.json)
er COMPLETE/konsumert; uavhengig kilde- og regnekontroll består. På identiske
4096 TRAIN-rader ble totaltapet 0,02123666 lavere i INTERLEAVED enn CONTROL:
M5-aktivitet −0,00405091, øvrig aktivitet −0,00742633 og øvrig retning
−0,00993436 oppveier svakere M5-retning (+0,00017495). Begge M5-retningsledd
ble dårligere. To lagrede arkiver, ingen modell eller trening; hele kommandoen
tok 0,439 s. FLAT-sonen og eksisterende sannsynligheter er uendret.

Dette forklarer sluttmålenes avveining, ikke den opprinnelige fullpassfeilen
(hvor også totaltapet økte). Ekstra fullpass er tidligere kjørt med korrigerte
M5-hoder og nullstilte Adam-momenter. Sammenhengende gjentatte pass på samme
representative utvalg er ikke målt; native konvergens er ikke dokumentert.
Neste avgrensede spørsmål er treningsdybde med uendret utvalg og mål.
Lagret CONTROL kan bare gjenbrukes til videre trening etter egen kvalifisering;
evidenskapselen har ingen kanonisk resume-autoritet. Ingen ny trening,
modell-/lossendring eller lansering er innrømmet. Alle treningsporter er stengt.

[Paret rekkefølgeforsøk](docs/TA_M5_PAIRED_BATCH_ORDER_REVIEW.json) er COMPLETE/
konsumert på boot636,CUDA250W. Alle trenings-/launchporter er stengt;
training_enabled=false og Windows-task deaktivert. Originale checkpoints bevart.
001 stoppet før modell fordi startvakten krevde12000s; marker/feilkvittering bevart.
002 rettet kun eksakt3000s/3300s-binding og kvalifisert guard-kildebro.
Ingen gammel plan ble relansert,ingen modell-/mål-/optimizerendring.

CONTROL og INTERLEAVED bruker samme originale INITIAL/tommeAdam/RNG og samme
4096 tidsfordelte TRAIN-rader,256 intakteB16-batcher perarm. Originalequeries,
fit_batch_start,loss og femTF beholdt. Innenbatchmedlemskap uendret;
dropout-tilordning til rader endres med rekkefølgen og inngår i effekten.
Én modell/optimizer/checkpointdekoding,to strict restores,512oppdateringer,
1048forwards,0nye baselineforwards,2private evidenskapsler,ingen canonicalresume.

Positivt avgrenset funn: CONTROL bedrer M5betingetS/F mot bådeINITIAL og
frystprior på alle4096TRAIN-radene og separat gjenbrukt senere192-panel.
Brier(laverebedre), INITIAL → CONTROL → INTERLEAVED / prior:
TRAIN S:0,247810→0,247566→0,247834 /0,249377 (1403known).
TRAIN F:0,248937→0,248458→0,248634 /0,249777 (1635known).
Senere S:0,241456→0,237100→0,240540 /0,247188 (65known).
Senere F:0,246894→0,243428→0,244046 /0,249382 (75known).
Logtap har samme fortegn. INTERLEAVED taper motCONTROL i allefire;
allefire forhåndsbundne interleavingporter feiler.

Samlet kanonisk TRAIN-loss faller1,929864→1,869982CONTROL/
1,848745INTERLEAVED. Lavere totalobjektiv gir altså svakereM5 ennCONTROL.
Dette er en målt avveining i dette utvalget,ikke isolert originalfullpassårsak:
CONTROL gjenskaper ikke den opprinneligeM5-forverringen. Datadekning og
treningslengde varierer fortsatt sammen sammenlignet med originalpasset.

FemTF/helprobabilitet begrenser funnet: CONTROL bedrer TRAIN-side9/10celler,
men senerebare4/10. Senere D1 ΔBrier motINITIAL: S+0,028470/F+0,088863
(64/55known); interleaving svekker enda mer. CONTROL F M5-treklassescore
svekkes bådeTRAIN(+0,004924) ogsenere(+0,010399). Bedre betingetUP/DOWN
er derfor ikke generelt bedreDOWN/FLAT/UP. SmåbevegelsersFLAT beholdes.
TRAIN bruker sampledS/nativeF; senere nativeS/F. AndreS-TF er ulikt querydomene.

UavhengigFP64cachedreview:180metric-countceller og12M5priorceller,
maxscoreavvik4,44e-16;6capturer,0modeller/forwards/checkpointdekoding.
Første review feilet fordi modellavhengigrepresentation ble behandlet som
observasjonsakse; bevart/konsumert. Nyreview sammenlignet bare ekte
akser/queries/truth og består. UavhengigJSON-review bekrefter konklusjonen.

Workerens monotontid1362,915s(~22m43s); CONTROLfit372,386s,
INTERLEAVEDfit374,132s:0,47%lengre,ingen hastighetsgevinst.
KampanjensUTC-intervall1339,068s er kortere enn internmonotontid;
ikke bland disse til en eksakt helkjøringsbenchmark. Årsaken er ikke undersøkt.
256baselineforwards gjenbrukt;forberedelse/reboot/review/venting er utenom.
MaksGPU57°C/minne64°C,effektgrense250W,makstrekk133,72W.

Ingen fullfoldrettelse,urørtgeneralisering,kalibrering eller økonomiskedge
er bevist. Målet er uoppnådd. Gjenbruk lagredeprognoser/tilstander for neste
presise spørsmål om datadekning/treningslengde og samsvar mellom totalobjektiv
ogM5-retning. Ingen nyfit eller gjentattrekkefølgetest er innrømmet.

Forrige fullførte kontroll følger nedenfor; dens engangsautorisasjon er stengt.

[Endelig-steg-kontroll](docs/TA_M5_COMMON_STEP_FINITE_REVIEW.json)
er COMPLETE/konsumert, med opprinnelige checkpoints bevart og gjenopprettet.
Alle trenings-/native-/launchporter er stengt; training_enabled=false.
Helkommando265,333s (4m25s), worker230,979s. Forberedelse/review er utenom.
Én modell,to checkpointdekodinger,to reelle parameterpåføringer,48candidateforwards;
0baselineforwards/backwards/optimizer/checkpointskrivinger.
48 ferdige baselineforwards gjenbrukt; ingen kontrollert hastighetsgevinst på
identisk helarbeidsmengde er målt.

Alle tolv lokale M5 S/F-celler bedres i faktisk BCE,Brier og log.
Dette omfatter bare26/28kjente S/F-sider i48anker-rader.
FINAL bedrer begge M5mål også på øvrige144TRAIN og senere192.
INITIAL svekker derimot øvrig-TRAIN F-Brier med3,448e-6 og senere S med1,537e-4.
Alle fire TRAIN192-scorene er fortsatt svakere enn fryst prior.
Senere FINAL F-Brier0,248774 er fortsatt svakere enn originalINITIAL0,246894.

Viktig fem-TF-grense: senere betinget D1-retning svekkes i S/F fra begge tilstander.
INITIAL D1 S/F ΔBrier +0,004244/+0,007585;
FINAL +0,007629/+0,003020 (S64/F55known).
Dette gjelder betinget side; FINAL F-aktivitet og F-treklasse bedres samtidig.
De tolv ankercellene deler observasjoner og er ikke uavhengige bekreftelser.
Ingen generell retensjonsrettelse er kvalifisert.
De senere panelene er gjenbrukt utvikling, ikke urørt generalisering.
Uavhengig cached FP64-scorekontroll0,594s:280side-countceller og56M5fortegn
består; maxBrier/logavvik2,22e-16/3,33e-16. Ingen ny modell/vektdekoding.
To uavhengige kildegjennomganger og separat JSON-regnekontroll består.

Ikke bruk fellesvektene som produksjonskoeffisienter, fortsett Adam-momenter,
relanser prøven eller start nye vekt-/optimizer-sveip.
Den lokale prøven er ikke et krav om at hvert enkelt SGD-steg skal forbedre
alle perioder og TF. Ny eksakt paret rekkefølgetest er avgrenset øverst.
Fullpass-årsak, samlet rettelse,kalibrering og markedspresisjon er fortsatt ubevist.

[Felles lokal retning](docs/TA_M5_COMMON_STEP_REVIEW.json) er eksakt sertifisert
fra lagrede skalartabeller: tre uavhengige Adam-forslag fra samme tilstand
kan kombineres med vektene 0,166679 / 0,337376 / 0,495945.
Alle tolv lokale S/F-derivater (tre perioder ved INITIAL og FINAL) er negative.
Minste sertifiserte margin er 2,28885e-5. Vanlig gjennomsnitt feiler sen S ved
INITIAL. Uavhengig rasjonell primal-/dual-kontroll har eksakt null gap.
Felles vekter betyr forskjellige parametersteg ved de to modelltilstandene.

Beregningen tok 0,157s, med 0 modeller/arrays/checkpoints/forwards/oppdateringer.
Dette beviser bare lokal førsteordensretning innen de tre eksisterende forslagene,
ikke bedre Brier, endelig stegeffekt, kalibrering eller senere generalisering.
Ingen produksjonskoeffisient eller videreføring av Adam-momenter er valgt.

Den separate finite-prøven er nå fullført og konsumert som beskrevet øverst.
Ingen ny epoke eller gjentakelse av gamle planer er innrømmet.


[Cached treningsforløpskontroll](docs/TA_M5_TRAINING_RETENTION_REVIEW.json)
er COMPLETE/konsumert:1,851s helkommando,1,591s worker. Toarkiver/tolvarrays,
0modell/forwards/fit/checkpointdekoding/senerearrays. Globalparitet eksakt;
uavhengig JSON-partisjonsregnskap max1,11e-16. Alle porter stengt.

FINAL svekker S/F Brier og log i hver av de første sju4096-radersblokkene.
Siste902rader bedres mot INITIAL og prior: S-Brier0,247228→0,244145,
F0,249103→0,246959 (S308/F365known). Siste246rader bedres også,
men er et nestet utvalg, ikke ny uavhengig bekreftelse.
Dette er en nylighetsassosiasjon ved INITIAL/FINAL, ikke isolert glemselsårsak.
Lagrede P28672→FINAL-paneler viser siste902updates svekker tidlig/midtre
S/F, bedrer sen S og svekker sen F litt. Ikke påstå at begge sene mål
bedres som umiddelbar følge av sin egen siste trening.

Neste direkte spørsmål er skadelig oppdatering på tvers av perioder.
Gjenbruk eksisterende gradienter og checkpoints; ingen gjentakelse av
konsumerte MIX24/ACCUM3/koordinattester eller ny fullfit er innrømmet.

Brukerpresisering10.11: Små/støyende bevegelser kan være FLATT; terskelen
skal ikke behandles som feil. Lær pålitelige sannsynligheter med det vi har.
Aktiv kode gir allerede sammenhengende DOWN/FLAT/UP-sannsynligheter:
DOWN=a*(1-u), FLAT=1-a, UP=a*u. S og F er separate fremtidsmål.
Ingen ny sannsynlighetsarkitektur eller feature-for-feature-prognose er innrømmet.

Ny presisering av eksisterende treningsbevis: Ved P4096 var bare64/192
faste panelrader besøkt i nativepasset (S23/75, F32/86 known). De øvrige
128radene og nyligtrent område4032:4096 er ikke en måling av nettopp lært
tilpasning. INITIAL var allerede TRAIN-tilpasset. Hele passets forverring
på29574rader/alle femmåneder står derimot fast.
Samme middle-blokk bedres vedP16384/P24576 og svekkes igjen motFINAL.
Dette viser reversering, men isolerer ikke den umiddelbare årsaken.

Cached-kontrollen ovenfor er konsumert; planen kan ikke relanseres.
Treningsporter er fortsatt stengt. Målet om bevist bedre retning er uoppnådd.

[Den sentrerte koordinattesten](docs/TA_M5_CENTERED_HEAD_COORDINATES_REVIEW.json)
er COMPLETE/konsumert på boot634, CUDA250W. Ingen retningsrettelse kvalifisert.
256 oppdateringer/4096rader, 328 forwards hvorav72 evalueringer, én modell/
optimizer, 2 checkpointdekodinger/3 strict restores. Én privat evidenskapsel;
ingen canonical checkpoint eller resume. Alle tekniske paritetskrav består.

Brier kandidat / original P4096 / INITIAL / prior, lavere er bedre:
TRAIN S:0,256475 /0,258153 /0,254605 /0,252092.
TRAIN F:0,253045 /0,254420 /0,252176 /0,250641.
Senere S:0,239123 /0,238781 /0,241456 /0,247188.
Senere F:0,247410 /0,248029 /0,246894 /0,249382.
Logtap gir samme konklusjon. Tre celler bedres mot originaltreningen,
men alle fire forhåndsbestemte krav mot samtlige referanser feiler.
Begge TRAIN-mål og senere F er fortsatt svakere enn INITIAL.
Senere S er litt svakere enn original P4096. Ingen større trening kvalifisert.

Worker559,016s, treningsblokk373,464s, kontroller560,043s.
Original replayblokk360,168s: omtrent3,69% lengre, ingen hastighetsgevinst.
Totalkjøringen har72 ekstra måleforwards; forberedelse og reboot er utenom.
Seks lagrede capturer er scoret uavhengig på0,311s, maksimaltavvik3,33e-16;
0modell/forwards/checkpointdekodinger. Uavhengig kilde-/rapportreviewPASS.
Originale resultater er bevart.
TRAIN192 har75/86kjente S/F-sider, senere192 har65/75; ingen urørt
generalisering eller fullpopulasjon. Windows-task er deaktivert,
treningsporter stengt og engangsplanen konsumert.

Brukeren presiserer at linjer bare var et eksempel på å se fremover.
Dagens S/F har allerede fremtidige sannsynlighetsmål. F under0,5nativeATR
lærer FLATT; brukeren godtar et slikt støyområde. Dette er ingen påvist feil.
Sluttpunkt og prisforløpet underveis holdes atskilt. Ingen ny arkitektur,
målendring eller fit er innrømmet; ingen metode er dokumentert best.
Ikke kjør flere koordinat-/optimizer-varianter på bakgrunn av dette forsøket.

[Head-koordinatkontrollen](docs/TA_M5_HEAD_COORDINATE_GEOMETRY_REVIEW.json)
er COMPLETE på3,303s total;0modeller/forwards/fit/checkpointdekodinger.
Faktisk dekning er29574TRAIN-rader,
S10010/F11909known. Alle lagrede logits gjenskapes innen1,10e-7.
To deskriptive FP64mu0 beregnes på alle INITIAL-TRAIN-rader; ingen senerearrays.

F known-meanlogit øker0,251474: weight·fastmu0 bidrar0,156273,
bias0,002114, midtpunktsrest0,022217 og representasjon0,070870.
S øker0,194193: tilsvarende0,062444/0,002061/0,015502/0,114185.
Vekt–middel-koblingen er derfor materiell forF; forS er upstream større.
INITIAL mean-energyandelen80,15% er uvektet geometri, ikke forklart varians.
I original256-trace kommer omtrent98,5% av fast-mu-headbevegelsen fra
vektene. Både sentrert og lokasjonsterm har likevel gunstig summert lokalt
gradient–steg-produkt. Dette er ikke bevis for skadelig Adam eller en rettelse.
001 importfeil førclaim/data er bevart;002 fikset kun import/bootstrap.
Uavhengig kilde-/JSONreviewPASS, uten ny arraydekoding. Alle porter stengt.

Neste konkrete testdesign er funksjonsekvivalente sentrerte M5-headkoordinater
med fastTRAINmu0, originalINITIAL/tomAdam og original256-stegskontroll.
SammeTF/features/loss/data/RNG/250W; ingen shuffle/LR/moment-/feature-sweep.
Fersk initialparitet og identiske forhåndsbundne TRAIN/senereTRAIN-paneler
motINITIAL/P4096/prior kreves. Den avgrensede testen er nå konsumert med negativ samlet sammenligning som beskrevet øverst.
Positiv korttest er bare en port til bredere verifikasjon; målet er uoppnådd.

[Original treningssporing](docs/TA_M5_ORIGINAL_TRACE_REVIEW.json):
Fullført på fersk boot633, CUDA250W, eksakt256steg/4096rader.
Første loss/norm og alle model-/buffer-/Adam-/RNG-statehasher matcher
originalP4096 eksakt. Samme numeriske treningsforløp er dermed gjenskapt;
ny kildeidentitet og serialiserte checkpointbytes påstås ikke like.
Én modell/optimizer,256forwards/backward/updates,2checkpointdekodinger,
0nye checkpoints/ekstra evalueringer. Original initialevaluering gjenbrukt.
TRACE.npz er bevart med faktiske gradienter, headaktiveringer og Adam-momenter.
Retningsforbedring eller årsaksrettelse er ennå ikke demonstrert.

Worker451,395s, treningsblokk360,168s, kontroller470,229s.
Originalblokk360,693s: ingen materiell kontrollert hastighetsgevinst.
Topptemperatur53°C/minne60°C; effektgrense250W, makstrekk130,46W.
Begge exitkoder0, source_unchanged=true. Windows-task deaktivert.
Engangsplanen er konsumert; alle treningsporter stengt.
Cached algebrakontroll fullført på0,090s;0modeller/forwards/fit.
Gradienter gjenskapes fra faktiske logits/mål/representasjoner innen5,5e-8,
AdamW-parametersteg innen2,4e-9. Ingen fortegns-/maske-/optimizerkodefeil.
Head-only BCE på samme fryste før-representasjon svekkes i60/250S- og
78/255F-steg; gammelt moment er ugunstig i alle disse. Summert head-effekt
på de skiftende treningsbatchene er likevel gunstig for begge mål.
Dette forklarer ikke fullmodellens forverring på et fast datasett og
kvalifiserer ikke fjerning av momentum eller ny optimizer.
Uavhengig kilde-/rapportreviewPASS; ingen ekstra NPZ-dekoding.
Ingen ny modellkjøring eller rettelse er admitted.

[Fullpopulasjonens månedsregnskap](docs/TA_M5_FULLPOP_MONTH_DECOMP_REVIEW.json)
er fullført med gjenbrukte prognoser på29574 TRAIN og17753 senere TRAIN.
Ingen modeller, forwards, tilpasninger, optimizersteg eller checkpointdekoding.
Helkommando2,092s, statistikk0,113s; forberedelse utenom, ingen målt
hastighetsgevinst for identisk arbeidsmengde. Alle porter stengt.

Native FINAL er svakere enn original INITIAL i alle16 måned/mål-celler.
Månedsvis middelavvik utgjør omtrent88-91prosent av økt Brier-feil i de
fire samlede S/F x TRAIN/senere-kombinasjonene. Økt spredning av
sannsynlighetene oppveies bare delvis av bedre samvariasjon med retningen.
Dette er feilregnskap, ikke årsaksandeler eller en modellrettelse.

Senere F: meanUP0,587281 mot observert0,513050; original0,514358.
Brier0,249856→0,256591, AUC0,508323→0,507775. S-ranking svekkes ikke samlet.
Korrigert readout gjenoppretter mye av sannsynlighetskvaliteten, men senere
F taper fortsatt mot prior. Denne readout-rettelsen var allerede målt.

Alle gamle globale scores/counts gjenskapes innen1e-12; alle måneders
vektede dekomponeringer består. Root har lest kilden; ingen separat ny
array-replay eller urørt generalisering påstås.
Rot: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FULLPOP_MONTH_DECOMP_20261110_001
RESULT SHA5ff8f93766d8c1d95225d69309f3e31ca521d9f51574bce70d03eb3ed4d3f03c.

Ikke gjenta fullpopulasjonsscoring, månedsregnskap, affin kalibrering eller
de mislykkede optimizervariantene. Ingen ny modellkjøring innrømmet.
Årsaken til oppdriften og en senere retningsforbedring er fortsatt uavklart.
Featurekontrollens manglende eksplisitte linjeprojeksjon/reaksjonsmål er et
separat designgap, ikke en påvist forklaring på denne treningsfeilen.

[ACCUM3 native-prøven](docs/TA_M5_ACCUM3_NATIVE_REVIEW.json) er fullført og
uavhengig verifisert; alle fire resultatporter feiler. Rot:
 /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_ACCUM3_NATIVE_20261110_001
RESULT SHA a6c3f6f3b492359a9960cfcad4fbf81a6341f97c7bc7a43bd15b22f65ac536cc.
REVIEW SHA d2390b037057aa35831460629c504c6b00dfe342471c745a166aa67b6b2b70dc.
Alle trenings-/engangsporter stengt; training_enabled=false.

Samme original ae772 CPUINITIAL og384SPREAD24-rader/order/RNG som CONTROL24.
24 canonical B16-loss/3-backwards,8 globalclip1/AdamW-steg. Alle fem TFs
side- og aktivitetslæring beholdt. Ingen core-/input-/mål-/samplerendring.
8 mot24 steg endrer også moment/decay/parameterbane; ingen ren
akkumuleringsårsak er isolert. Påfølgende SPREAD-tripletter er ikke de tre
vidt adskilte batchene i den tidligere gradientdiagnosen.

Brier kandidat / CONTROL / CPUINITIAL / prior:
TRAIN S 0,252980 /0,252419 /0,254605 /0,252092.
TRAIN F 0,251241 /0,250667 /0,252176 /0,250641.
Senere S 0,243190 /0,244365 /0,241456 /0,247188.
Senere F 0,247111 /0,247428 /0,246894 /0,249382.
Logtap gir samme konklusjon. TRAIN bedres mindre enn CONTROL og slår ikke
prior. Senere bedres mot CONTROL/prior, men er svakere enn CPUINITIAL.
Dette er mindre senere regresjon mot svakere TRAIN-tilpasning, ingen
kvalifisert felles retningsrettelse. Bare75/86 og65/75 kjente S/F-sider i
de to gjenbrukte192-panelene; ingen fullpopulasjon eller urørt generalisering.

Én modell/optimizer/INITIAL-dekoding/strictrestore,48 forwards,24 backwards,
0 ekstra autograd,8 step-callbacks og8 ekte oppdateringer. Alle570 parametre
oppdatert8 ganger. Alle8 akkumulerte steg er fortsatt klippet.
Første raw/loss og hele RNG-kjeden matcher CONTROL.
Første triplet:1710 readonly hooks, én per parameter/backward; None-mask
eksakt og gradientakkumulering avviker maksimalt2,980e-8. Hooks fjernet før
første steg. Divisjon3 er kildebevis, ikke en ekstra uskalert gradientoracle.

FINAL beholdes i eget privat evidence-capsule-schema:24micro/384rows/
8updates/pending0, faktiske theta/buffere/Adam/RNG/inventar.96,715MB,
SHA5444a64eb69538018c61f875a0fa870b490eaf5e4d7e4a72723c20a90bf7adbd.
Modell5be0bbc49fe399b04d82fa0c87d072500331f1a3f05d78af895a61ece22ff77c.
Canonical recipe1 brukes bare som ekte INITIAL-/optimizer-loss-referanse;
egen execution-route3. Ingen canonical resume eller promotion.

Helkommando357,256s (5m57s), worker353,655s; review7,272s, preflight4,904s.
PeakRSS5,725GB. Ingen baseline-/kontrollforward kjørt på nytt.
To weights_only-dekodinger i review, ingen modell/forward/optimizer.
Uavhengig scoreavvik1,111e-16. Forberedelse/venting utenom; ingen målt
akselerasjon av identisk arbeidsmengde.

Ingen ACCUM3-videreføring, fullfit eller accum-/LR-/momentumsveip.
Neste steg bruker lagrede original INITIAL/FINAL-fullpopulasjonsoutputs og
konvergert readout-bevis: avklar hva som allerede er målt per måned, og skill
periodevis middelavvik fra sannsynlighetsspredning og måltilpasning.
Ingen ny modellkjøring før en særskilt gjenværende mekanisme er isolert.

[AdamW-kryssperiodekontrollen](docs/TA_M5_ADAM_CROSS_PERIOD_REVIEW.json)
er fortsatt korrekt lokal evidens, men denne native prøven kvalifiserte
ingen fullpass-årsak eller retningsrettelse.
Originale checkpoints bevares. TEST, GC, Entry/Exit og handel er stengt.

MIX24 er fullført og uavhengig verifisert:
[docs/TA_M5_MIXED_BATCH_NATIVE_REVIEW.json](docs/TA_M5_MIXED_BATCH_NATIVE_REVIEW.json).
Rot: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_MIXED_BATCH_NATIVE_20261109_001
RESULT SHA6895ee933974dae209c04251719033ffdff0ab70dbe1179e529a80cc60767cea.
REVIEW SHAd4a3bbcd3c9fdce924f6400b76944f141d8227a01e1dcf7bed9a7b87ca73fdfb.

Samme INITIAL0, eksakt samme 384 observasjoner som SPREAD24, 24 oppdateringer,
48 forwards, ingen omkjøring av start eller kontroll. En 24x16-transponering
blandet tidspunktene innen batch; hver featurehistorikk forble kronologisk.
MIX24 er verre enn SPREAD24 i alle fire S/F x TRAIN/senere-TRAIN-celler.
F-Brier TRAIN: 0,250950 mot 0,250698; senere: 0,249055 mot 0,247459.
Begge TRAIN-mål bedres fra INITIAL0, men begge senere-mål forverres.
Ingen celle består kravet mot alle fire referanser. Ingen rettelse kvalifisert.

Helkommando 341,566 s mot SPREAD24 343,051 s; omtrent 0,43 prosent forskjell
er ingen vesentlig hastighetsgevinst. Worker 338,398 s; cachedreview 4,939 s.
Tre saved states, faktisk fullpermutasjon og 384-forekomstsett, start/slutt-RNG,
capture-akser og metrikker kontrollert; største scoreavvik 3,331e-16.
Ingen modell eller optimizer-replay i review. Kun 75/86 og 65/75 kjente
S/F-sider i de to gjenbrukte 192-raderspanelene; ingen fersk generalisering.

Minste API-utvidelse fit_batch_positions binder 16 unike int64-posisjoner
til originale S/F-akser med eksakt parent/tidspunkt. Gammel rute bevart;
query/forward/mål/tap byteidentiske. 50 fokuserte tester, metadata-preflight
og obligatoriske commitkontroller bestått. 84 av 85 runtime-kildeeiere
uendret; én eksplisitt gammel/ny kildebro. Featureauditens 35 øvrige eiere
er fortsatt hashidentiske. Modellens ONLINE-funksjon er uendret.
Gruppering endrer også dropout-tilordning og aktivitet/øvrige TFs batchvekter;
forsøket isolerer ikke bare gradientstøy. Alle porter er stengt, planen konsumert.
Ingen ytterligere sampler-/optimizer-sveip. Neste nye hypotese må bindes
til konkrete retningsmål; fullTRAIN-forverringen er fortsatt uforklart.

Cached hode-/representasjonskontroll er COMPLETE og uavhengig verifisert:
[docs/TA_M5_HEAD_REP_ATTRIBUTION_REVIEW.json](docs/TA_M5_HEAD_REP_ATTRIBUTION_REVIEW.json).
Rot: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_TRAJECTORY_HEAD_REP_ATTRIBUTION_20261109_001
RESULT SHA31e1eb1b7cb07360910ec28c5851049b62da9f9787f9da299dd9056047f50f9d.
REVIEW SHA30561999f3e18f82ba9ead35a7a98368ae468ce013b8209601def228636d6b91.
50 NumPy-affine kombinasjoner fra9bevarte checkpoints ogsamme192TRAIN-rader.
0modeller/backboneforwards/fits/backwards/optimizer. Ni headuttrekk, ikke
nye checkpoints. Native rålogitparitet max8,145e-8; uavhengig replay med
annen sumrekkefølge max2,776e-16,metrikker max3,331e-16. Helkommando15,909s,
worker13,024s,review0,377s,peakRSS0,724GB. Forberedelse ikke medregnet.

F-Brier Shapley er positiv for head8/8 og samlet oppstrøm7/8; P012288 har
oppstrøm-0,000514. FINAL Brier:INITIAL0,252176,bare nyrepresentasjon0,254368,
bare nytthead0,258727,begge0,266276;frystprior0,250641. Ingen enkelt
tilbakebytte kvalifiserer. FINAL to-faktor Shapley:head+0,009230,
representasjon+0,004870; dette er ikke optimizerårsak eller tapsandelbevis.
Tre64-raders perioder:FINALF innenperiode sentrertfeil bedres0,002891,
men vektet periodemiddelfeil forverres0,016991. P024576 har tilsvarende
-0,003774/+0,016262. Dette er realisert liten TRAIN-statistikk, ikke sanne
regime-sannsynligheter eller en deployerbar kalibrering. Ingen senereforbedring.

Tidligere fullTRAIN HEAD_DRIFT/FINAL_CALIBRATION/TRAIN_MONTHS er gjenbrukt:
global affin kalibrering reparerer myeTRAIN, men kvalifiserer ikke senereF.
TRAIN_MONTHS gjelder korrigertreadout, ikke nativeINITIAL/FINAL-periodedecomp.

Den fullførte MIX24-prøven ovenfor fant ingen gevinst fra batchgrupperingen.
Featurehistorikker forblir kronologiske. Grupperingen endrer også dropout-
tilordning og aktivitets-/øvrige-TF-tapsvekting innen batch; den isolerer
ikke bare gradientstøy. Ingen gammel engangsplan relanseres.

Ni bevarte fullpass-tilstander er nå kontrollert på de samme192 TRAIN-radene:
0:64,14272:14336,29504:29568. INITIAL ble gjenbrukt; åtte andre målt én gang.
Resultat og uavhengig kontroll:
[docs/TA_M5_FULLPASS_TRAJECTORY_REVIEW.json](docs/TA_M5_FULLPASS_TRAJECTORY_REVIEW.json).
Rot: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FULLPASS_TRAJECTORY_20261109_002
RESULT SHA02b686f0ae9a864892f9ac9500f1dd54110a654c708bc3d484ffbdac4a4df113.
REVIEW SHAcc7a642c965e8e388b6f1f89ec6b865b0c3c99d4ea3e54e3e3c8af6f7eca0992.
Scorer regnet uavhengig fra NPZ; største tallavvik2,22e-16. Outputparitet
INITIAL/FINAL består ny privat CPU/CUDA-port. Original001hidden-port er
fortsatt FAILED; internrepresentasjoner ble aldri transplantert.

F på dette utvalget: INITIAL Brier0,252176→første mellompunkt etter256updates
0,254420→FINAL0,266276; fryst prior0,250641. Alle åtte senere lagrede F-punkter
er verre enn INITIAL og prior. S forbedres enkelte steder, men ender verre.
Dette lokaliserer ikke de første skadelige oppdateringene før checkpoint256.
Kun75S/86Fknown-side mål av192; ingen checkpoint er valgt.

Bare gjennomsnittlig OPP-skjevhet forklarer ikke feilen: P024576 har F-mean
0,474798 mot faktisk oppandel41/86=0,476744, men Brier0,264664. Det sentrerte
feilleddet er derfor også svekket. Dette er ikke alene et rent rankingbevis.
Siste16batchers oppandel versus fastpanel-mean har Pearson S0,062/F0,348
overåtte punkter: beskrivende, ingen isolert årsak eller avvisning av alle
tidsregimeforklaringer. Ingen rettelse eller senere forbedring er kvalifisert.

002 worker297,157s,helkommando300,090s; setup77,018s,peakRSS5,726GB.
001 stoppet etter INITIAL,helkommando112,231s; begge modellkommandoer
tilsammen412,321s(6m52,3s). Cachedreviews1,536s+3,014s. Forberedelse er ikke
med i disse tidene; dette er ingen sammenliknbar trenings-speedupmåling.
00296forwards/8loads +gjenbrukt00112/1 =108/9;0fits/backwards/optimizer.
Ingen senereTRAIN eller TEST. Alle launch-/treningsporter er stengt.

Cachedattribusjon, MIX24 og moduskontrast er fullført; ingen ny kjøring er innrømmet.

Brukerens all-feature-kontroll er fullført i
[docs/TA_FORWARD_SEMANTICS_REVIEW.json](docs/TA_FORWARD_SEMANTICS_REVIEW.json):
701feltdefinisjoner/59grupper,8familier+61D1/makrofelt,ingen dekningshull.
Gjenbrukt konsumkontroll er bundet til uendrede aktive kilder. De fleste
features er nåtilstand/historikk; S/F gir framtidige mål. HTF-linjer er ikke
eksplisitt forlenget til aktuell M5-klokke/t+h, og linjespesifikke
kontakt/sprett/bruddmål mangler. F under0,5ATR trener aktivitet/FLAT, ikke side.
Historisk COT/makrovintage er fortsatt UNKNOWN. Ingen ny produsentfeil eller
fungerende retningsrettelse ble påvist. Ikke gjenta beståtte featurekontroller.
Alle launch- og treningsporter er stengt.

Siste læringsbevis beholdes: full29574 TRAIN fryst eval-tap1,948957162→
2,021683248; alle10sidekomponenter verre, aktivitet bedre. M5 Brier S
0,248473364→0,251117834 og F0,249327011→0,253903942. Importert
INITIALae772f og FINAL23ff34, sampledS/nativeF; ikke historisk dropoutreplay.
Resultat og uavhengig review finnes i
/home/andre2/GX1_RUNS/MARKET_STATE_TA_SF_OBJECTIVE_WEIGHT_AUDIT_20261109_001.
Det er fortsatt ingen isolert årsak til fullpassets forverring eller
dokumentert bedre retning på senere data. Målet er uoppnådd.

Forholdet mellom totalt ogprimærtmål er allerede sett iendpointmålingene:
SPREAD−CONTROL senere total−.055551840, mens M5-sidebidrag+.000295557.
AndreTFside−.031941814 ogaktivitet−.023905583 driver bedringen i totalen.
PåTRAIN INITIAL0→SPREAD kommer−.193691039 fraandreTFside,−.002054836
fraM5side og+.009448011 fraaktivitet. Dette er endpoint-evaluering på192,
ikke den faktiske samplede treningsloss; den nye kontrollen holder dem atskilt.

SPREAD24 er COMPLETE/konsumert; prosessen er bekreftet avsluttet og alle
trenings-/launchporter er stengt. Hel kommando343.051s(5m43.05s),
worker339.897s,peakRSS5.725GB.24updates/48forwards/24backwards,0skips,
én modell/optimizer/checkpointload og2strictrestores. CONTROL24 gjenbrukt.

Rot: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_SPREAD_NATIVE_20261109_001
RESULT SHAf0dd7678c6aed01a3add8c03ed19759c30be9e5bb3050ff0a09189f90224f35d.
REVIEW SHAc6b4d07fcd7995edc64b5374f221aedeac042e837af921bf2832896b9a58b7de.
Uavhengig review1.562s PASS_NEGATIVE_COMPARISON; maks scoreavvik3.331e-16.
Original→START endrer bare parent_order/source/seal. Full29574unik
permutasjon,384valgte prefixrader/29190ukonsumert suffix,alle24posisjoner
ogFINALcursor384/updates24 er kontrollert. Faktisk RNG matcher CONTROL.
Per-batchrecords ble ikke publisert av worker; faktisk gather bygger på
fryst kodeassert+COMPLETE, mens lagret fullordre/prefix er fysisk kontrollert.

Brier(laverebedre),SPREAD/CONTROL24/INITIAL0/prior:
TRAIN S .250228893/.250043408/.252926351/.252092094.
TRAIN F .250698470/.250848535/.253125394/.250641150.
Senere S .244415820/.244100787/.241010430/.247188030.
Senere F .247458543/.247039903/.246989478/.249382136.
Samme75/86kjenteTRAIN-sider og65/75senere sider. Alle fire krav feiler:
TRAIN-S svakereCONTROL; TRAIN-F bedreCONTROL men svakereprior; begge
senere mål svakereCONTROL ogINITIAL0. Begge evalueringscapturer er bevart.

Konklusjon: mer tidsmessig spredning ved samme24updates er ikke en påvist
rettelse. Ingen større trening eller permanent modellendring er kvalifisert.
Dette er en negativ liten sammenligning, ikke en generell dom over modellen.
Forholdet mellom faktisk samlet treningsmål og M5-retning er nå kontrollert
på lagrede full-TRAIN-komponenter, se siste resultat øverst. Tidligere
gradientbevis beholdes; de isolerer ikke hele forløpets årsak.
Ikke gjenta de konsumerte moment-/rekkefølgevariantene eller øk epoker.
Målet er fortsatt uoppnådd. TEST/Entry/Exit/handel er stengt.

M5-fasit/vekt-kontrollen er COMPLETE/konsumert,2.064s uten modell/targettape.
Uavhengig54celle-kontroll1.158s PASS; maksavvik2.665e-15. FullTRAINN/q,
sampletS k=1/native-likhet,24batcher og frosneINITIAL0-residualer stemmer.
PREFIX_TARGET_RESULT SHA498a6f3b076f54f6b39cfe2025c39f380796c7742bc62abbd5985c481cf4a522.
PREFIX_TARGET_REVIEW SHAfef19cce9388bd9669ac14b89afc6840a54be0dadfd32c69226fc910cf0bdd8f.
De384konsumerte observasjonene dekker 2024-05-01T00:05:00+00:00 til 2024-05-02T09:05:00+00:00.

Faktiske M5 OPP/side-kjent:
første16: S55/101(54.46%),F66/122(54.10%);
neste8: S18/38(47.37%),F21/52(40.38%);
alle24: S73/139(52.52%),F87/174(50.00%).
Koeffisient per kjent side S1799/(5×10010),F1844/(5×11909).
FrossenINITIAL0-biasderivert er negativ(favorisererOPP) i første16 og
positiv(favorisererNED) i neste8. Dette er evalreferanse, ikke replay av
faktisk treningsdropout/gradient. Ingen query-, maske- eller vektbindingsfeil
funnet; det forklarer ikke hele den delte representasjonens NED-forskyvning.

Den spredte24batch-hypotesen er nå utført og avkreftet som rettelse på
disse faste utvalgene, se resultatet øverst. Tidligere09.11order/accumulation
var frozen-head/0native; denne testen brukte faktisk hele native modellen.

Lagret shift/rank-kontroll COMPLETE/konsumert:0.135s,0forwards/updates/
checkpoints/refits. Alternativ skalar/rank-sum-kontroll0.029s,56scoreceller/
48kontraster, maksavvik1.11e-16. Alle tre faste blokker inngår.
Samme parede rot nedenfor: SHIFT_RANK_RESULT.json
SHA94dfc72a441b47bae539dffc9fa86a21d71ce9d2096c136a85e434422c688255.
SHIFT_RANK_REVIEW.json bevarer kontrollen. Denne kontrollen brukte ingen modell.

Senere CONTROL vs INITIAL0, eksakt Brier-dekomponering:
S total+0.003090357 = gjennomsnitt+0.003723185 + sentrert−0.000632828.
F total+0.000050424 = gjennomsnitt+0.000598355 + sentrert−0.000547931.
Gjennomsnittlig P(OPP) faller2.391/1.376prosentpoeng. RESET skyver videre
0.269/0.356prosentpoeng ned; samlet senere tap kommer også her fra dette
gjennomsnittsleddet. TRAIN-utvalget har45.33/47.67%OPP; senere58.46/53.33%.
Dette forklarer skåren algebraisk, ikke den unike årsaken i treningskoden.

Samlet senere AUC øker med CONTROL, S0.642300→0.663743,
F0.586429→0.592143. Dette er IKKE jevn forbedring innad i periodene:
S faller i2av3blokker og er lik i siste; F faller i første og øker i to.
Parvektet innen-blokk-AUC er S0.669211→0.659033 og F0.615711→0.622081.
Ingen kalibrator er tilpasset, ingen terskel eller ny modell valgt.

Den parede testen er COMPLETE/konsumert. Prosessen er bekreftet avsluttet;
alle trenings-/launchporter er stengt. Hel kommando353.792s (5m53.79s),
worker350.398s,peakRSS5.723GB. Ingen større kjøring er innrømmet.

CONTROL/RESET gjenopptok samme FINAL16 og trente de samme neste8batchene
hver:128rader, cursor256→384,16nye oppdateringer totalt,0skips.
64forwards,16backwards,4fullrestores; modell/optimizer bygget én gang.
Kun RESET nuller fire M5-hode exp_avg én gang. RNG er identisk ved start
og slutt. Originalcheckpoint og begge sluttstates/captures er bevart.
Alle genuine fem TF og kanonisk loss/optimizer er uendret.

Rot: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_HEAD_MOMENT_RESET_20261109_001
RESULT SHA375b372e66dcc66d6c10902152e9bacd438937f3ddbc96e253d5784f1b376b79.
Uavhengig kontroll PASS på3.481s, uten modellforward/refit. Fire momentfelt,
øvrig startstate, faktisk RNG og tellinger er kontrollert fra lagrede states.
M5 Brier/log reprodusert med maksavvik3.331e-16.
REVIEW SHA8617526a2ee26f0ce3ad42e292446b1d7d415c9ebea0601cf7ca9dd227a64a3a.

Brier (lavere er bedre),192TRAIN-rader med75S/86Fkjente sider:
S CONTROL0.250043408 → RESET0.249541961, prior0.252092094.
F CONTROL0.250848535 → RESET0.250518353, prior0.250641150.
RESET slår også POST16 og INITIAL0 og forbedrer logloss for begge.
Den forhåndsbestemte TRAIN-porten bestod; deretter ble begge armer målt
på192senere TRAIN-rader, med65S/75Fkjente sider.

Senere Brier:
S CONTROL0.244100787 → RESET0.244557725; INITIAL0=0.241010430,
prior0.247188030.
F CONTROL0.247039903 → RESET0.247192453; INITIAL0=0.246989478,
prior0.249382136.
RESET er altså dårligere enn CONTROL og INITIAL0 i begge senere mål,
også logloss. Begge armer slår prior i dette lille utvalget. Senere TRAIN
er gjenbrukt utvikling; dette er ikke uavhengig generalisering eller edge.

Konklusjon: den målte momentintervensjonen forbedrer TRAIN, men gevinsten
overføres ikke til det senere utvalget. Ingen permanent momentreset eller
større trening begrunnes av resultatet. Shift/rank-spørsmålet er nå tallfestet ovenfor.
Ikke gjenta de konsumerte armene.
Målet om bedre retningsprediksjon er fortsatt uoppnådd. TEST/handel stengt.

Hidden-kontrollen er COMPLETE/konsumert og numerisk verifisert; ingen jobb
kjører og alle porter er stengt.24encoderforwards,0optimizer/backwards/fit/
targettapes. Hel kommando130.491s (2m10.49s),worker127.215s,peakRSS5.723GB.
Rot: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_HIDDEN_SPLIT_20261109_001
RESULT SHA7c0c4a80f179381d96d6439597809ecfeba72f674cfda890ff52095bbf5a5486.
REVIEW SHAa08b267d6f286f210d030fd7807e69e7e91f65ceb2fbb4f378c5512e2a9891c0.
INITIAL_HIDDEN/FINAL_HIDDEN og8-kube er bevart.2.262s alternativ kontroll
av88scoreceller/direktepar-AUC/subset-Shapley: maks1.11e-16. Uavhengig
NumPy+math.erfFP64-decoder har maks1.706e-7logitavvik. Dette er alternativ
numerisk kontroll av integrator, ikke encoder-/optimizerreplay.
Faktisk tidligere capture matches innen1.192e-6repr/4.470e-8side-logit.

Siste TRAIN-blokk,3faktor-Shapley Brier:
S backbone+0.003734302,query+0.001017981,head−0.000532019.
F backbone+0.003154375,query+0.000828557,head−0.000040178.
Backbone står for mesteparten av representasjonstapet, også F-AUC-tapet.
Shapley bruker6rekkefølger og er ikke samme interaksjonsfordeling som
forrige2x2; diagonalscorene samsvarer. Ingen av8frosne kombinasjoner slår
F-prior samlet på192rader; lavestF Brier0.251015137,prior0.250641150.
Det er dermed ikke grunnlag for et enkelt komponentbytte/frys.

De faktisk trente256radene dekker2024-05-01T00:05–22:25UTC.
M5 OPP-andel55/101=54.46%S og66/122=54.10%F, mot heleTRAIN51.87%/51.11%.
Den målte bearish forskyvningen er altså ikke forklart av en DOWN-majoritet
i disse M5-labelene. Dette er beskrivende, ikke unik optimizerårsak.

Neste konkrete kontroll er faktisk delt gradient vedFINAL16, på samme
3TRAIN-batcher[0,14272,29552] som den ferdigeINITIAL-diagnosen. Gjenbruk
INITIAL-resultat/gradienter; ikke kjør baseline om igjen. Sammequeries,
vekting, CPUdropoutfrø og aktive parametergrupper; kun checkpoint endres.
Kvalifiser maks6forwards/21autograd,0optimizer/updates,600s/12G/tråd1.
Spørsmålet er om andre genuine mål nå motarbeider M5 i faktisk backbone/
querygradient, siden INITIAL-kontrollen var gunstig. Ingen slik jobb er
implementert/innrømmet ennå. Ikke bruk detached-hidden-proxy som svar,
endre tapsvekter eller frys moduler ut fra attribusjonen alene.

Den lagrede kontrasten er COMPLETE/konsumert og numerisk verifisert.
CONTRAST_V2_RESULT SHA57c407854940af56eb5c3c49ffb5cdd3a5a24cdf0787359f69902e128c59d2e2.
CONTRAST_V2_REVIEW SHA2bf702b63a8780b9831122a16a89a69576ed6c79a8caa3ca292ac861f83b425d.
Worker3.032s/hel capped kommando6.111s; review2.355s.70scoreceller er regnet
om via separat radvis FP64-projeksjon og direkte par-AUC for192rader;
maksavvik4.22e-15. Samme integrator gjorde alternativ numerisk kontroll,
ikke ny agentreview. FullTRAIN-AUC er ikke separat omregnet.0fit/forward,
ingen senerearrays/TEST. PID96001 er avsluttet; alle porter er stengt.

Ny konkret observasjon: ved16updates kommer sisteblokktapet fra pre-head
representasjon, mens hodeendringen delvis motvirker tapet.
S Brierbidrag head−0.000531816,representasjon+0.004752079,total+0.004220263.
F head−0.000039948,representasjon+0.003982702,total+0.003942754.
2x2-interaksjonen er halvfordelt; dette beskriver faktiske fryste
kombinasjoner, ikke en unik optimizerårsak.

Sisteblokk OPP-andel er70.59%S(n17)/68.18%F(n22), men modellens meanOPP
faller51.54→50.66%S og49.09→48.41%F. Bias²-endringen forklarer omtrent81%S
og66%F av Brierøkningen; resten er svakere kovarians/margener minus en
liten variansreduksjon. S-AUC er0.60→0.60, F0.61905→0.56190. Dette er små
TRAIN-utvalg, ikke bevis for et generelt regime eller senere edge.
Over192rader forbedres både score og AUC; forbedringen er dermed ikke
utelukkende en felles biasendring. Felles labelfri logitforskyvning er
−0.060604S/−0.040969F; den diagnostiske forskyvningen er ikke en valgt modell.

HeleTRAIN, uendretINITIAL-representasjon og bare faktisk0/16hode:
S Brier0.248201726→0.248207830;F0.249076427→0.249079063.
Hodedriften er liten etter16steg og har motsatt sannsynlighetsretning:
meanOPP øker0.518867→0.521427S/0.511068→0.512367F.
Dette er ikke faktisk global16-stegsmodellscore. Eldre256–1849hodebane
skal ikke brukes som forklaring på tidlig representasjonsendring alene.

Backbone/query-skillet er nå målt i den fullførte hidden-kontrollen øverst.
Den konsumerte planen skal ikke relanseres. Alle hidden-/kubeoutputs
bevares for billig videreanalyse uten nye encoderpasseringer.

Originalkontrasten feilet/konsumerte på representasjons-allclose.
Målt CPU/GPU-avvik2.927e-5 i én av24576koordinater, RMS3.939e-6;
begge faktiske hoder påvirkes <=2.67e-6logit. V2 beholder original
allclose=false, reproduserer avviket og består eksisterende2e-5-logitgrense.
Originale kvitteringer bevares; ingen modell-/scoreformel er endret.

Den nye korte CPU-prøven er COMPLETE/konsumert; controllerexit0 og PID95085
er avsluttet. Alle treningsporter er stengt igjen. Ingen automatisk
fortsettelse eller relansering. Uavhengig resultatkontroll PASS på3.031s.

Rot: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_SHORT_NATIVE_20261109_001
PLAN SHA126c9fb6b1059576870145ec9fe5ce652dd7791f5a39964665e438d65e79b15e.
RESULT SHA11937f7a4d180d88e06e298aa0ed368e5529a398d7ab8df51f8801c19e426376.
Kilde931021de;85kanoniske kildeeiere uendret.
16faktiske optimizersteg/256rader,40forwards,16backwards,0skips.
INITIAL_CPU_STATE og FINAL_CPU_STATE bevares med sann cursor0/256.
Ny CPU-RNG, ikke CUDA-replay; uendret modell/loss/AdamW/klipping.

Hel capped kommando289.5665s (4m49.57s),worker286.1096s,under600s.
Dette er vesentlig mindre arbeid enn fullpass, ikke en identisk-workload
speedup.192faste TRAIN-rader er målt ferskt før/etter:
S Brier0.252926351→0.250048769,prior0.252092094 (PASS).
F Brier0.253125394→0.251207498,prior0.250641150 (FAIL).
Begge forbedres mot INITIAL; F slår ennå ikke referansen. Derfor samlet
TRAIN-gateFAIL og ingen senere score-arrays/worker-score-gather/forward.
Kanoniske loadere dekoder hele TRAIN-targetbåndet for eksisterende
validering; dette er ikke senere modellscoring. TEST er ikke åpnet.

REVIEW SHA1cfb6be1e258df9a51a0e67d06ef8bcb85e451e749829319f93ba6af6614c341.
Uavhengig capture-regning har maks2.22e-16 metrikkavvik. Faktiske
CPU-checkpointseals, cursor/updates,29574-raders orden og AdamW er kontrollert;
ingen ny modellforward, fit eller senerearrays. Optimizerhistorien er ikke
replayet; kildekontroll og lagret sluttstate underbygger stegtellingen.

Begge mål forbedres i første og midtre64-raders TRAIN-blokk, men begge
svekkes i siste64. S Brier før→etter:
0.255763→0.247873(n23),0.257462→0.254431(n35),0.239750→0.243970(n17).
F:0.258842→0.252083(n32),0.249267→0.248161(n32),0.250423→0.254366(n22).
Begge slårprior i første/midtre blokk, taper i siste. Dette er en liten
tidskontrast innen TRAIN, ikke bevis for en generell regimesammenheng.

Den lagrede tidskontrasten er nå undersøkt og lokalisert som beskrevet
øverst. Ingen større fit er innrømmet. Kortløpet beviser ikke global
TRAIN-forbedring, generaliserbar retning eller økonomisk edge.

Den lagrede hodebanen er allerede COMPLETE og numerisk kontrollert:
15.692s,9checkpointlesinger,0forward/fit. Alle8lagrede hoder etterINITIAL
er svakere på samme gamle representasjon; ved256updates:
S Brier0.248201726→0.248591188;F0.249076427→0.250124852.
Dette lokaliserer tidlig hodedrift, ikke faktisk mellomliggende
helmodellscore eller én bevist optimizerårsak. Ikke gjenta kontrollen.

Null-update gradientdiagnosen er COMPLETE og uavhengig PASS.
220.756s worker/224.259s hel kommando,peakRSS7.316GB;6forwards/21autograd,
0optimizer/updates og bitlike modell/buffere. Evalparitet maks1.993e-6,
gradientsum maks1.192e-7. Aggregate M5train/evalcosine0.99815/0.99784.
Joint førsteAdamW-retning har negativ lokal evalBCE-endring for bådeS/F i
alle3batcher, også i backbone/query/M5hoder. Ingen initial antagonisme eller
dropout-fortegnsforklaring er påvist. Ingen Brier-/generaliseringsgevinst.
RESULT SHAf9876403a6ebc6881df9cd6164f199623a238bd07de0b257a6a2474d1b9b104d,
REVIEW SHA221b1030e9c1c336e5dbeb8fca1570946ee6398f26ef4561c71758cc8926527d.
Review3.724s gjenskaper alle5aggregatepartisjoner med maks2.085e-13 avvik.

Den ovenfor avsluttede hodebanen brukte HEAD_TRAJECTORY_PLAN i samme rot,
60s/4G. Ingen encoderforward,fit,senerearrays eller checkpointvalg.
Planen er konsumert og kan ikke relanseres.

Den fullførte null-update diagnosens konsumerte ramme:
MARKET_STATE_TA_M5_SHARED_GRADIENT_20261109_001, PLAN
SHA63097b54609ab3b6c3144e4190a708a513a88aaa9765cf54f00c42e1b4a32460.
Tre faste TRAIN-batcher/48rader,6fulle CPU-forwards,21autogradkall,
0optimizerkonstruksjoner/updates.12G/512M,én numerisktråd,600s.
KorrigertINITIAL og alle genuine5TF beholdes. Eval ogtrain bruker samme
sampledS/nativeF-query; M5-evalparitet måles mot lagrede korrekte prognoser.
Firetraingrupper til faktisk aktiv backbone/query/hoder, pluss to eval-M5-
gradienter, skiller målinterferens fra mulig train/eval-forskjell.
CPUdropout er egen frøbundet realisasjon, ikke CUDA-replay. Dotprodukter er
lokal førsteordens BCE-effekt, ikke anvendt steg eller målt Brier-gevinst.

Preflight og avgrenset uavhengig kildekontroll er fullført.85kildeeiere er
uendret; taps-/gradientsum består (maks1.86e-9). Første syntetiske fixture
manglet allow_unused for adskilteS/F-logits; rettet før genuineclaim.
Kildekontroll fant trådoverstyring og manglende originalplan i sluttkontroll;
begge er rettet før launch. Ingen produksjonsmodellkode er endret.

Brukerens føring er korte tester først. Den nye CPU-prøven ovenfor er
fullført/konsumert; native GPU-launch/full-epoch er stengt. Gjenbruk checkpoint, lagrede arrays og beståtte
kontroller. CPU-diagnostikk har60s budsjett. En eventuell nødvendig native
prøve må kvalifiseres separat, med høyst64updates/600s og små forhåndsbundne
initial/final/later-utvalg. Alle genuine features, femTF, kausalitet og
maskinvarevakter beholdes; fysisk GPU-grense250W. Den korte CPU-ruten er gjennomført innen600s;
resultatkontroll PASS. Ingen teknisk PASS åpner større trening.

To korte kontroller er nå fullført og uavhengig verifisert:
1. Faktiske INITIAL/FINAL M5-hoder krysset med gammel/ny representasjon på
   seks faste1024utvalg:6.258s beregning,2.842s review. Hodeendringen forklarer
   71.0%S/65.6%F av TRAIN-Brier-svekkelsen i disse utsnittene; resten følger
   representasjonen. Begge bidrar negativt senere. Interaksjonen er
   halvfordelt, ikke et ekstra additivt bidrag. Dette beviser ikke mekanismen.
   NATIVE_HEAD_ATTRIBUTION_RESULT/REVIEW i NEXT_PASS_READOUT_20261109_001.
2. Fast akkumuleringstest på lagrede INITIAL-representasjoner:5.035s worker,
   7.167s hel capped kommando. Samme64batcher/1024rader og samme AdamW/tap:
   vanlig64steg mot8steg med gradientgjennomsnitt over8batcher.
   TRAIN S Brier initial/vanlig/akkumulert:
   0.248652496 /0.248810289 /0.248686972.
   TRAIN F:0.249660644 /0.249802785 /0.249691001.
   Akkumulering demper svekkelsen omtrent78%, men begge mål er fortsatt
   svakere enn INITIAL. TRAIN-gaten feiler; ingen senerearrays dekodet.
   Dette kvalifiserer ingen native akkumulering eller lengre fit.

Akkumulering: MARKET_STATE_TA_M5_SHORT_ACCUMULATION_20261109_001.
RESULT SHA89a4b82fda6ce073e418620bc187c0c88c5b552125377dc1bf02ec0826bbf919.
REVIEW SHAf175d3032925bcc4509f9aea1d1637f68edb33e8270e4ada5512f0e8907cf2ec.
Review1.072s: åtte scoreceller med nullavvik, head-projeksjon maks7.45e-8.
64/8updates og lossdeling er kilde-/kvitteringsbevis, ikke optimizerreplay.
Færre optimizersteg inngår i kontrasten. Frossen representasjon og privat
tohodeklipping er ikke native bevegelig encoder eller helmodellklipping.

Neste direkte hull er faktisk delt encodergradient. Den eldre
DIRECTION_DECODER_DIAGNOSTIC_20261108_001 brukte detached-backbone derivatives,
og besvarer derfor ikke om andre mål motvirker M5-side i felles modell.
Kvalifiser høyst3 faste TRAIN-batcher/600s med null updates fra korrigert
INITIAL og uendrede produksjonstap/inputs. Den eksakte engangsdiagnosen er nå fullført/konsumert ovenfor.
Ikke gjenta negative L2-, intakt-batch-rekkefølge- eller akkumuleringstester.
Ingen retningsedge eller ferdig native rettelse er dokumentert.

Den ekstra native passeringen er fullført, og null-fit recovery002 er
COMPLETE/konsumert:2959 forwards,0 nye steg/INITIAL-forwards og bitlik full
modell/AdamW/RNG-tilstand. Guard/observatør exit0, task Ready/result0.
Begge målinger og M5-capture er lagret: FINAL29574 og senere TRAIN17753.
Opprinnelig capturefeil og recovery-forberedelse001 beholder FAILED-kvitteringer.

Fast readoutkontroll er COMPLETE og uavhengig verifisert:
MARKET_STATE_TA_M5_NEXT_PASS_READOUT_20261109_001/RESULT.json,
SHA686d489261fc787d9b8f52ec808449a1c52c00c00b5a7497f19273785d042b41.
REVIEW i samme rot, SHA2f61baa1b56d1c777ee1683f42a3865b2eeae34c2df836d0539920be66852118.
To forhåndsvalgte S/F-hodefit, alpha0.01, samme rå128D/vektmasse/solver/L2,
låst før senere arrayer.6.607s worker/8.663s capped kommando; ingen nativeforward.
Alle20 readoutceller,20 nativeTF-sideceller, akser/masker/vekter, koeffisientlås,
gradienter og fire ukeintervaller består uavhengig audit.2.091s/179.5MB topp-RSS.
Faktisk capture→checkpoint-headparitet har maks2.384e-7 logitavvik.

TRAIN S/F Brier før→fast readout etterpass:
0.248201726→0.247941037 /0.249076427→0.248704034.
Senere S/F:0.249767772→0.250054431 /0.249874165→0.250247306.
Begge senere scorer svakere enn prior0.249827988/0.249833416.
Forskjell mot førkontroll er positiv i begge95% UTC-ukeintervaller;
bare5/14 S-uker og3/14 F-uker forbedres. Dette er gjenbrukt utviklings-TRAIN,
ikke fersk generalisering. Hodekontrollen er diagnostikk; ingen ny native
vektimport eller promotion er innrømmet.

Ny kort screening er allerede utført på lagrede prognoser:
første/midterste/siste1024 rader i hver partisjon,6144 rader totalt.
SHORT_SCREEN_PLAN/RESULT ligger i samme readoutrot. Den tok2.969s,0 fit/forward.
Native etterpass er svakere enn før i2/3 blokker for både S/F på begge
partisjoner. Fast refit forbedrer senere førkontroll i bare1/3 blokker for
hver oppgave. Små scoringutvalg fanger altså problemet billig; dette beviser
ikke utfallet av64 treningssteg og er ingen generell dom over prosjektet.

Samlet registrert receipt-UTC fra originalstart til recovery-slutt:
9964.634324s (2t46m04.634), inkludert stopp og retting.
Worker-tider er7478.918166317s; gammel fullfit+bredere cache9733.087655347s.
UTC og worker-monotonic avviker; ingen identisk ende-til-ende speedup er bevist.
Det dokumenterte arbeidsbesparende grepet er capture i nødvendig scoring,
som unngår separat encoderpassering. Fullpresise tider/evidens står i REVIEW.

Den planlagte hode-/representasjonskontrollen er nå fullført som beskrevet
øverst. GC, TEST, Entry/Exit, broker, handel og spending forblir stengt.

TRAIN-månedskontrollen er COMPLETE/konsumert. Kun åtte eksisterende fit-arrays
ble lest; ingen senere arrays, fit eller nativekall. Hele capped kjøringen tok
3.655 s. Alle 30 scoreceller er uavhengig regnet om med nullavvik; samlet TRAIN
reproduserer den tidligere kontrollen nøyaktig. Første review feilet på str/Path
i hashhjelperen; bare review ble rettet. Original kode/logg er bevart.

Ny representasjon + korrigert hode forbedrer både Brier og AUC i ALLE fem
TRAIN-måneder og slår fryst prior i hver måned. Mai–september Brier:
0.249060 / 0.249367 / 0.249173 / 0.248665 / 0.249127.
Bare hodebyttet på ny representasjon forbedrer Brier i 4/5 måneder (august
svakere), AUC i 4/5 (mai svakere). Den samlede TRAIN-forbedringen skjuler
dermed ingen dårlig måned som forklarer desembertapet. Alle disse målingene
er in-sample; de beviser ikke at retningen generaliserer.

RESULT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_TRAIN_MONTHS_20261109_001/RESULT.json
SHA 727f2eef95dc5d294b01db4bbf4c6a8c1d1b64cc025326ffdd74b08c52127e60.
REVIEW i samme rot, SHA 5e6f0292d4001a4539d0b6d040ac9c416a8471f442e334ab8356e86ec916caaa.

Den tidligere planlagte native passeringen og capture-endringen er fullført;
resultatet og gjeldende korte arbeidsløp er beskrevet øverst. Den gamle
engangsplanen er konsumert og gir ingen adgang til ny trening.

Den faste 2×2-krysskontrollen er COMPLETE/konsumert: gamle/nye M5 F-hoder
på gammel/ny representasjon, uten fit eller nativekall. Beregning 5.074 s;
hele capped CPU-kommandoen 7.058 s. Ingen sammenliknbar native speedup.
Alle 25 scoreceller er uavhengig regnet om med nullavvik. Begge opprinnelige
FP64-prognoser reproduseres bitlikt; alternativ matrisemultiplikasjon avviker
maks 6.25e-17. Uke-/bootstrapresultater er eksakt kontrollert.

Desember AUC (gammel/ny representasjon × gammelt/nytt hode):
gammel/gammelt 0.502702; gammel/nytt 0.493803;
ny/gammelt 0.502190; ny/nytt 0.494426.
Rangeringstapet følger altså det nye hodet på begge representasjoner.
Den nye representasjonen øker samtidig gjennomsnittlig OPP-sannsynlighet:
gammelt hode 0.520864→0.543080; nytt hode 0.506274→0.528521,
mens observert OPP-andel er 0.502201.
Dette er to ulike feil: rangering i hodet og sannsynlighetsforskyvning
oppstrøms. Det er utfallsattribusjon på lagrede koordinater, ikke bevis for
én bestemt optimizerårsak eller at representasjonen har mistet informasjon.

Ingen krysskombinasjon løser senere retningspresisjon. Samlet senere Brier:
gammel/gammelt 0.249856; gammel/nytt 0.250111;
ny/gammelt 0.250290; ny/nytt 0.249874; fryst prior 0.249833.
Ny/nytt minus gammel/gammelt er +0.00001837, ukeblokkintervall
[-0.00026375,+0.00029686]; bare 4 av 14 uker forbedres.
På TRAIN forbedrer ny/nytt fortsatt Brier fra 0.249327 til 0.249076.
Ingen arm er valgt, importert eller promotert.

RESULT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_READOUT_TRANSFER_20261109_001/RESULT.json
SHA 0011322e3a920f640dc3702f7f5ce1c6a2bf9b9735674a862520d90271a0c454.
REVIEW i samme rot, SHA 2913476a9ee97f2614ccdf1159db3de6928ef20914e3906c0ca85ac9efcc3bc2.
training_enabled=false; de to auditene er konsumert, ingen ny kjøring innrømmet.

TRAIN-månedsspørsmålet er nå besvart over. Gjenbruk de fullførte kontrollene;
ingen tilbakeføring, terskel-/horisontseleksjon eller ny hodevariant er valgt.

Den kausale M5-historikkplanen er også COMPLETE/konsumert. November- og
desemberfit brukte henholdsvis 14484/16802 kjente utfall før prognosemåneden.
Hele kjøringen tok 7.288 s. Begge senere månedsprognoser ble svakere enn
det faste hodet: samlet Brier 0.249952 mot 0.249874, fryst prior 0.249833.
Kausal-minus-fast +0.00007747, ukeblokkintervall
[+0.00000904,+0.00016060]. 26 scoreceller er uavhengig kontrollert.
Det avviser denne konkrete kumulative hodeoppdateringen, ikke flere år
eller lengre native læring generelt.
REVIEW: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_CAUSAL_HISTORY_READOUT_20261109_001/REVIEW.json
SHA e26f87c562c0ee8c2af52b543fd63670eb5dbbff2924ff110d7a589b387d5866.

Fire faste F-hoder på bevart FINAL-cache fullførte på5.792s,0nativekall.
Alle fire TRAIN-Brier slår frystprior: M15/H1/H4/D1
0.248130/0.243525/0.231724/0.183697.
Senere Brier kandidat/prior:
M15 0.250136/0.249922; H1 0.250619/0.249393;
H4 0.253275/0.247974; D1 0.252789/0.255529.
D1 er bedre i gjennomsnitt,men bare7/14uker og ukeblokkintervall for
forskjellen[-0.040376,+0.048217]. Alle fire intervaller inkluderer null.
Ingen horisont er valgt/promotert. D1treff59.89% mot alltidLONG56.69%
gjelder kun6063framtidig sidekjente halvtATR-rader,ikke valgte handler.
Kandidaten predikererLONG på5609 av disse; separatSHORTpresisjon71.37%
på454 overlappende rader er diagnostikk,ikke uavhengige handler.
40scoreceller er uavhengig regnet om mednullavvik; kalenderuker/CI innen8.4e-17.
REVIEW: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FUTURE_HORIZON_READOUT_20261109_001/REVIEW.json
SHA877257893561d2d21e94da1066a0831c0f0b8cef1b65f6f091a563da7b1b0e40.
Planen er konsumert; training_enabled=false; ingen aktivjobb.

Den kausale månedsplanen er nå fullført negativt, som beskrevet øverst.

Den faktiske FINAL-modellen har nå de kvalifiserte globale M5-hodene.
Native kontroll:129.644s,3modeller/optimizere,6fullforwards,5918cachede
hodekall,0backwards/oppdateringer og bitlik lagring/gjenopptakelse.
Uavhengig checkpointkontroll bekrefter at kun fire hodetensorer er endret;
buffere,RNG,moduser og parentrekkefølge er uendret. Kandidaten har tom
optimizer/cursor0; originalfullfit med1849steg er bevart.
16lagrede scoreceller samsvarer innen9.46e-11. Maks logitavvik1.3411e-7
er innen forhåndsbundet absoluttgrense2e-6. Auditens gjenbrukte1e-7-sjekk
var for streng og ble rettet til den opprinnelige grensen; nativegate uendret.
Checkpoint: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FINAL_READOUT_IMPORT_20261109_001/RUNTIME/private-research/IMPORTED_M5_STATE.pt
SHA f33a9b3a2438d863925060d683291433a3827c9d313a276e9bd48197dd9a8ec1.
REVIEW i samme rot, SHA5325057a27641c06460e1234593147aa8d804a78a97d109751fd03a85be47acd.

Kampanjekontrolleren FEILET: Windows-observatørens nullstegsregel bandt
bare gammel importsti. Feilen ble reprodusert på den faktiske terminalen;
minste rettelse tillater også eksakt ny sti.10faktiske PowerShell-tilfeller
består,inkludert feil sti/tellere og gammel rute. Opprinnelig kampanjekvittering
mangler; ACTIVE_INVOCATION og originalterminal beholdes. Ingen modellomkjøring
eller syntetisk erstatningskvittering. Windows-task Disabled,result1.
Rettet observatørkilde er kvalifisert,men ikke kopiert over fryst Windows-mirror.
Trener/signedguard avsluttet0;250W,maks46°C/minne52°C/power130.99W.
Ingen fullført kampanje-walltid eller speedup hevdes.28rutetester og obligatoriske
commitkontroller besto. Planen er konsumert; training_enabled=false.

TRAIN S/F Brier0.248202/0.249076 er bedre enn gammel import og nativeFINAL.
Senere S/F0.249768/0.249874 mot prior0.249828/0.249833;F taper fortsatt,
og begge senere skårer er svakere enn gammelimport. Rettelsen er derfor
en verifisert TRAIN-/modellkoblingsforbedring,ikke dokumentert retningsedge.
Full ny encoderbaseline for heleTRAIN er ikke kjørt;32fysiske rader pluss
alle cachede hoder er kontrollert. Ingen ny encoderfit/cache.

Neste direkte spørsmål er separat retningsinformasjon ved de allerede
eksisterende lengre F-horisontene+15/+60/+240/+1440min i samme FINAL-cache.
Den eksakte faste TRAIN-fit/senere-score-sammenlikningen er nå fullført
som beskrevet øverst,med låste koeffisienter før senere data og fryste priorer. Begge kort/langt skal måles hver for seg,ingen beste-horisont-
seleksjon fra senere data. SenereTRAIN er gjenbrukt utvikling,ikke fersk TEST.

L2-kontrollen er COMPLETE/konsumert på6.009s. Isolert coupledL2 erstattet
decoupled decay ved samme1849 kronologiske steg, men begge TRAIN-hoder ble
svakere enn start. Ingen senerearrays åpnet; ingen nativeL2-endring kvalifisert.
REVIEW: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_COUPLED_L2_CONTROL_20261109_001/REVIEW.json.

Kalibreringskontrollen er COMPLETE/konsumert på2.759s:4små fits,0native.
Positiv affinekorreksjon fjerner99.81%S/99.59%F av TRAIN-Brierforverringen
mot importinitial. TRAIN-S/F Brier0.248478/0.249346;senere0.249765/0.249965.
SenereS slår prior svakt,F taper fortsatt. AUC er eksakt uendret av korreksjonen.
Det er overkonfidens/OPP-bias,ikke ny retningsinformasjon. Nativekode endres ikke.
REVIEW: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FINAL_CALIBRATION_20261109_001/REVIEW.json.

Full-FIT-gradientkontroll uten fit viser at importhodene er stasjonære for
BCE+coupledL2alpha0.01(maxgradient8.1e-10/2.1e-11),men ikke nativeBCE med
AdamWdecay0.0001. UtenL2 er globalBCE-gradient motsatt beta(cos≈-1).
En beregnet første global AdamW-oppdatering øker koeffisientnormen.
Dette isolerer et objektivavvik;det kvantifiserer ikke faktisk batchrekkefølge,
helmodellklipping eller bevegelig representasjon.
OBJECTIVE_ALIGNMENT.json ligger i samme rot.

Den isolerte L2-kontrollen ovenfor har fullført negativt. Hypotesen er testet,
og skal ikke føre til ny nativefit eller gjentakelse.

Ny fast avlesning av faktiskFINAL er COMPLETE/konsumert på5.383s:
2private128D-hoder,alpha0.01,like observasjonsvekter,samme Newton-eier;
ingen nativeforward eller parameteroppdatering. Begge hoder låst før
senereTRAIN ble scoret. Ingen ny hyperparameterseleksjon.
På29574TRAIN er Brier S0.248202/F0.249076 mot import0.248473/0.249327,
nativeFINAL0.251118/0.253904 og prior0.249651/0.249876.
På17753senereTRAIN er Brier S0.249768/F0.249874 mot import0.249652/0.249856
og prior0.249828/0.249833. Begge senereavlesninger er svakere enn gammelimport;
S slår prior svakt,F taper fortsatt svakt. Ingen retningsedge kvalifisert.
Lagrede Brier/log/mean-p er uavhengig regnet om mednullavvik.
REVIEW: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FINAL_READOUT_20261109_001/REVIEW.json,
SHA deff9b4774ad68e6306d598d2073f43e36b126cc29069122965b5319d4ef310d.

Dette avgrenser årsaken: samme avlesningsmetode henter bedre TRAIN-prognoser
fra nyrepresentasjonen,selv om native sluttprognoser er svakere. Nyrepresentasjonen
har derfor ikke bare mistet all TRAIN-retningsinformasjon. Readout-tilpasningen
er et konkret problem; generalisering består ikke. Fast råbasis-L2 påvirkes
også av representasjonens geometri; dette er ikke et fullstendig informasjonsmål.
Den nye nativeimporten er nå verifisert som beskrevet øverst. Originalimporten beholdes som sterkere
senere-referanse. Kalibreringsdriften er nå målt nedenfor/ovenfor; den eksakte nye
nullstegsimporten er konsumert som beskrevet øverst. Ingen ny fit er innrømmet.

Faktisk FINAL-cache er COMPLETE/konsumert:2959enkodinger,1modell,0backward/updates.
Alle20partisjon×oppgave×TF-sidemålinger er eksakt like lagret nativekjøring;
kilde136a8ca2 og modell er uendret. Controller2616.234s mot gammelcache2584.765s:
31.469s tregere,ingen målt speedup. Peak52°C/minne56°C,250W.
REVIEW: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_LOSS_FINAL_CACHE_20261109_001/REVIEW.json.
Cache/readout/kalibrering/optimizerkontroller er konsumert;ingen ny kjøring innrømmet.

Ny import/equal-loss-rekkefølgekontroll er COMPLETE/konsumert på10.914s:
3privateoptimizere/6småhoder,5547updates,0nativeforward. Samme start,data,tap
ogoptimizer; bare intakte batchers rekkefølge endret. TRAIN Brier:
S initial0.248473→kron0.248653/revers0.248480/permutert0.248977;
F initial0.249327→kron0.250115/revers0.249366/permutert0.249442.
Rekkefølgen påvirker kalibrering kausalt på fast representasjon,men alle
sluttpunkter er svakere enn importinitial. Forhåndsvalgt permutasjon feilet
TRAIN-porten; ingen senerearrays ble åpnet. Ingen native rekkefølgefiks kvalifisert.
REVIEW: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_LOSS_ORDER_20261109_001/REVIEW.json.

Den tidligere cacheforberedelsen er fullført som beskrevet øverst.
Fem fokuserte rutetester,syntaks og faktisk85-owner metadata-preflight besto;
bare cache-eierens checkpoint/root-binding og capped-rutens eksakte sti endret.
Modell,tap og dispatcher er uendret. Ingen cacheomkjøring.

Den avgrensede FIT-head-kontrollen er COMPLETE/konsumert på3.256s:
7396B16-lineærprojeksjoner,2checkpointlesinger,0fit/fullmodellforward/updates.
Importkontrollen gjenskapte lagrede logits bitlikt og nativeINITIAL-score
innen6.8e-12. Begge nyeM5hoder blir svakere på gammel representasjon, og
faktiskFINAL blir ytterligere svakere når oppstrømsrepresentasjonen også endres.
S Brierendring+0.002644=hodet+0.000329+oppstrøms+0.002315;
F+0.004577=hodet+0.001989+oppstrøms+0.002588 (avrundet).
F-hodets gjennomsnittligeOPP på gammel representasjon går51.10%→55.06%;
faktiskOPP51.11%. Dette lokaliserer drift,men beviser ikke optimizerårsaken.
Oppstrøms inkluderer encoder/query-projeksjon; attribusjonen er rekkefølgeavhengig.
REVIEW: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_LOSS_HEAD_DRIFT_20261109_001/REVIEW.json.

Originalimport616a22… beholdes; fullfitFINAL er diagnose,ikke forbedret kandidat.
Den nye rekkefølgekontrollen ovenfor avløser spørsmålet om en enkel
omstokking som rettelse; ingen native rekkefølgeendring er kvalifisert.

Fullfit002 er COMPLETE og konsumert; én modell/optimizer,6657forwards og1849updates,
ingen skips. Full29574INITIAL/FINAL og17753senereTRAIN målt. TRAIN S/F Brier
0.248473/0.249327→0.251118/0.253904; senere0.254579/0.256591 mot
prior0.249828/0.249833. Alle10TF×S/F-retningsskårer svekkes påTRAIN;
alle10taper motprior senere. Vektingsrettelsen løste ikke læringsproblemet.
Fullbaseline for imported ae772fd7 er nå kvalifisert; initial/final deler
observasjonskjede, masker/counts og prior. Alle lagrede scoremeans er kontrollert.
REVIEW: /home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_LOSS_FOLD1_FIT_20261109_002/REVIEW.json.

Native7121.942s,controller7144.960s mot7171.361s:26.401s/0.368%kortere.
Samme fulle måleomfang,men ulikt startmodell/loss;ingen vesentlig eller kontrollert
speedup.250W,maks56°C/minne64°C. Trainer/observer/guard/task0;taskDisabled.
Kilde84f0f47c var uendret. Originalcheckpoint og alle nativekvitteringer beholdes.
001feilet kun metadataentrypoint før noen nativejobb;planen er konsumert og bevart.


Brukeren har presisert begge: kort sniper og lengre long/shorttrend, med separate
prognoser/mål og ny vurdering hvertM5. F-mål finnes allerede ved+5/+15/+60/+240/+1440min;
S er framtidig lokal tilstand,F er nettopris fra beslutningen. Endpoint-side-score
måler ikke gunstig bevegelse først, motbevegelse før gevinst eller touch/brudd/retest-
rekkefølge. God M5-score er ikke i seg selv riktig inngang eller dokumentert økonomi.

Den nye [native equal-loss-kvalifikasjonen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_LOSS_MECHANICAL_20261109_001/REVIEW.json)
er COMPLETE og konsumert:3modeller/3optimizere,6forwards/4backwards/4steg.
Strict gammel restore av importcheckpoint616a22… og ny capture bevarte faktisk
modell/buffere/tomAdamW/RNG/modes/order bitlikt. Bare tapsbinding, source og
samlet seal endret. NyINITIAL5f91d264… er kontrollert direkte mot originalfilen.
2steg direkte mot1+lagring+resume1 ga bitlik fullstate; begge M5-hoder fikk
gradient. FINAL_PRIVATE e6751982… har2logiske steg og skal IKKE brukes i fit.
Gjenbruk originalimporten616a22… som startpunkt.

Trener/observer/signedguard og Windows-task avsluttet0; full campaign-receipt
finnes. Boot628,250W beholdt, maks46°C/minne52°C. Den gamle tasken er deaktivert; den senere fullfit er også
konsumert som oppsummert ovenfor. Kilde619c79da var uendret gjennom kjøringen.
Native155,619s; hele kampanjen175,810s mot tidligere171,595s (+2,46prosent).
Ingen målt speedup; dette er én workflow-sammenligning med ulik start/loss,
ikke isolert ytelsesbenchmark eller fullfit. Source/routing er kvalifisert
med110fokuserte tester og faktisk85-owner metadata. Én historisk testfixture
ble rettet til sin erklærte kildeversjon; produksjonsvaktene er uendret.

Native4stegsbeviset nedenfor gjenbrukes. Fullfit er nå avsluttet negativt som
oppsummert ovenfor; ingen ny fullfit eller mekanikkgjentakelse er innrømmet.

Den innrømmede [lagrede-logit-kontrollen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_OBSERVATION_LOSS_20261109_001/REVIEW.json)
er COMPLETE og konsumert:1849FIT-batcher,3698rålogit-autogradkall,213,206s.
Alle aktivitetsgradienter og øvrige TF-gradienter/tap er bitlike. Uavhengig
FP64-formel bekrefter alle gradienter (maks2,19e-8) og batchtap (maks3,62e-7).
Sidekjente S10010/F11909 får hver q/(5N); total M5-masse beholdes.
50/5 tomme sidebatcher og siste6rader følger samme regel. Masker/counts/skips
og gamle standardkontrakter er uendret.0modell/optimizersteg; ingen senere
arrays eller representasjoner åpnet. NPZ-filene er i sin helhet hashkontrollert.
Kilde269d9004 var uendret. Ingen ny retningsforbedring følger av en tapskontroll.

M5-sidevektingen er nå implementert som eksplisitt opt-in med fast fold1
TRAIN-binding. Hver kjent M5-observasjon får q/N før eksisterende TF-divisor;
samlet tapmasse beholdes. Aktivitet, øvrige TF, input, mål, scorer og gamle
standardkontrakter beholdes. Siste batch på seks rader behandles likt.
42 fokuserte syntetiske tester består, inkludert gradienter, akser før
forward, gamle/nye state-rundturer og avvist kryss-restore. Syntaks består.
[Kildekontroll](/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_OBSERVATION_LOSS_20261109_001/SOURCE_VALIDATION.json).

Native fullfit med rettelsen er nå kjørt og avsluttet negativt, se øverst.
Den kvalifiserte nye grenen unngår den gamle legacy-varmstarten som ville klonet gamle hoder.
Fullfit-planen er nå konsumert. AdamW er uendret; Newton med koblet L2 fra
readout-forsøket er ikke gjenskapt av denne vektingsrettelsen.


Den nye [nullstegsimporten](/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_WEIGHT_IMPORT_20261109_001/REVIEW.json)
har native PASS og et uavhengig verifisert checkpoint. Kampanjekontrolleren
feilet separat med exit1 og ga ingen campaign-receipt. Planen er konsumert,
den gamle oppstartsoppgaven deaktivert; importplanen gir ingen ny kjøre- eller treningsmyndighet.
Original RESULT/COMPLETE-terminal/progress/checkpoint/ACTIVE_INVOCATION er
bevart. Ingen gjenkjøring eller omskriving til kampanje-PASS.

Native kontroll tok128,227s: streng original FINAL CUDA-restore, eksisterende
dedicated-migrering, bare fire importerte FP32 M5-tensorer, seks fulle forwards
på de faste32TRAIN-radene og5918native head-kall på alle29574/17753cacherader
for begge mål.0backwards/optimizersteg. Modellkropp/cache-representasjon og
øvrige outputs var bitlike; checkpoint-rundtur og resume-outputs var bitlike.
Ny modeldigest ae772fd755cbfa07b1b7bcca8bccd2c8a8889806989a289ddb8267972c281b37.
Nytt checkpoint616a22e467ebf35bded61705e00f99531d28420db90c30b8d755d447c612679f.
Full fersk encoderbaseline dekker32TRAIN-rader; full head-baseline gjenbruker
kvalifisert originalrepresentasjon. Ingen ny full encoderpopulasjon påstås.

Uavhengig CPU-lesekontroll sammenlignet lagret state med kvalifisert dedicated
INITIAL: eksakt fire endrede tensorer, bitlike koeffisienter, øvrig modellstate,
buffers, tom optimizer, RNG, modes og parentorder.12scoreceller samsvarer med
verifisert likevektskontroll, maksimalt4,78e-11. Ingen nye forwards eller fit.
GPU↔CPU rålogitavvik var under7,83e-8.250W beholdt, maks47°C/minne52°C;
signed guard avsluttet0. Vellykket total kampanjetid/speedup er ikke målt
fordi kontrollerens sluttrapportering feilet.

M5 Brier etter faktisk import, med fryst prior:
TRAIN S0,248473364 mot0,249651008; F0,249327011 mot0,249876211.
Senere S0,249652488 mot0,249827988; F0,249855791 mot0,249833416.
Begge slår TRAIN-prior. S slår senere prior, F taper fortsatt svakt.
Senere TRAIN er gjenbrukt utvikling; robust generalisering/økonomi er ubevist.

Den målte rapporteringsdefekten er fysisk gjenskapt: Windows-observatøren
avviste total_units0 når trenerprosessen fortsatt eksisterte. Smal kildefiks
tillater bare eksakt importrecipe, korrekt fase, terminalstatus og fire
heltallstellere0.7faktiske PowerShell-tester og syntaks består; legacy positiv
progresjon består, feil recipe/optimizer/parent/terminal/bool avvises.
Controller og øvrige vakter er urørt. Ingen UNC-fallback er bygget: lesing
bestod ved repro, så en slik ny feilårsak er ikke etablert.
36tidligere fokuserte tester, fysisk metadata-/transportpreflight og
commitvakter gjenbrukes. Native kilde338f8b3f var uendret under kjøring.
Alle gamle capture/restore/migration-metoder forble byteidentiske.
Original Windows-kilde er bevart i den installerte speilingen; rettet observer
er testet separat. En framtidig eksakt plan må binde og installere riktige bytes.

**Direkte neste steg:** Korrigert M5-vektingsloss er implementert og syntetisk
kontrollert som beskrevet øverst. Full lagret-FIT-kontroll består; bind nå det importerte checkpointet
korrekt før eventuell native kvalifikasjon/fit. Ingen ny fit er innrømmet.
Det verifiserte importerte checkpointet og alle ferdige bevis skal gjenbrukes.
GC/TEST/Entry/Exit/handel forblir stengt; F-priorgrensen gjenstår.

Den [parvise likevektskontrollen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_EQUAL_WEIGHT_20261109_001/REVIEW.json)
er COMPLETE og konsumert. Kun vekten mellom kjente observasjoner ble endret:
original128D-representasjon, alpha0,01, startprior, masker, solver og total
M5-sidevekt ble holdt fast. To nye konvergerte hoder; vektet kontroll gjenbrukt.
Hele kontrollen tok4,091s, ingen nativeforward eller nativevektendring.

M5 FP32 Brier, vektet→lik observasjonsvekt, med fryst prior:
TRAIN S0,248546831→0,248473364, prior0,249651008.
TRAIN F0,249391063→0,249327011, prior0,249876211.
Senere S0,250021975→0,249652488, prior0,249827988.
Senere F0,250045801→0,249855791, prior0,249833416.
Alle fire Brier/log-målinger bedres. Begge slår TRAIN-prior; S slår nå også
senere prior, mens F fortsatt taper svakt. Senere AUC S0,521943/F0,508323.
Dette isolerer en konkret skade fra tapsvektingen ved frossen representasjon;
hele native treningshistorien eller robust generalisering/edge er ikke bevist.

Alle24 scoreceller er uavhengig kontrollert (maks2,23e-16), begge mål
(maks2,78e-17), begge optimumgradienter (maks8,13e-10), vektmasse og
FP32 B16-prediksjoner (bitlike). Kontrollprediksjoner ble gjenbrukt bitlikt.
Source6164a2f0 var uendret. Begge CPU-planer er lukket og konsumert.
Worker ble avgrenset delegert, lest av root og kvalifisert med fokuserte
vektkontroller; en overstreng float-literal-test ble rettet, uten workerendring.

Alpha0,01 kom fra [kronologisk indre TRAIN-valg](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_INNER_L2_20261109_001/REVIEW.json):
fem forhåndsvalgte styrker, tre splitter,32fits/5,537s. Valget ble låst før
senere cache åpnet. Én hel B16 ble purget, streng M5-modning kontrollert.
Encoder/normalisering hadde sett hele TRAIN: dette er betinget head-valg,
ikke encoder-OOF. Senere TRAIN er gjenbrukt utvikling, ikke forseglet TEST.

[Skår- og vektdiagnosen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_INNER_L2_20261109_001/CALIBRATION_DECOMPOSITION.json)
viste inntil14× ulik observasjonsvekt. Effektiv OPP-andel var S0,526853
mot faktisk0,518681, F0,515518 mot0,511126. Lik vekting fjerner akkurat dette
avviket med bevart tapmasse S0,194591671/F0,199459167.
Aktiv native rute er native_F_objective→side_normalized_loss_v1; den eldre
loss_v2 er ikke siste side-normalisering. Native kode/vekter er fortsatt uendret.

Likevektskoeffisientene og originalcache er grunnlaget for den nye
nullstegsimporten øverst. Den siste native FINAL har en annen kropp og brukes
ikke. F-priorgrensen er fortsatt uoppfylt.

Den nye [M5-fit-kjøringen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DEDICATED_M5_FOLD1_FIT_20261109_001/TERMINAL.json)
er COMPLETE og konsumert: original FINAL varmstartet til egne M5-sidehoder,
tom AdamW, fersk full INITIAL, én passering med1849updates/0skips,
samme29574 FINAL-rader og separat17753 senere TRAIN-rader.
Kilde8e1277f1 var uendret; trener, guard og observer avsluttet med0.
Windows-oppgaven er deaktivert; ingen ny nativefit er innrømmet.

M5 conditional-side Brier, lavere er bedre:
TRAIN S0,268542461→0,262282190, prior0,249651008 (N10010).
TRAIN F0,302929987→0,286292874, prior0,249876211 (N11909).
Senere S0,270574780→0,268627655, prior0,249827988 (N6096).
Senere F0,316566859→0,300100645, prior0,249833416 (N7165).
Log-loss bedres også, men begge M5-mål taper fortsatt mot prior på begge
populasjoner. Dette viser forbedring etter fortsettelsen, ingen kvalifisert edge.
Eget M5-hode og ekstra trening er ikke isolert fra hverandre i denne native
sammenligningen; tidligere frossen readout-kontrast isolerer kun head-begrensningen.

Før/etter og gammel/ny senere måling har identiske observasjonsbindinger,
alle60 score-antall og uendrede priorer. Separat leser har kontrollregnet alle
gjennomsnitt fra summer/antall; ingen nye forwards eller target-replay.
Fersk full INITIAL gjenskaper originale FINAL-aggregater med største
middelavvik9,71e-10; dette er ikke en ny rå per-rad-prediksjonskontroll.
[Resultat og tidsmåling](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DEDICATED_M5_FOLD1_FIT_20261109_001/PERFORMANCE_AND_LEARNING_REVIEW.json).

Hele kontrollerkjøringen tok7171,361s (1t59m31s), mot5051,843s for tidligere
recovery. Ny kjøring inkluderer full INITIAL (1691,751s) og256flere faktiske
updates. Ingen målt total speedup. Fit2623,855s/1849updates=1,4191s/update,
mot1,4067 tidligere (omtrent0,9prosent langsommere). FINAL1708,333s og
senereTRAIN1025,917s.250W ble beholdt; maksGPU56°C/minne64°C.

Den [faktiske M5-mekanikken](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DEDICATED_M5_MECHANICAL_20261109_002/RESULT.json)
er gjenbrukt: strictoriginalrestore, bitlike uendrede utganger, begge M5-hoder
med gradienter og bitlik2mot1+resume1.69 fokuserte fittester og commitvakter bestod.
Bare fem rute-/bindingsfiler endret; øvrige80 kildeeiere bevarte kvalifiserte bytes.


Den fulle [koblingskontrollen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_SHARED_OBJECTIVE_20261108_001/RESULT.json)
er COMPLETE og konsumert. Seks konvergerte readout-fits på samme frosne
FINAL-representasjon, samme pass0 sampled-S/native-F-fasit og eksakte
batch16-/TF-sidevekter; 7,992 s, null nye native forwards eller vektendringer.
Felles hode kontra eget M5-hode, med felles hode for de fire andre TF:
TRAIN M5 S Brier0,255734 mot0,246126; F0,252135 mot0,247188.
Felles hode taper altså mot prior0,249651/0,249876 på TRAIN selv ved
konvergens; eget M5-hode slår prior. Dette isolerer en konkret begrensning
i dagens readout ved den frosne representasjonen, ikke hele treningshistorien.
Senere S/F bedres fra0,258874/0,259311 til0,254622/0,253853 ved å skille M5,
men taper fortsatt mot prior0,249828/0,249833. Ingen kvalifisert edge.
Alle poeng/tellinger er kontrollert med0avvik; separat batchvis rekalkulert
treningsobjektiv avviker maksimalt1,05e-15. Alle seks løsninger konvergerte.

Minste native rettelse er implementert som TA-only opt-in
dedicated_m5_side=True: to egne 128→1 M5-sidehoder, 258 nye parametre.
Aktivitet, øvrige TF, genuine input/familier, mål og tapsvekter beholdes.
Standardkontrakter/state-nøkler/RNG og gamle checkpoints er bevart.
89 unike fokuserte tester består: gradientruting, vektmigrering, nye
optimizerparametre og bitlik syntetisk gjenopptakelse. Én testforventning
brukte ulik Q-form; den er rettet og kun egen testfil kjørt om, 26/26 PASS.
Faktisk metadatabinding bevarer opprinnelige felter, vinduer, ruter,
normalisering og priortall.
[Kildekontroll](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DEDICATED_M5_SOURCE_20261109_001/RESULT.json).

Varmstart kopierer alle gamle vekter og kloner de to sideradene; ny
modellidentitet beholdes. Dagens statekontrakt tillater ikke gammel AdamW-
historikk ved ny cursor0. Ny tom AdamW er derfor eksplisitt valgt; originalt
checkpoint bevares. Faktisk CUDA-migrering/RNG/gjenopptakelse er kvalifisert ovenfor; fersk
full initialbaseline og den innrømmede engangsfit-planen er nå fullført ovenfor.
Horisontspørsmålet står åpent; ingen target endres.

Kontrollen på [opprinnelige indikatorinput](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_INPUT_CONTROL_20261108_001/RESULT.json)
er COMPLETE og konsumert. 1 713 kolonner med alle genuine aktuelle felt og
siste lukkede M5/M15/H1/H4/D1-rad, samme frosne TRAIN-normalisering og samme
29 574 TRAIN-/17 753 senere TRAIN-rader. To faste LightGBM-modeller à256
grunne trær, ingen søk/early stopping, null nye native forwards/modeller.
Total 634,710 s; inputklargjøring 603,492 s, begge fit/score samlet25,868 s.
Inputcache og private modeller er bevart. Ingen hovedmodellendring.

M5 S: TRAIN/senere AUC 0,788409/0,510227; senere Brier0,254091 mot prior0,249828.
M5 F: TRAIN/senere AUC 0,769324/0,509349; senere Brier0,253952 mot prior0,249833.
Alle lagrede tellinger, Brier/log, gjennomsnitt og AUC er rekalkulert med
eksakt null avvik. De største TRAIN-gain-bidragene omfatter prisposisjon,
ATR-normalisert candle-geometri, EMA, trendlinje-/BOS-aldre og volatilitet.
De fleste av disse holder seg innen tidligere min/max senere i tid.
Dette er deskriptivt; verken feature-gain eller range-drift etablerer årsak.
Kontrollen svekker en ren representasjonsforklaring, men utelukker ikke verdi
i lengre historikk, andre tidsperioder eller modeller. Økonomisk edge er ukjent.

Brukeren er spurt om hovedmålet er neste 5-minutters bevegelse, en lengre
markedsreaksjon vurdert hvert 5.minutt, eller separate mål for begge.
Det skiller prognosetakt fra forventet reaksjonshorisont. Ingen target- eller
hovedmodellendring gjøres på grunnlag av en antagelse; gjeldende M5-fasit
og alle øvrige TF-mål er bevart. Retningsmålet er aktivt; ingen ny fit er innrømmet.

Full frossen FINAL-utlesning er COMPLETE og konsumert: 29 574 TRAIN-rader
og 17 753 senere TRAIN-rader, 2 959 enkodinger, null backward/optimizer.
Alle S/F-side-tellinger og Brier/log-gjennomsnitt på fem TF ble gjenskapt med
eksakt null avvik. Vekter og kilde var uendret. Trener og kontroller avsluttet
med 0; total kontrollertid 2 584,765 s (43 min 5 s), topp-RSS 5,85 GB.
250 W ble beholdt som brukeren krevde; ingen effektinnstilling ble endret.
[Full cache](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_FULL_CACHE_20261108_001/RESULT.json)
bevares med originale kvitteringer og to NPZ-partisjoner.

Den konvekse [M5-kontrollen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_CONVEX_M5_20261108_001/RESULT.json)
er COMPLETE og konsumert: fire konvergerte private readout-tilpasninger på
hele sidekjente TRAIN, 4,615 s, null nye native forwards. Senere TRAIN ble
kun scoret. Originale M5 S/F-prognoser hadde gjennomsnittlig OPP 0,630727/
0,741344 mot faktisk 0,513944/0,513050. S-rangeringen var svakt reversert:
AUC TRAIN/senere 0,476902/0,484692. TRAIN-fit kalibrering lærte S-slope
-0,249721 og F-slope 0,097952; dette er ikke bevis for en target-signfeil.

Senere Brier original/kalibrert/full128D/prior:
S 0,270575 / 0,249662 / 0,253804 / 0,249828 (N=6096).
F 0,316567 / 0,250001 / 0,252964 / 0,249833 (N=7165).
Full128D-readout bedret TRAIN til S/F 0,244985/0,246605, men senere AUC
var bare 0,505592/0,505114. Kalibrering fjerner mye feil sannsynlighetsnivå;
kun S har en liten senere punktgevinst mot prior, uten kvalifisert
statistisk/generell fordel. Fleksibelt readout løste ikke generaliseringen.
Hovedmodellen er uendret. Alle tellinger, Brier/log og vektrekonstruksjoner
er kontrollert separat mot cache, eksakt null måleavvik, ingen ny fit.

Brukeren ba om en annen smart angrepsvinkel. Neste direkte hypotese er en
liten kontrollmodell på de opprinnelige kausale indikatorinputene, med samme
M5-mål og kronologiske perioder. Den kan skille informasjonsmangel i input
fra informasjon som går tapt i den lærte representasjonen. Kontrollen er nå fullført som beskrevet ovenfor. Ingen promotering er tillatt.

Aktivt brukermål08.11: jobb bare direkte med retning. Første avgrensede
kontroll er fullført på eksisterende TRAIN-fasit på 1,24 s, uten modellkall.
M5 global/batchvektet OPP-andel er S 0,518681/0,526853 og F 0,511126/0,515518.
Denne statiske priorforskjellen alene forklarer ikke det målte Brier-fallet.
Checkpoint-prøven er COMPLETE: 246 TRAIN-rader, 32 CPU-enkodinger,
64 dekodergradienter, ingen optimizer, 153,21 s. Sluttmodellens TF-sidekorrelasjon
er minst 0,998620 for S og 0,999533 for F. På 5/18 rader med kjent motsatt
M5/D1-fasit ga S/F aldri motsatt side; totalt var det likevel 5/3 motsatte
prognoser av 246 rader. Total dekodergradient motarbeidet M5 S i 7/16 og F i
3/16 batcher; aggregert pekte den gunstig. Dette beviser svak TF-differensiering
og lokal konflikt i utvalget, ikke årsaken til hele treningsresultatet.
Den kontrollerte readout-sammenligningen er også COMPLETE og konsumert:
samme frosne backbone/projeksjon, identiske startprognoser, 480 private
oppdateringer per arm, 246 TRAIN- og 249 senere TRAIN-rader, 198,07 s.
På senere M5 S er Brier delt/TF-spesifikk/prior 0,270946/0,256937/0,249602
(N=75); F 0,265582/0,255479/0,250228 (N=107). Begge M5-mål bedres også
på TRAIN, men prior slår begge armer. Øvrige TF har blandede resultater.
Dette støtter delt readout som begrensning i denne kontrasten, ikke som
hele årsaken til tidligere native fit. Hovedmodellen er uendret.
Lagrede prognoser er rekalkulert mot cachede masker/counts; null nye forwards.
[Kontrollert sammenligning](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_READOUT_COMPARISON_20261108_001/RESULT.json)
og frosne dekoderinputs bevares for gjenbruk. Retningsmålet er fortsatt aktivt.

Korrigert TA S/F fold1 er nå **COMPLETE og konsumert**. 256 oppdateringer
ble gjenbrukt, 1 593 nye fullførte samme pass, totalt 1 849 uten skip.
Fersk initialparitet og full model/AdamW/RNG/order-restore bestod.
INITIAL/FINAL har identisk observasjonskjede, alle nevnere og frosne priorer.
På M5 TRAIN ble S-side-Brier dårligere fra 0,249789 til 0,268542 og
F-side-Brier fra 0,255374 til 0,302930. På senere TRAIN ble S/F henholdsvis
0,270575/0,316567 mot prior 0,249828/0,249833. Lavere er bedre.
Dette er sidekjente observasjoner, ikke treffprosent eller valgte handler.
Samlet TRAIN-tap falt fra 2,8812 til 2,4015 hovedsakelig via tilstand/aktivitet;
fallet dokumenterer ikke bedre retning. Bare D1 S slo senere prior på begge
sidemål; ingen statistisk kvalifisert gevinst eller årsaksforklaring følger.
[Resultat og tidskontroll](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_FOLD1_RECOVERY_20261108_001/PERFORMANCE_AND_LEARNING_REVIEW.json)
binder målinger, nevnere og filhasher. Trener/observatør/kontroller avsluttet
med 0, kilden var uendret, ingen TEST ble brukt. Windows-oppgaven er deaktivert,
ingen treningsprosess er aktiv, og resterende tre vinduer er ikke autorisert.

[Fullføringen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_INPUT_COMPLETION_20261108_001/RESULT.json)
er COMPLETE: 600 ekstra historikkrader × 94 felt = 56 400 sammenligninger,
null avvik. M5/M15/H1/H4/D1-gapene var 94/63/95/96/252 rader. Alle ni
modellflater har null ikke-endelige verdier på unike konsumerte rader.
315 lokale felt og 132 kontekstfelt er ikke konstante i dette utvalget;
noen tilstedeværelses-/sesjonsfelt er konstante på lengre TF.
Frosne S/F-priorer bruker eksakt 29 574 modne fit-rader, null score-fit.
385,86 s og 5,74 GB topp-RSS; null nye modeller, gather, normalisering,
lokal31-replay, backward eller optimizer. Én tidligere forward gjenbrukes
eksplisitt som evidens. Fullføringsplanen er konsumert.

Den tidligere lokal31-diagnosen er ACTUAL_SOURCE_MATCH: 47 697 rader ×
31 felt, 1 478 607 sammenligninger, null oracle/cache-/producer-avvik.
Den gamle auditreferansen startet i 2009, mens faktisk produsent startet
i juni 2010. Ingen produksjonsfeil ble demonstrert; ulike state-historikker
og absolutte koordinatstarter ble ikke isolert fra hverandre.

Ny [TA S/F-mekanikk](/home/andre2/GX1_RUNS/MARKET_STATE_TA_SF_MECHANICAL_20261108_001/RESULT.json)
er PASS: 32 modne fit-rader, seks forwards, fire backwards/vektoppdateringer,
fersk baseline og bitlik gjenopptakelse. 154,96 s, 5,75 GB topp-RSS og
1,79 GB toppallokert GPU-minne. Trener/observatør/kontroller avsluttet med 0.
Oppgaven er deaktivert og planen konsumert. Dette er ikke markedspresisjon.

Brukeren ba om å rette den målte ytelsesflaskehalsen straks. Ettpass-fit
ble derfor avbrutt før kildeendringen; planen er ABORTED_CONSUMED.
Full INITIAL på 29 574 rader og checkpoint ved 4 096 rader / 256 updates
er bevart. Checkpointets hele state- og komponenthasher er kontrollert.
Oppdateringer etter siste checkpoint i den avbrutte jobben er UNKNOWN;
dens FINAL og senere TRAIN-score ble ikke fullført. Startoppsettet ignorerte SIGINT; SIGTERM ga guard-exit143
og Windows-task-exit1. Ingen vellykket trenerterminal eller resume-cursor
ble publisert. Oppgaven er deaktivert, GPU/prosess/lås er ledige.
[Operatørterminalen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_FOLD1_FIT_20261108_001/OPERATOR_INTERRUPTION_TERMINAL.json)
bevarer dette som avbrudd, ikke PASS. Ingen gammel plan relanseres.

Den minste ytelsesrettelsen erstatter 2 700 gjentatte tuple.count-skanninger
med Counter. I 25 like metadata-målinger sank feltvalideringen fra
66,994 til 0,834 ms (80,34 ganger); hele TA-profilbyggingen fra
117,099 til 50,161 ms (2,33 ganger). Gyldige profiler, kontrakter,
feltrekkefølge og seks kontrollerte normal-/feiltilfeller er identiske.
Alle validatorer kjøres fortsatt; modell, features og regning er uendret.
Tre fokuserte TA-profil-/normaliserings-/konsumtester består. Tre nye
duplikatfeiltester følger den eksisterende obligatoriske pre-commit-gaten.
[Resultatet](/home/andre2/GX1_RUNS/MARKET_STATE_TA_DUPLICATE_CHECK_PERF_20261108_001/RESULT.json)
binder før/etter-målingene. Fasemåling og faktisk recovery-varighet står nedenfor;
ingen gammel fullført total finnes for en målt helkjøringssammenligning.

Den separate [inputfasekvitteringen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_INPUT_20261108_003/INPUT_TENSOR_PHASE.json)
er komplett innen sitt scope: 47 327 beslutningsrader og alle overlappende
inputvinduer, 7 726 606 020 råverdier bitlike, null normaliseringsavvik og
én faktisk forward med åtte familier/fem TF.
Disse bevisene er gjenbrukt i fullføringskvitteringen ovenfor.
003 stoppet i M5-lokalreplay; de øvrige MTF-historikkgapene er nå kvalifisert.
Alle originale `CLAIM/FAILED/TERMINAL` og logger for input001–003 og
diagnosen er bevart. Ingen konsumert plan relanseres.

Brukeren presiserte læring av berøring→brudd→retest på samme trendlinje.
Dagens representasjon har to nærmeste-linjeplasser uten linje-ID, ankere
eller byttemarkør; hendelsesfelt kan gjelde andre registerlinjer.
Tapet har S/F-framtidsmål, men ingen separat fasit/evaluering av samme-linje-
reaksjoner. Dette er en konkret representasjons-/målegrense; effekten på
prediksjonsevne er ikke målt. Ingen ny modellarkitektur er innført.
GC-registerplanen er utsatt og uklamet. Juni 2026 er brukt utviklings-VAL;
forseglet periode følger bare forward-journalregelen i GX1_RULES.md.

Den nye [recovery-planen](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_FOLD1_RECOVERY_20261108_001/CAMPAIGN/CAMPAIGN_PLAN.json)
fullførte i første vindu etter fersk Windows-boot624. Den bevarte modellen,
targets, normalisering, feltrekkefølge og optimizer; 85 kildeeiere var bundet.
Bare Counter og seks nødvendige recovery-eiere avvek fra originalen.
Input-/mekanikkjobbene og full INITIAL ble gjenbrukt; originalplanen relanseres ikke.

Hele controller-vinduet tok 5 051,84 s (**84 min 12 s**); runnerens monotone
måling til før terminalpublisering var 5 033,16 s. Resterende fit tok
2 240,82 s, full FINAL 1 681,83 s og senere TRAIN-score 1 013,11 s.
Første nye 256-blokk tok 363,60 s mot gamle 565,66 s: 35,7 % kortere tid.
Full evaluering av samme 29 574 rader tok 48,4 % kortere tid enn gammel
INITIAL på 3 260,23 s. Gamle grenser er filtid-intervaller; nye er monotone.
Fit bruker senere rader, INITIAL/FINAL ulike vekter. Dette er målt
operasjonell fasegjennomstrømning, ikke et identisk A/B-forsøk.
Gammel fullført total og gammel senere-score-tid finnes ikke; ingen målt
prosentgevinst for hele løpet påstås. [Tidsgrunnlaget](/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_FOLD1_RECOVERY_20261108_001/HISTORICAL_TIMING_BASELINE.json)
bevares. Ingen ekstra epoke eller lengre historikk er åpnet.

## Det som faktisk er målt

- **Eksplisitt TA-kandidat og korrigert læringsmåling:** Profilen
  `full_xau_ta_mtf_entry_research_v1` beholder 315 signal-, 132 kontekst- og
  253 MTF-felt, alle fem TF, åtte familier og 61 makrofelt. GC er fjernet
  før modellinput; modne fold1-statistikker projiseres uten refit.
  Faktisk FeatureView/checkpoint-identitet er kontrollert lesende:
  `3e2761afaf1d29e6537843e9e59b592a7c149a9cd99c97f866a2cd1e88611906`.
  Ny rute bruker korrigert S/F og hele samme 29 574-raders TRAIN-populasjon
  før/etter fit, med globale aktivitets-/sidemasker og frosne priorer.
  Senere 17 753 TRAIN-rader scores separat. Gammel initialkvittering
  med null forwards er ikke en læringsmåling. Inputkvalifikasjonen er komplett innen deklarert scope;
  ny fireoppdateringsmekanikk består. Den avbrutte fitens 256 oppdateringer
  er gjenbrukt, og den nye recovery-planen har fullført hele læringsmålingen.
  Retningen ble dårligere på M5/M15/H1/H4 for begge hoder allerede i TRAIN;
  bare D1 S/F forbedret begge sidemål fra initial og slo prior på TRAIN.

- **Funksjonsgjennomgang:** EMA bruker SMA-seed og alfa `2/(p+1)`;
  ATR bruker klassisk TR og Wilder14. SMC-pivot blir synlig etter bekreftelse,
  BOS bruker strict close-brudd, og alle inputbarene skal være lukket.
  Ingen ny produksjonsfeil er påvist i disse formlene ved kildekontroll.
  Modellen får indikatoravstander/helninger og candle-geometri.
  Prisoppdateringsantall er ikke omsatt volum.
  En reell makrodatofeil rundt høstens DST ble reprodusert og rettet:
  tilgjengelighet følger nå kalenderen D+2 ved NY-midnatt, ikke 48 elapsed timer.
  Alle 9 000 bundne kildedatoer fram til 2024-12-31 har identisk tilgjengelighet
  før/etter rettelsen; kandidaten trenger ingen ny materialisering for denne
  rettelsen. Kalender-/driverkontroll: 11 beståtte tester.

- **Test- og formelbevisgrense:** Ny TA-gather mistet C-kontiguitet og
  signaturrefleksjon mistet de fem gamle GC-argumentnavnene. Begge er rettet;
  sju berørte tester består.
  [Kilde-/testkvittering](/home/andre2/GX1_RUNS/MARKET_STATE_TA_SOURCE_QUALIFICATION_20261108_001/REVIEW_20261108T131458787986Z.json) SHA 68fcf96b270bf5c944fae4b3f6724ce25ad92c2ed25aaa3e1cdc329b55288a6d. Øvrige korrigerte taps-/batch-/state- og
  initial/final-tester bestod første fokuserte kjøring.
  Én eldre GC-relative-kildetest feiler på en uendret, foreldet
  `gc_order_flow_v1.py`-SHA; ingen GC-reparasjon åpnes i denne bølgen.
  186 lokale felt har eksakt M5-cache-alias. De 31 lokale trendlinjefeltene
  bruker egne 96-barsparametere og kontrolleres separat. Auditen inkluderer
  også 37 HTF/session-felt med eksplisitte asof-/avrundings-/klokkekontroller;
  den nye auditeierens kontroller er utvidet til 16 beståtte tilfeller.
  Fysisk tensorfase bestod; lokal31 er nå kvalifisert mot faktisk kilde som beskrevet over.
  61 makrofelt har eldre samme-eier-paritet/asof-bevis, ikke et uavhengig
  formelorakel; dette er en uttrykkelig bevisgrense.
   COT i 2024 bruker as-of +7 kalenderdager som tilgjengelighetsproxy.
   Faktiske publiseringstidspunkt og historisk datavintage er ikke kvalifisert
   per rapport. Ingen konkret 2024-lekkasje er påvist; full historisk
   tilgjengelighet er UNKNOWN og blir ikke bevist av teknisk input-PASS.
  Tidligere 253-feltsunion 158+31+1+63 og asof/postnorm-bevis gjenbrukes
  på deklarerte rader; de eldre faktisk konsumerte historikkgapene er nå også målt.
  [Tidligere kvitteringskontroll](/home/andre2/GX1_RUNS/MARKET_STATE_TA_INPUT_LOSS_REUSE_REVIEW_20261108_001/REVIEW_20261108T120132104946Z.json).

- **Input og mål:** 118 249 modellkoblede TRAIN-rader, 26 måneder GC MBP-1 og
  kausale fem-TF-flater er bevart. Foldprefikser, masker og normalisering er
  kvalifisert i sine deklarerte vinduer, ikke som bevis for prediktiv verdi.
  Hver genuine indikator trenger fortsatt kontroll helt til faktisk nytte;
  numerisk paritet og gradient på én rad er ikke edge. Lokal framtidstilstand
  `S_s(v)` bruker valgt aktivitet `R≥0,5 ATR`; i 0,5–1,0 ATR er side uavklart
  `{NED,OPP}`. Beslutningsforankret prisforskjell `F_s(t,v)` er et **separat**
  retningsmål med egen maske, ikke en ommerking av S-gråsonen. I senere TRAIN
  hadde 52 931 M5-rader S-side uavklart og gyldig F-endepunkt; bare 13 947
  hadde `|F|≥0,5 ATR(t)` og dermed meningsfull F-side.
  [F-dekning](/home/andre2/GX1_RUNS/MARKET_STATE_F_SUPERVISION_COVERAGE_20261108_001/RESULT.json)
  SHA `61b1e21a2dac1b86c8263e60fdcec5cd12b652135fcb991e231607493c942312`.

- **SMC/teknisk grunnlag:** Eksisterende 26-måneders M5-kilde og den eksakte
  modellklokken har 118 249 felles TRAIN-rader. Kausalt lukket M5 BOS og
  EMA20-helning gir 6 737 sidejusterte hendelser totalt; fem senere TRAIN-folds
  har henholdsvis 1 006/953/1 017/1 038/1 032 modne kandidater før noe
  framtidsutfall ble lest. [Kildedekning](/home/andre2/GX1_RUNS/MARKET_STATE_SMC_EMA_BOS_20261108_001/PREFLIGHT.json)
  SHA `f87a834f985cc53f1e5517f76753f6b00b4ef66d07735cc1d51bcc56594681b7`.
  Eldre SMC-familiefit kvalifiserte ikke retningsverdi på sitt tidligere mål;
  enkel EMA50/200-referanse ga ikke stabil sidegevinst på nytt S-mål. Ingen
  av dem testet den forhåndsbundne BOS+EMA20-sammenhengen ved 60 minutter.

- **Fast SMC+EMA20-forsøk, negativ retningsport:** Engangsplanen
  [MARKET_STATE_SMC_EMA_BOS_20261108_001](configs/research/MARKET_STATE_SMC_EMA_BOS_20261108_001.json)
  er konsumert. På 4 876 modne TRAIN-hendelser var ekstra signert
  60-minutters F-forskyvning mot EMA-kontroll i samme uke/side `−0,006385`
  H1-ATR, 95 % ukeblokkintervall `[−0,022610,+0,010314]`; bare 1/5 fold
  var positiv. Halv-ATR-kandidatens event-hit-rate var `0,488753`; det er
  ikke en låst markedstilstand eller handelsmåling. Uavhengig pandas-gruppering
  ga samme 4 876 rader og samme fem foldeffekter. Første auditforsøk tok med
  tre rader over foldcutoff; dette var auditens egen indeksfeil, rettet før
  kvittering, og påvirket ikke engangskjøringen. [Resultat](/home/andre2/GX1_RUNS/MARKET_STATE_SMC_EMA_BOS_20261108_001/RESULT.json)
  SHA `79fd9bccd08c78af6f4fb2de59f0416258e73546b0a302969ce152fd4822e40b`;
  [uavhengig kontroll](/home/andre2/GX1_RUNS/MARKET_STATE_SMC_EMA_BOS_20261108_001/INDEPENDENT_CHECK.json) SHA
  `d7a036c2065a29ea4f1a98c1c87cf4f62f2abc3ca51ae652243551ce5c36729d`.

- **Native tilstand, gammel v1:** Én fold-1-fit brukte 29 574 modne TRAIN-rader
  og 1 849 AdamW-steg. På 6 096 senere sertifiserte M5-siderader var
  side-Brier `0,277631` mot fryst prior `0,249828`; ingen retningsgevinst.
  Kilde/algebra viste at v1-tapet ga direkte skjev sidegradient når fasiten
  bare sa AKTIV; OPP-skjevheten er målt, men årsaken er ikke isolert.
  [Fit-resultat](/home/andre2/GX1_RUNS/MARKET_STATE_TIME_QUERY_FOLD1_FIT_20261107_002/RUNTIME/prefix-fit/window-0001/RESULT.json)
  SHA `08172fc2c17f8eaf4f0457e38bdd3aa750bfbbf9ae719cb587aabd4d93d21254`.
  Ingen ny v1-fit.

- **Ny S/F-rute:** Faktorisert v2-S og separat F-hode deler én native enkoding.
  På to ekte TRAIN-rader besto ett fysisk forward/backward: begge hoder og
  backbone fikk gradient, mens ukjent S-side og liten F-bevegelse fikk null
  direkte sidegradient. Vektene var bitlike før/etter; null optimizer/TEST.
  [S/F-nullsteg](/home/andre2/GX1_RUNS/MARKET_STATE_NATIVE_F_CPU_ZERO_STEP_20261108_001/RESULT.json)
  SHA `a2bb6cb8f586ea85f18ca4b313ffe28117dee67659216d00641d0a25cb0ccc7f`.
  Dette er teknisk kobling, ikke fit eller prognoseevne.

- **Enkle kronologiske referanser:** Femfolds S-LightGBM ga M5 side-Brier
  `0,249819` mot prior `0,249440`, men bedre aktivitet. En separat femfolds
  F-M5-LightGBM med alle 3 991 tabellfelt versus samme oppskrift uten de
  2 410 GC-navngitte feltene fikk **FAILED-terminal etter alle 20 fits** på
  grunn av skygget kilde-HEAD i siste bokendekontroll. Originalplan og
  terminal er konsumert; ingen RESULT eller omkjøring. Lesende hash-/mål-/
  prognosekontroll av de fem foldartefaktene ga på 34 760 meningsfulle
  M5-siderader Brier prior/full/uten GC `0,249755/0,250705/0,250821`.
  Prior slo full i alle fem folder; GC-tillegget var `+0,000115` Brier,
  3/5 positive folder og 95 % UTC-ukeintervall
  `[−0,000183,+0,000425]`. Den låste retningsporten feiler. På 88 397
  aktivitetsrader var prior/full/uten GC `0,238637/0,230894/0,231863`;
  aktivitet og GC-tillegget var positive i 5/5 folder. Dette er diagnostikk
  fra en feilet engangsjobb, ikke et fullført kampanjeresultat eller dom over
  native tidssekvenser.
  [Feilterminal](/home/andre2/GX1_RUNS/MARKET_STATE_F_M5_GC_ABLATION_20261108_001/TERMINAL.json)
  SHA `ce1b1862d1808fc67d63720b20e5c4075bf82b478ec5afbd2391b71d06f4e4a3`;
  eksakte foldreferanser, skår og begrensning står i policyens gate
  `market_state_F_m5_gc_ablation_20261108_001`. Den konkrete variabelskyggen
  i beholdt kilde er rettet etter kjøringen.

- **GC-tolkning:** Kjent aggressivt kjøps-/salgsvolum, sidevis gjennomsnittlig
  handelsstørrelse, OFI og GC-prisrespons når inputruten. Største registrerte
  handel har ikke egen sideetikett; dagens felt identifiserer ikke presist ett
  stort kjøp i aktiv modellvisning. De nye sidevise maksimumsfeltene er fysisk
  bygget, men ikke registrert eller konsumert av modellen. En enkel positiv
  GC-kjøpsubalanse ga ingen stabil framtidig XAU-retningsverdi; det tester ikke full native sekvensmodell eller absorpsjon
  ved struktur. [GC-flytreferanse](/home/andre2/GX1_RUNS/MARKET_STATE_GC_FLOW_REFERENCE_20261107_001/RESULT.json)
  SHA `256710c697cba9324cb4635aa323e99f4319ddbd1c418fa54ec549f04fde4946`.

- **GC-side-dekning på faktisk modellakse:** Av 118 249 M5 TRAIN-rader har
  117 626 gyldig GC-observasjon og global største handelsstørrelse; 117 541
  har både kjente aggressive kjøps- og salgsregistreringer. Sidevis
  størrelsesmaske finnes, men ingen sidevis største handel i den fysiske
  tabellen. Dette er presist informasjonstap for store handler, ikke mangel
  på GC-side generelt og ikke målt prediktiv verdi. 97 917 rader har også
  minst én handel med ukjent aggressorside (`N`); det kan sameksistere med
  kjente kjøp og salg. Kun eksakt M5-modellakse er kontrollert her.
  [Dekningskvittering](/home/andre2/GX1_RUNS/MARKET_STATE_GC_SIDE_LARGE_TRADE_COVERAGE_20261108_001/RESULT.json)
  SHA `7fdf2adb1138e0d777f85181c59d4272a50c6e832c6391d40ee9519ac684d184`.

- **Sidevis storhandel, kilde og fysisk data:** Eksisterende GC-samler bevarer nå
  største betrodde `T`-kjøp (`B`) og -salg (`A`) separat per M1 og reduserer
  dem kausalt til M5/M15/H1/H4/D1. Ti nye kandidater har samme sidevise
  observasjonsmasker og flow-eier; `N` gjøres ikke om til kjøp/salg. 77
  fokuserte kilde-/syntetiske tester består. Gamle fysiske GC-tabeller og
  nåværende native register/visning mangler fortsatt de nye feltene; nye
  fysiske tabeller er kvalifisert nedenfor. Prediktiv verdi og Entry er ubevist. Policyens
  `market_state_gc_side_max_source_20261108_001` binder dette kildesteget.
  Ny engangsplan [sidevis GC-bygg](configs/research/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001.json)
  SHA `57c076ffe2797297fa502a6fc32a0f0d38f4f347eaa71c9517e0d9c3b350bd47`
  er kjørt én gang og konsumert. Den gjenbrukte kvalifiserte råkilder og
  samme byggeeier: 66 filer/892 100 712 poster, 1 603 s under 1 800 s/4 GiB-
  kapsel. [Terminal](/home/andre2/GX1_RUNS/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001/RESULTS/TERMINAL.json)
  SHA `664d7b9aca3d969cfd7a72c34f1b5a186d2c37fd1f3a079a0bcacd5446abebd6`.
  Alle gamle 270 TF-felt er numerisk like på fem TF. Nye sidevise maksverdier
  har eksakte masker og likhet med maksimum over godkjente M1-minutter;
  [fysisk kontroll](/home/andre2/GX1_RUNS/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001/PHYSICAL_SIDE_MAX_AUDIT.json)
  SHA `9cefe83b23b7c02d848308ce756deeb8575205edfb89551478ad740db28c4dcc`,
  [M1–TF-paritet](/home/andre2/GX1_RUNS/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001/PHYSICAL_SIDE_MAX_M1_TF_PARITY.json)
  SHA `9bbcaf4bec20cd1e4bc642c8e60b8c4ee0cfd41ccfa2b803d51247a36942d7a0`.
  På de eksakte 118 249 M5-modellklokkene har 117 541 begge nye felt;
  [klokkedekning](/home/andre2/GX1_RUNS/NATIVE_GC_SIDE_MAX_RESTORE_20261108_001/MODEL_M5_SIDE_MAX_CLOCK_COVERAGE.json)
  SHA `3f8fcaa460b5f8e7f90d93cd84c504df914d1717bc1422bb8eb67f676d19289b`.
  Native register/layout/visning har ennå ikke konsumert de nye feltene.

- **Ny retningsvekt, teknikk og ett målt pass:** På de fem senere TRAIN-foldenes faste
  endepunkter hadde S entydig side i 33–44 % og F i 37–41 % av informative
  rader. Det gamle v2-tapet delte sidefeilen på alle informative rader. En
  separat side-normalisert tapsmodul deler nå på faktisk sidekjente rader;
  S/F-ruten bruker den for begge hoder. Den gamle v2-filen er byteidentisk
  med kvalifisert featurevisning.
  Den målte kandidatens F-vekt er bundet til 1,0 mot S; ingen scorevalgt vektsøk.
  11 fokuserte tester består. Ny fysisk nullstegskjøring på to ekte
  TRAIN-rader besto: S-/F-hode og backbone fikk gradient; ukjent S-side og
  FLATT F-side fikk null direkte sidegradient. Vektene var bitlike før/etter;
  én forward/backward, null optimizer, null TEST. 87,6 s og 5,35 GiB topp-RSS
  under 12 GiB kapsel. [Ny nullstegs-PASS](/home/andre2/GX1_RUNS/MARKET_STATE_NATIVE_F_CPU_ZERO_STEP_20261108_002/RESULT.json)
  SHA `7302d760741ae0329bd936a3bc0461dfedc2ada3a337fb23700bb2be7207b84b`.
  Dette beviser teknisk kobling, ikke læring eller retning. Policyens
  `market_state_native_F_side_normalized_design_20261108_001` binder designet.

- **Optimizer/gjenopptakelse, faktisk teknisk PASS:** Én guarded native S/F-prøve
  på de tidligere valgte TRAIN-radene 2–33 fullførte med seks forwards,
  fire backwards, fire AdamW-oppdateringer og tre modellkonstruksjoner.
  Avbrutt/gjenopptatt forløp var bitlikt referanseforløpet; trener- og
  observatørutgang var 0. 132,8 s, 5,76 GB topp-RSS og 1,98 GB toppallokert
  GPU-minne under 20 GiB/250 W-vakter. Planen er konsumert, og ingen
  prediktor-fit eller retningsverdi følger av dette.
  [Resultat](/home/andre2/GX1_RUNS/MARKET_STATE_NATIVE_F_PRIVATE_4STEP_20261108_001/RUNTIME/private-research/RESULT.json)
  SHA `55936ccfb06bddcebebfb10a6a591de29143fae3f64036594463f04985ccd97e`;
  [kampanjekvittering](/home/andre2/GX1_RUNS/MARKET_STATE_NATIVE_F_PRIVATE_4STEP_20261108_001/RUNTIME/receipts/invocation-0001.json)
  SHA `d274e2f0ea3d27842645105199b92761562965562cb2752d85169cd6fbc08f13`.
  Policyens `market_state_native_F_private_4step_20261108_001` binder
  engangsutfallet.

## Ikke gjort, hindring og neste steg

Native S/F har nå ett fullt korrigert fold1-pass med senere TRAIN-score,
men ingen ny femfolds OOF-kvalifikasjon. GC-merverdi i den native
tidssekvensmodellen, samme-linje-reaksjonsverdi, Entry-verdi og økonomisk edge
er ubevist. 61 makrofelt mangler uavhengig formelorakel; COTs faktiske
historiske publisering og vintage er fortsatt UNKNOWN.

**Neste steg:** Vurder matematisk målvalg for korte og lengre prisbevegelser i eksisterende modell. Koordinattesten er konsumert; ingen ny fit er innrømmet.
Målet er aktivt; senere retningsverdi er ubevist.
Brukerens spørsmål om 30 epoker og fem år er fortsatt åpne hypoteser.
Denne fitten dekker omtrent mai–september2024, senere score oktober–desember2024.
Ett pass og én fold er ingen generell dom over lengre læring eller historikk.

Den tidligere [GC-registerplanen](configs/research/MARKET_STATE_GC_SIDE_MAX_REGISTRY_20261108_001.json)
forblir bevart, men er utsatt og uklamet etter brukerens nye prioritering.
training_enabled=false stenger ny trening; firestegs kvalifikasjonen, koblingskontrollen og alle tidligere
retningsdiagnoser, input-kontrollen, den konvekse M5-planen, cacheplanen
og native fitplanene er konsumert.

Kilde/metode for denne fasen: [docs/NATIVE_LEARNING.md](docs/NATIVE_LEARNING.md).
Detaljert historikk og eldre beslutninger: Git, ikke nye arkivmapper her.
