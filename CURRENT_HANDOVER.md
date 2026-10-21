# CURRENT HANDOVER — XAUUSD M5 Entry, kompakt førsteprofil

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les GX1_RULES.md og kjør `bash scripts/gx1_handover.sh --check`.
NEXT_RUN_POLICY.json eier eksakte bindinger og arbeidsstatus. Én integrator,
én capped tung jobb; fryst kilde/policy under kjøring. Konsumerte planer
relanseres aldri. Originalkvitteringer og checkpoints bevares.

## Mål og beholdt omfang

M5 Entry lærer LONG, SHORT og FLAT fra observerbar markedstilstand.
Retning/netto endepunktsretur, signert MFE og konservativ MAE før første MFE
måles separat fra ett minutt til åtte timer; dager er støtte. Ingen Exit-
vurdering eller Exit-trening. Horisonter er ingen maksimal holdetid.
Behold lukket M5/M15/H1/H4/D1 og de rikeste26månedene med GC MBP-1.
Kvalitet og dekning gjelder også avståtte entries.

Første profil er `compact_mbp_smc_ema_atr_clock_macro_entry_v1`: MBP-1,
SMC/struktur, EMA/ATR og nødvendig klokke-/makrokontekst. Utvalgte candle-/
chart-pattern- og overlappende felt er parkert. Hele biblioteket, åtte familier
og niende D1/makrokontekst beholdes. Ingen felt er erklært støy uten måling.
Separate SMC/MBP-modeller, ensemble og M30 krever kronologisk OOF-merverdi.

TRAIN [2024-05-01,2026-01-01), VAL [2026-01-01,2026-04-01), senere
utviklingskontroll [2026-04-01,2026-07-01). Alle26måneder er gjenbrukt utvikling;
TEST fra juli er forseglet. training_enabled=false; ingen ekstraepoke/HPO/
fullVAL er åpnet. Ingen broker/live/paper/ordre/spending/promotion.
De to godkjente practice-leseoppslagene er konsumert.

## Gjeldende resultat og neste steg

133 er COMPLETE:470nye aktive cacheoutputs,94perTF, bestod fullhistorisk
uavhengig sammenligning på1326,730s/exit0.220kilder/policy og182native-
produksjonsfiler ble bevart.185diskrete outputs/60937483celler er eksakte;
285kontinuerlige outputs/93876663celler ligger innen forhåndslåste cellevise
F64/F32-grenser. Dette er ingen global bitidentitetspåstand.
Alle154814146cacheceller er finite,0NaN.469outputs varierer;D1session_change_flag
er målt konstant0 på4147rader. Egne warmup-/manglende-domener støttes av
håndtestene; denne trimmede cachen gir ingen fysiskNaN-vitne.

Fire eierblokker/TF ble sammenlignet og frigitt sekvensielt: swing20,local27,
level33,cross14.50atomiske immutable fremdriftsbunter matcher sluttresultatet;
alle lokale pendingflags er bevart, og sluttvakter/moduleterminal gir samletPASS.
12genuineOHLCV/q6kolonner,full2009prefix/1215514nativeM5rader,28filer med
1739102302B SHA én gang,94av253cachekolonner. Ingen råMBP/sekvens/PT/modell-
payload, fit eller nyepoke.619256584logiske featurebyte er ikke målt disk-I/O.
133ACTUAL_NUMERIC_REVIEW.json SHAdfe6d9047367b779cf37f89711532be34b37ca95c112c9902001d011068406e9.
133TERMINAL.json SHAbc200474849469addaff51fc7d8c94b27873b603f618186872e787607c1d2ecc.
Bevisroot: `/home/andre2/GX1_RUNS/NATIVE_M5_REMAINING_ACTIVE_CORE_NUMERIC_REFERENCE_20261021_133`.

133QUALIFIED_MTF_NUMERIC_INDEX.json erstatter129indeksen:127+128+129+133
nøyaktig disjunkt,790outputs/158perTF,695kompaktaktive+95parkerte,
260219522finiteceller. Alle139aktivecachefelt/TF er nå numerisk kvalifisert.
SHAf5022afbdcde059440a327d8ea2b9b06cb5bc07a077b482b45b61302232a292f.
95parkertefelt/TF er fortsatt utenfor tallkvalifiseringen. Dette er utganger
på flere tidsrammer, ikke790uavhengige indikatorer eller bevist modellnytte.
Ingen133relansering;130/131/132originaler og konsumerte planer bevares.

134 råMBP→M1 og valgt sluttmaskjoin er nå uavhengigPASS.66filer/
892100712records og62783740trustedT bestod alle20CSVkolonner/EOF/CRC/SHA
én gang perråfil.65nye+gjenbrukt pilot tok2150,829s/exit0;222kilder/policy
bevart.47råM1felt,37event/7Tsize og12997937pris/sidelevels bestod159399314
koordinatsammenligninger med låste heltalls-/F64-/NaN-domener. Alle60targeters
utenforfilunion0/NaN/null/dtype bestod; dagsvolum gir eksakt roll/basis/00:01akse.
Råinventaret20,039GBgzip/118,260GBCSV ble lest/parset én gang hver; bulk60
hash/decode og4pilotpass separat. Originalpartials og råkvitteringer bevares.
Eventwide-sumavrunding var ikke fysiskutøvd; syntetiskbevis er separat.

FINAL_JOIN er COMPLETE/exit0 på7,141s,222kilder/policy bevart.1.139.040rader,
157kolonner over54M1+13quality+40eventunmasked+40eventfinal+10Tsizekolonner:
178829280celler=72898560eksakteheltall/bool+78556709finiteF64bitkopier+
27374011likeNaN-domener. NaNpayloadbits ble ikke sammenlignet. Sluttmaskene,
incident/recovery,47→54feltkopien,alle37eventfelts OFImaske og7umaskerte
Tsizefelt bestod. Ni literalrader/fem slutttabeller og001priorvolumkopi bestod.

60cacheidentiteter ble gjenbrukt utenSHA;45mellomtabeller+5slutttabeller+
001prior3kolonner ble lest én gang hver,51footere førdecode og6nyeSHApass.
Ingen rå-/ladder-/TF-/modellpayload, fit ellertrening.1.441utenvalgtkontrakt-
rader beholder0/NaN/false;17clock-/18stateexclusions er eksakte. FINAL_JOIN_ACTUAL_REVIEW SHA1a43cb8fce38942bca7efc960ee2c139ba85776b43c518047fa644846bf16eef
og FINAL_JOIN_OUTPUT/RESULT SHA88403c0d32dd07bd13b1514a76995dcf1e6b115cccefedd2e937718ed2ed7da6.
Bevisroot: `/home/andre2/GX1_RUNS/NATIVE_M5_RAW_MBP_M1_NUMERIC_REFERENCE_20261021_134`.
Source/pilot/bulk/finaljoin-planer er konsumert/stengt; ingen relansering.
Ingen jobb gjenstår. Uavhengig FINAL_JOIN_ACTUAL_PEER_REVIEW bekrefter
alle157felt/masker/lesepass og ytrevakter; SHA55663fdb21e18fe26a0f0bcf1195f37cdcef9f747062b116e5e9bfb98e4c5ca2.

138 er COMPLETE/exit0 på0,932s,222kilder/policy bevart. Alle512lagrede
115initialbeslutninger/1024sidespor er revidert:1021singletonterminal−MAE-
score er eksakteF32;3sidehorisontties er ikke rekonstruerbare fra means.
Handlingsvalgene stemmer: TRAIN3LONG/0SHORT/253FLAT,CONTROL0/0/256,
0actionties. Dette er optimizer0-startmodell, ikke108s trente modell eller
bevist selektivitet. Egnekomponentnevnere/manglendeutfall bevares; like
terminal-LONG/SHORT/FLAT-counts krever ikke fullobservasjon. Hele480argmax,
individuelle fremtidsmasker og inputtensorer kan ikke rekonstrueres her.

16småfil-lesinger=7JSONlineage/observation+6kilder+3auditmetadata,
4466420byte;0nye forwards/updates/fits ogingen PT/Parquet/rå-/normaliserings-
payload. ACTUAL_REVIEW SHA693122b9039a007c78296f40ed98e097c2ce5d61b3de750c0870f76a51b4e2ed.
Uavhengig ACTUAL_PEER_REVIEW SHAcc5461e5c03f20f18e1314390b28c774da479f57333151309adb11217d0f0da4.
Bevisroot: `/home/andre2/GX1_RUNS/NATIVE_M5_INITIAL_DECISION_TRACE_RECEIPT_AUDIT_20261021_138`.
135/137er konsumert/stengt etter to rettede revisjonsfeil (canonicalLF og
ufundert fullFLATcoverage); originalkvitteringer bevares. Ingen produksjons-
eller datafeil er påvist.138planen er også konsumert/stengt, ingen relansering.

Neste er faktisk individuell modellpåvirkning på kompaktprofilens dynamiske
hoder. Eksisterende legacyQ/fullfeature-helper passer ikke denne fasen.
136/DESIGN.json SHA dd6eb9bed6412e4fc88436267a70d8144e956aa2b52a6ef61b0d68b72e129a9c
beskriver en audit-onlyadapter:882numeriske ruter,2525jointaliaser,7enum,
32familie/TF-grupper,13VJP og høyst218forwards etter eksakt caseplan.
Designet er IKKE implementert/kvalifisert/autorisasjon. Integratorreview,
kausaleTRAIN-maskvitner, gyldig value↔mask/aliasmanifold, batchparitet,
state/data/normalisering/cachebinding og fersk maskinvareadgang gjenstår.
136s gamle135PENDING-note overstyres av138s faktiske kvitteringer i policy.
Full480trace, LONG/SHORT/FLAT-kvalitet/dekning forvalgte/avståtte ogsenere
læring/generalisering er åpne krav. Ingen ny trening er åpnet. Ingen jobb kjører.
121–126/133/134/138 og produksjonsbaseline/paritet gjenbrukes uten omkjøring.

## Bestått — gjenbruk uten omkjøring

133indeksen ovenfor er gjeldende.127115tekniske+12885trend/state/kryss+
129120SMC/geometri+133470gjenståendeaktive bevares uten omkjøring.
132s AVWAP-/warmuprettelser og133s fem progresslivssykluscaser er bestått;
modell/normalisering/baseline er uendret og krever ingen omkjøring av auditkoden.

121–125 kvalifiserer alle1205aktive GCverdier på publisert M1→TF-lag.
126 kvalifiserer1205verdier+1205masker bit-eksakt mot faktisk bruktM5-context:
328119090celler/136149unike kilderader,118249TRAIN-/17284VAL-beslutninger
med96historikk. TRAIN dekker1205/1205;VAL1181/1205.14D1OB+10H1flow/CVD er
korrekt maskert0/0 iVAL, ikke finitt kvalifisert der. Rå→valgtM1/sluttmasker er nå bevist av134.
Tidligere kilde-/data-/normaliserings-/cachekvitteringer gjenbrukes.

Den bestilte hele nativeepoken108 er FULLFØRT:118249TRAINrader,
7391updates,6090,176s. Alle seks samlede TRAINkomponentfeil falt, men1minutt
ble dårligere på begge diagnoseutvalg; alle512kandidater valgte1minutt og
ingenLONG ble valgt. Dette er delvis læring, ingen kvalifisert Entry-kvalitet.
109 viste stor dagsstøtteskala;111 innførte TRAIN-onlyRMS.113 fjernet målt
redundant metadata-/hashlesing.114 ferskinit119,491s,115initial1991,766s,
116 state/RNG bevaring og117 fire private updates/fire paritetsforwards
3358,974s med eksakt resume er bestått. RMS-endringens læringsnytte er ubevist.
Ingen ny baseline/modellrunde kreves av uendret produksjon og ren auditkode.

## Åpent før ferdigmelding

RåMBP→valgtM1/sluttmasker er kvalifisert. Kontroller individuell faktisk
modellpåvirkning og beslutningssporbarhet uten ny råscan. Kvalifiser LONG/SHORT/FLAT, valgte og
avståtte entries, kvalitet/dekning og senere generalisering hver for seg.
Ingen ny treningskampanje er åpnet. Data-PASS, mekanikk, læring, generalisering
og økonomisk edge holdes atskilt; hele prosjektmålet er fortsatt ufullført.

Fysisk18.10-omstart er bekreftet; nyere Windowsboot20.10T20:15:55.500Z
ble bekreftet i117. Ikke gjenta gammel omstartsbestilling. CPUaudit krever
ikke ny omstart; fremtidig GPUadgang følger faktiske maskinvareporter.
