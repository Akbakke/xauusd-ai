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
Ingen134jobb gjenstår. Uavhengig FINAL_JOIN_ACTUAL_PEER_REVIEW bekrefter
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
Integratorreview er ferdig med presisering om at hele siste sekvensrad og
snapshot er samme fysiske markedstilstand. Også ikkealiaser må få summert
seq+snap-derivat før absoluttverdi; contextaliaser summerer alle tre kopier.
139 er COMPLETE/exit0:28eksakte CPU-håndtester bestod på9,392s inkludert
signert alias-/ikkealias-kansellering,13analytiske derivatfasiter, egne masker,
enum/familiecaser, F32-overflow og lazyB16/paritet/padding.224kilder/policy
ble bevart. Ingen produksjonsfunksjon, dataset, normalisering eller baseline
ble endret;0nativeforwards ogingen PT/Parquet/rå-/normaliseringspayload.
139ACTUAL_REVIEW SHA36c5926c28dc8ba4d12f7400c4f73e362e8114b8cdcb86186da08bf62178eb75;
ACTUAL_PEER_REVIEW SHAe1d13c9ebf5835160810ed7424903870ead45159e9692d47f32bd69f0556e59b.
Bevisroot:`/home/andre2/GX1_RUNS/NATIVE_M5_DYNAMIC_COMPACT_INPUT_INFLUENCE_SOURCE_QUALIFICATION_20261021_139`.
139planen er konsumert/stengt. Dette er syntetisk mekanikk, ikke faktisk
modellpåvirkning eller læring.

140 er COMPLETE/exit0 på47,906s:118249TRAINrader, kun time/ctx_cont,
2525/2525aktivecurrentalias-maskeiere dekket av3unikevitner,0udekkede.
Parentrader[45713,14387,25188] gir nye eiergevinster[2465,40,20]; valget
brukte kun tilgjengelighetsmasker, ingen utfall/prediksjoner. FrystB16 fyller
13uscoredeplasser med FØRSTEvitne; de teller ikke som nye observasjoner.
Alle2525eierne har36..118249gyldigecurrentTRAINrader. Én fersk TRAIN-hash
på19146223003byte tok33,579s før eneste kolonneprojeksjon; ingen batchhash.
117s SHA/stat er kun lineage, ingen varig cache/unntak.0nativeforwards,
PT/rå/VAL/TEST/target/provider-/normaliseringspayload, fit ellerupdates.
224kilder/policy og dataidentitet/klokke ble bevart.140ACTUAL_REVIEW
SHA0d9d4dbc5ac3e53a2c48ba96cf1d27c3c7dd8c662cd476d3bd85d77cd54a81aa;
ACTUAL_PEER_REVIEW SHAed415d69bae1d777a66d0d4c944ac11f015e11b43615279b0a8b4f632d14bf1a.
Bevisroot:`/home/andre2/GX1_RUNS/NATIVE_M5_DYNAMIC_COMPACT_TRAIN_WITNESSES_20261021_140`.
WITNESS_OUTPUT/WITNESSES.json SHA3eb5fbf35c895bde814de205b794978d71158cdafa906c0f81370dded3aed233.
140planen er konsumert/stengt. Gamlecarry-scope/reviews er overstyrt førlaunch.
Beviset gjelder currentaliasers masker, ikke komplette inputtensorer eller
96historie/MTF/genericsnap/kategorisk påvirkning.

141 er COMPLETE/exit0 på 103,633 sekunder: tre TRAIN-inputs er satt sammen
med ekte 96-raders M5-historikk og alle fem tidsrammer. Alle aktuelle aliaser,
seq-last/snapshot og 140s tre F32-context-hasher bestod. Alle 15 valgte
MTF-vinduer er lukket ved vitnenes faktiske beslutningstid. Ingen fremtidsinput.

INPUT_RENDER_OUTPUT/INPUT_PACKET.npz inneholder 9 unike rå inputarrayer,
parentrader, B16-radkart og scoremaske: 3 ekte + 13 uscorede kopier av første
vitne. 4.793.460 logiske arraybyte er komprimert til 1.280.070 byte. Gjenåpning
med allow_pickle=False bevarte eksakte nøkler, dtype, shape og bits. Pakken skal
GJENBRUKES i neste kontroll uten ny råscan eller dekoding av TRAIN/M5-radene.
SHA 6c287b8fbd857a233cd3e3019dea68a1e445099651c942f8cd82fecf8a8ecbe0.
Metadata SHA 1819b343a74a8aacaa7eac2c2130c6bad2ae678a2c81ab7eb2315beda3e2e656.

CPU-producerens cgroup verifiserte 8G/512M, 64 tasks og én numerisk tråd.
224 kilder/policy ble bevart. Begge Parquet-footere bestod før hash/cache/decode;
største projiserte rowgroup var 5.922.839 TRAIN-byte og 1.047.786 M5-byte.
Tre TRAIN-rader krevde 3 rowgroups/1.536 dekodede rader; 288 M5-historierader
krevde 6 rowgroups/576 dekodede rader. Disse byte er metadata, ikke målt disk-IO.
Cachen ble lastet én gang; én fersk TRAIN-hash og én fersk valgt M5-hash.
Cachekildens M5-fil var en annen fil, så dens hash ble ikke feilaktig gjenbrukt.
Cache og hasher tok 95,155 sekunder. Ingen PT, modell, targets, provider,
Dataset, fit, updates eller TEST; ingen trening ble åpnet.
Eksternroot: `/home/andre2/GX1_RUNS/NATIVE_M5_DYNAMIC_COMPACT_INPUT_RENDER_20261021_141`.
ACTUAL_REVIEW.json SHA 13cb327ce8db9d3973e44885e8af8196452710b4c45c54c1d57eef382d620cf6.
INPUT_RENDER_TERMINAL.json SHA 63853609c001e977d2b212737fc715013286f418e4b11d7ff1f9fd13031eed05.
Planen er konsumert/stengt; ingen relansering. Uavhengig ACTUAL_PEER_REVIEW
er PASS, SHA 2f4d425b1314177fde257e770e6ec08d2546e29a9af5bb532b087e445cc9014b.
Dette beviser inputpakken, ikke faktisk modellpåvirkning eller læring.

Etter inputpakken gjenstår eksakt modell-/caseplan: eksisterende fixed_step._model
og trainer._CandidateTrainingSession(read_only=True).load_checkpoint; kun
model_state fra 115s optimizer0-state med ONLINE-digest, ordinær checkpoint-policy,
055-inputnormalisering og 111-loss-RMS i sine separate roller. Ingen bypass/EMA/
teacher. Fersk maskinvareadgang kreves før faktisk modelljobb. Deretter måles
individuell inputpåvirkning; input-PASS er ingen lærings- eller nyttepåstand.
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
