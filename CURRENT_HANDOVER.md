# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Les GX1_RULES.md helt. NEXT_RUN_POLICY.json er eneste arbeidsstatus;
originale prosess-/terminalkvitteringer overstyrer prosa. Historikk finnes i Git.

## Gjeldende status 20.11.2026

Nytt brukerbestilt mål: bygg og klargjør stor test med TRAIN[2021-01-01,2026-01-01)
og utviklingsVAL[2026-01-01,2026-07-01). Kontroller hele den aktive pipelinen
kritisk underveis og ved slutt, og meld brukeren ved startstreken. Selve
stortreningen skal IKKE startes i dette arbeidet. Appmålet og policy-målet er aktive.
Kvalifisert cache003 stopper2024-10-01. Utvidet kilde/inputgrunnlag, ferske
helefemårsTRAIN-stats/prior/full initialbaseline, B64optimizer/kaldreplay og
én eksakt hovedplan må kvalifiseres. NativeTEST er fortsatt forseglet.
Tidligere YEAR_FIT001 er supersedert utenCLAIM og skal ikke relanseres;
årskvalifikasjon/profiler er bevart som historisk omfangsbundet evidens.
Alle globale treningsporter står stengt. Ingen ny tung jobb eller storfit startet.

M1 brukes ALDRI til retning: ingen input, fasit, metadataforelder, rader,
prior, normalisering, evaluering eller warmstart. Bare senere oppfølging
av allerede åpnet trade. M5 med lukket M15/H1/H4/D1 er retningsgrunnlaget.
GC er utsatt; eksisterende kode/data beholdes. TEST fra01.07.2026,
Entry/Exit, broker, paper/live, ordre, spending og promotion er stengt.
Juni2026 og øvrige gjenbrukte perioder er utvikling, ikke urørt test.

## Ny bred revisjon før større test

Alle2160sporede filer er inventarisert:1406PythonAST,650JSON og9shellfiler
består syntaks/parse.18M1/049-metadatafiler er bare navnelistet, aldri åpnet.
321statiske importnoder er ikke bevis for faktisk modellkonsum. Dypkontrollen
følger21formeleiere og den aktive provider→target→modell→loss→checkpointkjeden;
hele historiske funksjonsbiblioteket er ikke atferdstestet på nytt.
Fire faktiske formel-/spread-eiere er nå eksplisitt kildebundet. Originale
Gitbytes matcher cache/pipeline/mekanikk, så ingen feature-rebuild trengs.
Full21-eiervalidering kommer før eksplisitt gammel17-key kvitteringsprojeksjon.
Makroparseren får read_end-predikat før DataFrame-eksponering; den gamle
resample-cutoffen er uendret.2syntetiske seal/journal-tilfeller består.
Dette beviser ikke at Arrow aldri dekoder en blandet rowgroup fysisk.
5closure/kvitteringskontroller,8nye guardvarianter og5profilkildetilfeller består.
To GPU-profiler er fullført med0optimizerupdates og eksakt samme logits, tap
og klippede gradienter. Første profil målte B16=1,194s/B64=1,418s; det er
3,37ganger flere rader/s på disse panelene, uten AdamW. Andre profil reduserte
per-parameter finite-synkroniseringer: aten::item1435→106 og CUDAstreamSync1537→208.
B64-referanser1,281/1,276s mot1,140/1,154s med per-device-bools; rundt10%
computegevinst på samme panel, ikke bevist total treningsgevinst. Alle finite-vakter
består; ingen loss-/optimizerformel eller sikkerhetsgrense er fjernet.
Større test er klargjort som ett helt input-TRAIN-år, én B64-pass
og påfølgende septemberCONTROL. Nye heleTRAIN-stats, prior og baseline er fysisk kvalifisert.
Hovedtreningen er ikke startet; gammel kortplan skal ikke relanseres.
Ny årsroute er kildeklar: TRAIN[2023-09-01,2024-09-01)=70935rader,
CONTROL[2024-09-01,2024-10-01)=5789rader; B64gir1109updates/énpass.
33literal-/SHA-/guardtester og16fold-/mask-/tail-/coldcursor-tester består.
GPUforberedelse er1200forwards/0updates med ferske heleTRAIN-stats/prior.
Tekniskmekanikk er to ulikeB64-batcher + kald replay:2modeller/7F/3updates/6prober.
Hovedplanen skal ha1ferskmodell/2309F/1109updates, gjenbrukt faktisk initialscore,
fullTRAIN+CONTROL-finalscore og3600s hardgrense. Årsforberedelsen er fysisk
fullført på566,864s:1modell/1200F/0updates,1ny normalisering og1ny matchedprior.
Alle initiallogits er eksakt nyeTRAIN-priorlogits; modell/RNG og source/policy
er uendret, firewall0avvisteforsøk. Hele70935rader brukes i normaliseringen.
M5close-fasit70934celler:21210NED/27751NEUTRAL_NET/21973OPP
(29,90/39,12/30,98%). Dette er støymarginfasit, ikke bevist ikke-tradbart marked.
B64optimizer/kaldmekanikk fullførte på85,383s:2modeller/7F/3updates/6prober,
bitidentisk andre-batch-replay i ny prosess, uendrede stats/prior og firewall0.
FørsteM5-sharedgradient0 fra nullreadout er forventet; andre=0,000794187,
auxiliary→shared er0. Peakreserved6,91GB. Hovedfit bruker ingen slike prober.
Hovedplan YEAR_FIT001 er publisert utenCLAIM:70935TRAIN/5789CONTROL,
1ferskmodell/2309F/1109updates/1pass; tidligere1200initialforwards gjenbrukes
etter eksakt tilstand/input/target/norm/prior-kontroll. Aldri tekniske vekter.
Hardgrense3600s, ingen hovedfit startet. Endelig metadata-preflight består;
originalkvittering og samlet LARGER_YEAR_TRAINING_READY_20261120.json ligger
i cache003-roten og er SHA-bundet fra NEXT_RUN_POLICY. Uavhengig JSON-kontroll
av20predictive proper-scoreceller gir maksavvik4,37e-11. Ingen nye modeller/arraydecode.
Gamle current-baseline/statusfelt i policy pekte fortsatt på liten råbar/GC-funksjon;
disse er korrigert til faktisk årsbaseline/tekniskmekanikk. Historiske kvitteringer består.
52målrettede tester består. Faktisk hovedpass-tid og læring er ennå umålt;
3600s er hardgrense, ikke garanti for fullføring. Ingen endring i markedsvaktene.

## Ferdig seksårs inputgrunnlag

NATIVE_M5_TA_HISTORY_EXPANSION_20261120_003 fullførte exit0 på1181,156s
(19m41), CPU12G/512M/1tråd/64tasks. Source/policy var uendret.
Issuevindu[2018-10-01,2024-10-01):425320issues/425415lokalrader,
samme fulle kausale native2009-prefiks og frosne2013-parametre.
Alle8TA-familier,61D1/makrofelt,5TF og9modellinputflater beholdes.
Signal318,ctx132+1kategorisk,MTF256. Ingen modell/norm/paramfit/targetdecode.
Ny seksårs modellbaseline eller læring er IKKE målt.

| År | Inputklare issues | Manglende |
|---|---:|---:|
|2018, fra oktober|17788|0|
|2019|70692|0|
|2020|71033|0|
|2021|70855|0|
|2022|70881|0|
|2023|70684|0|
|2024, til oktober|53387|0|
|Sum|425320|0|

Alle61makrofelt er finite på alle425320issues, med gjeldende kausale
publiserings-/asofregler. Fysiske verdier/NaN og ekte readiness bevares.
Dette viser inputdekning, ikke hvert felts prediktive verdi.

Én uavhengig NumPy/stdlib-audit under audit4G bestod exit0 på16,377s,
peakRSS875,09MiB: alle27arraydtype/shape/SHA, globaleM5/TF-eier-IDer,
tilgjengelighetsklokker/vindusunioner,132kontekstaliaser og rullende
readiness er kontrollert. Ingen Torch/provider/modell/norm/target/råpriser.
Alle27gamle mai–september2024-overlappsarrays er byteeksakte,
inkludert NaNs/klokker/globalIDer og29964issues.41source/policy/artifact-
bookends var uendret. CACHE_QUALIFICATION.json og original audit/script
er hashbundet i policy; originalproduksjonskvitteringer er urørt.
Fullt27arrayvolum1563164006B, arkiv614460598B.
primary_matrix_bytes1533220514B er delsum, ikke totalvolum.
Fasekvitteringer målte7054,47MiB maxRSS gjennom reload; dette er ikke
påstått som måling av alle etterfølgende revisjons-/årsrapportfaser.

## Målte blokkeringer som er rettet

Gammel aktiv inputcache dekket bare mai–september2024; full kildeakse
var ikke ferdig featuredekning. Native råhistorikk fantes allerede.
CACHE001 ble drept med ytre137 uten NPZ/RESULT/nativeTERMINAL; kernel-
OOM i kjøreperioden, men gammelCLAIM manglet direkte cgroup/PID-binding.
OriginalCLAIM og ekstern OPERATOR_EXIT_OBSERVATION er bevart, ingen
terminal er fabrikert. Gammel dødsfase/COW-setting er ukjent.

En liten audit4G-probe viste Pandas2.3.3/COW-off som cachet3fulle attrs-
matriser ved3kolonnevalg (64→192B). Den aktive M5-aliasløkken hadde
samme Series-kall med store featureattrs. Nå leses samme float32kolonne
som NumPyview, uten Series/itemcache/attrs-deepcopies. Bytes/NaN,
navn/rekkefølge/lineage og view-eierskap er bevart. Eksakt helper+ett
RHS-bytte normaliseres i smalASTbro; øvrige17numeriske eiere er uendret.
Ingen HTF-/writerrefaktor eller capøkning ble gjort.

CACHE002 passerte aliasløkken, men feilet ved makro._research sin eager
learner/model_free-import til walkforward/direction_decision/Torch.
Native FAILED_CONSUMED-exit1 og ekstern kildebookend er bevart separat;
terminalens manglende sourcefelt ble aldri diktet. Kalenderen var ikke
årsaken. Kun to importgrupper i research_ta_campaign er flyttet til14
runtimebrukere. Makro-/asof-/fit-/porteføljemattematikk er uendret.
Testreferanser til importerte implementasjonsnavn peker nå faktiske eiere.

11fokuserte cachetilfeller fra alias-/livsløpsbølgen og3nye import-/002-
tilfeller består;9berørte research-regresjoner og obligatoriske modell-
native precommit-kontrakter består. Tidligere beståtte kontroller gjenbrukes.
Ren fresh-process makro/asof under Torch-nekt bestod uten datalesing.
Uavhengige kilde-/planreviews og eksakte CLI-preflight bestod før hver
engangsinnrømming.003 bevarte12G/3570s og har cap/cgroup/PID iCLAIM
samt10små immutable før/etter-fasekvitteringer.001/002relanseres aldri.

## M5-resultat før seksårsutvidelsen

Alle følgende målinger bruker samme faste1536FIT/512senere september2024-
CONTROL; gjenbrukt utvikling, ikke uavhengig test. Fast pass4 er slutt.

| Referanse | FIT logloss | CONTROL logloss | CONTROL Brier | CONTROL AUC |
|---|---:|---:|---:|---:|
|Matchet TRAIN-prior|0,709405|0,716887|0,504676|0,500000|
|BLOCK_ORDER001|0,708095|0,720930|0,506435|0,476531|
|M5-primary-backbone|0,679772|0,732819|0,520350|0,478913|

Rekkefølgeforsøket endret bare hele TRAIN-blokkers lokalePCG64-rekkefølge.
Shared-gradientdiagnosen målte aux/M5-norm24,91 ogcosinus−0,28891 på4
fasteTRAIN-paneler,0updates. Dette var rå evalgradient, ikke historisk
AdamW-årsaksbevis. Optional m5_close_only_backbone_v1 lar M5-close styre
fellesdelen, mens øvrige9hoder fortsatt lærer fra detached hidden.
Alleinputs/5TF/fasit/native loss/Ntotal/optimizer beholdes; default-ruten
er uendret. Fersk baseline/gradient/cold-resume var genuint kvalifisert.

M5-primarymain:1fersk modell/4fastepass/384updates/544forwards på578,831s,
ingen norm-/priorrefit, gamle vekter eller CONTROL-basert best-passvalg.
FIT Brier0,474310/AUC0,653148; senere M5 taper fortsatt prior.
CONTROL-meanP(NED/LIK/OPP)=49,7707/0,2336/49,9957prosent,
argmaxOPP256/511 mot faktisk243/511. Nær50/50 er ikke kalibrering/edge.
Andre4TF-close slår prior, men bare175/48/18/12unike CONTROL-mål.
D1-open har0prediktivstøtte. Ingen falsk kreditt fra kjent-open/ukjent-tail.
Uavhengig scorekontroll av alle faser/hoder og384batchhashes ga maksavvik0.
Source/policy var uendret og originalcheckpoint/captures er bevart.

## Målt seksårs fasitstøtte uten ny fit

Én audit4G/512M fullførte på4,840s/RSS232,20MiB. Alle425320inputissues
matcher både source_row og issue_ns i AKTIV target003 (SHA49be65ca…).
Policyens eldre002-felt er historisk; active_native_bar_target_tape i
current_work peker nå eksplisitt003. Alle future-fortegn er kontrollert
mot faktabarene, med strict validity<2024-10-01, ekte predecessor/mature-
masker, kjent-open separat og ukjent/cutoff aldri som LIK.

| TF | Kjente framtidige sluttmål | NED / LIK / OPP | Unike målbarer | Prediktive åpninger |
|---|---:|---|---:|---:|
|M5|425319|210329 / 1186 / 213804|425319|425319|
|M15|425317|210077 / 561 / 214679|141824|141771|
|H1|425308|208474 / 287 / 216547|35472|35425|
|H4|425297|204840 / 228 / 220229|9275|8239|
|D1|425297|196786 / 0 / 228511|1550|507|

M5-fordeling49,4521/0,27885/50,2691prosent; én cutoffcelle. OPP/NED-
masse er nær balansert, LIK er sjelden eksakt prislikhet. Dette er ingen
fasit for sideveis marked. D1-open har507prediktive celler i HELEvinduet;
det endrer ikke nullstøtten i gammel1536FIT-kohort og beviser ikke læring.
Nye priors/støtte må bindes til framtidig deklarert TRAIN-fold/kohort.

Års-klassetellinger/cellesummer stemmer; høyere-TF-årsunikheter overlapper
vedårsgrenser og summeres ikke som globalunikheter. Fast last-closed
persistens var kausal (available<=issue), M5hit48,2313prosent mot
forhåndsvalgt alltidOPP50,2691. Ingen fortegns-/modell-/terskelutvelging
ble gjort fra dette. Klassifikasjonsreferanser er ikke økonomi/edge.

SIX_YEAR_BAR_SUPPORT_AUDIT/REVIEW/script/cappedlog er immutable og
hashbundet i policy. Evidence-agenten reviderte script+aggregataritmetikk,
uten egen nyarraydekoding. Hashbookends gjelder policy/inputartefakter,
ikke en NY41-kodefilkontroll. Kilden ble holdt urørt; tidligere kvalifikasjon
beholdes. Første /tmp-scriptstart manglet repo i sys.path og stoppet før
data; canonical importrute ble rettet, deretter én fullført lesning.
Ingen prisarrays/modell/fit, ny klassevekt/prior/fold eller fasitendring.

## Liten støymargin — målprodusent kvalifisert

Brukeren ber om en liten bunnmargin for støy. Target-eieren har nå en
separat eksplisitt ATR-nøytralsone for netto åpning-/sluttendringer.
Ingen skjult standardkoeffisient er innført. Vanlig build, dataclass,
råkontrakt og gather er urørt; hele opprinnelige AST og råfeltbytes er
bevist uendret. Ny producer-only gather legger til noise_sign/known/
predictive-eligible/delta og kausal ATR-/marginmetadata, aldri modellinput.

19nye testtilfeller under audit4G og uavhengig statiskreview består.
En genuin24175-issueTRAIN-projeksjon bestod på2,934s/RSS487,77MiB:
uavhengig WilderATR-orakel på alle5TF, alle nye masker/tegn/delta,
asof<=issue og hver rågatherbytes er kontrollert. Beregning dekodet hele
lagrede preTEST-prefix; bare kausale TRAIN-asofverdier ble brukt.
0.5 var teknisk kvalifikasjonsverdi fra gammel valgt eier, IKKE valgt
produksjonsverdi for denne lille marginen. Den påfølgende ATR03-ruten har nå egen persistert målidentitet og
kvalifisert modell/prior/loss/score/checkpointkobling, se gjeldende status.

Readonly TRAIN-deskriptorer målte gammel FNET:.5ATR gav59,8995prosent
nettonøytrale M5-utfall av24087eksakte issue+5min-endpoints; medianbånd
0,75995USD. Dette er ikke range-FLAT eller den nye neste-TF-bar-projeksjonen.
Alle24175TRAIN-issues hadde positive faktiske nativebid/ask-spreader,
median0,40USD, median spread/ATR0,257713. En hel observert quote-spread
gav35,1642prosent nettonøytrale utfall i samme gamle FNET-domene.
Quote-spreaden er ikke dokumentert statistisk støygrense eller realisert
kostnad. Ingen koeffisient er valgt fra disse klasseandelene, og ingen
CONTROL/TEST-basert utvelging er gjort. Script/aggregataritmetikk er
leserevidert uten nyarraydekoding. Alle audits/script/log/proofs er
immutable og hashbundet i NOISE_MARGIN_SOURCE_QUALIFICATION/policy.
Alle trenings-/kjøringsporter er fortsatt stengt.

## Brukerforeslått0,3 mot0,4ATR — TRAIN-sammenligning

Én readonly audit4G/512M/1tråd fullførte på2,950s/RSS488MiB.
Begge marginer brukte samme24175issues fra deklarert TRAIN[01.05,01.09)2024,
samme kausale ATR14-skala og masker i neste faktiske nativeTF-bar-domene.
Alle rågatherfelt er byteuendret.0,4 flytter bare flere utfall til nøytral,
aldri fra OPP til NED eller omvendt. Source/input/policybookends bestod.

| TF-close | Nøytral ved0,3ATR | Nøytral ved0,4ATR |
|---|---:|---:|
|M5|39,1909%|50,1779%|
|M15|42,7401%|53,6361%|
|H1|45,7392%|56,4615%|
|H4|40,1613%|52,7363%|
|D1|36,2854%|46,5605%|

M5 har24174eligible mål: NED/nøytral/OPP=7175/9474/7525 ved0,3,
5922/12130/6122 ved0,4. HøyereTF er issuevektet, ikke uavhengige unike
målbarer; D1-open har fortsatt0prediktivstøtte i denne TRAIN-kohorten.
0,3 er assistentens første kandidat fordi den er det mindre inngrepet,
deklarert før modell-/CONTROL-resultater. Ingen klassekvote ble valgt.
Dette er netto-prisendringsnøytralitet, ikke bevis på rolig prisbane eller
hele markedets sideveis-FLAT. Ingen seksårs marginmåling eller modellfit.
ATR_MARGIN_03_04_TRAIN_COMPARISON.py/.log/.json og lesereview er immutable
i cache003-root og hashbundet i policy. Uavhengig script-/aggregatreview
bestod uten nyarraydekoding. Den valgte treningsmarginen er nå0,3;
persistens, modell/prior/loss/score og fersk initialbaseline/kald mekanikk
ble først kvalifisert på kort rute og nå på heleTRAINåret ovenfor. Ingen serving-/handelsmargin
er innført. Alle kjøringsporter er stengt.

## Gjenværende full markeds-FLAT-grense

Brukeren bekrefter FLAT=sideveis marked og at eksakt like priser inngår.
Dette viderefører eksisterende forståelse; ingen ny lokal range-terskel er valgt.
Den tidligere manglende-avklaring-blokkeringen var unødvendig og er fjernet.
Ikke spør om samme betydning igjen. Rå åpning-/sluttfakta beholdes fullt.

Kilderevisjon bekrefter et mål-mismatch i den siste barretningstesten:
native target bruker np.sign, objective labels=raw_sign+1 og samme tegn
går til prior/score. Modellens tre logits beskriver da pris-NED/LIK/OPP,
ikke hele sideveis-FLAT-målet. ATR03-ruten bruker nå egne noise-fortegn og sanne taskmasker gjennom
hele kjeden; gamle EQUAL-vekter/resultater ommerkes ikke til marked-FLAT.

Tidligere valgt lokal aktivitet er R<0,5ATR14=FLAT, R≥=AKTIV, med egne
sannferdige sidemasker. En ren nativeM5-v1-eier er bevart i Git og
native001-planen; M1-loadere gjenåpnes ikke. Den bruker1/3/12/48/288
nativeM5-barer på femTF og kausalt lukket ATR14. FLAT_NET inkluderte
eksakt null nettoendring; lokal range og netto må fortsatt ikke blandes.
Dette er eksisterende definisjoner, ingen ny parameterbestilling.

Gammel nativev1 resetter span på ethvert5min-gap og hadde0kompletteD1-span.
Dekning/gap og framtidsreferanse kan derfor ikke ommerkes som kvalifisert.
Full range-/markeds-FLAT utover netto-støyproxyen krever fortsatt riktig
native eier og dokumenterte pause-/ukjentregler. Rå barfakta beholdes. Alle8TA-familier,
61makro og femTF består; GC utsatt og M1 aldri til retning.
Ingen hovedfit er innrømmet. ATR03-nettofunksjonens ferske initialbaseline
og kaldmekanikk er kvalifisert. Årsplanen krever egen eksakt engangsadmisjon;
den starter ikke som del av klargjøringen. Tidligere kortplan er supersedert.
## Bevart grunnlag og oppryddingsgrense

Hele native prisfasit2009–2026 før TEST er tidligere merket og alle76arrays
uavhengig gjenskapt; M5/M15/H1/H4/D1-barer1215513/406109/102354/26427/4417.
Første uten forgjenger perTF er ukjent, gap annoteres, bar som lukker på
TEST-sealet utelates. Denne faktahistorikken er større enn seksårsinputene.
26måneders MBP/GC-korpus/femTF-pakke beholdes, ikke brukt i TA-runden.
M1-avhengig EMA/S/F-evidens er historisk og gir ingen retningsberedskap.
Entry-v3 er dokumentert kontrakt, ingen ny kjørbar fit/handelsautorisasjon.
Ingen fast tapsgrense/maksimalholdetid er innført.

Ingen repo-/DATA-/RUNS-filer er slettet i denne bølgen. Ingen kode slettes på
mistanke; referanse-/import-/test-/eierskapskontroll kreves. Bare retention-
eieren rydder DATA/RUNS etter GX1_RULESregel9 med alle referanser/åpne filer
kontrollert. Alle originalkvitteringer, caches og checkpoints bevares.
Ikke gjenta eldre fullført retention/Windows-/cache-opprydding.
Kode/docs/interne stier/aggregerte bevis kan pushes til autorisert branch;
aldri rådata, vekter, hemmeligheter eller force-push.

## Historisk kort ATR03-kvalifikasjon — supersedert

PREPARE002 og ATR03_MECHANICS001 var kvalifiserte små forgjengere med
1536FIT/512CONTROL,32initialforwards og67mekanikkforwards. De er fullført,
bevares immutable og skal ikke relanseres. Den tidligere korte hovedplanen
bff73ed6…20b7d ble aldri claimet og er SUPERSEDED_UNCLAIMED_BY_BROADER_SOURCE_AUDIT.
Den gir ingen kjøringsrett. Små gammelstats fra source_axes944b brukes ikke
på årsgrunnlagets source_axes2173; året har nye heleTRAIN-stats/prior.
Gjeldende hovedplan og faktiske årskvitteringer står øverst og i policy.

## Målt fart før større test

THROUGHPUT001 og002 er begge COMPLETE_CONSUMED, aldri relansering.
Første målte3,37ganger flere rader/s ved B64 ennB16 på faste paneler.
Andre beviste eksakt samme logits/tap/klippede gradienter etter finite-reduksjon,
med10,25% lavere gjennomsnittlig compute-tid på to samme-B64-par.
Ingen optimizerupdates i profilene; faktisk hovedpass-tid er fortsatt umålt.
PROJECT_TRAINING_THROUGHPUT_REVIEW_20261120.json binder begge terminaler og
presise målinger. Guardendringer, geometri og årsforberedelse står ovenfor.
