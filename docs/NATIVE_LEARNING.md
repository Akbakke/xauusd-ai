# Native læring — gjenværende metode

Nåstatus, utført arbeid og aktuell hindring står bare i CURRENT_HANDOVER.md
og NEXT_RUN_POLICY.json. Dette er metodekrav, ikke en launchordre eller
historikk over daglige forsøk. Bruk eksisterende eiere og genuine resultater;
ikke relanser ferdige inputs, kapasitetstester eller engangsplaner.

## Før en modell kan startes

Aktuell ren/committet/pushet kilde, fysiske inputs, preprocessing og
TRAIN-/CONTROL-koordinater må være eksakt bundet i kanonisk recipe/campaign.
Dynamisk Entry-only binder faktisk hele TRAIN-populasjonen og kanonisk
ordre via eget vitne; dette er ingen målt Exit-samplerbenchmark.
Manifestets split-end er eksklusiv og kan være lik cutoff. Faktiske
fit-/pris-/pointerklokker må fortsatt være strengt før riktig cutoff. Et separat finite native-vindu og eksisterende
cgroup-/host-/GPU-vakter må faktisk godkjenne kjøringen.

Fysisk omstart krever at alle prosjektjobber, writers, låser og GPU-beregninger
på maskinen er ledige. Etterpå verifiseres ny Windows-boot, WSL, kilde og
artefakter. En terminal eller lav temperatur alene er ikke slik dokumentasjon.

Fersk nullstegs ONLINE/TARGET skal bruke samme nåværende funksjon og delte
moduler. Historiske checkpoints, constructor metadata eller identiske
vekthasher erstatter ikke genuine initial-/funksjonsparitetsmålinger.
Endret ONLINE-funksjon krever en ny faktisk initialbaseline.

Dynamisk Entry-only bruker dagens feltdimensjoner, feltrekkefølge, åtte
spesialistruter og cachebindinger i fersk constructormetadata. Seks eksplisitte
arkitekturskalarer kan videreføres fra hashbundet metadata; historiske vekter,
gammelt feltkart og gammel normalisering brukes ikke. Faktisk nullstegs
ONLINE=TARGET=EMA0 og funksjonsparitet må deretter måles på nye kilder.

Dynamiske nettotargets bruker bid/ask-priser pluss separat hashbundet
deklarert kurtasje, slippage og finansiering. Kostautoriteten skal dekke
samme TRAIN/control-periode og samme M1-Parquet og manifest som målgrunnlaget.
Aktuelle vilkår er prospektive antakelser; de beviser ikke historiske kostnader.
Ingen utdatert autoritet relabeles, og ingen manglende kostnad settes til null.
Regelen om stengt broker gjelder også nye leseoppslag uten særskilt tillatelse.

Frosne256TRAIN- og256CONTROL-entries måles separat. Percohort er profilen
16encoderforwards med batch16,256entries,10800s og progress hver16forwards.
Alle480horisonter deler samme encoderforward. Disse computegrensene lager
ingen Exit-stateviews eller handelsregel. Native-/yttergrensene12000/13800s
og eksisterende maskinvarevakter beholdes.

## Teknisk kvalifisering av dynamisk Entry

Før epoken må faktisk dynamisk komponentparitet og Entry-only optimizer/resume
bevises på samme kompaktprofil og frosne data. Legacy entry_action_q_bps-smoke
er ingen slik kontroll. Initialmåling bruker null optimizersteg og
training_enabled=false. En særskilt hashbundet teknisk kvalifisering må ha
training_enabled=true, full_epoch_training_allowed=false og eksakt avgrenset
policy/recipe/campaign før noen teknisk update tillates.

Paritet bruker første16 i de samme256TRAIN/256CONTROL-kohortene, evalmodus
med og uten gradientsporing, fire forwards og null updates. Alle480queryer,
komponenter, forecasts, kandidatretning og ties kontrolleres; toleranse er
forhåndsvalgt1e-4bps/rtol0. Modell, RNG, optimizer/EMA og originalpointer
bevares. Batchenes inputs, queries, targets og masker bindes separat.
Faktisk097 viste ulike fused hostdispatch-veier mellom no_grad og grad
med samme CUDA/eval/FP32/input/vekter; første komponent overskred låst
toleranse. Felles _model_forward_fp32 avgrenser nå MHAfastpathFalse til
dynamisk Entry/deterministiskFP32 rundt selve modellkallet og gjenoppretter
forrige verdi i finally. SDPAvalg, autograd og andre faser/presisjoner
beholdes. Faktisk CUDA-paritet skal fortsatt verifiseres på ny kilde. Toleransen beholdes; hostdispatch alene er ikke
kausalt bevis for rettelsens effekt.

Ved paritetsfeil publiseres PARITY_FAILURE.json atomisk før samme feil kastes
videre: rolle/kohort, fullførte forwards, faktiske sammenligningstall og
gjenopprettingsobservasjoner. De samme to forwardene per rolle observeres
med CPU-only Torch-profiler uten shapes, stacks eller minneprofil. Kun
ATen-attention-/transformer-operatørnavn og antall lagres. Dette viser
hostdispatch, ikke CUDA-kernel eller treningshastighet; overhead er umålt.
Ingen ekstra forwards, endret toleranse eller backendvalg følger av diagnosen.

Resume-kontrollen bruker første32rader i057s kanoniske TRAIN-ordre, batch16:
to sammenhengende steg mot ett steg, faktisk varig lagring/gjenlasting og
samme andre steg. Fire tekniske updates totalt; samme eksisterende loss,
AdamW, EMA, scheduler, clipping og gradvakter. ONLINE/anker/optimizer/EMA/
scheduler/RNG/orden/cursor skal være ekvivalente. Ingen CONTROL-updates,
vektsortering eller gjenbruk av kvalifiseringsvektene i hovedepoken. Dette
er én serialiseringskontroll, ingen ekstra økonomisk treningsarm eller HPO.
Kildens faktiske innrømmelse og målte terminalkvitteringer må fortsatt finnes;
metodebeskrivelsen starter ingen jobb og åpner ingen port.

Den verifiserte dynamiske M5-prefixfasen bruker bare Entry-supervisjon og
har ingen Exit-lifecycle-adapter. Coordinatoren setter derfor lifecycle-epoch
bare for øvrige faser. Manglende adapter i en slik øvrig fase skal fortsatt
feile lukket; adapterfravær alene gir aldri adgang til å hoppe over kontrollen.

## Omstart, mellomlagring og effektivitet

Operatøren ba 09.10 om hyppige trygge segmentgrenser og mellomlagring. Bruk
samme native campaign, checkpoint-eier og vedvarende inputs/cacher. Omstart
kommer etter varig checkpoint/terminal og dokumentert maskinvid ledighet.
Kontrolleren returnerer REBOOT_REQUIRES_MACHINE_WIDE_IDLE_REVIEW ved denne
grensen. Agenten kontrollerer Windows/WSL-prosesser, writers, prosjektlåser
og GPU før eksisterende prepare-reboot/confirm-reboot og fysisk SSH-omstart.
Neste Windows-oppstart fortsetter bare det allerede bundne finite vinduet.

Kildekontroll: deterministic_fp32 lagrer hver 64 optimizersteg, ved tidsgrense
og ved avgrenset slutt. Checkpointet bevarer ONLINE/TARGET, optimizer, EMA, scheduler, RNG,
epoch-order, neste batch og fremdrift. Eksisterende to-slot-eier bruker
temporær fil, fsync, atomisk rename, hash og atomisk aktiv peker. Diskcacher
gjenbrukes; RAM-/GPU-cache må lastes på nytt. Ved krasj kan arbeid etter siste
fullførte checkpoint gå tapt. Gjeldende resume-evidens og dens avgrensning står i CURRENT_HANDOVER.md.

Operatøren valgte 09.10 eksisterende 12000s (3t20) etter å ha opplevd heng
etter 12 timer. Behold denne driftsgrensen; ingen forkorting er nødvendig.
Tidspause vurderes etter hvert fullførte optimizersteg og skjer først etter
varig checkpoint. Windows-oppgavens grense er 4t slik at native 3t20,
ytre guard 3t50 og terminalbokføring får plass. Smoke skal måle step-/checkpoint-tid,
kald gjenlasting og samme neste batch/tilstand etter resume, og kontrollere
at siste batch, lagring og terminal får plass innen eksisterende ytre vakt.
Rapporter beregningstid og lagre-/laste-/oppstartstid separat.
Ingen kalenderjobb avbryter aktivt arbeid. Omstart beviser ikke løst BSOD-årsak.

## Gjeldende M5/MTF/MBP-læring

Prosjektmålet i GX1_RULES.md er M5 Entry med lukket M15/H1/H4/D1-kontekst,
alle genuine indikatorer og de eksisterende26månedene med MBP-1. Åtte
spesialister og niende D1/makro-kontekst beholdes. Alle ruter, masker,
formler, tidsrammer og faktisk konsum gjennomgås før ny modellkvalifisering.

Den native epoken som er bestilt17.–18.10 erstattes ikke av en kort CPU-
studie eller en gammel256-stegsmetode. Små mekanikkprøver kvalifiserer
mekanikk; de avgjør ikke om en fullt trent Transformer kan ha verdi.
En epoke må omfatte hver kvalifiserte TRAIN-rad nøyaktig én gang, med faktisk
partial siste batch, varige checkpoints og det eksisterende finite vinduet.
Ingen automatisk ekstra epoke, HPO eller seleksjon på evalueringsutfall.

Den fulle dynamiske M5-epoken bruker eksisterende `entry_m5_path_single_epoch_run`,
ikke en kort chronological_learning_run eller native_learning_calibration.
Bind fysisk population witness, strict audit av faktisk initialmåling og samme
profil/kildeclosure. TRAIN-taket er7391batcher/118249rader med batch16 og siste9,
drop_last=false, epochs=1 og endelig ONLINE-modell. Et endelig antall forhåndsbundne
12000s-vinduer kan gjenoppta samme cursor/epoke; ingen automatisk ekstraepoke.
Private mekanikkvekter fra uavbrutt2 mot1+reload+1 skal aldri brukes som start.
Etter durable TRAIN7391 kreves fast sluttmåling og læringsreview; ordinary
complete=false ved TRAIN-grensen gir ikke adgang til ny epoke. Bevar skillet
mellom fysisk radkonsum, sluttmåling, læring, generalisering og økonomisk verdi.
Eksisterende sluttmåling deler invokasjonens tid og har frist11940s, mens vanlig
train-wallpause er12000s. En delvis sluttmåling gjenopptas ikke automatisk av
current campaign. Dette er en kjent betinget begrensning; faktiske terminaler og
bevarte checkpoints styrer håndtering, uten uautorisert relaunch eller grenseendring.

Operatøren valgte18.10 eksplisitt M5-prisforløp og liten MAE før MFE som
hovedmål, med lengre horisonter som støtte. Entry-target, tap, handling og
evaluering må bindes til dette før launch. Et h20-terminalmål måler20observerte
sesjoners nettoavkastning og er ikke lenger hovedmålet. En targetendring krever én navngitt
kontrakteier med kausale executable BID/ASK-utfall og deklarerte kostnader,
ikke en gjettet horisont/vekt eller hindsight-valgt optimal exit. MFE/MAE,
terminalavkastning, timing og handelsfrekvens rapporteres separat.

Den dynamiske etterfølgeren bruker eksplisitt `m5_dynamic_observed`-emisjon
og `entry_only_dynamic_m5_path`. Ett minutt til mange timer er primært;
1..480min er første deklarerte beregningsdomene, hentet fra eksisterende
96×M5-skala, ingen målt optimal tid eller handelsgrense. Alle minuttqueryer
bruker samme encoder; en liten delt dekoder lærer retningsmarkout, signert
MFE og konservativ pre-MFE-MAE separat. Dager er maskert læringsstøtte.

158 velger ett kildeavgrenset conditioningforsøk for denne delte dekoderen:
`q=(h−240.5)/sqrt((480²−1)/12)`, analytisk fra det kjente querygridet.
Fersk initial queryvekt multipliseres med `σ/480`; bias økes med original
queryvekt ganger `240.5/480`. Alle andre initialbytes og RNG-trekk beholdes.
Dette matcher tidligere startfunksjon i eksakt aritmetikk; faktisk F32-paritet
og kandidat-/action-ties må bestås med fersk current-function baseline før
trening. Det er en endring av optimizerens koordinater, med samme parametertall,
funksjonsklasse, mål, masker, TRAIN-RMS, tap og score. Uvektet gridortogonalitet
er ikke ortogonalitet under reelle masker/RMS. Svak terminalhorisontrespons
mot MAE er målt i154/156; bedre retning eller treningsårsaken er ikke bevist.
Policy binder eventuell observasjon og ett sammenlignbart7391stegs-forsøk;
selve designet gir ingen launch- eller treningsautoritet.

Fysisk datasett binder original M1 BID/ASK-kilde én gang og har target-only
fill-/clock-/censorpointers. Ingen tett entry×480 framtidsbane lagres. Hver
komponent og horisont får egen maske. Eksakte endpointquotes kan gi
terminalmarkout selv om mellomliggende MAE/MFE sensureres av et ukjent hull.
Closed-bar-only-observasjoner og terminal-open-inkluderende baner har ulike
endepunktkontrakter; de er aldri stille substitutter. Lavere og konservativ
øvre pre-MFE-MAE beholdes fordi M1OHLC ikke viser intraminuttrekkefølgen.

Ingen largest-horizon-/legacy37-/dags-union får redusere primærpopulasjonen.
Tidligere `m5_primary_observed` med fast19M5-bars-fasit og legacyemisjon
beholder sine egne historiske kontrakter; de er ingen kjøreautoritet for ny
fase. Fire aktive M5-forecasts har egne observerte masker på1/5/12/24
M5-barer; bare uobserverte celler lagrer0 bak false-masker. Seks dagsstøttemål
har egne masker og uobserverte fysiske NaN-triader som blir0 først i modellbatch.
Endret populasjon krever ny faktisk TRAINklokke, fersk TRAINregistrytilpasning
og gjeldende immutable kontraktbindinger før inputadmission.

Prisbanens MFE minus pre-MFE-MAE alene kan gi positiv forventning på begge
sider uten forutsigbar retning. Den er ingen tilstrekkelig trade/FLAT-fasit.
Modellens nye beslutningsverdi må la både retning og kvalitet påvirke valg,
og må kvalifiseres på valgte/avståtte entries, dekning og stabilitet. En
komponentdifferanse av predikert netto endepunktsmarkout og predikert
pre-MFE-MAE upper bruker eksplisitt kvalitets-bps. Koeffisient1 er et valgt
lik-enhets kvalitetsmål, ingen empirisk optimalitet eller cash. Ulike
observasjonsmasker identifiserer ikke ubetinget samme forventningspopulasjon.
Den tidligere ratioen kunne ikke velge FLAT på grunn av høy MAE alene når
noen predikert retur var positiv; den egenskapen er algebraisk bevist.
Ingen kandidat gis native autoritet ved bare å flytte formelen inn i forward.
Fremtidens beste horisont er aldri fasit for en udokumentert scalar.


Første reelle smoke skal bevise samme fysiske batch gjennom profilens eksplisitte aktive familier,
GC-value/mask-par, alle MTF, normalisering og modell, samt fersk initialbaseline
og train/serve-paritet. Entry-only-fasen må bevise Exit-grad=None og uendrede
Exit-vekter. Bruk allerede beståtte uendrede kontroller; en ny funksjon krever
ferske funksjonsbevis. Nøyaktig nåstatus og eventuelt uavklart målvalg finnes
bare i CURRENT_HANDOVER.md/NEXT_RUN_POLICY.json.

## Sammenligning av læring og loss-enheter

Før/etter-readout bruker samme frosne kohorter, targets, masker og counts.
Rapporter rå MAE/MSE separat for hver komponent, side og query, sammen med
valgte/avståtte kandidater og dekning. Samlet forbedring over alle480horisonter
beviser ikke forbedring ved ett minutt. Prediksjons-MAE er absolutt prognosefeil;
den må ikke forveksles med targetets adverse excursion før første MFE.

Rå feilrapportering bruker maskert rawbps-MSE. Optimeringens nye loss-only
enhet er hver koordinats fryste TRAIN-RMS: sqrt(sum(target_float32² i float64)
/observed_count), uten sentrering. Alle fysiske TRAIN-rader inngår én gang;
inklusive/lukkede endepunkter og hver side/query har egne4810skalaer sammen
med forecast-koordinatene. Kvadrer residual/RMS før dagens maskerte pooling.

Inclusive/closed MFE deler fortsatt én task og denominator, tilsvarende for
preMAE. Terminalens FLAT0 bypasser divisjon og inngår fortsatt som kjent null
i denominator; skill dette fra LONG/SHORT alene. Fire M5-forecasts og seks
dagskoordinater deler forecast-tasken. De fire lærte ONLINE-logvariansene
bruker exp(−s)·dimensjonsløst_tasktap+s. Rawbps-diagnostikk og unitloss skal ha
separate felt; ingen rå bps²-metrikk skal stille endre enhet. Cellevektet pooled
diagnostikk er ikke nødvendigvis lik evaluatorens radvektede batchloss.

En genuin lærbar koordinat med count0, RMS0 eller ikke-finit skala stopper før
optimizer. Ingen epsilon, klipping eller unitfallback. Fit-lineage binder
stabile data/klokker/populasjon/targets/økonomi/M1-indeks, aldri en framtidig
fresh-PLAN. Samme artifact-/kontrakt-/lineageidentitet må følge recipe,
initialmåling, target-only batch, session, sluttmåling og resume. Modellens
inputgeometri og rawbps-heads endres ikke. Kildekvalifisering og faktisk
TRAIN-fit er ulike bevis; gjeldende adgang står bare i NEXT_RUN_POLICY.

Forecast-gruppebidrag beregnes fra gruppens SSE delt på hele forecast-count,
multiplisert med samme lærte presisjon. Gruppens egen MSE er en annen størrelse.
Det felles+s-leddet kan ikke entydig fordeles mellom horisontgruppene.
Store residualandeler beviser ikke parametergradientdominans eller årsak til
en konkret prognosefeil. En fallende lært presisjon kan redusere vektet tap
samtidig som rå prognosefeil øker; rapporter råfeil, unitloss og faktisk presisjon.

Et nytt loss-enhetsdesign krever en eksplisitt kontrakt før fit. Eventuelle
skalaer skal tilpasses bare eksakt wholeTRAIN-targetpopulasjon med egne
koordinat-/horisontmasker, counts og lineage. VAL-targetstd er diagnostikk,
aldri fitgrunnlag. Ingen skala velges etter CONTROL-prestasjon. Rawbps-targets,
prediksjoner og beslutningsenheter beholdes ved loss-only skalering; en ny
fit skal ikke trigge ny feature-/inputnormalisering. Håndtering av nullskala
og grouping må være deklarert; ingen gjettet epsilon eller relativ taskvekt.
Ny kode/fit er ikke kvalifisert før faktiske bindinger og nødvendige kontroller
foreligger. Ingen ny treningsadgang følger av et diagnose- eller designnotat.

## Bevarte datakrav og økonomi

Alle genuine featurefamilier, ordnede felt, M1/M5/MTF-klokker, targets og
normaliseringseiere består. Geometri, reference-policy og cutoff hentes fra
de faktiske bundne eierne, ikke gjettede defaults eller gamle dokumenttall.
Normalisering tilpasses én gang på hele fysisk TRAIN; ingen VAL/TEST-refit.
For canonical `entry_only_dynamic_m5_path` er fit-populasjonen eksplisitt
Entry-only:unionen av faktisk konsumerte96bars M5-vinduer, alle fysiske TRAIN-
contextbeslutninger før targetmasker og fem kausale MTF-laner. Ingen Exit-
populasjon eller lærer. Source-backed normalisering krever en ekte bundet
seq/snap-kilderekonstruksjon og eksakt wholeTRAIN-registryklokke. Den gamle
joint-populasjonen beholder Exit-kravet; modellfase og fit-populasjon må samsvare.
Den eksisterende produsenten
`gx1.scripts.materialize_unified_exit_pilot_base_normalization_v1` har en egen
`--entry-training-phase entry_only_dynamic_m5_path`-gren. Den krever
`--train-parquet`, `--train-manifest`, `--train-sequence-source-audit` med
`--train-sequence-source-audit-sha256`, og forutgående `--full-input-liveness`
med `--full-input-liveness-sha256`. `--m5-prebuilt` må være den faktisk bundne
M5-kilden i MTF-cachemanifestet; `--mtf-cache-dir` må stemme med splitmanifestet.
Bare den planbundne hele TRAIN-populasjonen er lovlig. Output publiseres én gang
med `--output` og `--publish` under eksisterende capped audit4G/512M og prosjektlås.
Pilot-/Exit-/subsetargumenter avvises. Fit og etterfølgende streng validering
har én felles planbundet tidsfrist; den fornyes ikke per trinn. Et inputfit-PASS
gir ingen nativeadmission. Canonical MTF-vinduslengder hentes fra
produksjonseieren. Source-backed Dataset bruker tre diskarrays for snap/ctx,
ingen dense96bars speilsekvens og blokkvis eksakt paritet med M5-kilden.
Transformert TRAIN-støtte skal bruke samme observasjonsmasker både ved fit
og etter deling av temporal-alias-statistikk. Manglende lagringsnuller skal
hardnulles etter transformasjonen; de er ingen ekstra markedsobservasjon.


Eksisterende full input-liveness skal bestå før markedsfit. Inputfit krever ikke
labels/design; den fysiske målkvitteringen må senere binde det faktiske fulle
native-designet med genuint epochwitness/budsjett før nativeadmission. En
midlertidig data-design gir ingen nativeadgang og skal ikke kjøres for dette.

Det faktiske dynamiske native-designet binder inputs.full_input_liveness og
inputs.sequence_source_audits for både TRAIN/VAL. Samme targetreceipt/designeier
beholdes. Dynamic prefix bruker den direkte Entry-only-normaliseringsartefakten
og sin base_contract_sha256; composite-/summary-/Exit-first-state-bevis hører til
legacygrenen. Entry-cohort bruker frosne fysiske parent-ID-er og klokker, ikke
Exit-statekoordinater. Begge grener må være eksakt fasebundet.
Dynamisk single-epoch bruker fast siste ONLINE etter hele den bundne TRAIN-
epoken. Egen checkpointpolicy avviser VAL-/økonomisk utvalg og metric-score;
ingen top-K eller tidligstopp bestemmer denne modellen. Varige state-slots,
lagringsfrekvens, optimizer/EMA/RNG og resume-eier beholdes. Initial/final-
kurvemåling har samme frosne parentrader og separate target-/maskehash, én
encoderforward perbatch, og ingen Exit-factory eller Exit-forward.



Original target-M1 og nyere komplett input-M1 er ulike kilde-/artefaktroller.
Gjenbrukt uendret-kilde-/algebrabevis er ikke en ny raw quote-måling.
Ukjente gap forblir ukjente/right-censored; ingen carry, imputering eller
nye forskningsperioder. Ingen fast tapsgrense eller maksimal holdetid.

Senere økonomi må inkludere alle valgte handler, åpne posisjoner, kostnader
og én-posisjonskapasitet mot forhåndsvalgt samme-risiko alltid-LONG.
Lokale prospective kostreceipts er ikke historisk kostfasit.

Regler: GX1_RULES.md. TEST, broker, live/paper, handel, spending og promotion
er stengt. GC og andre forskningsarmer åpnes ikke av denne metoden.


Den dynamiske dekoderen har to eksplisitte endpointtyper fra quote-eieren:
`closed_m1_prefix` og `terminal_open_inclusive`. Begge lærer signert MFE og
konservativ pre-MFE-MAE med samme encoder/heads, egne mål og egne masker.
Path-tapene bruker observerte celler fra begge typer i sin egen denominator.
Ingen closed-path-fasit erstatter inclusive-path-fasit. Den kjente endpointkoden
er queryinput; faktisk framtidig tilgjengelighet er aldri input. Dermed forblir
et observert siste1minuttsforløp læringsaktivt selv når neste open mangler.
Kvalitetsdifferansen bruker kun modellens eksplisitte inclusiveprognoser.
Maksimum velges over predikerte horisonter, aldri fremtidens beste fasit.

## Kompakt førsteprofil og effektiv validering

Første navngitte profil er `compact_mbp_smc_ema_atr_clock_macro_entry_v1`:
MBP + SMC/struktur + EMA/ATR + nødvendig klokke-/makrokontekst. Alle 2410
GC-verdi-/maskekanaler og 61 makrofelt beholdes. Hele biblioteket og standard-
fullmodellen beholdes. Profilreferansen `{name,contract_sha256}` følger
metadata, recipe, modell og state; ingen implicit profilerstatning.

Kompaktprofilen har 2622 signal-/2525 context-/139 MTF-felt, 7 lokale og 5
MTF-familier. Bare aktive encodere og projeksjoner konstrueres og kjøres;
fullprofilen beholder 8 lokale/40 familie×TF-ruter/253 MTF-felt. Full fysisk
055-inputvalidering og normalisering kommer før intern gather. Rapportering
bruker samme aktive profilbredder. MBP-felt må aldri droppes fra et familienavn;
74 genuine GC-kanaler ligger fortsatt i den aktive candle-ruten.
Kilde-/syntetiske rutetester beviser beregningsgeometri. Faktisk modellkonsum,
latency, minne og læringsfart må måles på den bundne ekte datakjeden.

Individuell påvirkning for de dynamiske hodene eies av
`gx1/contracts/entry_dynamic_compact_input_influence_v1.py`. Hele siste
sekvensrad og snapshot må endres sammen; contextaliaset følger samme verdi.
Signer summeres før absoluttverdi, slik at kansellerende delderivater ikke
blir falsk påvirkning. Historikk og egne observasjonsmasker rapporteres separat.
Manglende observasjon kan aldri erstattes av et kunstig observert felt.

Kildekvalifisering bruker analytiske CPU-fasiter. Faktisk modellpåvirkning
krever senere en egen bundet TRAIN-vitne-/caseplan, ONLINE-state, frosne
normaliseringsdata og maskinvareadgang. Tre kandidatmarginer og fem separate
komponenter undersøkes med 13 input-VJP; nullrespons er ikke alene død kode.
Fysiske ruter er ikke uavhengige markedstilstander. Gyldige endringscaser
pakkes lazily i B16, padding gir ikke bevis, og kildetaket er 218 forwards
først etter en eksakt plan. Følsomhet beviser ingen læring eller lønnsomhet.

Inputpakken gjenbrukes uten nye TRAIN-/cache-/rådatadekodinger. CPU-planen
registrerer alle 3446 eieretiketter og velger bare ekte vitner i fast rekkefølge
etter beslutningstid og parentrad. Den bruker egne kildeobservasjonsmasker;
padding, utfall, modellprediksjoner og gradienter inngår ikke i vitnevalget.
F32-visningen av frosne sentre/skalaer gjenspeiler modellbufferne og endrer
ikke originalnormaliseringen. Manglende, kildekonstant og ugyldig endringscase
rapporteres separat; ingen automatisk reparasjon eller prediksjonsstyrte skips.

Den eksisterende allowlistede tekniske validatoren får en egen inputmodus med
`--dynamic-compact-input-influence-plan` og
`--dynamic-compact-input-influence-plan-sha256`, samt `--device cuda` og absolutt
`--out-dir`. Modusen krever egen eksakt plan/policy og gjeldende capped CUDA-
producer-/maskinvarevakter. Den går før legacy VAL-/Exit-parseren og bruker
ikke legacyvalideringen, Dataset, targets, provider eller optimizer. Checkpoint
leses gjennom eksisterende read-only session/policy med ferske filkontroller;
ONLINE `model_state`, normalisering og innmatingspakke bindes hver for seg.
Dette er en ny auditinngang, ingen trenings- eller serving-adgang.

Gjentatt fysisk hashing av samme store fil innen én valideringsoperasjon
unngås gjennom kontrakteierens eksplisitte `file_verification_scope()`.
Dette gjelder eksisterende CLI/materializer og handover-/native-window-/
full_train-preflight, samt avgrenset dynamic-prefix-, initial-, technical-
og learning/epoch/continuation-metadata. Den sene prefix-sessionbindingen
har sitt eget korte scope før optimizerkonstruksjon. Den rene dynamiske
cohort-eieren har også ett endelig scope, lukket ved retur før modellarbeid;
hele øvrige kohort-AST og alle SHA/stat-/identitetsvakter beholdes.
Coordinatorens sammenhengende konstruksjon og kontroll av checkpointpolicy
har ett kort scope: begge semantiske kontroller kjøres, men samme stabile
filidentitet hashes én gang av den felles eieren innen operasjonen. Scopet
lukkes før normaliseringsbinding, separate artifact-hasher og Session/state.
Dette beviser ingen per-batch-gevinst eller eliminering av øvrige råhasher.
Liveness-fildigest delegerer til samme verifiserte eier. Metadata-scopene
lukkes før Dataset-/modellkonstruksjon og før initialstate restore; de går
aldri gjennom modell-, batch- eller checkpointarbeid.
Første faktiske digest bindes til canonicalpath, dev,
inode, mode, size, mtime_ns og ctime_ns, kontrollert før/etter lesing. Hvert
kall beholder filtype-, symlink-, identitets-, forventet-SHA- og semantiske
kontroller. Scope lukkes før budsjettpublisering og modell-/statearbeid;
ny operasjon eller endret fil krever ny fysisk hash. Ingen varig cache.

Checkpointpolicyens runtime-bevis er separat fra hash-memoet: én fersk strict
policy-/receipt-/pointervalidering per Session fanger alle fysisk verifiserte
immutable data- og kostfiler og revaliderer samlet identitet før forsegling.
Det binder hele sessionkontrakten, prefix, monitor og eksakt policy-JSON.
Gjenbruk krever alle identiteter uendret og gir en ny kopi av metadata uten ny
wholeTRAINstream/pointeraudit. Endring feiler lukket. Beviset serialiseres ikke,
overlever ikke ny Session/resume/read-only-admission og brukes aldri av
filhasheren. Checkpointfilenes egne SHA-/load-/save-vakter består.

Bruk den kanoniske hasheieren for den aktuelle rollen. Campaign-eierens
kanoniske JSON inkluderer avsluttende newline; fil-SHA og andre kontrakters
semantiske SHA er ulike roller og er ikke utskiftbare. Et kontrollert stopp
bevarer alle originalkvitteringer og konsumerer den claimede engangsplanen.
En feil i etterkontroll må rette etterkontrollen, aldri omskrive originalbevis.

Endret modellkildeclosure krever ny faktisk kildebundet CPU-fresh-publisering
før initialmåling, også når modellmatematikken er uendret. Gjenbruk uendret
normalisering/data, aldri gamle vekter. Endring utenfor closuren krever ingen
slik republisering når alle bundne bytes/stat og kontraktroller er verifisert.
Bind alltid en ny eksakt campaign og gjeldende maskinvare-/bootporter.
Dynamic-coordinatorens kontrollcutoff velges fra samme SHA-bundne DESIGN-
kalender som kohorteieren: `(train, train)` bruker `train_control_cutoff`,
`(control, val)` bruker `development_control_entry_end_exclusive`. Andre
rolle-/splitpar avvises. TRAIN-prefixets cutoff beholdes. Fysiske deklarerte
vinduer kan slutte tidligere enn kalendergrensen; de erstatter derfor ikke
kohortens targetcutoff. Per-horisont sensurering og øvrige bindinger beholdes.
Klargjøringstid, preflight-tid, encoderforward-tid og treningstid rapporteres
hver for seg. Aktuell status og kvitteringer står bare i NEXT_RUN_POLICY og
CURRENT_HANDOVER; historikken finnes i Git.

Ved målt ekstraskriving i en kildekaller som bare beholder M5-signal, vurderes
signal-only retention i eksisterende loader før større cachearkitektur. Hele
bounded kilde-/verdi-/domenevalideringen og uendret standardrute skal beholdes.
Per-prosess memmapbygging må skilles fra per-batch arbeid; logiske lesebytes
beviser ikke host-disktrafikk eller tidsfordeling. Bevar kvalifisert kilde gjennom
aktuell baseline/mekanikkbølge, og la målt treningsgjennomstrømning styre neste
minste I/O-rettelse. En uimplementert kandidat åpner ingen ny kjøring.

## Entry-tidsramme og etterfølgende prisfasit

Entry-beslutninger tas på M5. M1-data gir finere observasjon av prisforløpet
etter en slik beslutning; prognoser fra 1 til 480 minutter er framtidige
målehorisonter, ikke M1-entry, Exit-regler eller pålagt holdetid. M1-Exit
vurderes først i neste bølge etter Entry-edge. Rapportér derfor
`entry_timeframe=M5` atskilt fra `query_horizon_minutes` og `exit_evaluation=false`.
Den eksisterende diagnostiske kandidaten er
`q[h,side] = pred_terminal_net_markout_bps - pred_pre_mfe_mae_upper_bps`,
med maksimum over alle1–480min for LONG/SHORT og analytisk0 for FLAT.
MFE trenes aktivt, men er ikke direkte operand i denne kandidatscoren.
Komponentregresjonen har ingen direkte action-/rankingloss. Bedre komponentfeil
beviser derfor ikke bedre kandidatvalg. Et maksimum ved1minutt endrer ikke
M5-kadensen og begrunner ikke alene en ny horisontgrense eller handelsregel.
Vurder terminal og pre-MFE-MAE hver for seg, deretter forskjellen på felles
observerte celler. De separate maskene må ikke gjøres til framtidsinputs.
For feildekomponering brukes samme frosne FINAL-radgrupper og prognosevalgte
horisonter for initial/slutt: `eD=e_terminal-e_MAE` og
`SSE(eD)=SSE(e_terminal)+SSE(e_MAE)-2*sum(e_terminal*e_MAE)` på fellesmasken.
Native F32-subtraksjonsavrunding rapporteres separat fra F64-komponentmomentene;
ingen gjettet epsilon avgjør matematisk PASS. Eksisterende enkeltkomponent-MSE
med ulike masker kan ikke erstatte disse felles momentene.

Forecast-headen har fire skalarer for1/5/12/24 etterfølgende M5-barer, deretter
seks dagstøttekoordinater: LONG/SHORT ved1/5/20 observerte handelsdager.
Fire basehorisonter betyr fire koordinater, uten ekstra sidefaktor. Bar-offset
er ikke alltid veggklokketid over gap; dagstøtten er ingen Exit-regel.

Fullkurve-diagnostikk lagrer hver M5-entrys fem allerede beregnede
1–480minuttkurver med egne observasjonsmasker og `null` for uobserverte mål.
Base-/dagsprognoser har separat layout. Validatoren gjenkobler alle observerte
celler til eksakte maskhasher, tellinger og batch16/F64-feilaggregater.
Original rå target-storage-SHA beholdes separat; den rekonstrueres ikke fra
nullable rapportverdier. Kurvene skrives én gang per TRAIN-/CONTROL-rapport;
checkpoints beholder bare filreferanser med SHA. Ingen ekstra modellkall
eller Exit-beregninger følger av dette. Endret observer-/valideringssemantikk
krever riktig kildeclosure og kvalifisert faktisk fullkurvebaseline før trening.
Gjeldende baseline-, state- og terminalbevis eies av CURRENT_HANDOVER/policy.

Etter den avgrensede epoken sammenlignes lagrede initial-/finalrapporter på
identiske TRAIN-/CONTROL-kohorter. Kontroller rå target-storage-identitet,
nullable mål og separate masker før råbps-/TRAIN-RMS-feil og alle480horisonter
sammenlignes. LONG/SHORT/FLAT og valgte/avståtte kandidatutfall rapporteres med
egen observerbar dekning. Framtidsmaksima er kun merket hindsightdiagnostikk;
ingen terskler eller nye policyer velges fra sluttmålingen. Dette er gjenbrukt
utvikling, ikke ny generalisering eller realisert P&L.

Gjenbruk eksisterende canonical observasjonsvalidator under CPUaudit. Den kan
åpne de fire eksplisitt bundne parentpointer-kolonneprojeksjonene (initial/final
for TRAIN/CONTROL); ingen feature-/target-/sekvenskolonner, råMBP eller PT/modell
trengs for denne sammenligningen. Ingen nye forwards, refits eller Exit-arbeid.
