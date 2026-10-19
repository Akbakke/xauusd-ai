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

Resume-kontrollen bruker første32rader i057s kanoniske TRAIN-ordre, batch16:
to sammenhengende steg mot ett steg, faktisk varig lagring/gjenlasting og
samme andre steg. Fire tekniske updates totalt; samme eksisterende loss,
AdamW, EMA, scheduler, clipping og gradvakter. ONLINE/anker/optimizer/EMA/
scheduler/RNG/orden/cursor skal være ekvivalente. Ingen CONTROL-updates,
vektsortering eller gjenbruk av kvalifiseringsvektene i hovedepoken. Dette
er én serialiseringskontroll, ingen ekstra økonomisk treningsarm eller HPO.
Kildens faktiske innrømmelse og målte terminalkvitteringer må fortsatt finnes;
metodebeskrivelsen starter ingen jobb og åpner ingen port.

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
fase. Fire M5-forecasts lagrer0 bak false-masker; seks dagsstøttemål har
uobserverte fysiske NaN-triader og blir0 først i maskert modellbatch.
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

Gjentatt fysisk hashing av samme store fil innen én valideringsoperasjon
unngås gjennom kontrakteierens eksplisitte `file_verification_scope()`.
Dette gjelder eksisterende CLI/materializer og handover-/native-window-/
full_train-preflight, samt avgrenset dynamic-prefix- og initialmetadata.
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
