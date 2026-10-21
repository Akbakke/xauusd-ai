# XAUUSD M5/MTF/MBP — individuell kilde- og dekningsrevisjon

Dette dokumentet følger det varige målet i GX1_RULES.md. Arbeidsstatus,
kjøregrenser og terminalbindinger eies av NEXT_RUN_POLICY.json og siste
CURRENT_HANDOVER.md. Tidligere rapportversjoner finnes i Git. Ingen tidligere
kort regel-/CPU-/smoketest avgjør verdien av en ny korrekt MBP-Transformer.

Det varige målet er M5 Entry med gunstig prisforløp og liten MAE før første
MFE, alle åtte featurefamilier, niende D1/makrokontekst og kausalt lukket
M5/M15/H1/H4/D1 fra de26rike månedene. Gjeldende dynamiske querydomene er
1..480forhåndsdeklarerte, forløpte minutter; dagshorisonter er separat støtte.
95minutter er ikke nytt primærmål, og480minutter er ingen maksimal holdetid.
Kildeimplementasjon, fysisk dataadmission, læring og økonomisk edge er egne bevis.

## Revidert flate og bevisklasse

Inventaret på73aa04f0 omfatter770 sporede filer og600 Python-filer.
Alle600 kan AST-parses. Import-, CLI-, test- og eierskapsreferanser er indeksert;
dette er et statisk inventar, ikke et matematisk korrekthetsbevis for alle linjer.
Hvert navngitt signal-, kontekst- og MTF-felt har formelkilde, eier, kildeklokke,
TF-dekning og manglende ruter i det komplette maskinlesbare feltinventaret.

Inventarer og hashmanifest:
`/home/andre2/GX1_RUNS/NATIVE_GC_MTF_DERIVATION_20261018_018/AUDIT/MANIFEST.json`.
Filene heter gx1_repo_inventory_20261018.json,
gx1_feature_audit_20261018.json, gx1_mbp_audit_20261018.json og
gx1_model_repo_audit_20261018.json. De beskriver utgangspunktet før rettelsene;
ny funksjon/datasett må kvalifiseres separat.

| Utgangspunkt | Antall og semantikk |
| --- | --- |
| Native signal |735felt:25base +162mandatory +548candidate;96M5-steg |
| Kontinuerlig kontekst |552aliaser i signalflaten, ingen552nye uavhengige signaler |
| Kategorisk kontekst |1sessionfelt |
| GC |210verdier +210egne observasjonsmasker;102kilde-/TF-felt ikke innrømmet |
| D1/makro |61felt i eksisterende session/regime-spesialist; ingen niende encoder |
| MTF |190felt beregnes selvstendig på lukkede M5/M15/H1/H4/D1-candles |

| Lært familie | Lokale signalfelt | MTF-felt per klokke |
| --- | ---: | ---: |
| Struktur/swing |30|20|
| SMC/likviditet |299|50|
| Trend/EMA |40|29|
| Volatilitet/kompresjon |18|5|
| Momentum/flow |179|22|
| Session/regime |80|4|
| Chartgeometri |32|38|
| Rå candlegeometri |57|22|

Antall felt er ikke antall selvstendige indikatorer. Masker, sammeformelaliaser,
kildeklokkebundet kontekst og lærte representasjoner telles eksplisitt hver for
seg. Omtrent150 er et brukeranslag; de eksekverbare eierne bestemmer faktisk flate.

## Påviste koblings- og målgap

1. Entry konsumerte bare M15/H1/H4/D1-MTF-sekvenser. M5s190-feltscache gikk
   til Exit;27MTF-formler manglet en direkte lokal M5-motpart. Rettelsen bruker
   eksisterende M5-rute/moduler også for Entry. Entry lukkes fortsatt på+5min,
   Exit på+1min. Samme-clock-aliaser er kildeavledede og må fortsatt være
   bitlike på ekte beslutningsrader. Endret funksjon krever fersk baseline.
2.190perTF er ikke den komplette unionen av alle lokale genuine formler.
   Blant gapene er basic/KAMA/TEMA/BB/returprimitiver, sweep-ankret aktivitet,
   enkelte foundation-/mikro-/spreadfelt og clockspesifikke kontekstformler.
   Feltinventaret navngir hvert gap; komplett all-TF-konsum er ikke bevist.
   Makro publiseres as-of; den skal ikke resamples til fiktive intradagnyheter.
3. H20-terminal nettoavkastning er aktivt Entry-target i den tidligere fysiske
   generasjonen. Dette måler ikke direkte liten MAE før MFE på M5. Et vedtatt
   target må være eksplisitt før ny trening. Operatøren valgte18.10 M5-path og
   MAE før MFE primært, med lengre horisonter som støtte; auxiliary MAE/MFE-hoder
   alene oppfyller ikke dette vedtaket.
4. Fysiske016-manifester beholder legacy fitted-Q-basismetadata og et separat
   observed-h20-avsnitt. Ny recipe/bundle skal binde faktisk fase og uavhengighet
   eksplisitt; de immutable016-manifestene omskrives ikke.
5. M30 er ikke aktivert, og tilført verdi er ikke målt.

## Ny kildeflate under kvalifisering

Den nye MTF-eieren deklarerer253felt per lukket tidsramme: opprinnelige190,
40lokale genuine formler, ni generiske HTF-formler og14Group-A-nivåavstander.
Dette gjenbruker eksisterende formel-/pivot-/likviditetseiere. De åtte familiebreddene
er24/76/39/13/29/9/39/24;217lokale M5-par deler eksakt formel/kildeklokke. Ekte BID/ASK
føres gjennom inputprojeksjonen. H4/D1 bruker den eksisterende session-klokken;
et krav om midnattmodulo kunne ikke validere de virkelige offsetbarene.
Gjenbruk av rå OHLC-oppsett fjerner cache-selvreferansen for14Group-A-feltene;
den opprinnelige serving-cachegaten består. Kildeendring er ikke fysisk paritet.

Kildekartet etter019/020 og trusted-T-utvidelsen deklarerer995kandidater,
199perTF, med eksplisitt formelkolonne, spesialisteier, validitetsmaske og
kildeklokke. Dette er et implementasjonskart, ikke endelig nativeadmission.
Det omfatter70% sammenhengende VAH/VAL, kjent/ukjent kumulativ T-volumfordeling,
T-sidetall og sidemiddel, total størrelsesstandardavvik/min/maks, observerte
BBO-/ordreantall og intraminutt spread-/event-/kildeklokkestatistikk.
Trade-recordstørrelse identifiserer ikke kundens overordnede ordre.

019 er fysisk publisert med gammel kildeadmission.020 beviste TRAINstøtte
for935daværende kandidater og rettet observerte H4/D1-SMC; den foreløpige1130-
registryen er ikke aktivert. Påvist signaturregel avviste78322ellers gyldige
M1 med25064729kontrakter. Identiske Tposter dedupliseres ikke uten kildesviktbevis.
Databentos dokumenterte PTP-hendelse17.09.2025 har eksplisitt overlapps-/anker-
sensur. Historiske non-T-actionlabels innrømmes ikke som kanselleringsflow.
Kildehendelser og semantikk er bundet i
`configs/research/GC_VENDOR_SOURCE_INCIDENTS_20261018_001.json`.

024er COMPLETE og har gjenopprettet hashbundne001reduksjoner. Ett målrettet
eventpass kompletterte tidligere maskert informasjon og nye T-statistikker,
med eksakt uint64-/volum-/antallsparitet. Netto etter PTPsensur er78305M1 og
25059620kontrakter gjenopprettet. Alle210gamle registrydeklarasjoner er også
rebindet til samme reparerte tabeller.

025s opprinnelige109507TRAINrader er bevart som produksjonsprovenans.
048ferdigstilte faktisk dynamisk TRAIN118249/VAL17284 etter begge uttømmende
seq/snap-kilderekonstruksjoner.049gjorde fersk registryfit på alle118249TRAIN-
beslutninger før taskmasker, uten outcomes/VAL/TEST. Alle995nye+210gamle felt
har genuine observasjoner:1205verdier+1205masker,30observerte konstanter,
0uobserverte og0formelaliaser. Ordnet verdi-/maske-/eier-/kilde-/formel-/rolle-/
lagringsekvivalens er målt mot025; bare diagnostisk fitklokke/statistikk endres.
Ny immutable config049 og kun GC-eierens REGISTRY_PATH/SHA256 aktiverer dette;
alle andre modul-AST-noder er identiske.050fullførte gjeldende signal- og
M1/M5hardlinkmetadata samt faktisk ny existing-owner v5overlapaudit19.10
kl00:11UTC,686.736s, uendret kilde.051fullførte typed fysisk splitmetadata
og begge nye eksakt bundne seq/snap-rekonstruksjoner01:28UTC,1315.079s,
exit0 og uendret kilde, uten numerisk rebuild. TRAIN118249/VAL17284.
053fullførte eksisterende full input-liveness19.10 kl02:09UTC,640.168s,
exit0/uendret412011fc: alle118249TRAIN/17284VALrader og alle253felt perTF.
Streng validator/finisher PASS. ATR/OOD SHIFT_OBSERVED beholdes som diagnostikk;
ingen VAL-fit.054stoppet før publisering etter336.301s: aliasdeling overskrev
maskerte ekstremastatistikker med støtte fra manglende lagringsnuller.
GC M15 minimum tradestørrelse er ekte konstant1 på116782observasjoner,
1467mangler.055s minste maskebevisste rettelse er uavhengig revidert og
kvalifisert i51fokuserte caser; én separat ny faktisk fit avventer ren binding;
strenge konstant-/binærvakter beholdes.055publiserte deretter den faktiske transformasjonen.
Physicaltargetreceipt skal binde det faktiske fulle native-designet før
nativeadmission; ingen midlertidig data-design. TRAIN-fit trenger ingen targets.
055bestod faktisk wholeTRAIN-normalisering og streng finisher02:53UTC,
357.914s, exit0/uendretfed117b1: context118249/lokalunion118800 og femMTFlaner.
Ingen Exit-/VAL-/TEST-fit. Den publiserte transformasjonen gjenbrukes.
Nativeadmission, faktisk modellkonsum og læring gjenstår.
Originalkvitteringer, datasettbytes og025s gamle populasjonsbevis omskrives ikke.

022M5/cache/model-source og026M1 er fysisk COMPLETE, exit0, uendret kilde og
TEST=false.022publiserte1140982M5rader/2576enrichedkolonner og
2725signal-/2542ctx-cont-/1ctx-cat-felt.026publiserte1111058M1rader/
2576kolonner fra2023-05-09T22:03Z til2026-06-30T23:59Z, med1205GCverdier
og1205observasjonsmasker i samme deklarerte inputkontrakt.026brukte6092.253s
på HEADb669d8f05c79504c1dc18a81079b53ae6f6c937f. Publikasjonene ligger i
`/home/andre2/GX1_DATA/data/data/prebuilt/XAU_GC_NATIVE_26M_20261018_003`.
M1-parquetSHA er7928886648820c6e2ddb5057628c980f75264efbab77d94623732f7a88e1efcd;
sidecarSHA er f4e3ab32955cbf7b16ba3e279d070f5b4b39d764f122dd63c0b7415e6810136d;
026terminalSHA er0294e5c02093c991e008e24c77fe5cb3e5393f89f1af63ded263379bac05ee3e.

De ekte022/026featurebytene skal gjenbrukes ved bevist uendret ordnet
felt-/kilde-/verdi-/maske-/rolle-/evaluatorkontrakt. En ny TRAINfit med lik
verdifunksjon krever fortsatt eksplisitt ekvivalensbevis og gjeldende immutable
signal-/datasetmetadata; gammelt registryhash i metadata ignoreres ikke.
Featurepublisering er ikke dataset-, normaliserings-, modell- eller læringsadmission.

15/60min/NYsession-profiler beregnes på fysisk M5-klokke med as-of-projeksjon,
separat dekning og faktisk kildealder.15/60betyr minutter også i H4/D1.
Absolutte identiteter og ns-tidsstempler er metadata; eventtidsforskjeller er
signerte målinger. Negative forskjeller klippes ikke, og clamped ts_in_delta
har egen maske. [Leverandørens tidsstempelkonvensjoner](https://www.databento.tech/docs/standards-and-conventions)
skiller kildeklokkene; verdiene er ikke automatisk fysisk markedslatens.

Endelig registry måles fra faktiske hele TRAIN-beslutningsklokker uten
labels/VAL/TEST. Strukturmasker har binær identitet. Lagringsnull bak missing-
maske fjernes etter normalisering, og profilalder regnes ved beslutningen.
Observerte konstanter/like TRAIN-vektorer rapporteres; samplelikhet beviser
ikke like formler, og finite input er ingen alpha-/nyttepåstand.

## MBP-1: hva de eksisterende bytene faktisk gir

Rå001 omfatter66MBP-1-filer og892100712records. Den fullførte rebuilden bevarer
9173893M1-prisnivårader med eksakt buy/sell/unknown-konservering. GC velges
kausalt fra forrige fullførte UTC-døgns observerte volum med00:01-tilgjengelighet,
ingen tilbakegang i utløp og ingen substitusjon ved dagens datamangel.
GC-priser er ujusterte outright-priser. Kilden er en futuresinput, ikke vår
XAUUSD-utførelse. GC/spot-basis må bruke samme observerte kildevindu.

Kildens B betyr buy-aggressor og A sell-aggressor på Trade; N forblir ukjent.
F/passive fills er ingen ekstra aggressorhandler. BBO order counts er aggregert
antall viste ordre, ingen individuell ordreidentitet, køposisjon eller dypere bok.
[Leverandørens MBP-1-skjema](https://www.databento.tech/docs/schemas-and-data-formats/mbp-1)
og [kontraktsymbologi](https://www.databento.tech/docs/standards-and-conventions/symbology)
beskriver de observerbare feltene og ujusterte kontraktene.

| Strict komplett bøtte i rå001 | M5 | M15 | H1 | H4 | D1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Volum/profil |98594|18545|862|2|0|
| Full kjent aggressorside |19091|933|3|0|0|

Etter kildebevist signaturrettelse og PTPsensur i024 er strict volum/profil
M5/M15/H1/H4/D1:151773/50074/12145/2445/0. Observed gyldige volumbøtter er
152991/51001/12753/3334/547. Strict signedkompletthet følger sin egen maske;
ukjent aggressor blir aldri konstruert som kjent side.

Årsaken er kildebevist: hver M1 må ha handler, passere kvalitetsmasken og ha
samme kontrakt. D1 krever dermed aktivitet også i stengte minutter. Dette er
ikke bevis for at D1-voluminformasjon mangler i rådata. Nytt observed-domene
summerer bare aksepterte minutter og følger faktisk dekning, kjent/ukjent side,
kontrakt, første/siste observasjon og uendrede strict-masker. Histogrampasset
bruker de fullførte M1-/prisgridbytene og avstemmer hver side, uten ny råskanning.

Observed-SMC er eksplisitt en proxy på observerte candles; en intern manglende
prisbane eller institusjonell ordreblokk blir ikke rekonstruert. L1/ordreantall
aggregeres med egne masker og faktisk observert tid.024bevarer ekte intraminutt
spread-/eventreduksjoner; et M1middel alene rekonstruerer ikke slike hendelser.
Verken reduksjonene eller prisgridets volumtabeller bevarer hele råhendelsenes
felles rekkefølge, alle T-størrelsesfordelinger eller full BBO-bane. Full neural
price-grid-ingestion og faktisk alle-TF modellkonsum er fortsatt egne ubeviste
funksjoner, selv om de deklarerte featurebytene er fysisk publisert.

## Beholdt korrekt eierskap og verifikasjonskrav

GC-normalisering bruker bare observed==1 på fysisk TRAIN. Lagringsnull bak
observed==0 er ingen markedsverdi og maskes etter normalisering før projeksjon.
Entry-only-trening skal ikke bruke Exit-teacher eller Exit-gradienter. Ny smoke
må også bevise uendrede Exit-vekter, ekte batchgjennomløp, familie-/TF-påvirkning
og samme bundne train/serve-beregning. Kildelesing erstatter ikke disse målingene.

Alle26månedene er gjenbrukt utvikling. Kronologisk TRAIN/VAL/kontroll og censur
ved utfallsgrensene beholdes; en ny oppdeling gjør ingen gammel periode urørt.
TEST fra juli2026 brukes ikke til tilpasning. Senere økonomi må inkludere alle
valgte handler, åpne posisjoner, spread, finansiering og øvrige deklarerte
kostnader mot samme-risiko referanse. Ingen profittpåstand følger av datakvalitet.

## Opprydding

Ingen Python-fil var bevist uten repo-referanser når import, CLI, tester og
beholdt eierskap tas med. v12_trade_state er bare testimportert, men SYSTEM_MAP
beholder offline persistenseieren eksplisitt; dette er ikke slettebevis.
Gamle aktive rapportfortellinger er erstattet av denne någjeldende revisjonen;
Git og originale maskinkvitteringer bevarer faktiske målinger. Rådata/RUNS og
checkpoints slettes bare gjennom retention-eieren. Ikke opprett nye historikkmapper.

## Grenser som gjenstår å bevise

Individuell numerisk kontroll av øvrige genuine tekniske formler, oppstrøms
råevent→M1 og individuell modellnytte gjenstår.126har nå kvalifisert alle1205
GCverdier/masker TF→lagret M5-context med beslutningsalder og96historikk-radunion.051rekonstruksjon,053liveness,055helTRAIN-inputnormalisering og115/117
initial-/paritets-/resumemekanikk er ferdige bevis med egne grenser.108fullførte
én nativeepoke med blandet komponentlæring; Entry-kvalitet, generalisering og
nettoøkonomi er ikke kvalifisert. Gjeldende terminal og adgang leses fra den
eneste overleveringen og policyen; dette åpner ingen ny trening.

## Dynamisk M5-hovedmål og fysisk måltilgjengelighet

Gammel fysisk emisjon krevde både legacyM1-labels og alle37auxmål selv når
M5-path var hovedmål. Unionen inneholder inaktive96-M5-bars-hoder; fire aktive
M5forecasts var dessuten merket alltid observerte. `legacy_complete` beholder
sin opprinnelige kontrakt. Den separate `m5_primary_observed`-rettelsen har egne
forecastmasker, men dens faste95minuttersmål er ikke den nye primærruten.
Gammelt027utkast skal ikke kjøres.

Ny eksplisitt `m5_dynamic_observed`-emisjon og fase
`entry_only_dynamic_m5_path` er implementert i kilde og avventer samlet
kvalifisering. Eksisterende eiere er `entry_causal_m1_outcomes_v1.py`,
`entry_observed_market_v1.py`, `entry_model_native_training_objective_v1.py`,
`entry_model_native_aux_targets_v3.py` og
`build_entry_v10_ctx_training_dataset_v3.py`. Queryeieren avgrenser M1-klokken
før prislesing ved splitens eksklusive grense.1..480minutter er deklarert før
utfallet observeres, og en liten horisontbetinget dekoder deler encoderforward.
Det er ingen hindsight-valgt beste exit og ingen Exit-target eller Exit-gradient.

Terminalmarkout bruker eksakt M1 BID/ASK ved fill og terminaltid. Denne kan være
observert over et ukjent mellomrom mens MFE og MAE er sensurert. Den primære
banesemantikken inkluderer lukkede M1barer førT og bare eksakt open vedT,
aldri high/low fra Tbaren. Closed-M1-prefix har en egen endpointtype og maske.
Signert MFE og konservativ øvre MAE før første MFE har egne mål; ukjent
intraminuttrekkefølge gir observerte grenser, ingen oppdiktet tickbane. Retnings-,
MFE- og MAE-tap måles uavhengig. MFE minus MAE alene er ikke retningsbevis.

Den fysiske flaten lagrer `dynamic_entry_m1_row` som global quote-radpeker,
`dynamic_first_prefix_observed`, separate LONG/SHORT-masker for første
terminal og closed-prefix-path samt deres censorgrunner. Manifestets
`extra.dynamic_m5_supervision` binder en delt immutable M1-quoteparquet med
SHA/størrelse/radantall/klokkehash/autoritet, global indeksmapping og eksklusiv
splitgrense. Denne quotepriskilden er separat fra026s berikede M1-featureflate.
Ingen tett entry×480framtidsbane lagres. Queryradindeksen refererer til den
forespurte fillsurface; den er ikke den lagrede globale M1-quotepekeren.

Fysisk cohort krever genuine inputs, kausal warmup og minst én genuin komponent
ved første1minuttsprefiks: terminal ELLER separat closed-prefix-path. Den krever
ikke alle480horisonter,95minuttersbanen, legacy37unionen eller dagshorisontene.
Senere komponenter får egne masker, og closed-prefix-observasjon blir ikke
automatisk terminal-open-inkluderende observasjon. Fremtidige masker, censor,
MFE-tid eller realisert elapsed kan ikke være modellinput. Fire M5forecastceller
K1/5/12/24 og seks1/5/20d-støtteceller beholder sine individuelle masker; dagers
mål gir ingen egen primær Entry-autoritet. Manglende støtteceller er ikke nullutfall.

Ukjente spotkildehull sensurerer banen. Godkjente lukkingsintervaller krever
eksakt ekstern XAU-spotkalender med riktig kilde-/klokkebinding; GC/CME eller
prisform beviser ikke OANDAs historiske åpningstider. Full feed-/periodebinding
av en slik kalender er ennå ikke bevist. Terminalmarkout etter deklarerte
kostnader er ikke realisert økonomi. En navngitt kvalitetsdifferanse
(netto endepunktsmarkout minus konservativ pre-MFE-MAE upper, begge
prediksjoner i bps) har ingen native autoritet før separat kontrakt og måling.
Lik bps-vekt1 er valgt semantikk; ulike masker gir ingen ubetinget felles
forventningsclaim. Ratioen kunne ikke avstå på grunn av høy MAE alene.

Dynamisk fysisk TRAIN118249/VAL17284 og fersk wholeTRAIN-registryfit er
fullført i047/048/049. Gjeldende051fysisk metadatabinding og053full input-liveness er fullført.
055wholeTRAIN-normalisering er bestått. Aktuell initialbaseline står i policy;
paritet og dynamisk læring krever fortsatt egne målte bevis. Den ene
bestilte epoken krever fortsatt alle eksisterende kildescope-/input-/mål-/
optimizer-/resume-/maskinvareporter. Dette dokumentet og handovervelgeren gir
ingen launchautoritet; status og eksakte bevisbindinger eies av policyen.

## Kildevis makroavklaring

De61D1/makrofeltene er44OANDA-prisfelter,9Treasury-kurvefelter,2VIX,
2GLD-beholdningsfelter i tonn og4COT-felter. En generell påstand om at alle61
har et påvist vintage-/lekkasjeproblem er ikke dokumentert.44er markedspriser
og CFTC opplyser at historiske COT normalt ikke oppdateres etter publisering;
offisielle korreksjonsunntak finnes. Ingen konkret GC088691-korreksjon i det
aktuelle26månedersvinduet er identifisert. Den eksisterende eieren binder også
13forsinkede2025-rapporter. Full historisk tilgjengelighet/korreksjonshistorikk
er samtidig ikke bevist for hver Treasury-/VIX-/GLD-/COT-verdi.

Specens revised_after_publication=false er en kildeerklæring, ikke en målt
vintage-sammenligning. D+1-laget for Treasury/VIX/GLD er konservativt under
normal publisering og beviser ikke hver historiske release. Driverendringer
teller dessuten rader i en felles, forward-filled datokalender; de betegner
ikke nødvendigvis hver kildes egne handelssesjoner. Disse grensene skal
rapporteres kildevis og ikke omskrives til en udokumentert blanket-lekkasje.
[CFTC FAQ](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm),
[CFTC historiske kunngjøringer](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm),
[Treasury-metode](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology).

Aktivert1205registry er etterkontrollert i75unike forbrukercaser. En konkret
normaliseringsfeil i min/max for maskerte binærverdier er rettet uten svekkede
domene-/statsvakter. Dette er syntetiske kilde-/modellmekaniske kontroller;
ny fysisk003inputkjede, faktisk inputparitet og læring er fortsatt egne krav.

## Gjeldende individuelle bevis19.10

060/INDIVIDUAL_FEATURE_EVIDENCE_INDEX.json binder253×5MTF-felt,1205GCverdier
og1205separate masker,2542signal/context-aliaser og183andre lokale signaler.
Indeksen skiller kilde-/klokkebevis, shared-owner-paritet, uavhengig numerisk
referanse og ennå umålt aktuell modellforward. Ubeviste unit-/warmup-/
referansefelter er eksplisitte; dette er ikke en full individuell formelPASS.
276runtimealiaspar og217historiske sourcepar er forskjellige tellinger.
53GCkontekstfelt perTF beholder fysisk M5profilklokke.

Første kompakte profil er ett kildeforslag, ikke en målt støyseleksjon:
alle2410GCkanaler beholdes. Lokal candle-rute inneholder74GCkanaler og
volatility-ruten102; familienavn alene gir ingen trygg slukking av MBP.
061implementerer2622/2525/139 og7lokal/5MTF som én canonical profil med
streng navn/SHA-reference, full fysisk055validering før gather og reelt modulskip.
061har380godkjente source-/syntetiske modellcaser. Aktuell initialbaseline
måler ekte512entries; full individuell konsum- og latencymåling gjenstår. Legacy wholefeature LIVE-only-audit skal fortsatt avvise
conditional GC-/maskestatus; dens forventede avvisning er separat fra faktisk053liveness.
Fullbibliotek og eksakte rawkilder beholdes; ingen ekstra fit/HPO følger av dette.

Den avgrensede13felt-kontrollen i065/MACRO13_READ_ONLY_REVIEW.json binder
aktuelle source-/051-/055-metadata. Treasury9 bruker tre daglige yieldserier
med tilgjengelighet etter slutten av D+1 New York. COT4 bruker maksimum av
as-of+7d og slutten av offisiell, eventuelt forsinket publiseringsdato;
kalenderen binder13forsinkelser i2025. As-of-join er strengt bakover med
available_at<decision. Ingen faktisk framtidslekkasje ble påvist i denne
avgrensede kontrollen. Treasuryflagget revised_after_publication=false er
kopiert fra manifestet, ikke målt mot originale publiseringer. Årsarkiver
og publiseringskalender beviser ikke alle originalverdier eller rettelser.
Dette er et konkret originalvintage-bevisgap, ingen full PIT-kvalifisering.

Fire avgrensede GC-referansetester er lagt til i eksisterende testeiere:
alle sju kjente kumulative felt ved UTC-dato/gap/roll, tidsvektet mikropris
og vist ordrestørrelse, volum/antall ved ujevne T-antall og spread-event
count/sum/max med filankertilbakestilling. Hver har håndberegnet tallfasit på
alle femTF (20caser). Alle20casene bestod i065s faktiske707PASS-kjøring. Dette er uavhengige
tallreferanser, ingen aktuell fysisk26måneders formel-/modellkonsumPASS.
Produksjonsformlene er uendret.

## Supplerende formel-, enhets- og warmupbevis

Samlekartet i
`/home/andre2/GX1_RUNS/NATIVE_M5_READONLY_FORMULA_UNIT_WARMUP_20261020_002/GC_ALL1205_SOURCE_UNIT_WARMUP_COVERAGE.json`
har SHA2dd5c8c4b1dbe16b959d5ae83e145ef123a28db8a9e808e9575a8c11c21026a5.
Det binder seks rapporter: tidligere350verdier og det disjunkte855komplementet
(210bevarte felt,275bok-/event-/ordrefelt og370observerte trade-/profil-/SMCfelt).
Unionen er nøyaktig alle1205registrerte GCverdier og deres1205separate
observasjonsmasker, uten overlapp. Kildeformel, enhet, domene, klokke, reset,
warmup og eksisterende referansers konkrete grenser er nå kartlagt for alle.
Komplett kildekart er ingen full numerisk formelPASS eller bevis for aktuelle
fysiske verdier, individuell modellattribusjon, læring eller økonomisk edge.
Revisjonen gjorde ingen rådatalesing, prosjektimport, testjobb eller modellforward.

060s indeks og001/MANIFEST.json bevares; sistnevntes SHA er
b4919b15710a957b34c406d04a9175963272c0748fa35f68f29ebddf93773911.
001s23tekniske formler er fortsatt beskrevet på alle femTF:115outputposter,
ikke115uavhengige indikatorer. Masker og signal/context-aliaser er ikke nye formler.

Profilrapporten skiller GCkontraktvolumvektet tradeVWAP fra OHLC/tickVWAP;
15/60minutters profil bruker3/12M5bøtter også ved lukket høyereTF-kontekst.
POC-ties velger laveste pris;70%valuearea utvides sammenhengende med eksplisitt
tie-regel. Trade-recordstørrelse er ikke kundens parentordre. Ingen egen
POC-konsentrasjons-/HHI-skalar ble funnet; dette er et mulig utvidelsesforslag,
ikke en målt feil som åpner en ny rebuild.

Bokrapporten skiller tidsvektet BBO-sizeimbalance/mikropris, vist antall/
størrelse og eksekverte T-poster. Spreadutvidelse er count/sum/max i ticks,
ikke en kalibrert shock-zscore. Full uavhengig raw→M1→alleTF-fasit for hele
OFI/refill/depletion-vektoren og kombinerte reset-/feildomener gjenstår.
121kontrollerte uavhengig370aktive publiserte event-/bok-/ordrefelt:70per
M5/M15/H1/H4/D1 og20bevarte strikte M5-felt.1139040M1-rader fra01.05.2024
til eksklusiv01.07.2026 ga finitt dekning for alle370:350varierende og20
observerte konstante nulltellere, ingen heltNaN-felt. Feltposter, masker og
metadata telles ikke som uavhengige indikatorformler. Math.fsum-referansen
bruker analytiske produkt-/reduksjons-/divisjonsgrenser, fullgrid, samme kontrakt
og egne observed/strict-/NaN-domener. BBO og ordre beholder egne tidsvekter.
329felt hadde eksakt nullavvik;41hadde små avrundingsavvik innen de cellevise
grensene. Største målte avvik var5,68434e-14ticks i striktM5 tw_spread.
Alle34fokuserte kildecaser bestod, inkludert faktisk bokskjema-regresjon og
370felt matchet mot eksekvert registry-/formeleier. Produksjonsformler er uendret.
Bevis: `/home/andre2/GX1_RUNS/NATIVE_M5_GC_EVENT_BOOK_NUMERIC_REFERENCE_20261020_121/ACTUAL_NUMERIC_REVIEW.json`,
27bf991c808cad3018092ea6dca4ceaf76ada6216e6d2cd255aee36a6aa66a80.118s tre klokkemidler×5TF er et overlappende delbevis,
ikke15ekstra felt. Publiserte OFI-/incidentmaskerte M1-statistikker gjenbrukes;
råevent→M1, native tilgjengelighetsprojeksjon og individuelt modellkonsum
bevises ikke av denne kontrollen. Klokkelabel alene beviser ikke kalibrert
fysisk nettverkslatens. Roll-/reset- og øvrige formelgrupper beholder sine
separate ubeviste referansegrenser.

122supplerer121med211disjunkte aktive trade-/størrelse-/kumulativfelt:
34observed×5TF +41retained (12M5/M15/H1,5H4,ingenD1). Alle1139040M1-rader
ble kontrollert for eksakt volum-/sidetallparitet før publisert volumemaske.
Pythonheltall bevarer nQ−S², og sidegjennomsnitt bruker egen recordnevner.
Begge kumulative M1-ledgere beregnes én gang, resetter ved UTC-dato/roll/
ugyldigM1/hull og projiseres til egne strikte eller siste observerte endepunkter.
Alle211aktive felt hadde finitt dekning og nullavvik;205varierer,6er konstanter.
16heltNaNstrikte hjelpekolonner er bare maskekontrollert, ikke formelPASS.
Strikt H1-CVD har bare3gyldige bøtter; numerisk paritet beviser ikke læringsdekning.
58kildecaser bestod, fysisk audit tok31,810s/exit0 med206kilder bevart.
Bevis: `/home/andre2/GX1_RUNS/NATIVE_M5_GC_TRADE_CUMULATIVE_NUMERIC_REFERENCE_20261020_122/ACTUAL_NUMERIC_REVIEW.json`,
e7e1d56cbbfbbe5b0e7a26738de4a93c4d04516bf7dcf10c4effcb8b5647ef3c. Samlet121/122kontrollerte581aktive GCfelt på
publisertM1→TF-stadiet, uten dobbeltelling av118. Originalstats er bevisst
UNMASKED; egne publiserte volumemasker brukes etter all-M1-paritet. Ingen
råevent→M1-, uavhengig TF→beslutning-, individuell sensitivitet-/nytte- eller
læringspåstand følger av dette. Ingen produksjonsformel er endret.

SMCrapporten beskriver mekaniske observerte proxyer: trehøyrebarsbekreftede
pivoter, engangsBOS/sweep, tidligere motsatt candle som OB-opprinnelse og
senere wick-retouch. Age teller observerte candles. Hull nullstiller soner;
latest-invalid maskeres uten fallback til eldre gyldig signal. H4-klokken
starter02/06/10/14/18/22UTC og D1 ved22UTC. Volum følger opprinnelsescandlen;
det er ingen påstand om institusjonsidentitet eller volumterskel for BOS.

123 har i tillegg kontrollert198 disjunkte aktive SMC-/prisfelt fra publisert
M1 til fem tidsrammer:161 SMC og37 OHLC/close. Alle har finitt dekning;
196 varierer. D1 active=1 har bare15/17 endelige bull/bear-observasjoner.
Begge sider har positive BOS/sweep/opprettelse/retouch på alle femTF;
D1 opprettelser9/7 og retouch1/2 er sparsomt. 135felt har nullavvik;
63 USD-felt har avrunding innen cellevise analytiske grenser, maks9,094947e-13USD.
44 heltNaN hjelpekolonner er kun maskekontrollert. 77kildecaser bestod;
fysisk audit31,086s/exit0,206kilder bevart, ingen produksjonsformelendring.
Bevis: `/home/andre2/GX1_RUNS/NATIVE_M5_GC_SMC_PRICE_NUMERIC_REFERENCE_20261020_123/ACTUAL_NUMERIC_REVIEW.json`,
df2f2d9b6dd6e0bfd7287d640e5ad3fed693eaf4a1b41721167e9a26a61fade5.
Ved123 var union121/122/123779av1205 registrerte GCverdier uten overlapp; dette teller
felt med tidsramme-/observasjonsvarianter, ikke779 uavhengige indikatorformler.
123/PENDING_NUMERIC_FIELD_GROUPS.json binder de daværende426 utestede verdiene:
316 dynamiske profiler,64 histogram,32 footprint og14 basis.124oppdaterer dette
til14basisfelt nedenfor; den eldre inventaren gir ingen ny kjøreautoritet.
Råevent→M1, uavhengig tilgjengelighetsprojeksjon og individuell modellnytte
bevises ikke av denne publisertM1→TF-kontrollen.

124 supplerer med412 disjunkte profil-/histogram-/footprintfelt. Alle har
finitt numerisk dekning;407varierer og fem snapshotaldre er0 ved eksakt M5→TF.
111felt har nullavvik;301 har avrundingsavvik innen analytiske cellegrenser,
maks9,094947e-13USD/1,267609e-11ticks.14allNaN strikte hjelpefelt er ikke
numerisk kvalifisert. StrictH1-footprint har bare3endelige observasjoner/felt.
Én passering av11285831qualifiedladderrader ga seen=admitted763347M1minutter;
1139040M1-rader,116kildecaserPASS,368,026s fysisk/exit0 og206kilder bevart.
NY18/DST, rolling3/12M5, session-/segment-/gap-/roll-domener, sparse70%VA,
ukjent side og eksakt M5-snapshot uten tilbakefylling er fysisk kontrollert.
Bevis: `/home/andre2/GX1_RUNS/NATIVE_M5_GC_PROFILE_FOOTPRINT_NUMERIC_REFERENCE_20261021_124/ACTUAL_NUMERIC_REVIEW.json`,
6ebe32c512ce0820bdcfa8253ffd7a706138db3064024db6564cf5172ad560de.
124/PUBLISHED_ROW_COUNT_RECONCILIATION.json forklarer at024s11286145 er
inputteller før incidentfilter, ikke11285831publiserte qualifiedrader.
Originalt DESIGN og alle receipts/data er bevart; ingen produksjonsformel er endret.

125 fullfører de siste14 basisverdiene med uavhengig samme-M1 GC-/spotreferanse
på alle femTF:657521endelige celler,14varierende, nullavvik i alder/masker/klokker.
Største bps-avvik3,524292e-12 er innen analytiske cellegrenser. StrictD1close/
basis er heltNaN hjelpekolonner, ikke kvalifiserte verdier.164kildecaserPASS,
fysisk21,969s/exit0,208kilder bevart; native117s182filer er uendret.
Bevis: `/home/andre2/GX1_RUNS/NATIVE_M5_GC_BASIS_NUMERIC_REFERENCE_20261021_125/ACTUAL_NUMERIC_REVIEW.json`,
43050645ff1bd3d3be144f5ca325e6c68cb00fda497297a68dee2dfe93821646.

Samlet121–125 er1205av1205 unike GCregistryverdier, uten manglende/fremmede/
dupliserte felt.125/ALL_1205_PUBLISHED_M1_TF_NUMERIC_INDEX.json binder hver verdi
med registrynavn, tabell/kolonne/maske, faktisk resultat og statistikkpointer.
SHA49de00d10d3d861aae53e212bc41e179a99689f400fda06ff415cbcf69ca6406.
1172varierende/33observerte konstanter;791nullavvik/414analytisk tillatte avrundinger.
Finitt dekning3–227808 per felt er ikke lik læringsdekning.124s14-felts pending-
inventar bevares som historisk bevis, ingen gjenværende M1→TF-verdi.
Råevent→M1 og individuell modellnytte har fortsatt egne ubeviste lag.
TF→lagret M5-projeksjon er nå kontrollert i126 nedenfor.051rekonstruksjon og
055normalisering gjenbrukes; ingen ny sekvens-/PT-/råTB-lesing eller baseline
følger av inputkontrollen.

126 er fysiskCOMPLETE:1205GCverdier+1205masker stemmer bit-eksakt mot
050M5-context på136149unike kilderader,328119090verdi-/maskeceller,0bitavvik.
TRAIN118800+VAL17443kilderader overlapper94; eksakt96historikk-radunion er
målt fra tidskolonner. Beslutning=barstart+300s, selvstendig latestavailable/
ownvalid/sourceclock/alder/f32. Alle1205har observerbar TRAIN-/uniondekning;
VAL1181/1205med24heltmaskertefelt (14D1OB+10retainedH1flow/CVD), korrekt0/0.
Min36videreførte kontekstrader er ikke36uavhengige TF-observasjoner.
225kildecaser bestod på6,888s; fysisk146,957s/exit0 og211kilder bevart.
13filer24645265193byteSHA én gang,10GCprojeksjoner,136224ctx-rader/1419batcher,
75ekstrarader. Fysisk disk-I/O er ikke målt.050s2410faktiske hashpar og051s
bit-identiske sekvens/snapshot gjenbrukes; ingen ny sekvens-/signal-/PT-payload,
modell eller normaliseringsfit.182native117filer er uendret.
GC-footerTESTbarrieren bruker barstart; partialH4/D1bevares uten for tidligkonsum.
Bevis: `/home/andre2/GX1_RUNS/NATIVE_M5_GC_NATIVE_PROJECTION_REFERENCE_20261021_126/ACTUAL_NUMERIC_REVIEW.json`,
724a4616cb4e11ca94a8fed1c4342e175927b835fb68c06cc8494f7b19c08abc.
Råevent→M1, andre tekniske formler, individuell lært nytte, Entry-kvalitet og
økonomisk edge bevises fortsatt ikke av denne inputkontrollen.

Teknisk rapport beskriver WilderATR/RSI/ADX-seeding, EMA/SMA-seed og slope-lag,
stack/stateage, quote-volumeVWAP, BBpopulationddof0, enheter og første gyldige
rad perTF. Gyldige nullverdier skilles fra udefinerte normaliserte kvotienter.
Formel-/warmupbevis alene erstatter ikke uavhengig numerisk kontroll av alle
felt, faktisk modellkonsum eller etterfølgende kvalifisert læring.
