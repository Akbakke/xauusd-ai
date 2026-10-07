# Gjeldende overlevering — 07.10.2026

Gjeldende bestilling: ny kjøreplan, gjennomgang av hele repoet for feil/mismatches,
ferskt datasett, sletting av utdaterte genererte artefakter og liten smoketest
før større trening. GC/order-flow og nye order-block-/footprint-utvidelser
forblir paused. Planen heter HISTORY2009W_NATIVE_V38_20261007 og er bundet
i NEXT_RUN_POLICY/native_v38_rebuild_20261007. Eldre repo-opprydding er fullført,
men den nye DATA/RUNS-oppryddingen er ikke utført.

## Nåstatus

Ingen treningsjobb kjører eller er startet av den nye bestillingen.
Hele kildeinventaret er kontrollert: 640 tracked filer / 581 Python-filer;
ingen syntaksfeil, lokale importhull, doble toppnivådefinisjoner, brutte
Markdown-lenker, JSON-duplikater, shell-syntaksfeil eller avhengighetsmismatches.
Dette er mekanisk repo-dekning og risikoprioritert manuell gjennomgang, ikke
manuell revisjon av hver linje. Source-review/hasher står i policyen.
Én fullsuite startet fail-fast og stoppet etter 19 beståtte/1 feil på foreldet
v37-smokebredde 242 mot utført v38-kontrakt 254. Rettet til kontrakteide mål.
En fokusert build-test avslørte også for kort syntetisk SMC-warmup: ny
tosidig AVWAP var først definert på rad 262, ikke price-prefix 219.
Fixture måler nå egen SMC-prefix; produksjonsvakter/NaNs er ikke svekket.
Endrede flater: 368 fokuserte tester bestod; siste retentiongruppe har 155 PASS.
Samlet case-sensitiv JUnit-dedup av gruppene gir 544 unike cases med siste PASS; det er ikke
fullsuite-PASS. Den første fullsuiten var avbrutt fail-fast som beskrevet over.
8 endrede/eksterne Python-filer og tre JSON-autoriteter består syntakskontroll.
Tre F821-lintvarsler er allerede vurderte lovlige closures; 66 F811 er
fixtureimports/parametre. Ingen kode slettes kun på disse lintvarslene.
Slettingsvernet manglet dagens policyrot. Eksisterende launch-register er nå
hash-bundet til NEXT_RUN_POLICY; retention støtter eksakte {path,sha256}-
bindingsformer og samme exact-target-eier for DATA/RUNS. Foreldres TEST-rolle
bevares også i nested bindings før child-metadata kan åpnes. Kanonisk kilde-
dir følger sin reelle policy; dette er ikke et nytt generisk manifestunntak.
Faktisk read-only closure stopper nå på den ennå uregistrerte nye run-rooten,
ikke på CURRENT-kildedir. Etter replacement må genuine run-registrering og
full closure bestås; ingen håndlaget graph-stub. Ukjent closure og TEST
blokkerer fortsatt, uten payloadlesing eller bypass. Ingen sletting nå.
Rå-/kalibreringsreview er faktisk utført under capped audit: uendrede pair-
bytes og seks squeeze-klokker er validert, og komplett pre-TEST M1-råkilde
har 5959045 rader. Dette er kun avhengighets-/klokkeidentitet, ikke modellbevis.
Én kildebundet supervisor bygger først eksisterende chain, deretter komplett
M1 og selvstendig clock-oracle samt eksisterende post-build readiness. Den
har felles 64800s budsjett, immutable START/PROCESS/stage-terminal/sluttterminal
og stopper før ny normalisering, retention og native modellsmoke. Alle tunge
faser bruker eksisterende capped-eier; kilden må være fryst hele løpet.
Faktisk start/ferdigstilling er bare bevist av prosess-/terminalkvitteringer i
INPUT_BUILD_001, ikke av at dette dokumentet eller planen finnes.
Handover 07.10 kl. 04:31 UTC viste ren CURRENT-kilde og ingen Python-workload.
Den eksekverte signaleieren bekreftet v38 og 254 signalfelt. Alle eksisterende
features, spesialister og tidsrammer bevares; «order block og
det» tolkes som utsatte nye utvidelser, ikke fjerning av dagens SMC-primitiver.
Forrige benchmark er terminal og konsumert. Ny samplerplan med egen endelig
budsjettautoritet må bindes før kjøring; deretter følger læringsløpet i
docs/NATIVE_LEARNING.md. training_enabled=false; ingen full epoch/full VAL.
GC-kildemangelen er ikke en avhengighet for dette eksisterende v38-oppsettet.

Prioriteringsendringen er verifisert med 200 fokuserte status-/handover-/
protokolltester under capped audit, JSON-/AST-kontroll, avgrenset stale-path-
scan og git diff --check. Sammenligning mot forrige policy bekreftet uendrede
input-/normaliseringsbindinger, konsumert native autoritet, GC-fremdrift og
gratisprøvekvitteringer. Dette er konsistensbevis, ikke ny modellmåling.

## Bevart GC-arbeid — på pause

Den tidligere brukerpresiseringen bestilte en kostnadsfri kildeundersøkelse:
AlgoSeek Sandbox US6011 først, deretter offentlig Databento CME MBP-1 og
Portara GCE2019V kun for teknisk innlesing. Ingen abonnementer eller belastninger.
Resultatet og alle kvitteringer er bundet i policyens free_source_investigation;
det er ikke en erstatning for de fire empiriske trinnene.

AlgoSeeks gjestekatalog viser US6011 i pakken til USD 0/måned, med januar–mars
2023 og hele symboluniverset. Konkrete gratis GC-utløpsfiler, handelsdagantall,
kontokvote og demoens lokale trenings-/backtestrettigheter er ikke verifisert.
Pakkens «No Download Fees» overstyrer ikke Sandbox-vilkårenes §4 om kvoter og
mulig overforbruksgebyr. Ingen AlgoSeek-GC-fil er hentet; ESZ3-forhåndsvisningen
fra august 2023 er ikke bevis for gratis GC. Ingen konto, nøkkel, abonnement
eller eksplisitt lisensaksept er opprettet. Målrettet plugin-søk ga ingen
AlgoSeek-kobling; kontrollerte miljø-/dotenvflater viste ikke AlgoSeek-nøkkelnavn.

To offentlig lenkede filer er faktisk hentet uten autentisering/betaling,
hash-bundet og inspisert under capped audit. Hele Databento-filen, 350169102
byte og 2185295 rader, inneholder bare ESZ5, ingen GC. Mottaksdato er
22.09.2025 UTC; filen slutter kl. 15:59:59, altså ikke en hel handelsdag.
Portara har 1999 hendelser, hvorav 11 handler, 1112 bid og 876 ask, fra
06.08.2019 kl. 00:00:00.664–00:05:39.776 uten bevist tidssone. Ingen
aggressorside, sekvens-/mottaksklokke eller eksplisitt bokresettfelt finnes;
den er kun teknisk evidens, ikke profitt-/aggressorfasit. Ingen tilstrekkelig
GC-strategihistorikk er kvalifisert. Originale prefiks-/feil-/metadatafiler
er bevart; en feilskrevet prisetikett er korrigert i ny immutabel kvittering.
Ingen forskningsperiode, native kode, TEST, fit eller backtest ble endret/åpnet.

Alle fire utsatte trinn og deres ferdigkriterier er registrert i den eksisterende
GC-protokollen. NEXT_RUN_POLICY/current_work.gc_goal_progress er eneste
fremdriftsstatus. Ingen trinn er merket fullført uten genuine bevis.
Blokkeringsrevisjonen 13:09 UTC bekreftet samme manglende genuine GC-kilde/
brukbare leverandørtilgang i tredje påfølgende målomgang. Kildeplanen har
fortsatt null bundne filer; nytt avgrenset inventar og nøkkelnavnsjekk ga ingen
input/tilgang. Handover viste ren kilde og ingen live CURRENT-jobb å vente på.
Alle fire trinn er ufullførte. Videre empirisk arbeid krever lisensierte
pre-TEST GC-filer eller faktisk brukbar leverandørtilgang konfigurert lokalt;
ingen hemmeligheter skal sendes i chat. Omfang og ferdigkriterier er uendret.
Ved overtakelsen 12:46 UTC ble ingen GC-/DBN-/flow-fil funnet i et avgrenset
filnavnsøk i DATA/RUNS uten TEST/SOURCE_CHUNKS; det dekker ikke alle mulige
lagringssteder. DATABENTO_API_KEY var ikke deklarert i CURRENTs .env eller
Windows-prosessmiljøet og var ikke satt i WSL-prosessmiljøet. Bare nøkkelnavn/
tilstedeværelse ble kontrollert; ingen verdi ble skrevet ut. Ingen tilgjengelig
markedsdatakobling ble funnet i verktøyoversikten. Ingen vendor-API, kjøp,
lisensaksept eller ny datanedlasting ble utført i disse tidligere kontrollene.
Videre kontroller dekket avgrensede Windows-filnavn, målrettet plugin-søk og
offisiell CME-prøve-/tilgangsmetadata. En annonsert gullprøve fra 02.01.2020
er ikke lastet ned, lisensavklart eller kvalifisert for A/B/C. Se protokollen
og policyens source_access_readiness; dette er ikke en ny markedskvittering.
Brukeren har presisert historisk TRAIN/VAL/TEST og backtest med alle avtalte
features. Dette er dokumentert i protokollen som adskilte tidsperioder og
bevisnivåer; ingen native launch eller TEST-åpning følger av presiseringen.

GC_ORDER_FLOW_RESEARCH_001 har en kildebundet protokoll og implementert lokal
`audit-gc-source` hos eksisterende research-eier. Se
docs/GC_ORDER_FLOW_RESEARCH.md og configs/research/GC_ORDER_FLOW_RESEARCH_001.json.
81 fokuserte tester i GC-kilde-/eksisterende research-eier bestod under capped
audit. Syntetiske fixtures beviser mekanikk, ikke GC-kvalitet eller tradingverdi.
Ingen fil er bundet til selve GC-strategitesten. Den tomme kildeplanen rapporterer
BLOCKED_NO_BOUND_GC_FILES, ikke godkjent dekning. Ingen empirisk A/B/C,
native endring, trening eller indikatorfjerning er gjennomført. Den separate
gratisprøveinspeksjonen beskrevet over gir ingen slik adgang.
En eventuell niende spesialist er ikke vedtatt; dagens åtte bevares.

Siste native terminal er bevart nedenfor og er ikke en GC-resultatkvittering.
SAMPLER_BENCHMARK_001/ATTEMPT_003 er kontrollert avbrutt, ikke fullført.
Verifisert worker PID 10886 fikk SIGINT; KeyboardInterrupt ga exit 1.
Supervisoren ventet inn prosessen og skrev FAILURE og TERMINAL.
Siste rapport var 3200 av 8192 Entry-par for første kandidat (32768 overganger),
etter 2868,03 sekunder. Andre kandidater er ikke fullmålt.

Terminal: /home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/SAMPLER_BENCHMARK_001/ATTEMPT_003/EVENTS/TERMINAL_20261006T065527088266Z.json
SHA-256: 87f3faf08038727f182dd591ca3e09a7092d139469adb030fe3c02b9c9dbdc20.
Kilden var uendret ved terminalen; TEST ble ikke brukt. Originale logger,
planer, delresultater og kvitteringer er bevart utenfor repoet.
Prosessene er reaped og prosjektlåsen kontrollert ledig etter stopp.
En aktuell prosessobservasjon må fortsatt tas ved ny overtakelse.

Ingen full benchmark, valgt sampler, koordinatpublisering, fersk initialmåling
eller 256-stegs læringsprøve er etablert. training_enabled=false, full epoch
og full VAL er stengt. Den claimede engangsplanen må aldri relanseres.

## Tidligere inputs — beholdes til fersk replacement er akseptert

NEXT_RUN_POLICY.json binder fullførte v38-inputs under
/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001:
652552 TRAIN-rader og 70880 utviklings-VAL-rader, fysisk separate M1-visninger,
fullført normalisering, økonomiautoriteter/indekser og sekvenskontroller.
Signalflaten har 254 felt i åtte familier. Disse tallene beskriver det bundne
bygget; feltene og dimensjonene eies fortsatt av kontraktene, ikke dokumentet.
Den eksplisitte nye bestillingen er grunnen til ny output-identitet, ikke
relaunch/relabel av dette gamle bygget. Fersk fysisk TRAIN-normalisering og
komplett pre-TEST M1-coverage er obligatorisk. Råkilder/kalibreringsforeldre
beholdes også når genererte gamle outputs kan pensjoneres.

## Opprydding

Oppryddingen er fullført og verifisert: 842 → 635 repo-filer,
136 → 20 Markdown-filer. Alle 257 beholdte gx1-filer er byteuendret.
Ingen lokale importhull eller brutte dokumentlenker er funnet.

Doble/historiske statuskilder erstattes av current_work i NEXT_RUN_POLICY.
Statusleseren kontrollerer eksplisitt terminal/hash og CURRENT-prosesser;
manglende autoritet feiler lukket, uten checkpointfallback.
Claude-vaktene peker nå på CURRENT og er synkronisert med de installerte
kopiene etter særskilt brukerautorisasjon. Andre globale innstillinger er urørt.
Økonomiske enhetstester bruker isolerte syntetiske fixtures, ikke et gammelt worktree.
Omfang, slettinger, kontroller og ubeviste grenser står i docs/REPO_REVIEW.md.

I den tidligere repo-oppryddingen ble ingen DATA/RUNS slettet. Den nye
bestillingen tillater nå eksakt retirement av utdaterte genererte outputs,
etter replacement-aksept, komplett avhengighetsbevis og retention-ruten.
Råkilder/provenance som fortsatt trengs, .env, .venv, .git og GC-pausen bevares.
Historiske repo-filer kan gjenopprettes fra Git ved behov; ingen ny arkivmappe opprettes.

## Neste grense — eksisterende v38-læring

Bind ren kilde og fersk CPU-build gjennom eksisterende eiere, deretter
genuine acceptance/komplett M1/ny normalisering og retention før modellsmoke.
Den pair-alignerte M1-lifecycle-flaten alene dekker ikke komplett rå M1-klokke;
den komplette state-flaten er derfor en planlagt obligatorisk byggefase.
Ingen featurekassering eller historiske modellvekter. Ny full TRAIN-only
samplerbenchmark krever sin egen finite
plan/budsjettautoritet; gammel engangsgodkjenning fornyes ikke automatisk.
ATTEMPT_003 og originalbevisene bevares, aldri relanseres.
Deretter valgt sampler/fryste koordinater → fersk initialmåling → avgrenset
256-stegs prøve → separat TRAIN/CONTROL256-review → bare betinget, endelig
utvidelse. Punktene 1–6 og eksisterende launch-/maskinvareporter står i
docs/NATIVE_LEARNING.md; dagens policy åpner ingen modelltrening.

GC/order-flow/footprint og nye order-block-utvidelser venter på uttrykkelig
senere gjenopptakelse. Ingen flere kildesøk, nedlastinger eller GC-fits nå.
Protokoll, kildekrav, perioder, prøver, hasher og ufullført fremdrift bevares.
Ingen kilde eller niende familie er innført i den native modellen.
Full makro-B er separat og ufullført; den må ikke reduseres til MACRO_CORE.
TEST, broker, live/paper, handel, spending og promotion er stengt.

Kjør bash scripts/gx1_handover.sh --check ved overtakelse. Den maskinlesbare
nåstatusen og faktisk prosess-/terminalbevis overstyrer dette dokumentet.
