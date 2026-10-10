# CURRENT HANDOVER — 10.10.2026

Brukerens fire avgrensede Entry-edge-steg er fullført. Ingen Entry-edge er dokumentert for den undersøkte hypotesen. Kilden er GX 1_CURRENT / work/gx 1-current. Ingen jobb eller ny launch-autorisasjon gjenstår; training_enabled=false og TEST er fortsatt forseglet.

1. Seks CPU-målinger på tre fryste TRAIN-batcher viste at Entry-kontrasten når 538 delte parametertensorer. Samlet hjelpegradient er svakere og ikke motrettet. Ingen tapsvekt-/gradientrettelse er begrunnet.
2. Én fast ridge-probe med 14 additive felt mot 38 felt med samspill ble gjennomført over 12 purgede kronologiske TRAIN-perioder: 48 CPU-fits,529545 evaluerte rader. Samspillets MSE-forbedring mot historisk konstant var bare 0.007524%; månedsintervallet inkluderer null.
3. Hypotesen avvises for native oppfølging. Native modell, tapsvekter,254 inputfelt, åtte familier og checkpoints er uendret.
4. Begge prober valgte FLAT 529545/529545. Kontinuerlige M1 BID/ASK-ledgere med én maksimal posisjon, alle kostnader og åpne sluttposisjoner ga 0 handler/0 netto. Dette er ingen selektivitet eller handelsfordel.

Full metode, usikkerhet, økonomi og SHA-bundne bevis står i [docs/ENTRY_EDGE_RESEARCH_20261010.md](docs/ENTRY_EDGE_RESEARCH_20261010.md). Sluttkontrollen bekrefter alle mål, perioder, ledgere og bevart originalcheckpoint. Autorisasjonen i NEXT_RUN_POLICY.json er brukt opp og stengt.

En fremtidig undersøkelse trenger en særskilt kausal informasjonshypotese og egen avgrenset forhåndsregistrering. Ingen automatisk treningsutvidelse, terskel-/horizontsøk, CONTROL-tuning eller relansering av disse planene.95-minuttersmålet er valgt med hele TRAIN; dette er betinget utviklingsevidens, ikke urørt OOS eller endelig bevis for at alle Entry-hypoteser feiler.

## Ferdig smoke og målt læring

NATIVE_ENTRY_OBSERVED_SMOKE_20261009_001 kjørte på kilde 0ec2364c.
Native terminal 20:10:18.879 UTC /22:10:18.879 Oslo: guard PASS,
trainer 0, observer 0, optimizer 256, epoch 0, next_batch 256, RESUMABLE.
Windows-tasken fullførte med 0 og er deaktivert. RESUMABLE beskriver lagret
tilstand; det er ingen tillatelse til å fortsette. Ekte resume ble senere
verifisert i en separat avgrenset kontroll, beskrevet nedenfor.

PAIRED_LEARNING_REVIEW.json, CHECKPOINT_READONLY_DIAGNOSTIC.json og
COMPLETION_REVIEW_002.json i run-root binder originale mål, kohorter,
outputs, ONLINE/TARGET, checkpoint, gradientflagg og native/Windows-kvitteringer.
Review brukte ingen nye modellforwards, fits eller optimizersteg.

| CONTROL256 | Initial MSE | Slutt MSE | TRAIN-konstant |
|---|---:|---:|---:|
| LONG |992.9300|964.0220|962.7421|
| SHORT |993.6276|957.0055|955.2976|
| LONG−SHORT |3841.7931|3839.3332|3834.1503|

92–93 prosent av side-MSE-forbedringen på CONTROL er redusert kvadrert bias.
Retningskontrastens korrelasjon er −0.0123. Alle forhåndsdeklarerte CONTROL-
sammenligninger mot begge referansene feiler den samlede numeriske porten.
TRAIN har svakt forbedret tilstandsavhengig punktestimat; intervallene mot
konstant inkluderer null. Det er ikke dokumentert stabil retningslæring.

Alle perioder er rapportert: TRAIN 168 måneder, 126 med 1–5 kohortrader og 42
uten rader; CONTROL 13 måneder med 14–29 rader. Alle observerte uker er med.
Handlingene er FLAT i samtlige perioder med rader. Små periodeutvalg gir
ikke selvstendig stabilitetsbevis. CONTROL er gjenbrukt utvikling, ikke urørt OOS.

ONLINE endret seg fra 52cd7442 til 2d49594b; TARGET forble 52cd7442.
Alle 150 Exit-eide ONLINE-statefelter er bit-like starttilstanden.
Checkpointen bekrefter 256 optimizer-/EMA-steg og gradient/supervisjon for
Entry og åtte hjelpeoppgaver; Exit-gradientflagg er false. Mål og kohorter
er uendret. Ingen konkret implementasjonsfeil er påvist som forklaring på
svak læring. 256 steg kan heller ikke bevise at modellen aldri kan lære.

## Hva Entry faktisk lærer

Entry bruker kausale OHLC-/prisavledede inputs, alle features, åtte familier
og tidsrammer. Fasit er observert eksekverbart BID/ASK-markout etter deklarerte
kostnader for LONG/SHORT og FLAT 0; ingen Exit-modell inngår i Entry-targetet.
TRAIN-eid referansehorisont er 19 M5-barer/95min, ingen maksimal holdetid.
Negative utfall beholdes. Framtidsutfall er aldri beslutningsinputs.

Entry-only-fasen kaller ikke TARGET-forward eller Exit-tap.47 genuine
hjelpeutfall og delte encodere består. Exit-eide parametere har grad=None
før AdamW. Delte encodere kan endre Exit-output; Exit-funksjonen er ikke
frosset. Senere Exit-trening må bevare godkjent Entry og er uimplementert.
Raw bps er ikke en kalibrert sannsynlighet; «sikker retning» er ikke bevist.

NATIVE_ENTRY_OBSERVED_INITIAL_20261009_001 er ekte nullmåling på kilde 30708263.
Dens separate audit rekonstruerte alle 512 Entry-targets fra originale fysiske
utfall, klokker og kostnader. Fersk start fra NATIVE_COMPONENTS_20261009_003:
9637663 parametere, ONLINE/TARGET/EMA 52cd7442. Originale tilstander bevares.

## Omstart, checkpoints og effektivitet

Siste verifiserte fysiske Windows-boot 485 var 21:11:56.500 UTC /
23:11:56.500 Oslo; WSL-boot e5b3d977. Alle 182 kode-/recipe-/tilstandsbindinger
bestod før/etter omstart. Resume-kjøringen avsluttet med native guard PASS,
trainer/observer 0 og Windows-task 0; tasken ble deaktivert 21:43:24 UTC.
Resume-guard målte maks 49C core,56C memory-junction,131.44W og2164MiB VRAM.
Den opprinnelige smokens maksnivåer var 53C,62C,136.14W og3488MiB.

Checkpoint ble skrevet ved 0/64/128/192/256 steg; lagretid 0.656–1.881s.
Alle 256 steg ligger i gyldig atomisk to-slot-checkpoint med optimizer,
EMA, scheduler, RNG inklusive CUDA, epoch-order og neste batch.
Kanonisk hashkontroll og CPU-deserialisering tok 0.697s med eksisterende
filesystem-cache. Dette er ikke kald GPU-gjenlasting etter reboot.

Målt komponentoppbygging 1003.140s; native elapsed 1571.497s.
Logget treningssløyfe fra første hentede batch til siste step_done tok 142.631s
for 4096 rader, 28.717 rader/s, inklusive warmup og mellomliggende checkpoints.
Sluttlagring og sluttmålinger er utenfor denne sløyfetiden. Dette korte
forsøket domineres av oppstart; tallet er ikke en langtidsbenchmark.

Native-vindu 12000s/3t20, ytre guard 13800s/3t50, Windows-task 14400s/4t.
Tid kontrolleres etter hvert steg; checkpoint hvert 64. steg og ved pause.
Omstart skjer først etter varig terminal og ferskt maskinvidt idle-/writer-/
lås-/GPU-bevis. En WSL-shutdown er ikke fysisk reboot, og vellykket reboot
beviser ikke at årsaken til tidligere blå heng er løst.

Windows-staging C:\Users\Andre\GX1_CURRENT_NATIVE_30708263 er byte-verifisert;
gjenbruk bare når aktuelle skripthasher matcher. Native task er nå deaktivert.
SSH gx1-3090-lan virker. Ingen bakgrunnstrening eller automatisk utvidelse.
Ingen aktiv kjøring skal følges videre. Ved senere godkjent trening brukes
sjelden resultatkontroll og eksisterende automatiske maskinvarevakter.
Ingen idle-omstarter uten et konkret behov.

## Verifisert resume og lukket omfang

NATIVE_ENTRY_OBSERVED_RESUME_EQUIVALENCE_20261009_001 kjørte på 073fdd0c.
Native terminal 21:41:09.345821 UTC /23:41:09.345821 Oslo. Den separate
sesjonen gjenopprettet originalt steg192 og gjenskapte de samme64 stegene til256.
Ingen nye unike TRAIN-rader, CONTROL-forwards, full epoch eller full VAL.
Originalt steg192/256 og opprinnelig peker er eksakt bevart.

RESUME_EQUIVALENCE_REVIEW.json ga BITWISE_NATIVE_RESUME_EQUIVALENCE_PASS:
alle tilstandsfelt matcher, inklusive ONLINE/TARGET, optimizer, EMA,
scheduler, Python/NumPy/Torch/CUDA RNG, epoch-order, checkpointindeks og
fremdrift. Bare ny session_contract_sha256 skiller. COMPLETION_REVIEW.json
binder dette til native/Windows-kvitteringer og avslutningen av tillatelsen.

Kanonisk CUDA-restore, tilstandsoverføring og TRAIN-order tok73.803s etter
fysisk omstart; komponentklargjøring tok1033.667s separat. De1024 gjentatte
TRAIN-radene tok34.080s fra første hentede batch til siste step_done,
30.047 rader/s inklusive warmup, men uten sluttlagring. Lagring tok1.602s.
Native-kvitteringens totale varighet var1530.654s. Korte kjøringer domineres
av oppstart; disse tallene er ikke en langtidsbenchmark eller læringsbevis.

Forberedelsen endret bare eksisterende resume-/scope-kontroll. Alle andre
trenerfunksjoner/klasser var AST-like, og treningssløyfen og resten av filen
var byte-like originalen.151 dedupliserte fokuserte tester bestod med3 SKIP;
seks kontroller av review-operatøren bestod. Den genuine testen bekrefter nå
likhet etter fysisk omstart, utover syntetisk og CPU-basert evidens.

Første preflight brukte feilaktig heltallsinterseksjon mellom TRAIN-/CONTROL-
rad-IDer. IDene er lokale til forskjellige fysiske parqueter. Korrigert bevis
bandt begge kilder og strengt atskilte originale klokkeområder. Feilet rapport
er bevart; datasett og modell ble ikke endret.

## Videre læring er ikke godkjent

Handover uten bundet native-vindu rapporterer fortsatt manglende generelle
fullprofilartefakter i required_evidence, inklusive resume_equivalence. Disse
krever egen evidens for VAL-profil256/8. Suffix-testen ovenfor verifiserer
bare det navngitte Entry-prefixet; den fyller eller frafaller ikke fullportene.

Bevar det negative læringsresultatet. Ingen flere native optimizersteg, nye CONTROL-
terskler eller blind budsjettutvidelse. Den nye bestillingen 10.10 åpner bare
de fire avgrensede forskningsstegene ovenfor, med separat forhåndsbundet scope. Neste eventuelle hypotese må bygge på bevart TRAIN-bevis;
stor trening krever fortsatt bestått læringsport. Ingen konkret kodefeil er
påvist som forklaring på svak læring. Drifts-PASS endrer ikke dette utfallet.

## Helrepo-revisjon og bevart historikk

De opprinnelige 638 sporede filene er inventert; Python-AST, JSON,
shell/PowerShell, lokale importer og Markdown-lenker bestod. Fullt testutvalg
ga 6290 PASS, 3 SKIP og 6 kartlagte feil. Alle seks ble rettet og kontrollert
fokusert. REPO_AUDIT_20261009_001 binder originalt bevis og senere rettelser.
Entry-bølgen ga deduplisert 244 PASS, 3 SKIP og ingen gjenstående feil.
Genuin datakobling for 652552 TRAIN/256 CONTROL bevarte alle 47 hjelpekolonner.

Minimale rettelser omfatter os-import i skrivevakten; fysisk foreldremanifest;
kanonisk JSON-familierekkefølge; checkpointpause mellom steg;4t Windows-task
og riktig klokke-launcher; fersk fysisk campaign uten gammel modellautoritet.
Dobbelt scope i sampler-adgangen ble fjernet etter måling. Post-record inspect
fikk 180s etter målt 99.547s mot 90s; oppstart/record 90s og vindusgrenser består.
Tre komplette genuine native-sykluser bekrefter nå Windows-rettelsen.

NATIVE_INITIAL_20261009_003 med Exit-basert Entry-fasit er historikk.
NATIVE_SMOKE_20261009_001 ble aldri kjørt og er erstattet.
Ingen fullført eller claimet plan relanseres. Én etterfølgende lesediagnose
feilet JSON-rapportering av eksisterende ±inf checkpoint-sentineler;
rapporteringen ble rettet uten endring av checkpoint eller treningskode.
Begge operatørversjoner og terminaler er bevart.

HISTORY2009W_NATIVE_V38_20261007 har 5523147 M1-rader,652552/70880 TRAIN/VAL,
254 features, åtte familier, alle tidsrammer og47 kontrollerte targetfelt.
Whole-TRAIN-normalisering, 71 aliaspar og fysiske koordinater består.
Inputs, caches, originale checkpoints og fullførte analyser bevares.
TEST, broker, live/paper, handel, spending og promotion forblir stengt.
GC/order-flow, de fire ufullførte GC-trinnene og separat full makro-B er på pause.
