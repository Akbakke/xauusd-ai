# GX1 — siste overlevering, 09.10.2026

Den ekte native nullmålingen av den tidligere target-definisjonen er fullført.
Ingen optimizersteg eller smoke er startet. Operatøren har09.10 vedtatt at
Entry skal finne stabile retningsmuligheter uavhengig av Exit, som deretter
håndterer posisjonen. Dette krever nytt mål og tydelig gradient-eierskap før
smoke. Stor trening er fortsatt portbundet.
App-målet i thread01a1203d-4a95-7d03-bbd9-84017db6b29c er aktivt.

Kun /home/andre2/src/GX1_CURRENT, branch work/gx1-current, én agent og
én tung CURRENT-jobb. Mac er overleveringskopi. NEXT_RUN_POLICY.json eier
eksakte bindinger og autoritet; kjør scripts/gx1_handover.sh --check.
Fullførte planer/receipts er bevis og skal aldri relanseres.

## Nåstatus og ekte modellbevis

NATIVE_COMPONENTS_20261009_003 fullførte fersk CPU-konstruksjon gjennom
eksisterende eiere med9637663 parametere,652552 TRAIN og70880 VAL Entries.
Initialstate116539510 bytes er varig lagret. ONLINE/TARGET/EMA startet
med samme hash52cd7442 og null optimizer-/EMA-steg. Ingen refit/TEST.

NATIVE_INITIAL_20261009_003 fullførte ekte guarded native-vindu med
terminal16:57:08 UTC, guardPASS og både trainer/observer exit0.
TRAIN256 og separat CONTROL256 er målt:256 Entry- og256 Exit-anchor-
observasjoner pluss1024 samplede Exit-tilstander i hver gruppe.
INITIAL_MEASUREMENT_AUDIT.json bekrefter bit-lik modell, TARGET,
optimizer, EMA, scheduler og CPU/Python/NumPy RNG etter native lagring.
CUDA RNG er separat seedet og lagret. Checkpointet står på optimizersteg0.
Dette beviser starttilstand og lagring, ikke læring eller lært resume.

Samme-run guard målte maks49C core,56C memory-junction,134,76W og2764MiB
VRAM. Workload-only klokkeprofil1395–1695/9751MHz er faktisk kvittert
på boot482. Hele campaign-vinduet inklusive kontroller tok omtrent26min.
Les faktisk state/profilering før konklusjoner om NN-treningsytelse.

Nullmålingens opprinnelige prediksjoner og targets er frosset. Initial
Exit valgte HOLD på begge sider overalt; dette er ikke lært tålmodighet.
Ingen profitable beslutninger, OOS-generalisering eller stortrening er bevist.

## Gjeldende designbeslutning og neste arbeid

Brukeren presiserte: «entry skal være uavhengig av exit og heller finne
muligheter der den er sikker på retningen til markedet ... det viktigste
er en god og stabil entry». Entry skal trenes fra observerte markedsutfall,
uten Exit-lærer i fasiten. Exit-tap skal heller ikke endre Entry gjennom
delte encodere. Lær og kontroller Entry først; senere Exit-trening må bevare
den godkjente Entry-funksjonen innen samme bundle og genuine featuregrunnlag.

Dette er vedtatt retning, ikke ferdig implementasjon. Kontroller eksisterende
direkte M1-utfall, multihorisont-forecasts og risiko-heads før den minste
nødvendige mål-/gradientendringen. Konfidens må dokumenteres på senere
kontrollperioder; rå bps eller softmax er ikke i seg selv kalibrert sikkerhet.
En target-horisont er aldri maksimal holdetid eller en ny lukkeregel.

Gammel fasit brukte observerte referanseutfall pluss fryst, utrent Exit TARGET
ved backup-grensen. Ved TRAIN256 Entry-ankre var gjennomsnittlig absolutt
markedsbidrag13,24/13,35bps LONG/SHORT og bootstrap0,053/0,062bps. Lite
gjennomsnittsbidrag er ikke uavhengighet eller bevis for hvert enkelt tilfelle.
REPO_AUDIT_20261009_001/ENTRY_INDEPENDENCE_DECISION_20261009.json binder
kilde, design, uendrede observasjoner og brukerens beslutning.

NATIVE_SMOKE_20261009_001 fikk publisert/verifisert recipe og campaign, men
ble aldri kjørt. Det gamle scope er flyttet til bevart historikk i policy;
ingen chronological_learning_run er åpen. Planen skal ikke relanseres eller
stille få nye targets. Nullmålingen og checkpointet bevares som historisk
bevis. Ny funksjon/mål krever en ny bundet initialbaseline og smoke-recipe.

Neste steg er å fullføre/teste den uavhengige mål- og gradientkontrakten,
binde eksakte targets/kostnader/kausale klokker, deretter måle ny baseline
og ett begrenset native smoke-forsøk. Samme3t20-vinduer og omstartskontroll.
Større trening krever faktisk læring, stabilitet og kostnadsjustert evidens;
TRAIN-tilpasning, senere kontroll og samlet økonomi holdes atskilt.
CONTROL/Juni2026 er gjenbrukt utvikling; TEST forblir forseglet.

## Omstart og varig fremdrift

Fysisk Windows-omstart via administrativ SSH er verifisert:
boot482 kl.16:27:20.500 UTC /18:27:20.500 Oslo, WSL69a27500.
Alle164 kode-/recipe-/starttilstandsbindinger bestod etter omstart.
Native-vindu12000s, ytre guard13800s, faktisk Task Scheduler-grense14400s.
Tidsbudsjett sjekkes etter hvert fullført optimizersteg; vanlig checkpoint
hvert64.steg samt ved tids-/sluttpause. Ingen blind Windows-omstart:
kontrolleren beholder terminalen og stopper for maskinfelles idle-review.
Andre prosjekter kan eksistere; omstart må ikke avbryte dem.

Den fullførte initial-tasken med gammel plan er nå deaktivert og XML/exit1
bevart i REPO_AUDIT_20261009_001/COMPLETED_INITIAL_TASK_DISABLED.json.
Dette endrer ikke GPU-vakten eller telemetribroen. Ingen ny task er installert.
Gammel Windows-staging C:\Users\Andre\GX1_CURRENT_NATIVE_6D726F7A er historikk;
ny campaign må stage og verifisere dagens controller/observer.
Installer med UseNativeClockProfile og korrekt brukerAndre.
Den eksisterende GPU-effektvakten og signerte telemetribroen skal bestå.
SSH gx1-3090-lan fungerer. Ikke gjør omstart om til wsl --shutdown.

## Helrepo-revisjon og bevarte rettelser

De opprinnelige638 sporede filene er inventert. Python-AST, JSON,
shell/PowerShell, lokale importer og Markdown-lenker bestod.
Hele testutvalget ble kjørt i to deler:6290 PASS,3 SKIP og6 kartlagte feil.
Alle seks ble rettet og kontrollert fokusert uten å gjenta fullsuiten.
Fokuserte kampanje-/initialkontroller etter siste rettelse:49 PASS.
Se REPO_AUDIT_20261009_001 og current_work.repository_review_20261009.

Minste observerte rettelser: skrivevaktens manglende os-import; foreldremanifest
mot eksakte verifiserte pre-TEST-kilder; JSON-rutingers kanoniske rekkefølge
for alle åtte familier; varig pause mellom steg;4t Windows-task med riktig
klokke-launcher; fersk fysisk campaign uten gammel full-VAL-modellautoritet.
Tidligere dobbelt scope i sampler-adgangen ble fjernet: byggingen falt fra
204 til107s og inspect før første receipt tok57s. Etter receipt validerer
sluttkontrollen også det lagrede cursor-scope. Begge scope-kall ble nå målt
til99,547s, over Windows-fristen90s. Kun post-record inspect har fått180s;
oppstart/record90s, treningsvindu12000s, guard13800s og task14400s består.
16 fokuserte tester og ekte Windows-harness bestod. Tasken fra nullmålingen
hadde exit1 etter ekte trainer/observer0 og bevart receipt; dens eksakte
exception er ikke bevart. Målt fristfeil er rettet, men en ny komplett
Windows/native-syklus er ennå ikke verifisert.

CPU-forsøk001/002 og metadataforberedelser INITIAL001/002 bevares med
ekte feil/terminaler; ingen relaunch. Completion-review binder originale
kvitteringer som manglet noen rapporteringsfelt, uten å omskrive dem.

## Datasett og fortsatt avgrensning

HISTORY2009W_NATIVE_V38_20261007:5523147 M1-rader,652552/70880 TRAIN/VAL,
254 features, åtte familier, alle tidsrammer og47 kontrollerte targetfelt.
Whole-TRAIN-normalisering,71 eksakte aliaspar og fysiske koordinater består.
Valgt CPU-sampler målte8192 Entry-par på7275,518s. Dette er ikke NN-
treningsytelse eller en hel TRAIN-epoch. Ingen full datarekonstruksjon nå.

Originale datasett, caches, checkpoints, failed receipts og ferdige analyser
bevares. GC/order-flow og nye footprint/order-block-utvidelser er på pause.
Alle fire ufullførte GC-trinn og separat full makro-B består. TEST er
forseglet; ingen broker/live/paper, handel, spending eller diskopprydding.
