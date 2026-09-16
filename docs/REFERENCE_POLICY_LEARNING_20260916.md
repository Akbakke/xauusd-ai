# Referansepolicy32 — målt, ingen utvidelse

Referanse32 er ferdig og gir **ikke grunnlag for større trening**. På512 trente
Entries er samlet Exit-MSE omtrent0,019% lavere; på128 separate TRAIN-Entries
omtrent0,0034% høyere. Endringene er små og spriker mellom måneder. LONG holder
omtrent90% av tilfellene; SHORT velger alltid EXIT. Entry er fortsatt FLAT på
alle640, med litt bedre trent tilpasning og dårligere separat TRAIN-tilpasning.
Dette er ikke bevist tilstandsavhengig beslutningsforbedring eller profitt.

Målrettelsen ga balanserte positive/negative Q_mu-mål og samme referanse for
begge sider, men løste ikke verdilæringen i dette avgrensede forsøket. Entry-
læreren ble bevisst bevart.32 steg beviser ikke at modellen aldri kan lære;
resultatet åpner heller ikke flere blinde steg eller teacher-refresh.

Fysisk boot448, guardPASS/trainer0/observer0, terminal2026-09-16T18:52:41Z.
Checkpoint96/global5809/offset1728 er bevart; dette er referansepolicyens96,
ikke de eldre96-forsøkene. Windows-task er deaktivert og ingen native prosess
observert. Den avgrensede launch-tillatelsen er fjernet og avvisning kontrollert.
Les [måling og eksakte bindinger](docs/REFERENCE_POLICY_LEARNING_20260916.md).

Neste arbeid bruker eksisterende tap-/gradientlogger, targets, outputs og
checkpointtilstand til å avgrense den svake verdioppdateringen før mer kode.
Ikke gjenta måloppretting, kandidat32 eller den ferdige før/etter-målingen.
Alle200 features/åtte familier/tidsrammer, kausalitet og vakter er bevart.
TEST er forseglet; ingen full epoch, VAL eller handelskjøring er åpnet.

## Samme frosne mål før og etter

| Utvalg/side | MSE før | MSE etter | Korrelasjon før → etter | HOLD etter |
|---|---:|---:|---:|---:|
| trained LONG | 1012.408122 | 1012.515807 | 0.0327 → 0.0472 | 89.84% |
| trained SHORT | 1010.072086 | 1009.584579 | -0.0293 → 0.0818 | 0.00% |
| separate_train LONG | 1070.624416 | 1070.544078 | 0.0004 → 0.0384 | 90.04% |
| separate_train SHORT | 1069.265899 | 1069.418265 | 0.0212 → 0.0318 | 0.00% |

Trent LONG/SHORT bedret MSE i6/7 av12 måneder; separat LONG/SHORT i7/5.
Sentrert feil bedret seg i5/8 trente måneder og6/6 separate måneder.
Begge trente sider er fortsatt verre enn hvert sitt trente konstantgjennomsnitt.
På separat TRAIN er LONG litt bedre og SHORT litt verre enn konstant0.
Svak høyere korrelasjon alene er ikke dokumentert kalibrert handlingsverdi.
Entry-MSE trent15,52216→15,45082; separat6,44416→6,49612. Alle er FLAT.
Målingens tap/regret er mot Q_mu-læringsmål, ikke realisert netto handels-PnL.

Utvalget er512 faktisk trente og128 separate Entries,0 overlapp,12 måneder.
Tidligere «separat128» inneholdt16 trente. Disse ble erstattet med cachedbatch1
basert bare på rad-ID; de112 øvrige beholdes. Separate TRAIN er ikke holdout.

Frosne mål/resultater, original95, ny96, native receipt og alle outputs er bevart.
CPU-evalueringen gjenbrukte før-output/inputs/mål og tok225,37 sekunder;
ingen nye lærerforwards, backward, GPU, VAL eller TEST. Native tilstandsovergang
bevarte lærer, Adam, EMA, RNG og rekkefølge før de32 faktiske oppdateringene.

Eksakte artifact-/checkpointbindinger og månedstall finnes i
[maskinrapporten](../handover_snapshot/REFERENCE_POLICY_LEARNING_20260916.json).
Fullt før/etter-resultat ligger under dens result.path; nye outputs er lagret
ved siden av. Target-preparation brukte kilde9980f710; native/før-etter brukte
f3b5eca8. Modellsourcen er identisk. Første planforberedelse stoppet på JSON-
linjeslutt i kildehashen; bare planbindingen ble rettet, mål/resultater ble bevart.
Omstart ryddet midlertidig eval-script; dette ble lagt i varig artifactmappe før
første evaluering. Ingen av hendelsene gjentok trening eller modellmåling.

Eksisterende førstebatchlogg viser at forecast har gradient til de fire målte
familie-/tidsrammerutingsparametrene, mens Entry/Exit ikke bruker disse direkte.
Dette var også del av tidligere arkitektur og er ikke alene en ny bevist feil.
Det eldre diagnostikkfeltet frozen_bootstrap_bps er target minus første reward;
for120-stegsreferansen omfatter det også senere observerte rewards. Bruk de
frosne target-filenes eksplisitte bootstrap_component_bps ved videre analyse.
