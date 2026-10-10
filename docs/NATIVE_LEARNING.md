# Native læring — gjenværende metode

Nåstatus, utført arbeid og aktuell hindring står bare i CURRENT_HANDOVER.md
og NEXT_RUN_POLICY.json. Dette er metodekrav, ikke en launchordre eller
historikk over daglige forsøk. Bruk eksisterende eiere og genuine resultater;
ikke relanser ferdige inputs, kapasitetstester eller engangsplaner.

## Før en modell kan startes

Aktuell ren/committet/pushet kilde, fysiske inputs, valgt målt sampler,
preprocessing og TRAIN-/CONTROL-koordinater må være eksakt bundet i den
kanoniske recipe/campaign. Et separat finite native-vindu og eksisterende
cgroup-/host-/GPU-vakter må faktisk godkjenne kjøringen.

Fysisk omstart krever at alle prosjektjobber, writers, låser og GPU-beregninger
på maskinen er ledige. Etterpå verifiseres ny Windows-boot, WSL, kilde og
artefakter. En terminal eller lav temperatur alene er ikke slik dokumentasjon.

Fersk nullstegs ONLINE/TARGET skal bruke samme nåværende funksjon og delte
moduler. Historiske checkpoints, constructor metadata eller identiske
vekthasher erstatter ikke genuine initial-/funksjonsparitetsmålinger.
Endret ONLINE-funksjon krever en ny faktisk initialbaseline.

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

## Avgrenset smoke og review (opprinnelig 256-stegsmetode)

1. Bruk den særskilt bundne native prøven: høyst256 optimizersteg og4096
   TRAIN-Entries, batch16, seed20260911, eksisterende AdamW/scheduler,
   FP32 og TF32 av. Ingen full epoch, full VAL eller automatisk utvidelse.
2. Bevar initial/sluttstate, mål, masker, alle valgte samples og genuine
   terminalkvitteringer. Bruk samme fryste TRAIN256 og separat CONTROL256
   før/etter; aldri utfallsvelg kontrollrader eller flytt periodene.
3. Mål Entry/Exit LONG/SHORT mot fersk initialisering og TRAIN-konstanter,
   med bias, kontrast, varians, handlinger og ukeblokk-usikkerhet.
   FLAT/HOLD eller biasflytting alene er ikke tilstandsavhengig læring.
4. Rapporter TRAIN-fit og CONTROL-generaliseringsmåling separat etter
   docs/LEARNING_GATE.md. Juni2026 er gjenbrukt utviklings-VAL, ikke urørt OOS.
   Input-/mekanikk-PASS er ikke læring, generalisering eller profitt.
5. Bare ved bestått samlet læringsport kan én ny forhåndsbundet finite
   utvidelse vurderes. Feilet port krever diagnose, ikke en ekstra runde.

## Bevarte datakrav og økonomi

Alle genuine featurefamilier, ordnede felt, M1/M5/MTF-klokker, targets og
normaliseringseiere består. Geometri, reference-policy og cutoff hentes fra
de faktiske bundne eierne, ikke gjettede defaults eller gamle dokumenttall.
Normalisering tilpasses én gang på hele fysisk TRAIN; ingen VAL/TEST-refit.

Original target-M1 og nyere komplett input-M1 er ulike kilde-/artefaktroller.
Gjenbrukt uendret-kilde-/algebrabevis er ikke en ny raw quote-måling.
Ukjente gap forblir ukjente/right-censored; ingen carry, imputering eller
nye forskningsperioder. Ingen fast tapsgrense eller maksimal holdetid.

Senere økonomi må inkludere alle valgte handler, åpne posisjoner, kostnader
og én-posisjonskapasitet mot forhåndsvalgt samme-risiko alltid-LONG.
Lokale prospective kostreceipts er ikke historisk kostfasit.

Regler: GX1_RULES.md. TEST, broker, live/paper, handel, spending og promotion
er stengt. GC og andre forskningsarmer åpnes ikke av denne metoden.

## Separat avgrenset læringskurve

En eksplisitt bestilt læringsstudie kan binde et eget endelig budsjett etter
diagnose, uten å omskrive gammel negativ smoke til PASS. Planen
configs/research/ENTRY_LEARNING_STUDY_20261010.json binder to ferske
initialiseringer: 1024 steg med 256 faste TRAIN-rader og 16384 steg med
262144 unike rader fra den opprinnelige epoch-rekkefølgen. Ingen full epoch.
Arkitektur, alle 254 felt, åtte familier, tapsfunksjoner og optimizer er like.

Nullmåling og faste mellomtrinn skjer i eval-modus på deklarerte rader;
modelltilstand og RNG skal være uendret etter måling. Kontroll av 4096 senere
CONTROL-rader skjer bare ved initialisering og det forhåndsvalgte sluttsteget.
Ingen checkpoint eller parameter velges på CONTROL. Alle CONTROL-resultater
er gjenbrukt utvikling; TEST åpnes ikke. HGB-referansen bruker alle 254 rå
snapshotfelt og samme 262144 fit-rader, med purget indre kronologisk TRAIN-valg.
Den representerer ikke sekvensmodellens fulle inputflate.

Eksisterende atomiske to-slot-checkpoints eier resume. Mellomobservasjoner
bevarer modellhash, RNG-hash, pekerinnhold, rader, mål og prognoser; en gammel
mellomvekt beholdes ikke som separat checkpoint etter at eieren roterer sloten.
Sluttcheckpoint og initial TARGET-state består. Review-metoden og eventuell
videre kvalifisering er registrert før noen av de nye fits kjøres.
