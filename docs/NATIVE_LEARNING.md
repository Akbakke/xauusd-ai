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

Kildekontroll: deterministic_fp32 lagrer hver 64 optimizersteg og ved avgrenset
slutt. Checkpointet bevarer ONLINE/TARGET, optimizer, EMA, scheduler, RNG,
epoch-order, neste batch og fremdrift. Eksisterende to-slot-eier bruker
temporær fil, fsync, atomisk rename, hash og atomisk aktiv peker. Diskcacher
gjenbrukes; RAM-/GPU-cache må lastes på nytt. Ved krasj kan arbeid etter siste
fullførte checkpoint gå tapt. Dagens v38 resume-ekvivalens er ennå ikke målt.

Vinduseieren krever eksakt 12000s; tidspause vurderes etter checkpoint. Dette
er ikke et bevist trygt fysisk rebootintervall. Smoke må måle step-/checkpoint-
tid, kald gjenlasting og samme neste batch/tilstand etter resume. Bind deretter
kortere endelige segmenter hos eksisterende recipe/campaign-eiere med margin
til siste batch, checkpoint og terminal. Rapporter beregningstid og lagre-/
laste-/oppstartstid separat. Kortere vinduer er ikke implementert her.
Ingen kalenderjobb avbryter aktivt arbeid. Omstart beviser ikke løst BSOD-årsak.

## Avgrenset smoke og review

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
