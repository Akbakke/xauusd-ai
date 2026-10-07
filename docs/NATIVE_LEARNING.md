# Gjenværende avgrenset v38-læring

Brukeren autoriserte punktene 1–6 05.10.2026, men prioriterte 06.10 deretter
kontrollert benchmarkstopp og repo-opprydding. Den 07.10 ble GC/order-flow og
nye order-block-/footprint-utvidelser satt på pause og ny trening med dagens
tekniske indikatoroppsett prioritert. Dette dokumentet beskriver det gjenværende
læringsløpet; faktisk kjøring krever fortsatt en ny eksakt plan i NEXT_RUN_POLICY.

ATTEMPT_003 er terminal med exit 1 etter operatøravbrudd. Første kandidat
rapporterte 3200/8192 Entry-par. Ingen full benchmark eller sampler er publisert.
Engangsclaim og godkjenning er konsumert; må aldri relanseres.
Eksakte terminal-, plan-, input- og operatorbindinger står i NEXT_RUN_POLICY.json.
Ingen ny benchmark/trening er autorisert av oppryddingen. Den nye prioriteringen
er heller ikke en fornyelse av ATTEMPT_003s konsumerte engangsbudsjett.
GC-kvalifisering er ingen forutsetning for det eksisterende v38-oppsettet.

## Beholdt rekkefølge

1. Full workload-matchet TRAIN-only CPU-samplerbenchmark gjennom capped producer,
   faktisk full-TRAIN-indeks og alle 254 felt/åtte familier. Behold eksisterende
   kandidater 32768/65536/131072 overganger, batch16, én full repetisjon og
   utfallsblind rangering. Godkjent tidligere eligibility var 10800s per epoch,
   total hard wall 64800s; dette er ikke en ny kjøringstillatelse.
   Original tracemalloc, 2GiB allokasjons-/1GiB padded-inputgrenser og øvrige
   eiergrenser består. Partial/småbatchbevis er ikke full kapasitetsmåling.
2. Publiser valgt sampler og fryste epoch0/first4096/TRAIN256-koordinater
   gjennom eksisterende eiere. CONTROL256 forblir fryst på separat fysisk VAL.
3. Bind eksisterende native recipe/campaign til genuine sampler/koordinater
   og mål fersk nullstegsinitialisering. ONLINE/TARGET bruker samme nåværende
   funksjon. Historiske vekter eller constructor metadata er ikke initialbevis.
4. Én eksplisitt bundet native prøve: batch16, 256 optimizersteg, høyst4096
   Entries, seed20260911, eksisterende AdamW/scheduler, FP32 og TF32 av.
   Bevar initial/sluttstate, mål, masker og terminale kvitteringer.
   Ingen full epoch eller full VAL.
5. Mål Entry/Exit LONG/SHORT mot fersk initialisering og TRAIN-konstanter,
   identiske fryste mål og ukeblokk-usikkerhet. Rapporter TRAIN-fit og
   CONTROL256-generaliseringsmåling separat. Biasflytting/FLAT/HOLD alene
   er ikke tilstandsavhengig læring. Bruk docs/LEARNING_GATE.md.
6. Bare ved bestått eksisterende samlet læringsport: forhåndsregistrer én
   kontrollert utvidelse med endelig, kapasitetsbegrunnet budsjett og eksakte
   checkpoint-/stopp-/evidenskrav. Ingen automatisk uavgrenset trening.
   Feilet port krever diagnose, ikke en ekstra treningsrunde.

## Forutsetninger for eventuell gjenopptakelse

Bruk ferdige v38-data, normaliseringer, sekvens-/feature-/økonomikontroller;
ikke relanser avsluttede inputjobber. Gjeldende design bruker Seq96 og
M5=16/M15=64/H1=96/H4=96/D1=252 fra den faktiske bundne geometrien.
Reference-policy og cutoff kommer fra frosset DESIGN.json.

Ny eksplisitt plan må binde aktuell ren/committet/pushet kilde, inputs,
operator, output-identitet og finite budsjett før launch. Engangsclaim og
supervisor skal avslutte/reape ved hard wall og kreve genuine eiergodkjente
receipts før grønn terminal. Ingen håndlaget seleksjon eller gammelsti-fallback.
Én tung jobb, etablerte cgroup-/GPU-/host-vakter, fryst kilde under kjøring.

Alle genuine features/tidsrammer/targets bevares. Ingen fast tapsgrense eller
maksimal holdetid. TEST, broker, handel, spending og promotion er stengt.
Dette dokumentet demonstrerer ikke læring, full kapasitet eller lønnsomhet.
