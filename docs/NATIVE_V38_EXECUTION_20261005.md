# Avgrenset v38-læring — operatørvedtak 05.10.2026

Brukeren ba: «Enig med deg. Lag deg et mål for å gjennomføre punkt 1 til 6 nå og knekk i gang».
Dette autoriserer gjennomføring av nedenstående avgrensede løp, ikke bare forberedelse.
Eksisterende design, eiere og vakter gjenbrukes. TEST, broker, handel og spending er stengt.

1. Én full workload-matchet, TRAIN-only CPU-samplerbenchmark gjennom capped producer.
   Bruk faktisk full-TRAIN-indeks, 652 552 Entry-rader, 254 felt og alle åtte familier.
   Kandidater og utfallsblind rangering kommer fra eksisterende eiere; ingen nye søk.
   Batch16, én repetisjon, hele hver kandidat, ingen smoke-avkorting.
   Seq96, M5=16, M15=64, H1=96, H4=96, D1=252 kommer fra den ferdige
   normaliseringens faktiske sekvensgeometri. Reference-policy og cutoff kommer fra
   frosset `LEARNING_DESIGN_001/DESIGN.json`; ingen modellforwards eller fits.
2. Publiser valgt sampler og faktiske epoch0/first4096/TRAIN256-koordinater gjennom
   eksisterende eiere. CONTROL256 beholdes uendret fra designet, på separat fysisk VAL.
3. Bind eksisterende native recipe/campaign til disse artefaktene og mål fersk
   nullstegsinitialisering. ONLINE og TARGET skal bruke samme nåværende funksjon.
   Ingen gjenbruk av historiske vekter eller initialmålinger med annen funksjon.
4. Kjør én bundet native prøve: batch16, 256 optimizersteg, høyst4096 Entries,
   seed20260911, eksisterende AdamW/scheduler, FP32 og TF32 av. Ingen full epoch
   eller full VAL. Publiser og bevar initial/sluttstate, mål, masker og terminale kvitteringer.
5. Mål Entry og Exit for LONG og SHORT mot fersk initialisering og TRAIN-konstanter,
   med identiske fryste mål og ukeblokk-usikkerhet. Rapportér TRAIN-tilpasning og
   CONTROL256-generaliseringsmåling separat. Ren biasflytting eller alltid-FLAT/HOLD
   er ikke dokumentert tilstandsavhengig læring. Teknisk PASS er ikke lønnsomhet.
6. Bare ved bestått eksisterende læringsport: bind én kontrollert utvidelse med
   begrunnet endelig budsjett fra målt kapasitet og populasjonsdekning. Commit budsjett,
   checkpoint-/stoppregel og evidenskrav før kjøring. Ingen automatisk uavgrenset trening.
   Hvis porten feiler, fullfør diagnosen og dokumenter hvorfor utvidelsen ikke åpnes.

Benchmark-plan/operator og evidens publiseres immutabelt under
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V38_20261001/SAMPLER_BENCHMARK_001`.
Kilde og inputhasher verifiseres før/etter kjøring. Én tung jobb om gangen.
Global `training_enabled=false` beholdes mens forberedelsen kjøres; den eksisterende
native eieren skal få eksplisitt avgrenset autoritet ved hver faktisk måle-/treningsfase.

Fullførte input-, normaliserings-, kostnads-, indeks- og featurekildereviews relanseres ikke.
Ingen features kasseres, ingen tidsrammer fjernes, ingen maksimal holdetid eller fast
tapsgrense innføres. Ingen edge-, økonomi- eller train/serve-påstand før egen måling.
