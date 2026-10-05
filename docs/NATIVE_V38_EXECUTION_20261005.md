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

## Observert CPU-blokkering og avgrenset diagnose

ATTEMPT_002 er terminalt stoppet, ikke fullført eller selektert. Første kandidat
hadde brukt1800,23 sekunder på1104/8192 Entries; dermed kan denne kandidaten
ikke bestå den eksisterende30-minuttersgrensen. Andre kandidater er uundersøkt.
Den observerte timeren inkluderer tracemalloc. Ikke utled native kapasitet,
featureverdi eller lønnsomhet fra denne ufullstendige instrumenterte målingen.

CPU_WORKLOAD_PROFILE_001 er nødvendig diagnose innen dette gjennomføringsmålet:
ett uendret TRAIN16-batchutvalg per eksisterende kandidat, tre faste pass
(uinstrumentert CPU-tid, cProfile, opprinnelig tracemalloc-benchmark), høyst144
Entry-materialiseringer. Eksisterende benchmark-/data-/kollasjonseiere brukes.
Alle batchbytes må være like; paritetshashene beregnes etter det målte intervallet.
Den begrensede kvitteringen er aldri samplervalg eller fullbudsjettbevis.
Kjør capped producer20G/512M med fryst kilde, nettverk og TEST stengt,
null modellforwards, fits og optimizersteg. PLAN_002 og operatorhasher står
i NEXT_RUN_POLICY.json. Originale logger og begge uferdige oppstarter bevares.
Diagnostiser først; verifiser minste rettelse før ny full benchmark bindes.
Ingen nye kandidater, mål-/featureendringer eller flytting av godkjenningsgrenser.

## Målt CPU-flaskehals og avgrenset paritetskontroll

CPU_WORKLOAD_PROFILE_001 fullførte tre pass per kandidat, men sluttlagring
feilet fordi heltallsnøkler ikke er objektlike etter JSON strict-load.
Original rød terminal beholdes; logger, staging og pstats er kontrollert uten
ny tung kjøring i den hash-bundne PUBLICATION_FAILURE_REVIEW i NEXT_RUN_POLICY.

Målt på genuine TRAIN16: 12,25–13,01 sekunder uten instrumentering og
25,81–28,13 med tracemalloc (1,98–2,27x). Første profilerte batch utførte
18884 små økonomiprojeksjoner og 113304 array-hasher. Ikke fullbudsjettkapasitet,
samplervalg, modellkvalitet eller økonomibevis.

Minste rettelse utvider bare eksisterende data-/state-view-eiere: beregn den
samme observerte referanseøkonomien samlet per side, avled samme forseglede
og validerte stegutsnitt, og kod identiske kanoniske skalare JSON-bytes uten
JSONEncoder-objekt per skalar. Ingen validering, hash, mål, maske, familie eller
tidsramme fjernes. 51 fokuserte syntetiske kontrakttester består; ekte bytes
og ytelse er fortsatt ubevist etter rettelsen.

CPU_WORKLOAD_PROFILE_002 forhåndsregistrerer den nødvendige kontrollen:
samme tre kandidatbudsjett og gamle TRAIN16-rader; tre faste pass, høyst144
materialiseringer; alle batchhasher må være eksakt like de bevarte målingene.
Ingen forwards, fits, optimizersteg eller samplervalg. Kjør capped producer
20G/512M, ren/fryst kilde, nettverk og TEST stengt. Plan/operator står i
NEXT_RUN_POLICY.json. JSON-nøkler er strengkodet før strict-load. Et ustartet
planutkast som brukte rikere kildebindinger enn kontrollen er bevart; PLAN_002
binder samme fullstendige kildeclosure med eksakte path/sha256-felt.

Vurder ekte tidskostnad etter paritetskontrollen før ny full benchmark. Alle
opprinnelige kandidat-/tids-/minnegrenser består. Hvis kapasiteten fortsatt
ikke passer, dokumentér dette; ikke flytt en grense eller kasser signaler.

CPU_WORKLOAD_PROFILE_002 stoppet før første kandidatmåling. Den første
state-view-rettelsen kolliderte med den eksisterende inputautoritetens bundne
kildehash; gaten feilet korrekt. Feil/terminal beholdes i NEXT_RUN_POLICY.
Ingen genuine ytelses- eller paritetskonklusjon er målt etter rettelsen.

Minste videre rettelse beholder state-view-kilden byteidentisk
(sha2564315ac48e860cba4c83e246e5688564342e97da4a89376f015a5b1c552104436),
bekreftet mot FINAL_BINDINGS_BUNDLE.json. Adapteren deklarerer det eksisterende
referansevinduet for økonomileverandøren. Leverandøren beregner det lazy én gang
per side, etter original state-view sine cutoff-/klokkekontroller, og emitter
de samme forseglede slices. Parent og hver child valideres fortsatt av original
projeksjonseier. Vinduet og høyst to sideprojeksjoner nullstilles per view.
Ingen ny fit, indeksbygging, bindingsoverstyring eller svekket gate.

CPU_WORKLOAD_PROFILE_003 har samme avgrensede oppgave og høyst144 besøk som
002, men binder denne minimale plasseringen av rettelsen. Samme originale
TRAIN16-rader/batchhasher er obligatoriske. Plan/operator står i NEXT_RUN_POLICY.
51 fokuserte syntetiske tester består, inklusive alle120×2 byteidentiske
cacheutsnitt, lazy beregning, avvisning utenfor eksplisitt vindu og reset til
original usamplet projeksjonsvei. Ekte batchparitet og kapasitet gjenstår.

## Terminal CPU-paritet og gjenstående budsjettvalg — 06.10 lokal tid

CPU_WORKLOAD_PROFILE_003 fullførte med exit0, uendret kilde9ca9153c og alle
originale genuine batchhasher like i alle tre pass. Resultat/terminal og
CAPACITY_PLANNING_REVIEW er hash-bundet i NEXT_RUN_POLICY. Ingen relansering.

Målt på én TRAIN16-batch per kandidat: native7,169/6,995/6,526 sekunder og
instrumentert13,893/15,533/15,758 sekunder for32768/65536/131072 overganger.
Native forbedring1,81–1,88x; padded inputs79124544 bytes uendret. Null fits,
forwards, optimizersteg og TEST. Dette er ikke målt full kapasitet eller læring.

Lineær planleggingsfremskrivning gir instrumentert1,98/4,42/8,96 timer per
full kandidat og15,36 timer samlet uten oppstart. n=16 per kandidat, ingen
repetisjon/konfidensgrense; geometri/cache/belastning kan variere. Ingen kandidat
er erklært eligible/ineligible etter denne rettelsen. Gjeldende eier har fortsatt
1800-sekunders eligibilitycap; ingen ny full plan eller sampler er autorisert.

Før videre utføring må operatøren velge CPU-budsjett. Forslag er3 timer CPU
per samplerepoch og høyst18 timers total hard wall for én full benchmark,
med uendrede kandidater, batch16, én full repetisjon, mål, felt, kvalitetsporter
og maskinvarevakter. Forslaget er IKKE godkjent/implementert og kan ikke gjøre
småbatch- eller partiale bevis grønne. Alternativet er å beholde30min og
avklare videre CPU/designarbeid. Ingen stille grenseflytting eller feltkassering.

Målet punkt1–6 er fortsatt aktivt og ufullført: full benchmark, sampler,
koordinater, fersk native initialmåling,256-stegs prøve og læringsreview gjenstår.
Den betingede treningsutvidelsen er ikke åpnet. TEST, broker, handel og spending
forblir stengt. Ingen tung/native jobb kjører ved sluttkontrollen.
