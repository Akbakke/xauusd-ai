# Gjeldende status - 06.10.2026

Operatøren svarte «Ja jeg godkjenner» på spørsmålet om 3 timers
CPU-eligibilitetsgrense per samplerepoch og maksimalt 18 timer samlet for
én full benchmark. Budsjettblokkeringen er løst; punkt 1–6 er ufullført og
arbeidet gjenopptas. Tidligere blokkeringsaudit bevares som historisk bevis,
ikke som gjeldende stoppordre.

Én fersk SAMPLER_BENCHMARK_001/ATTEMPT_003 bindes nå gjennom eksisterende
benchmarkeier. Bare CPU-eligibilitet endres: 1800 til 10800 sekunder.
Alle tre kandidater (32768/65536/131072 overganger, 8192/16384/32768 Entries)
skal måles fullt én gang på batch16, med original tracemalloc-måling.
3-timersgrensen brukes ved utfallsblind rangering etter full måling,
ikke til tidlig stopp eller avkorting. En supervisor kjører én måleprosess
i samme verifiserte capped producer20G/512M-scope og dreper/venter på denne
ved totalfrist 64800s, også hvis den sitter i C-kode. Feil/terminal bevares.
Python-allokasjon2GiB, padded-input1GiB, CPU0–7, én numerisk tråd og TasksMax64
er uendret. Ingen modellforwards, fits, optimizersteg, TEST eller nettverk.

Plan/operator, eksakte input-/kildehasher og godkjenningsomfang står i
NEXT_RUN_POLICY.json og CURRENT_RESTART_POINT.json. Kode/dokumentasjon
testes, committes og pushes før den ene tunge kjøringen starter. Fryst
kilde endres ikke under kjøring; claim hindrer relansering. Ved overtakelse
må faktisk prosess/lås og nyeste terminale bevis kontrolleres før ny launch.
På dette forhåndsregistreringsstadiet er benchmarken ennå ikke fullført.

Fullførte forutsetninger gjenbrukes:
- CPU_WORKLOAD_PROFILE_003: exit0 og uendret kilde, alle originale genuine
  TRAIN16-batchhasher like. Uinstrumentert6,53–7,17s mot12,25–13,01s før
  rettelsen (1,81–1,88x); instrumentert13,89–15,76s. Dette er batchparitet,
  ikke full kapasitet, samplervalg, læring eller økonomi.
- CAPACITY_PLANNING_REVIEW: lineært småbatchanslag2,0/4,4/9,0 timer per
  kandidat, samlet15,4 timer uten oppstart. n=16 per kandidat, ingen
  konfidensgrense; faktisk fullmåling kan avvike.
- VAL_SEQUENCE_AUDIT_001: exit0, uendret kilde7056bf56, alle70880
  time/seq/snap-rader verifisert mot bundet M5, Seq96×254, capped audit4G.
  Ingen utfall eller modellevaluering. TRAIN- og VAL-audit relanseres aldri.
- Komponentutkastets døde ATTEMPT_002-samplersti er rettet. prepare() tar
  eksakt selected_sampler fra coordinate-COMPLETE og kaller eksisterende
  komponent-/samplereier før videre klargjøring. 10 syntetiske bindingstester
  består; ekte semantisk sampleradgang gjenstår. Metadata er uendret.

Den minimale CPU-rettelsen gjenbruker ett eksplisitt referansevindu per
view, lazy etter original cutoff-/klokkekontroll. Parent og hver child
valideres fortsatt; original state-view-kilde er byteidentisk til
inputautoriteten. Alle254 felt, åtte familier og tidsrammer beholdes.
Ingen normalisering, økonomisk indeks, mål eller ferdige data bygges om.

Tidligere feil og partiale forsøk bevares og relanseres ikke:
- Første benchmarkoppstart: feil kostnadsfilrolle, stoppet før måling.
- ATTEMPT_002:1800,23s ved1104/8192 Entries, stoppet uten samplervalg.
- CPU_WORKLOAD_PROFILE_001: målt, men JSON-sluttpublisering feilet;
  original rød terminal og verifiserte stagingbytes består.
- CPU_WORKLOAD_PROFILE_002: inputbundet state-view-kildehash avviste
  første rettelse før måling; ingen binding eller gate ble omgått.

Ingen genuine sampler, koordinater, fersk native initialmåling eller
256-stegs prøve er ferdig. Konstruktørmetadata er ikke modellmåling.
native_component_preparation_authorized=false; hver native fase trenger
sin eksakte bundne autoritet. Først full benchmark, så valgt sampler og
fryste koordinater, fersk nullbaseline, én256-stegs prøve og Entry/Exit-
review. Større endelig trening er betinget av den ekte læringsporten.

Authority er /home/andre2/src/GX1_CURRENT på work/gx1-current.
Les CURRENT_RESTART_POINT.json og docs/RESTART_POINT_20261005.md.
Global training_enabled=false; TEST, broker, live/paper, ordre og spending
er stengt. Læring, generalisering, positiv økonomi og faktisk train/serve-
paritet er fortsatt ubevist. CPU-godkjenningen er ingen kvalitetsgodkjenning.
