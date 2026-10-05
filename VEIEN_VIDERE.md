# Veien videre - oppdatert 06.10.2026

Punkt 1–6 er nå autorisert for faktisk gjennomføring i
docs/NATIVE_V38_EXECUTION_20261005.md. Benchmarken er forhåndsregistrert;
følg den bundne planen i NEXT_RUN_POLICY.json før den øvrige kjeden.
ATTEMPT_002 er terminalt stoppet uten samplervalg etter bevist overskridelse
av første kandidats tidsgrense. CPU_WORKLOAD_PROFILE_001s målinger er bevart
og gjennomgått etter en sluttpubliseringsfeil; original rød terminal består.
CPU_WORKLOAD_PROFILE_002 er bevart som feilet før måling: inputbundet kildehash
ble korrekt avvist. Original state-view kilde er gjenopprettet byteidentisk;
økonomileverandøren gjenbruker det eksplisitte referansevinduet i stedet.
CPU_WORKLOAD_PROFILE_003 er fullført med ekte batchparitet og exit0; ikke relanser.
Målt CPU-tid er rundt1,8x bedre, med alle felt/mål/masker og originale hasher.
Full kapasitet/samplervalg er ubevist. Avklar operatørens CPU-budsjettvalg før
en ny full benchmark: gjeldende30min/kandidat er uendret; småbatchfremskrivning
gir rundt15,4t samlet, ikke målt full kjøretid. Forslag3t/kandidat og maksimalt18t
for én full benchmark er ikke godkjent eller implementert. Behold alle øvrige
grenser. VAL_SEQUENCE_AUDIT_001 er nå fullført med exit0 og uendret kilde:
alle70880 time/seq/snap-rader er verifisert mot bundet M5-kilde, Seq96×254,
capped audit4G/512M. Ingen utfall, fits/forwards/optimizersteg/TEST/nettverk
eller VAL-modellevaluering. Kontraktverifisert audit/resultat/terminal er
hash-bundet i NEXT_RUN_POLICY.json; autoriteten er konsumert. Gjenbruk dette
og TRAIN-auditen; ikke relanser. CPU-grensen og trening er fortsatt stengt.
Deretter koordinater/fersk nullbaseline/fixed256 når faktiske porter består.

Komponentutkastets feilaktige ATTEMPT_002-samplersti er fjernet. Fremtidig
prepare() må bruke eksakt binding fra coordinate-COMPLETE og eksisterende
samplereier før videre arbeid. 10 syntetiske koblings-/filtester består;
ingen genuine koordinater, klargjøringsplan, initialisering eller måling
er kjørt. Uendret metadata gjenbrukes. Autoriteten er fortsatt false.

1. Kjør obligatorisk read-only handover og kontroller branch, HEAD, status,
   prosesser, låser og terminalkvitteringer.
2. Ikke relanser INDEX_FEATURE_SOURCE_REVIEW_001; den er fullført.
3. Les eksisterende benchmarkeier og SAMPLER_BENCHMARK_CANDIDATES.json.
4. Forhåndsregistrer én workload-matchet samplerbenchmark mot faktisk
   indeksbundet 254-felts TRAIN-kilde.
5. Kjør den bare gjennom capped producer med uendret kilde og etablerte vakter.
6. Velg sampler fra resultatet og publiser immutable epoch0-, first4096-,
   TRAIN256- og CONTROL256-koordinater.
7. Kjør fersk nullstegs initialmåling for eksakt ONLINE/TARGET-funksjon.
8. Åpne høyst én avgrenset native v38-læringssammenligning hvis porten tillater det.
9. Mål senere kronologisk generalisering, full økonomi og paritet.
10. Kvalifiser operativ restart og brokeravstemming offline før TEST eller handel.

Eksakte stier, hasher, fremdrift og sperrer står i
docs/RESTART_POINT_20261005.md og CURRENT_RESTART_POINT.json.
training_enabled=false. TEST, live/paper, broker, ordre og spending er stengt.
