# GX1: effektivitet og kortere treningstid

Dato:20.11.2026. Gjennomført i CURRENT, branch `work/gx1-current`,
fryst kodebase `1a03fa47`. Hovedplan002 er klargjort; hovedtrening er ikke startet.

354550 TRAIN-issues og34934 utviklings-VAL-issues beholdes. TRAIN er fortsatt
B64, én kronologisk pass og5540optimizerupdates. Kvalifisert B256/equalTF ved
sluttscoring reduserer denne fra6086 til1522forwards; samlet main fra11626 til7062.
10800sekunder er hardgrense, ikke dokumentert ferdigtid.

## Gjennomført og kvalifisert på ekte data

| Tiltak | Målt / bevist |
|---|---|
| Lokal bindingsvalidering | Én fersk validering fremfor to i hvert bindingkall. Samme kontraktberegning, ny mutasjonsnektelse og ingen varig cache. Uavhengig fullmodul-AST-restitusjon består. |
| Grouped finite-forward | TRAIN/EVAL-B64 output/loss/gradient/RNG bitidentiske mot autentisert gammel kontrollflyt. Alle finitevakter ferdigstilles før loss. |
| B256 + equal-length-TF-scoring | 256TRAIN-rader:2,110990→0,534416s, omtrent3,95ganger throughput i dette panelet. Ingen feature-/TF-/presisjonskutt. |
| Reelle optimizersteg og cold resume | Mekanikk002:3updates/7forwards/6autograd; modell, optimizer, RNG, cursor, schedule og stegrapporter matcher original001eksakt. Kald gjenopptakelse i ny prosess er bitidentisk. |
| Målrettede tester | 295uendrede/påvirkede tilfeller +8nye positive/negative porttester består; originale testfunksjoner og52andre kildeeiere er bevist uendret ved append. Obligatoriske pre-commit-kontrakter består. |
| Maskinvarevakter | Diagnostikken holdt peak CUDA-reserved7331643392bytes under12GiB. Signerte47samples: maks50Ccore/48Cmem/179,28W. Mekanikkens67samples:50Ccore/50Cmem/189,58W. Ingen vakt svekket. |
| Telemetri etter restart | Eksisterende portproxyregel manglet aktiv lytter38128. Eksakt etablert delete/add av samme private regel gjenopprettet signert transport. Sertifikat, brannmur og tjenester ble ikke endret. Årsaken til manglende runtime-lytter etter boot er ubevist. |

B256/equalTF er **toleranseparitet**, ikke bitidentiske CUDA-resultater: største
logitavvik1,1920929e-7, hidden2,6226044e-6, innen på forhånd bundet eksisterende
`atol=1e-6, rtol=1e-5`. Tellinger, confusion og ties var eksakte; state/RNG
uendret. Reell korttail består. Dette er panel256+tail2, ikke en full ny
numerisk sammenligning av alle framtidige sluttoutputs. Ingen hovedfit,
generalisering eller økonomisk edge er bevist av hastighetskvalifikasjonen.

## Behold det som allerede sparer arbeid

- Ferdige kausale featurearrays, targets, TRAIN-normalisering, prior og full
  initialbaseline gjenbrukes. Main refitter ikke statistikk. Initialbaseline
  kostet 2745,427 sekunder og 6086 forwards; main laster den ferdige målingen.
- Featurearkivet lastes én gang til RAM, uten featureberegning eller
  fullscan/hash i batchløkken. 27 arrays utgjør 1433216758 råbytes;
  NPZ-arkivet er 563844990 bytes. Basisarrays og avgrenset gather beholdes.
- CUDA-aktiveringscheckpointing er allerede deaktivert for å unngå
  ekstra encoderforwards under backward. Minnegrensen beholdes.
- Gradient-/parameterkontroller aggregeres allerede per enhet med samme
  finite-predikat. `zero_grad(set_to_none=True)` brukes allerede.
- AdamW har `foreach=None, fused=None`; installert PyTorch2.6.0+cu124 velger
  normalt foreach for CUDA. `foreach=True` er ingen dokumentert ny gevinst.
- SDPA-ruten finnes allerede via TransformerEncoderLayer med
  `need_weights=False`. Faktisk attentionbackend er ikke målt i de lagrede
  profilsammendragene.
- Scoring skjer i eval/no_grad, etter én pass. Checkpointintervallet er
  1024 oppdateringer: seks periodiske filer, inklusive siste update, samt
  initial/final. Mekanikkens checkpoint med AdamW er omtrent95MB.
- Eksisterende beståtte kontroller gjenbrukes innen eksakt uendret omfang.
  Ny Session, også etter restart, gjør fortsatt nødvendig fersk fysisk og
  semantisk inputvalidering gjennom eksisterende eier.

## Målt B64-trening med samme matematikk

THROUGHPUT003 brukte den autentiserte originale bindingsmetoden og hovedrutens
allerede aggregerte grad-/parametervakter som referanse. Gammel AST ble
kompilert én gang før timerne. Seks forwards/backwards, ingen optimizerupdates:

| Løp | Veggsekunder |
|---|---:|
| Referanse |1,131552 /1,111187|
| Lokal binding + grouped-forward |1,054665 /1,063698|

Gjennomsnittlig paneltid falt1,121370→1,059182s, omtrent5,5prosent. Output,
loss, klippet gradient og RNG er bitidentiske. Profilert løp på2,629148s brukes
ikke i hastighetsberegningen. Dette er ikke målt AdamW- eller fullfitgevinst.

Ny mekanikk002, på samme B64TRAIN-oppskrift og kilde, kvalifiserte deretter
faktiske updates og kald resume samt eksakt gammel modell-/optimizer-/RNG-bane.
Gammel fullinitial på6086forwards gjenbrukes gjennom eksplisitt kildebro,
uendrede modellberegninger og samme ferske initialstate. Kvalifiseringsvekter
brukes ikke til main. Konsumerte planer relanseres ikke.

## Øvrige kandidater og minste nødvendige bevis

| Prioritet | Tiltak | Status, konkret grense og kvalifikasjon |
|---|---|---|
| Utført | Lokal gjenbruk av bindingskontroll | Innført og kvalifisert. Fersk deep-copy valideres én gang per kall, fortsatt likhet kontrolleres før lokal gjenbruk. Alle mutable buffer-/normaliserings-/head-/prior-/checkpointkontroller beholdes. Ingen vedvarende cache. |
| Utført | B256 ved sluttscoring | Valgt etter genuine output-/hidden-/score-/state-/RNG-paritet og målt VRAM/tid. B64TRAIN/5540updates beholdes. Hele TRAIN/VAL-raddekningen og ukjente haler beholdes. |
| Utført | Equal-length MTF ved scoring | Kvalifisert bare i eval/no_grad sammen med B256. H1/H4s96barer samles langs batchaksen, og splittes tilbake uten padding eller featuretap. TRAIN/ONLINE-default uendret. |
| 3 | Fused AdamW | Umålt. Profilér optimizer separat først. Krever eksplisitt recipe, faktisk update-/optimizerstate-/cold-resume-kvalifikasjon og kontroll av peakminne. Ikke forveksle dette med foreach som allerede kan være aktivt. |
| 3 | Redusere små kernelkall / avgrenset kompilering | Én instrumentert B64-profil registrerte25961kernel-launches. Undersøk bare målte dominerende tensorblokker. Inkluder kompileringstid, graph breaks og minne; bevis output/loss/gradient/RNG/resume. Ingen ny rammeverks- eller helmodellombygging uten behov. |
| 4 | Samle D2H-evidenskopier | Undersøk samlede kopier av små logits/counts fremfor mange synkroniseringer. Bevar samtlige verdier, hasher, finiteporter og kontrollrekkefølge. Mål kostnad før endring. |
| 4 | Pinned buffere, avgrenset prefetch og asynkron H2D | Nåværende overføring er synkron. Profilert B64: gather13,14ms, H2D6,89ms, forward820,67ms og backward616,76ms. Dataoverføring er ikke påvist hovedflaskehals i dette panelet. Krever korrekt strøm-/bufferlevetid, samme batch/RNG og målt RSS/VRAM innen vaktene. |
| 4 | NPY/memmap, mindre kopiering og flere workers | NPZ lastes bare én gang. Vurder format eller workers først ved målt oppstarts-/RSS-/gatherproblem. Flere workers/tråder kan øke kopiering og minnebruk. Behold felt/dtype/akse/hash og kausalitet. |
| Betinget | Større treningsbatch | B64 er kvalifisert. Mekanikk peak reserved var6,912GB under12GiB-grensen; dette beviser ikke at B128 passer. Ny batch endrer antall updates og læring, så krever nytt konkret læringsspørsmål, recipe/schedule og mekanikk/resume. Gradient accumulation fjerner ikke forwardarbeidet. |
| Betinget | BF16/FP16/TF32 | Faktisk fit-owner bruker deterministisk FP32 med TF32false. Native hidden og logits/loss krever FP32. Autocast krever eksplisitt avgrensning og FP32-head/loss; FP16 må også kvalifisere skalering. Bevis numerikk, masker, gradienter, optimizer/resume og TRAIN/serve-paritet. Endret ONLINE-funksjon krever fersk initialbaseline. Ingen målt gevinst her. |
| Betinget | Attentionbackend eller annen kernel | SDPA finnes allerede. Mål faktisk backend og tid før endring. Behold determinisme, dropout, posisjonering, masker og gradienter; ny kernel må kvalifiseres numerisk. |
| Betinget | Rå TF-indeks-/vindusgjenbruk | Kun uendret råinformasjon før modellavhengige gates kan være aktuell. Bevar causal asof, felt, skalaer, eier-ID-er og vindusgrenser. Ingen sampling eller featurefjerning. |
| Betinget | Asynkrone checkpoints | Bare ved målt checkpointflaskehals. Krever frossen og komplett snapshot av modell, optimizer, cursor og RNG, samt atomisk publisering/fsync og strict readback. Levende mutable `state_dict` kan ikke skrives parallelt. Dagens sparsomme sikre rute beholdes inntil da. |

Profilert vegg-/CUDA-eventtid omfatter instrumentering og eventuelt CPU-submit-
og køgap; det er ikke ren GPU-kerneltid. Nested kumulative tider skal ikke
summeres. Throughput001/002 hadde ingen AdamW-oppdateringer. Ingen robust
fullfit-speedup eller ETA er dokumentert.

## Tiltak som ikke bør brukes som snarvei

- Ikke materialiser alle overlappende vinduer. B64-input er omtrent42,27MB;
  alle389484vinduer ville utgjøre omtrent257GB. Behold basisarrays og gather.
- Ikke cache trente TF-embeddings/hidden mellom M5-eksempler eller updates.
  Gates avhenger av M5-representasjonen; vekter og dropout/RNG endres. Lik rå
  H4-/D1-historikk er ikke bevis på likt modellresultat.
- Ikke erstatt FP64-mellomregningen før `asinh` med FP32 uten domenebevis.
  Den beskytter hele finite FP32-inputdomene ved svært små TRAIN-skalaer;
  resultatet går allerede tilbake til FP32.
- Ikke bruk helsløyfe-CUDA-graphs som bryter. Nåværende `.item()`/CPU-vakter,
  datastyrte feilveier, allokeringer og AdamW `capturable=False` krever en
  særskilt kvalifisert avgrensning. Vakter fjernes aldri for å få capture.
- Ikke anta at precisionbrytere i gammel `entry_v10_ctx_train_v3` styrer
  denne main. Aktiv eier er auditens `fit_step`/`fresh_fit_model`.
- Ikke reduser genuine features, åtte familier, fem TF, kausalitet,
  målmodning eller økonomisk/evalueringsmessig dekning for å pynte throughput.
- Ikke innfør persistent hashcache på tvers av Session/restart, svekk
  checkpoint/readback eller les forseglet TEST som ytelseskontroll.

## Runtime og maskinvare

Brukerbestilt restart gjøres én gang ved trygg jobbgrense. Restart har ingen
målt ytelsesgevinst her. Bekreft ny Windows-boot, WSL/tilkobling, uendret kilde
og intakte artefakter gjennom eksisterende vakter. Unngå rutineomstarter mellom
kvalifiserte delsteg. Kontroller signert telemetritransport etter ny boot;
kjørende Windows-oppgave alene beviser ikke aktiv WSL-portlytter. VHD-komprimering er et separat lagringstiltak, ikke en
dokumentert treningsoptimalisering.

Behold én tung CURRENT-jobb, capped-wrapper,20G RAM,512M swap,128tasks,
8numeriske tråder,12GiB GPU-reservasjonsgrense og eksisterende CPU-affinitet,
temperatur-/effekt-/minnevakter. Ikke overklokk, tving maksimal effekt eller
opphev grenser. Mål faktisk utnyttelse, clocks og eventuell throttling under
den reelle jobben før maskinvaretiltak vurderes. Unngå konkurrerende tunge
prosesser. Ingen nye kjøp, betalte tjenester eller driver-/bibliotekoppgradering
begrunnes av umålt antatt gevinst.

Lang stabil kjøring kontrolleres etter prosjektets avtalte intervall,
omtrent15–30minutter eller sjeldnere ved stabil drift. Ingen minuttvis
modellpolling, ekstra fits eller arbeid bare for å fylle ventetid.

## Gjenværende tiltak krever ny målt begrunnelse

Bruk eksisterende profil-/trenereier til å skille oppstart/inputvalidering,
modellkonstruksjon, gather/H2D, forward, backward, optimizer, evidenskopier,
scoring og checkpoint. Registrer peak RSS/VRAM, faktisk disklesing mot logisk
lesing og eventuelle guard-/throttlinghendelser. Profiler bare et avgrenset
representativt panel under ny eksakt admisjon; de konsumerte001/002 relanseres
ikke. Velg deretter ett tiltak etter målt dominerende kostnad. Aktuelle målte rettelser
er gjennomført; øvrige AMP/TF32/fused/compile/graphs/I/O-tiltak er ikke aktivert
uten nødvendig matematikk-/lærings-/ressursbevis.

Endringer begrenses til konkret målt flaskehals. Samme funksjon må bevises;
endret ONLINE-funksjon krever fersk initialbaseline. Teknisk throughput,
TRAIN-læring, senere generalisering og økonomisk edge er separate spørsmål.

## Kilde- og målegrunnlag

- `gx1/audit/qualify_native_m5_full_gc_direction_v1.py`: `native_fit_forward`,
  `native_fit_score_forward`, `fit_step`, `fit_score`, `throughput_*`,
  `qualified_runtime_throughput` og `runtime_mechanics_trajectory`.
- `gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py`: finite-scope,
  aktiveringspolicy, normaliseringsbinding, FP64/asinh og eksisterende evalsamling.
- `gx1/contracts/market_state_ta_integrated_model_v1.py`: lokal bindingsvalidering;
  `market_state_ta_integrated_objective_v1.py`: FP32-output/loss og float64-score;
  `market_state_native_m5_feature_inputs_v1.py`: gather/kausalitet;
  `gx1/audit/build_native_m5_ta_feature_cache_v1.py`: engangsloading.
- `RESULT.json`/`TERMINAL.json` i `NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_003`
  og `NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_002` under GX1_RUNS.
- Fullinitial: `NATIVE_M5_TA_ATR03_FIVE_YEAR_PREPARE_20261120_001`.
- Source-/unit-/telemetri-/sluttbevis:
  `GX1_FULL_PROJECT_AUDIT_20261120_001/EFFICIENCY_QUALIFICATION_001`.
  Eksakte aktuelle SHA-bindinger finnes i NEXT_RUN_POLICY.json.
- Lokal installert PyTorch2.6.0-kilde kontrollerte foreach/SDPA-rutingen.
  [AdamW2.6](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/adamw.py)
  og [CUDA graphs](https://docs.pytorch.org/docs/stable/notes/cuda) er
  primærkildene for de oppgitte umålte bibliotekalternativene.
