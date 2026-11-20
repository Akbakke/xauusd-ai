# GX1: effektivitet og kortere treningstid

Dato: 20.11.2026. Lesende vurdering av CURRENT, branch `work/gx1-current`,
kildegrunnlag `f1446337`. Dette er forslag til prioritering, ikke utførte
ytelsesendringer, modellkvalifikasjoner eller ny treningsadmisjon.

Den eksisterende ene femårspassen er klargjort: 354550 TRAIN-issues,
34934 utviklings-VAL-issues, B64, 5540 optimizeroppdateringer og 11626
modellforwards. Av disse er 6086 sluttmåling på hele TRAIN og utviklings-VAL.
10800 sekunder er hard kjøretidsgrense, ikke bevist ferdigtid.

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

## Målt første kandidat: samle finite-kontroller i forward

Eksisterende `model_finite_check_scope('grouped')` brukes i diagnostikken,
men ikke rundt modellkallet i main eller sluttscoring. Kontroller beholdes;
detached predikater samles og ferdigstilles før scope returnerer output.

THROUGHPUT002 målte samme B64-panel uten optimizeroppdateringer:

| Løp | Veggsekunder |
|---|---:|
| Referanse | 1,2806 / 1,2758 |
| Grouped | 1,1399 / 1,1544 |

Output, loss, klippet gradient og RNG var byteidentiske. Vektene var uendret.
Profilerte synkroniseringer falt fra1537 til208 og `item` fra1435 til106
sammenlignet med THROUGHPUT001. Diagnostikken endret både forwardkontroller
og gradient-/parameteraggregasjon; main har allerede sistnevnte. Hele
tidsforskjellen kan derfor ikke tilskrives ubrukt forward-scope. Dette er
ikke en målt AdamW- eller fullfitgevinst.

**Prioritet1, foreslått kvalifikasjon:** Gjenbruk eksisterende scope i den
aktive eieren; behold FP32, batchrekkefølge, dropout, RNG og alle vakter.
Kvalifiser identisk output/loss/gradient/RNG, første ikke-finite feilmelding,
nøstede scopes og feil før output når loss. Bevis deretter faktiske native
updates, optimizer/state og kald resume. Integrator må binde eventuell ny
kilde og plan; eksisterende konsumerte diagnostikkplaner relanseres ikke.
Tiltaket er ikke gjennomført i denne bølgen.

## Øvrige kandidater og minste nødvendige bevis

| Prioritet | Tiltak | Status, konkret grense og kvalifikasjon |
|---|---|---|
| 2 | Samordne gjentatt bindingskontroll | `_require_binding` kostet0,387s kumulativt i én profilert B64-forward, inklusive nestede normaliserings-/inputkontroller. Isoler kostnaden; undersøk samordning av identiske kontroller innen samme forward. Behold kontroll av mutable buffere og alle kontraktsbindinger. Ingen persistent modell-/state-/checkpointcache. Krever feilinjeksjon og dynamisk paritet. |
| 2 | Større scorebatch | B128/256 kan undersøkes separat for de6086 sluttmåleforwardene, med treningsbatch og5540updates beholdt. Krever eksakt raddekning, output-/score-/state-/RNG-paritet, målt VRAM og eksplisitt ny planbinding. Ingen målt gevinst. |
| 3 | Samle like lange MTF-encodercall ved scoring | Eksisterende `batch_equal_length_mtf_eval_scope()` er avgrenset til eval/no_grad. H1/H4 har begge96barer. Hver sekvens beholder egne normaliserte/gatede inputs. Krever output-/scoreparitet og peakminnemåling. Ikke slå dette på i trening uten egen dropout-/RNG-/gradientkvalifikasjon. |
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
kvalifiserte delsteg. VHD-komprimering er et separat lagringstiltak, ikke en
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

## Måling før neste endring

Bruk eksisterende profil-/trenereier til å skille oppstart/inputvalidering,
modellkonstruksjon, gather/H2D, forward, backward, optimizer, evidenskopier,
scoring og checkpoint. Registrer peak RSS/VRAM, faktisk disklesing mot logisk
lesing og eventuelle guard-/throttlinghendelser. Profiler bare et avgrenset
representativt panel under ny eksakt admisjon; de konsumerte001/002 relanseres
ikke. Velg deretter ett tiltak etter målt dominerende kostnad.

Endringer begrenses til konkret målt flaskehals. Samme funksjon må bevises;
endret ONLINE-funksjon krever fersk initialbaseline. Teknisk throughput,
TRAIN-læring, senere generalisering og økonomisk edge er separate spørsmål.

## Kilde- og målegrunnlag

- `gx1/audit/qualify_native_m5_full_gc_direction_v1.py`:1185–1210
  (presisjon/optimizer),1328–1370(fit),1373–1421(score/checkpoint),1481–1615
  (hovedløype),2786–2842(profil),3060–3095(bundet recipe).
- `gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py`:140–245
  (finite-scope),303–358(aktiveringspolicy),1910(normaliseringsbinding),
  2198–2217(FP64/asinh),2635–2750(evalsamling).
- `gx1/contracts/market_state_ta_integrated_model_v1.py`:327–428;
  `market_state_ta_integrated_objective_v1.py`:178;
  `market_state_native_m5_feature_inputs_v1.py`:300,397;
  `gx1/audit/build_native_m5_ta_feature_cache_v1.py`:286.
- Eksisterende `RESULT.json` i `NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_001`,
  `..._002`, `NATIVE_M5_TA_ATR03_FIVE_YEAR_PREPARE_20261120_001` og
  `NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_001` under GX1_RUNS.
- Lokal installert PyTorch2.6.0-kilde kontrollerte foreach/SDPA-rutingen.
  Offisiell versjonskilde: [AdamW2.6](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/adamw.py).
  [CUDA graph-begrensninger](https://docs.pytorch.org/docs/stable/notes/cuda)
  forklarer hvorfor synkronisering og dynamisk kontroll krever avgrensning.
