# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og neste steg

TA_DIRECTION_EDGE_CAUSE_EXPOSURE_RETAINED_REVIEW_ADMITTED. Edge-målet er fortsatt uoppnådd.
Ny eksponeringskontrast er COMPLETE/CONSUMED. ROOT
/home/andre2/GX1_RUNS/MARKET_STATE_TA_COVERAGE_BENCHMARK_20261112_001.
RESULTd5ba8fa9, TERMINALe43be2ee, runtimekvittering75bf0f16. Ren fryst kilde67fa0d2a.
768faktiske updates/0skip,2304forwards,1modell/optimizer/INITIAL-dekoding,0checkpointwrites,
1evidenskapsel og alle6målinger bevart. Felles INITIAL/pass1 er bit-eksakt på begge paneler;
modell, RNG og256treningsposter matcher kontrollen. Alle pass har lik RNG-strøm.
Input-/kilde-/måleporter og faktisk sluttcursor/kvitteringskjede består. Windows-boot646,
controllerexit0 og engangsoppgaven er deaktivert. Ingen tung jobb kjører.

Kandidaten bruker12288unike TRAIN-rader mot4096×3 i bevart FULL, med samme768updates.
Faste4096TRAIN/LATER-paneler, FULL-mål, INITIAL, genuine features,5TF og normalisering beholdes.
De736komplementradene er ikke brukt til nye updates. Radinnhold, repetisjon og rekkefølge endres
sammen; forsøket isolerer ikke unik-radtall alene. Gamle forbrukte planer forblir stengt.

Foreløpige punktestimater: senere M5F Brier0,251796→0,249511; M5S0,250519→0,248787 mot FULL.
Alle10S/F×TF-sideceller får bedre Brier/logloss mot FULL. Senere samlet tap2,470884→2,360385,
men fortsatt dårligere enn INITIAL2,350915 og prior2,316114. M5F er nesten lik INITIAL/prior.
Gamle TRAIN-panelets tap1,789263→1,832358. Dette støtter en eksponeringsforklaring på sterkere
paneltilpasning og dårligere senere sannsynlighetsskår; rangering/usikkerhet gjenstår.

Neste: én admittert gjenbruksvurdering av6lagrede INITIAL/FULL/COVERAGE TRAIN/LATER-captures.
REVIEW_EXPOSURE.py er bundet i policy; capped audit4GiB/512MiB,300sek og2000dag-/ukereplikater.
Metadata/import kontrolleres før CLAIM; bootstrap og skåringsfunksjoner gjenbrukes uendret.
Gjenbruk kvalifisert AUC-/skåringskode og dag-/ukebootstrap for begge oppgaver og5TF. Behold
frossen prior og alltid-LONG. Ingen ny modell, checkpointdekoding, featurelesing eller fitting.
Global training/native_training er false. Modellen er ikke et dokumentert handelsoppsett;
S/F-prognoser beviser ikke Entry-kvalitet, prisbane eller økonomisk edge. TEST forblir forseglet.

## Målt resultat og presise grenser

ROOT /home/andre2/GX1_RUNS/MARKET_STATE_TA_CURRENT_COMPLEMENT_20261112_001.
Samlet bevis: docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json, current_complement_result.
Native RESULTb4ae60a3/TERMINALdae3afbc: FAILED/CONSUMED etter253,78sek,97forwards,0updates.
Retained RESULTbf5d6a51/TERMINALe57f14d1: tostatssammenligning COMPLETE/CONSUMED,12,12sek,
to eksisterende captures,0checkpoint-dekodinger,0modeller/forwards/updates.

736faste komplementrader fra46hele B16-batcher;49UTC-dager/22uker.32andre kontrollrader permodell
er utelatt fra skårene. Dette er originalTRAIN med tidligere læringshistorie, ikke urørt holdout
eller full25478-raderskomplement. Siste6TRAIN-rader er utenfor dette deklarerte utvalget.
Alle genuine features og5TF bevares; sampled-S/pass0 og native-F, sammeCPU1/no_grad for begge.

| Native F | Kjent side | Brier INITIAL → FULL | AUC INITIAL → FULL | Brier-differanse95dag | Brier-differanse95uke |
| --- | ---: | --- | --- | --- | --- |
| M5 | 280 | 0.249783 → 0.250101 | 0.505413 → 0.534416 | [-0.004438, 0.005657] | [-0.004720, 0.005255] |
| M15 | 230 | 0.253980 → 0.282337 | 0.518207 → 0.528889 | [-0.000658, 0.062578] | [0.006930, 0.050816] |
| H1 | 228 | 0.262638 → 0.284652 | 0.451783 → 0.501125 | [-0.012866, 0.061731] | [-0.012680, 0.061730] |
| H4 | 239 | 0.256703 → 0.311271 | 0.444292 → 0.383645 | [0.017131, 0.094742] | [0.020314, 0.092041] |
| D1 | 189 | 0.207414 → 0.171439 | 0.699691 → 0.744599 | [-0.081595, 0.009884] | [-0.080140, 0.009015] |

H4s Brier-forverring er positiv i begge punktvise95intervaller. M5F inkluderer null i begge,
og alle F-AUC-differansene er usikre. M15F-dagintervallet inkluderer null, H1F begge; D1F har
bedre punktestimat men begge intervaller inkluderer null. Ikke påstå at alle horisonter blir verre.
M5S-Brier øker0,005598: dag[-0,000114,0,011795], uke[0,001225,0,010210]. Ikke entydig M5-dom.
M15F-logloss blir verre motINITIAL i begge intervalltyper. StateH4/D1 blir bedre på Brier. Dag/uke-bootstrap2000 er punktvis/utforskende med overlappende mål.
Frossen TRAIN-prior og eksakt alltid-LONG for alle nativeF beholdes separat; dette er ikke PnL.
UNKNOWN og umodne mål holdes eksplisitte. Fknownside280/230/228/239/189; UNKNOWN0/6/28/80/139.

## Den tredje armen og integritet

INITIAL og FULL fullførte48forwards hver med eksaktmodell-/bufferrestore og passive evalueringer.
M5-only ble eksakt lastet, men første kontrollbatch fikk future-raw-avvik2,9087e-5 mot GPU-referansen,
over grensen2e-5. Stoppet før komplementmåling. Ingen toleranse ble lempet og ingen relaunch utført.
De32numeriske kontrollradene gir ingen global CPU/GPU-feilgrense. SammeCPU sammenlignes direkte.
Original native source_unchanged=false beholdes: closing bookend ble ikke nådd etter stoppet.
Retained review verifiserte etterpå alle bundne kilde- og statfil-hasher på nytt uten checkpointdekoding.
En ferdig tostatsreview betyr ikke at den planlagte trearmskjøringen er ferdig.

## Tidligere hovedfunn og arbeidsgrenser

Femhorisontreviewen er COMPLETE/CONSUMED. FULL TRAIN-AUC M15–D1 var0,62–0,71, senere0,47–0,51.
Låste støtte-/EMA-/HTF-oppsett viste ingen avklart merverdi mot alltid-LONG på femhorisontene.
AUC er ikke winrate. Alle tidligere kvitteringer og checkpoints beholdes uendret.

Global training/native_training, full epoch og fullVAL er false. Ingen tung jobb kjører.
Root eier policy/launch. TEST, Entry/Exit-evaluering, broker, handel, spending og promotion er stengt;
GC er på pause. Bare benchmarkorkestrering, kontrakt/ruting og fokuserte tester er endret; numeriske eiere er uendret.
Korte runder kvalifiserer feil og hypoteser; de dømmer ikke endelig kapasitet.
Full kvalifisert TRAIN-historikk og30–40epoker er senere skalering, ikke autorisert nå.
