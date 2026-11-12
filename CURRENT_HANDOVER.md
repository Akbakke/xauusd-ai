# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og neste steg

TA_DIRECTION_EFFECTIVE_SIDE_WEIGHT_MEASURED. Edge-målet er fortsatt uoppnådd.
Eksponeringskontrast og retained review er COMPLETE/CONSUMED. ROOT
/home/andre2/GX1_RUNS/MARKET_STATE_TA_COVERAGE_BENCHMARK_20261112_001:
RESULTd5ba8fa9, TERMINALe43be2ee, runtimekvittering75bf0f16, EXPOSURE_COMPARISON3e25678a.
Native kilde67fa0d2a; retained kilde1a735c2a.768faktiske updates/0skip,2304forwards,
1modell/optimizer/INITIAL-dekoding,0checkpointwrites,1evidenskapsel og alle6målinger bevart.
INITIAL/pass1 modell/RNG/256poster og begge panelarrays matcher kontrollen bit-eksakt.
Alle sluttporter og runtimekvitteringer består; source_unchanged=true. Windows-boot646,
controllerexit0, engangsoppgaven er deaktivert, ingen tung jobb kjører.

Kandidaten bruker12288unike TRAIN-rader mot4096×3 i bevart FULL, med samme768updates.
Faste4096TRAIN/LATER-paneler, FULL-mål, INITIAL, genuine features,5TF og normalisering beholdes.
De736komplementradene er utenfor nye updates. Innhold, repetisjon og rekkefølge endres sammen;
ikke tilskriv hele effekten unik-radtall alene. Ingen gamle engangsplaner gjenåpnes.

| Senere native F | FULL AUC → coverage | FULL Brier → coverage | Frossen prior Brier |
| --- | --- | --- | --- |
| M5 |0,513672 →0,528606|0,251796 →0,249511|0,249696|
| M15|0,509511 →0,521081|0,307638 →0,279244|0,250709|
| H1 |0,495398 →0,513097|0,316309 →0,280750|0,251352|
| H4 |0,499505 →0,520348|0,309307 →0,270971|0,249256|
| D1 |0,470691 →0,488312|0,303766 →0,263401|0,252789|

Alle5F-horisonter har lavere Brier/logloss enn FULL i begge punktvise95dag-/ukeintervaller.
M5F AUC-økningen motFULL inkluderer null; motINITIAL0,511290 er ΔAUC0,017316,
dag[0,004153;0,031747], uke[0,004639;0,029871]. M5Brier motINITIAL/prior er usikker.
M15/H1/H4F er fortsatt dårligere ennprior i begge Brier-intervaller. Ingen robust edge er etablert.
Alle10S/F×TF får bedre punkt-Brier/logloss motFULL; ikke påstå at alle har avklarte intervaller.
M5S AUC0,551444, Brier0,248787; Brierforbedringen motFULL inkluderer null i begge intervaller.

Metadataregnskap uten nye capturelesinger: M15/H1F anslår meanPup0,615858/0,624343 mens
kjent-side-UP-frekvensen er0,491060/0,500000. Feil gjennomsnittsnivå forklarer51,8%/48,0%
av punkt-Brier-underskuddet motprior. Resten er Var(p)−2Cov(p,y), ikke separat identifiserte
varians-/kovariansledd. Denne nivåfeilen ble større motFULL selv om Brier ble bedre.
Ingen kalibrering ble tilpasset. På TRAIN-panelet er H1coverage meanPup0,573676 mot UP0,583275;
modellen øker altså UP-prognosen på LATER samtidig som den observerte UP-andelen faller.

Retained review:6captures/0checkpoint/0modell/0forward/0update,7,98sek. Uendrede kvalifiserte
scorere/2000parrede dag-/ukereplikater;70dager/14uker. Gamle M5 FULL-minus-INITIAL-intervaller
replisert; største native skårreplayfeil1,11e-15. Alle5TF/S/F, prior og alltid-LONG er beholdt.
Dette er gjenbrukt utviklingsdata, punktvise utforskende intervaller og overlappende mål.
Kjent-side-støtte kommer fra realiserte labels; ingen selekterbare Entry-handler eller PnL bevises.

Effektiv labelvekting er nå målt, COMPLETE/CONSUMED:
/home/andre2/GX1_RUNS/MARKET_STATE_TA_EFFECTIVE_SIDE_WEIGHT_20261112_001,
RESULT89c0a1f5, kilde1f981f28.2,61sekunder,9arkivdekodinger,
0features/checkpoints/modeller/forwards/updates. Alle10bevarte TRAIN-paneltall matcher;
alle5native-F-priortall og S/M5-profilen matcher. Uendrede bindere, causalitet og5TF.

| Coverage native F | Radandel UP | Effektivt konstant tapsmål | Senere modell-Pup | Senere UP-fasit |
| --- | ---: | ---: | ---: | ---: |
| M5 |50,519%|50,519%|53,398%|51,924%|
| M15|50,871%|52,169%|61,586%|49,106%|
| H1 |54,334%|54,372%|62,434%|50,000%|
| H4 |56,150%|55,652%|62,807%|53,904%|
| D1 |66,813%|63,605%|63,816%|57,509%|

M5 har lik kjent-radvekt q/N. Non-M5 bruker1/n i hver ikke-tom B16-batch, med opptil16×
forskjell per kjent rad. Retningen er ikke universelt UP: H4/D1 trekkes ned av vektingen.
Tellingen etablerer ingen tapsfeil eller årsak til hele senere UP-skjevheten. TRAIN-andel
og senere modell-P er forskjellige populasjoner; konstant tapsminimum er ikke AdamW-banen.
På H1 var senere Pup allerede57,459% ved INITIAL; coverage øker det til62,434% mot50%fasit.
På M15 var det57,820% og blir61,586% mot49,106%. Skjevheten fantes før og forsterkes i forsøket.
Siste6-radersbatch inngår bare i full29574TRAIN-referanse, ikke gamle/nye updateplaner.
Ingen modell-/tapskode er endret. Kilde/formel/resultattolkning er uavhengig gjennomgått.

Neste: kvalifiser én mulig INITIAL/COVERAGE-faktorsammenligning med beholdte non-M5-F-representasjoner
og readouts. Kontroller eksakt lagringsgrense og eksisterende svar først; gjenbruk ferdig metode.
En2×2replay er kun kandidat til eksakt native-logitreplay og minste artefaktomfang er kvalifisert.
Ingen ny data-/checkpoint-/modelljobb, trening eller LATER-kalibrering er admittert.
Global trening/native/full epoch/fullVAL er false. Alle engangsporter er konsumert/stengt.
TEST, Entry/Exit, broker, handel, spending og promotion er fortsatt stengt.

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
