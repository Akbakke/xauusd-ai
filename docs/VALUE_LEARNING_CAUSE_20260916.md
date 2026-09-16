# Hvorfor GX1-verdilæringen er skjev — 2026-09-16

**Skjevheten er lokalisert til lærer-/målkjeden og modellens verdilesing.**
Entry lærer hovedsakelig et negativt grunnnivå fra en Exit-lærer med nesten
ingen fortsettelsesverdi. Samme lærer bestemmer hvor langt femstegsmålene
faktisk følger markedet, forskjellig for LONG og SHORT. ONLINE-Exit flytter
deretter sidevise grunnverdier mer enn den lærer å skille tilstander.
Dette dokumenterer mekanismene; det beviser ikke at én enkelt kodefeil,
én bestemt optimizerparameter eller mangel på markedsfordel forklarer alt.

Gjeldende kilde ved målingen var `0dd34ff8352a3b56746a00b4a2564d128fb56ac1`
i `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`. Ingen modell- eller
treningskode er endret. Trening, full epoch og full VAL forblir deaktivert.

## 1. Entry lærer nesten første lukking, ikke en dokumentert videre mulighet

Den eksisterende brede TRAIN-dekomponeringen på 1 024 Entries viser:

- Første gjennomførbare likvidasjonsverdi: gjennomsnitt −5,85327 Bps.
  Dette inneholder både prisbevegelsen frem til første tilstand og kostnader;
  tallet skal ikke omtales som ren spread eller realisert strategiprofitt.
- Frossen lærers tillegg for fremtidig verdi: bare +0,02456 Bps i snitt,
  maksimum +0,09434 Bps.
- Å fjerne dette tillegget endrer bare 3 av 1 024 lærerhandlinger.
  Læreren velger 914 FLAT, 49 LONG og 61 SHORT. ONLINE95 og96 velger alle FLAT.

Koden bygger Entry-målet som første likvidasjonsverdi pluss maksimal gyldig
HOLD/EXIT-verdi fra den frosne læreren; FLAT-målet er null. Med så liten
fortsettelsesverdi ligger LONG/SHORT-målene nær et negativt grunnnivå.
Ved checkpoint96 er alle 1 024 LONG- og SHORT-prediksjoner negative; selv
maksimum er henholdsvis −3,988 og −4,073 Bps. Prediksjonenes standardavvik er
omtrent 0,50 Bps, mot 4,36 Bps i målene. LONG-MSE er 19,2013, mens en konstant
lik samme utvalgs målgjennomsnitt gir 19,0460. SHORT har en liten fordel over
denne beskrivende konstantbaselinen, men ingen valgte handler.

Dermed er allFLAT forenlig med målberegningen og den svake tilstandsavhengige
tilpasningen. Det er ikke dokumentert selektivitet. En in-sample konstant er
bare en diagnosebaseline, ingen validert handelsmodell.

## 2. Fem beregningssteg er ikke fem steg med likt læringssignal på begge sider

Den lagrede native batchen har alle fem successors tilgjengelige for samtlige
64 overganger. Den gjeldende mål-eieren reproduserte lagrede targets eksakt.
Likevel blir effektiv lengde:

| Side | Ett steg | Fem steg | Lærerens HOLD-verdi ved første successor, snitt |
|---|---:|---:|---:|
| LONG | 59 | 5 | −0,06342 Bps |
| SHORT | 0 | 64 | +0,10568 Bps |

Årsaken er den eksplisitte regelen: bare en entydig HOLD fra den frosne
læreren tar med neste observerte reward. EXIT eller likhet stopper sporet
med lærerens verdi. Her er dette lærerpolicyens valg, ikke manglende data,
en tidsgrense eller utilsiktet terminalisering.

Femstegs LONG-target har standardavvik 5,736 Bps, SHORT 8,461 Bps. Samme
beregningsbudsjett gir dermed forskjellig målhorisont og målfordeling.
Dette er målt på én batch; fordelingen i alle 32 oppdateringer er ikke målt.

Femstegsendringen gjelder Exit-target. Entry bruker fortsatt det frosne
førstetilstandsestimatet; største observerte Entry-targetforskjell mellom
ett- og femstegsberegningen er 2,98e−8 Bps. Lærer91 er uendret gjennom95→96.
En bedre ONLINE-Exit kan derfor ikke straks endre Entry-supervisjonen.
Tidligere FQI-refresh endret heller ikke de 64 målhandlingene og forbedret
ikke Exit-fit samlet. Automatisk lærerrefresh er derfor ikke en dokumentert løsning.

## 3. LONG-HOLD kommer av grunnverdiforskyvning, med kjent parameterbidrag

I den brede cachen stiger LONGs HOLD-verdi fra 0,05001 til 0,20493 Bps.
Alle 4 096 verdier stiger og blir positive, altså alle HOLD mot EXIT=0.
En konstant forskyvning forklarer 94,70 % av kvadrert prediksjonsendring.
Korrelasjonen med samme ettstegsmål går fra −0,00159 til −0,00689.
Dette er diagnostikk mot gamle ettstegsmål, ikke den nye femstegs-lossen.

To CPU-evalueringer av den allerede lagrede 16-entry/64-overgangsbatchen
reproduserte Entry-, Exit- og forecast-output for95 og96 eksakt. Ved å lagre
representasjonen rett før siste Exit-hode kunne endringen deles algebraisk:

`ΔQ = Δw · gjennomsnitt(h) + gjennomsnitt(w) · Δh + Δb`

Her er `w` og `b` differansen mellom rå HOLD- og EXIT-rad i siste lineære hode.
Fordelingen er symmetrisk mellom checkpointene; restfeilen mot FP32-output
er maksimalt 8,59e−8 Bps.

| Gjennomsnittlig bidrag, Bps | LONG | SHORT |
|---|---:|---:|
| Siste hodets vekter | +0,094622 | −0,000646 |
| Representasjonen foran hodet | +0,063794 | −0,039401 |
| Felles biasendring | +0,001507 | +0,001507 |
| Samlet endring | +0,159923 | −0,038541 |

Omtrent 60 % av LONGs gjennomsnittsøkning kommer fra hodet inklusive bias,
40 % fra representasjonen. SHORT-fallet kommer fra representasjonen, svakt
motvirket av hodet. Det er derfor ikke en enkel feil i biasparameteren.
«Konstant outputforskyvning» og «endret biasparameter» er forskjellige ting.

Denne dekomponeringen identifiserer hvor endringen uttrykkes. Den identifiserer
ikke bidrag fra hver upstream-modul eller Adam-historikken. Lærerens skjeve
spor er heller ikke alene bevis for fortegnet på ONLINE-oppdateringen:
LONG-targetsnittet i denne batchen er −0,02855 Bps, samtidig som ONLINE stiger.
Oppdateringen brukte 32 batcher og bevart optimizerhistorikk.

## 4. Prognosesignal finnes på TRAIN, men har en annen vei til Entry

På 1 024 TRAIN-rader har forecast95 korrelasjon 0,176/0,219/0,308/0,425 med
observerte bruttoavkastninger etter nominelt5/25/60/120min. Ved96 blir disse
0,119/0,162/0,293/0,395. Alle fire MSE blir verre. Dette er verken senere
VAL, kostnadsjustert handelsverdi eller dokumentert generalisering.

`head_forecast(z)` lærer observerte utfall. Entry-Q får lokale og MTF-
representasjoner som input, men ikke forecast-output direkte. I V4 er denne
representasjonen detached ved inngangen til Entry-Q-mikseren. Entry-Q-lossen
trener mikseren og handelsverdihodet mot Exit-læreren; den kan ikke selv forme
Entry-representasjonen tilbake gjennom dette grensesnittet. Prognosene kan
påvirke Entry indirekte gjennom representasjonen, men de erstatter ikke
handelsmålet. Bedre forecast alene løser derfor ikke målproblemet.

## 5. Kontrollerte alternative forklaringer og beslutning

- Sampleren bruker samme tidslinje for LONG/SHORT og velger uten pris, reward
  eller side i seed. Av 4 096 lagrede overganger er497 ved tilstand0,
  1 082 ved tilstand≤5 og2 035 ved tilstand≤60. Hypotesen «Exit ser aldri
  tilstander rett etter Entry» støttes ikke. Separate lærerankre har no-loss,
  men ordinære treningssamples omfatter også tilstand0.
- Eksisterende gradientanalyse på checkpoint87 viste ingen direkte Exit-
  gradient i private Entry-encodere. 99,9966–99,9987 % av kvadrert rå Exit-
  gradientnorm lå i Exit-blokkene og verdihodet. Dette er historisk gradient-
  geometri, ikke dagens Adam-oppdateringer; den gir fortsatt ikke grunnlag for
  generell backbone-, gradient- eller tapsvektendring.
- Resume-paritet og nye eksakte outputankre støtter at vi sammenligner riktige
  tilstander. Ingen ny teknisk kontroll, epoch eller full VAL trengs for å
  gjenta disse funnene.

**Beslutning:** Behold treningsstoppen. Før en kodejustering må neste avgrensede
hypotese rette den konkrete lærer-/verdisupervisjonen: Entry trenger et lært
estimat av videre nettoverdi som faktisk skiller tilstander; femstegs Exit
arver fortsatt en svak, sideavhengig stopp-policy. Ikke nullstill bias/Adam,
oppdater læreren automatisk, koble bruttoforecast direkte til handel eller
endre tapsvekter ut fra disse resultatene. Ingen slik endring er valgt eller
utført. En eventuell målendring må bevare kausalitet, observerte successors,
bootstrap, kostnader og fravær av fast holde-/tapsgrense, og sammenlignes mot
samme frosne mål/baselines før en ny avgrenset native trening kan begrunnes.

## Reproduserbart bevis

Artifactrot: `BASE/NATIVE_REAL_TRAIN_TO_VAL_FDD70E5C/OPERATOR_OBSERVATIONS/CACHED_VALUE_CAUSE_20260916`,
der BASE er `/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912`.
Plan, operatorer, kjørelogger og resultater er bevart der og kopiert til
`handover_snapshot/CACHED_VALUE_CAUSE_20260916/`.

- Cacheanalyse:1,04s, RSS584 432KiB, ingen forwards.
  RESULT.json SHA256 `de50f50bf2d81817cbfa0a033aa9143c213a415b71d67b692972bf7ff2f266e2`.
- Hodeanalyse:9,75s, RSS1 955 728KiB, to Entry- og to Exit-forwards på CPU,
  ingen lærerforward. ATTRIBUTION_RESULT.json SHA256
  `98cbd1b223542f5a678265ba48f4910ed9596d1079f8937a5a5c5d12e0853559`.
- Begge brukte `gx1_capped_run.sh`, audit/4GiB/512MiB-swap, verifisert cgroup.
  Ingen backward, optimizersteg, checkpointendring, GPU, ny materialisering,
  VAL, TEST eller ordre. Et første overføringsforsøk traff Windows'
  kommandolengdegrense før kjøring; SSH standard input løste overføringen.

Eksisterende bevis: `BROAD_TRAIN95_134_TARGET_DECOMPOSITION_20260916.json`,
`FIXED_TARGET_LEARNING_RESULT_20260915.json`, `FQI_TARGET_LEARNING_RESULT_20260915.json`,
`SHARED_TASK_GRADIENT_REVIEW_20260915.json` og
`FROZEN_TRACE_LEARNING95_96_REVIEW_20260916.json` under `handover_snapshot/`.

Kildeeiere: `unified_exit_random_access_training_v1.py:485,690`,
`entry_v10_ctx_hybrid_transformer.py:3100,3710`,
`entry_v10_ctx_train_v3.py:9392,14296`,
`unified_exit_random_access_sampler_v1.py:162` og
`unified_exit_random_access_model_v1.py:26`. Filhashene fra kildegjennomgangen
ligger i cacheanalysens `input_bindings`.
