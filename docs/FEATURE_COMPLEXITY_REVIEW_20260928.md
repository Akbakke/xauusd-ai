# Foreløpig vurdering av feature- og modellkompleksitet

Kilde verifisert mot alle 850 Git-objekter ved `736728929e011767675baccad0b6b80bc591aaa7`.
Dette er del av den bestilte repo-gjennomgangen, ikke en fullført revisjon eller en treningsbeslutning.

Vi bør vurdere en enklere modell. Påstanden «altfor avansert» var sterkere enn
bevisene: ingen kontroll har isolert featureantall som årsak til manglende edge.

## Målt eller kontrollert i v36-kilden

- M5-signalet ved denne målingen hadde 241 felt: 24 basis, 150 obligatoriske og 67 kandidater.
  De 71 kontinuerlige kontekstfeltene er allerede representert i disse 241.
- Den ferdige TRAIN-rangeringen har 994 500 rader og 67 kandidater. Ingen av
  kandidatene er helt konstant eller et eksakt duplikat av en annen kandidat.
  Det sier ikke at alle tilfører prediktiv informasjon, eller at de ikke overlapper
  med basis, obligatoriske felt eller andre tidsrammer.
- Alle 67 beholdes av gjeldende kontrakt. Rangeringen er diagnostikk, ikke et
  kriterium som velger et bestemt antall features.
- Modellen utelater allerede eksakte kontekstaliaser fra den generelle
  snapshot-projeksjonen, og kontrollerer at aliasenes verdier er identiske.
  Tidsserien beholder historikken. Dette er allerede håndtert og skal ikke
  fremstilles som en ny, urettet dupliseringsfeil.
- Åtte spesialistfamilier, flere tidsrammer og separate Entry-/Exit-forløp gir
  betydelig strukturell kompleksitet. Faktisk samlet parameterfordeling skal
  måles fra dagens konfigurasjon før en størrelsesanbefaling.
- Entry har ett beslutningshode og åtte hjelpehoder. Beslutningshodet får fire
  lærte representasjoner direkte; hjelpeprediksjonene mates ikke inn i Q-valget.
  Hjelpetapene påvirker likevel den delte representasjonen under trening.
- De åtte hjelpehodene er lineære outputprojeksjoner med totalt 50 utganger.
  Med kodens bredde 128 er det 6 450 vekter/biaser i disse outputlagene.
  Å fjerne dem forenkler først og fremst læringsoppgavene; det fjerner ikke
  de delte sekvens- og tidsrammeenkoderne.

Kildesteder: `gx1/contracts/entry_model_native_signal_v1.py`,
`gx1/contracts/entry_exit_production_architecture_v1.py`,
`gx1/contracts/entry_model_native_aux_targets_v3.py`,
`gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py` (projeksjoner rundt
linje 777 og 1190; aktiv Entry-forward rundt 3690).

Den eksisterende `entry_exit_feature_usefulness_v1.py` skiller allerede
inputtilgjengelighet fra senere prediktiv nytte. Negative nytteverdier er
gyldige funn, og diagnostikken gir ikke automatisk tillatelse til featurefjerning.
Gjenbruk denne eieren der den passer; ikke bygg et nytt parallelt rangeringssystem.

## Anbefalt neste avgjørelse

1. Avklar den målte SMC-kontraktkonflikten før videre inputbygging. Rett
   observerte feil i diagnostikken før den brukes til å velge bort modellkomponenter.
2. Kartlegg redundans på TRAIN og parameter-/beregningsfordeling, og vurder
   hvilken dokumentert funksjon hver hjelpeoppgave har. Ingen vilkårlig grense
   som 15, 20 eller 50 features.
3. Begrunn og bind én forenkling på én akse: informasjon, læringsoppgaver eller
   modellkapasitet. Ikke endre alle tre samtidig.
4. En senere tillatt sammenligning må bruke samme kausale inputs, kronologi,
   kostmodell og budsjett. Mål senere LONG/SHORT/FLAT-valg og netto utfall mot
   kausale baselines. Lavere treningsloss eller lavere minne er ikke edge.

Min prioriterte forenklingshypotese er **færre hjelpeoppgaver først**, med samme
inputs og Entry/Exit-økonomi. Vi må da uttrykkelig tillate en forskningsvariant
som ikke trenger å bestå kravet om at alle hjelpehoder brukes. Det er et forslag
til en kontrollert sammenligning, ikke et påvist bedre oppsett eller tillatelse
til ny trening. Hvis hensikten er mindre beregning, må vi vurdere encoderne
separat; hjelpehodene alene er små.

Featurefjerning bør begrunnes med redundant informasjon eller fravær av senere
beslutningsverdi. Korrelasjon/rangering på TRAIN alene kan brukes til å formulere
hypotesen, men ikke erklære en forbedring. Bevar rådata og tidligere schema slik
at reduksjonen kan etterprøves.

## Byggfeilen før SMC-rettelsen

Ingen feature, familie eller modellhode er fjernet. Native trening er deaktivert.
Arrow-rettelsen ved `10c78d70` er bekreftet i fullkjøringen: faktisk RSS falt
9,52 → 6,85 GiB, og alle 1 382 Group-A-chunks ble fullført. De bevares.
Dette var en buffer-/minnefeil, ikke dokumentasjon på for mange features.

M1 stoppet deretter på `smc_pivot_envelope_position`: fire bekreftede pivotpriser
er like på sju TRAIN-rader etter oppvarming (2012 og 2019). Formelen deler på
intervallets bredde og returnerer tilsiktet NaN ved null bredde, mens input-
kontrakten krever endelige tall. Ferdig M1-datasett foreligger derfor ikke.

Denne konflikten må løses ved feature-eieren: enten eksplisitt representasjon
av utilgjengelighet med en tilhørende indikator, eller en begrunnet revisjon/
fjerning av målingen. Å fylle null uten indikator, slette de sju radene eller
endre pivotdefinisjonen for å få grønt er ikke en dokumentert løsning. Begge
reelle alternativer påvirker skjema/lineage og må ha konsekvensoversikt før bygg.
En feature som er vanskelig å bygge, er ikke dermed prediktivt ubrukelig.

M3-diagnostikken er nå rettet for masker, entropi, eksakte konstanter, sesjoner
og ugyldige hjelpeetiketter; 140 fokuserte tester besto. Én samlet suite ga
5 628 bestått, 18 feil og 3 hoppet over. Alle de 18 feilede tilfellene består nå
etter triage (133 tester i berørte filer). En ny fullsuite er ikke kjørt.
Inputkontrakten og siste parameter-/redundansvurdering gjenstår før eventuell
arkitekturendring. Teknisk kontroll er ikke bevis for lønnsomhet.

## SMC-rettelse og avgrensning av oppryddingen

Brukerens etterfølgende vedtak 28.09 autoriserer retting og opprydding. Den
målte SMC-konflikten er nå implementert som et eksplisitt posisjon-/breddepar:
positiv bredde bevarer rå posisjon, kjent null bredde kodes `(0, 0)`, og ukjent
oppvarming beholder NaN. Lokal bredde manglet og er lagt til; MTF hadde den
allerede. V37 får dermed 242 signalfelt, ikke færre. Det er en korreksjon av
inputsemantikk, ikke et valg om å øke modellkapasitet eller en ny edge-påstand.

Det tidligere forslaget om en binær tilgjengelighetsindikator er erstattet av
den allerede brukte kontinuerlige bredderepresentasjonen. En ny indikator
ville vært konstant på tidsrammer uten nullbreddehendelser og måttet endre
liveness-kontrakten. Nå gjenbrukes én eier og eksisterende MTF-felt.

Kontrollstatus: 189 fokuserte og 92 integrasjonstester består. Faktisk lokal
M1-kontroll dekker 6 019 349 rader; MTF-paritet er kontrollert på alle sju
berørte TRAIN-rader. Den tidligere maskinfelles låsen er erstattet av en
prosjektlås etter brukerens vedtak; ressursvaktene består. Ingen ferdig v37-datasettrebuild, ablasjon,
parameterfordeling eller ny redundansmåling foreligger. Gamle v36-artefakter
og alle checkpoints er bevart. Se gjeldende status i CURRENT_HANDOVER.md.
