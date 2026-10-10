# Featureaudit og Entry-resultat — 10.10.2026

Revisjonen er fullført med konkrete rettelser. Den eksisterende læringsstudien
fant **ingen kvalifisert Entry-edge**. Korrigert kilde er kontrollert, men gamle
datasett og vekter må ikke behandles som oppdatert modellgrunnlag.

## Hva er kontrollert

Alle254 lokale signalfelt,71 eksakte kontekstaliaser, ett sesjonsfelt og190 MTF-felt
på M5/M15/H1/H4/D1 er navngitt enkeltvis i
[FEATURE_AUDIT_FIELDS.json](FEATURE_AUDIT_FIELDS.json). Hver rad har indeks,
familie, formelkilde, kildeplassering eller dynamisk formelansvarlig, tester,
ekte verdistatistikk, gjenberegning, modellpåvirkning og avgrenset konklusjon.
Dette er516 registrerte overflatefelt, ikke516 uavhengige markedssignaler.

651 sporede prosjektfiler ble inventert. Alle aktuelle17 Markdown-dokumenter,
funksjonene som beregner featureverdiene, åtte-familierutingen, hele hybridmodellen,
normaliseringen og relevante materialiserings-/forbrukerbaner er gjennomgått.
Tidligere statisk gjennomgang og uendrede featuretester er gjenbrukt med hashbinding.
Dette er ikke en påstand om at hver linje i alle651 filer har et uavhengig matematisk bevis.

## Ekte målinger

| Kontroll | Omfang | Resultat |
|---|---:|---|
| TRAIN-livlighet |994500 M5 +4872183 M1-rader,2011–mai2025|Ingen konstante felt eller eksakte dubletter innen flaten;71 aliaser bitlike|
| MTF-livlighet |994500/331959/83608/21631/3615 rader|Alle190 felt varierer på alle fem klokker|
| Lokal eierberegning og pakking |16165 M5 +16094 M1-rader, alle254 felt|Null avvik; base/kontekst gjenbrukt på dette trinnet|
| Uavhengig rå M5-rekonstruksjon |2009–2011 innlest;42414/14268/3622/915/152 TRAIN-rader sammenlignet|Null avvik i190 MTF-felt på hver klokke|
| Rå lokal-/kontekstrekonstruksjon |83 felt,112204 rader|Null avvik|
| Group-A-avstander |14 felt,61282 TRAIN-rader|Ti historiske ekstremumavstander eksakte; fire dagsnivåer har den målte døgnskiftefeilen|
| Modellkoblinger |8 ekte TRAIN-tilstander, sluttcheckpoint|1120 numeriske gradientfelt,71 aliaser,6 kategorier og40 familiekoblinger har målbar effekt|

MTF-ruten for Entry er M15/H1/H4/D1; M5-MTF brukes i Exit. Modellen har både
lokale tidssekvenser, åtte lærte familier og lærte forbindelser mellom familier
og tidsrammer. Numerisk påvirkning på åtte tilstander beviser forbindelse,
ikke informasjonsverdi eller lønnsomhet. CPU-kontrollen gjenskapte native
GPU-prediksjoner med største forskjell0.0001321bps.

## Feil og minste rettelser

1. **Trendlinjens siste berøring gikk bakover.** Forsinket pivotbekreftelse
   overskrev nyere berøringer.131072 ekte M5-rader viste4637 hendelser,
   hvorav1423 innen TRAIN. Gammel beregning matchet130853 lagrede rader
   nøyaktig. `max(siste_berøring,pivot_bar)` bevarer nyeste observasjon.
   Etter rettelsen: **0 regresjoner** på samme utvalg.
2. **Dagsnivåer brukte åpningsklokken.** Baren som lukker22UTC fikk én dag
   for gamle pivotnivåer. Alle fire felt endret seg på128 av61282 TRAIN-rader;
   ingen endring utenfor døgnskiftet. Både M1 og M5 bruker nå beslutningens lukketid.
3. **Fremtidig line-hold-mål kunne gjelde en allerede brutt linje.** Samme bar
   kunne registrere både berøring og brudd og likevel få «held=1» uten senere
   informasjon. Slike linjer utelates før målets maske beregnes. Speilede
   støtte-/motstandstester og ekte datakontroll er gjennomført.
4. **Sesjonsklokken antok nanosekunder.** Gyldige UTC-indekser med mikrosekunder
   fikk feil dags-ID. Eksplisitt enhetskonvertering retter funksjonen. Aktuelle
   datasett hadde allerede ns; feil i disse dataene er ikke påvist.
5. **Beskrivelser og metadata var misvisende.** EMA50/200-slope er5/20 barer,
   warmup219; utgåtte spreadderivater er fjernet fra formelmetadata. BID/ASK-
   ekstremumfeltene er proxyer fra separat aggregerte priser, ikke observerte
   samtidige spreads. «Unswept liquidity» er historiske ekstremumavstander uten
   sweep-identitet. Retestvindu, rå aldre, VWAP-vinduer, klokker og128-dimensjonal
   fusjon er beskrevet i samsvar med faktisk kode.

TrendlinjekontraktV7, MTF-matriseV24/cachev34, Group-A-checkpointv5 og oppdaterte
lokal-/kontekstkontrakter sperrer gammel semantikk. Den faktiske v33-cachen ble
avvist før arraylesing. Ingen rådata, gamle kvitteringer eller checkpoints er endret.

## Læringsstudien

Den lille gjentatte TRAIN-prøven viste fit: sentrert retningsfeil3092.50→241.16,
korrelasjon0.96094. Den separate ferske kurven fullførte16384 optimizersteg på
262144 unike TRAIN-rader, med uendret Exit og godkjente maskinvakter.

På senere CONTROL4096 ble retnings-MSE7357.64 mot konstantens7360.93, omtrent
0.0447% lavere. Ukeblokkforskjellen var−2.46 med familiejustert intervall
[−35.09,+31.35], altså ingen påvist forbedring. LONG var svakere enn konstanten;
modellen valgte4095 FLAT og1 SHORT. Sju av13 måneder hadde lavere retnings-MSE,
men dette kvalifiserer ikke samlet Entry. HGB-referansens100 trinn slo heller
ikke konstanten ved det avtalte indre TRAIN-valget.

Beslutning: `REJECT_ENTRY_QUALIFICATION`. Begge engangsplaner er konsumert,
Windows-oppgaven deaktivert, videre trening og Exit stengt. CONTROL var gjenbrukt
utviklingsdata; markouts overlapper og er ikke en realisert portefølje. TEST er urørt.
De nye featurefeilene beviser ikke at reparasjonene vil skape edge.

## Verifikasjon og gjenstående grenser

141 fokuserte tester består etter rettelsene; uendrede tidligere featuretester
ble gjenbrukt. Ny ekte datakontroll bekrefter monotont berøringsminne og korrigerte
line-hold-mål. All kjøring brukte eksisterende kapasitetssperrer, én jobb av gangen.
Auditadapterfeil og én korrigert formelhash-test er bevart sammen med vellykkede
kvitteringer. Stats-operatørens kildebinding har en egen kontroll som dokumenterer
at en endring i en ikke-kjørt diagnostikkfunksjon lot hele den utførte AST-en være lik.

Rekonstruksjon er avgrenset; alle historiske verdier er ikke uavhengig beregnet
på nytt. Null for første nivåankers recurrence har den eksisterende dokumenterte
fraværs-/eksakt-treff-tvetydigheten. HTF-memoen forutsetter uforanderlige frames.
Ingen nye hypotetiske funksjonsendringer er gjort for disse begrensningene.

**Etterfølgende rebuild og baseline er fullført.** Generasjonen
`HISTORY2009W_FEATURE_REPAIR_20261010` har korrigerte avhengige MTF-/M5-/M1-felt,
berørte labels, én fersk normalisering på hele fysiske TRAIN og beståtte komplette
input-/sekvenskontroller. Fersk komplett ONLINE/TARGET-modell ble målt på de samme
4096 TRAIN- og 4096 CONTROL-tidspunktene, med bitidentiske netto-targets og null
optimizersteg. Alle aktive head-kontrakter og ONLINE/TARGET-paritet består.

Bare tre av 8192 argmax-handlinger endret seg fra gammel initialbaseline; dette
omfatter også tidligere GPU mot nåværende CPU-aritmetikk. Det er ingen målt læring
eller ny edge. Beviset gir verken trenings-/samplerautoritet eller økonomisk
aksept. Ukjente kildegap, gjenbrukt utviklings-CONTROL og historisk kostusikkerhet
består. Samlet aksept:
`/home/andre2/GX1_RUNS/FEATURE_REPAIR_REBUILD_20261010_001/ACCEPTANCE_REVIEW_001/RESULT.json`.

Autoritative maskinbevis: `/home/andre2/GX1_RUNS/FEATURE_SEMANTIC_AUDIT_20261010_001`.
Læringsresultat: `/home/andre2/GX1_RUNS/ENTRY_LEARNING_CURVE_20261010_001/STUDY_REVIEW.json`.
Arbeidsstatus eies bare av [NEXT_RUN_POLICY.json](../NEXT_RUN_POLICY.json).
