# Feature- og modellkompleksitet — fullført forberedelsesvurdering 29.09.2026

Konklusjon: vi har et stort og sammensatt oppsett, men ingen kontroll har vist
at antall features er årsaken til manglende edge. Bevar inputene. Første
forenklingshypotese er færre hjelpeoppgaver i en senere avgrenset sammenligning.
Hvis målet er vesentlig lavere beregningskost, må encoderne vurderes separat.
Ingen feature, familie, tidsramme, head eller tapsfunksjon er endret her.

Kilde: `ca6106a9a2af6258b29006115e072b79a1abe23a`, faktisk v37-build
`HISTORY2009W_NATIVE_V37_20260928`. Sluttrapport med hasher:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/FINAL_PREPARATION_REVIEW_20260929/FINAL_PREPARATION_REPORT.json`.
Tall nedenfor er enten målt på ekte TRAIN-inputs eller utledet fra gjeldende
kilde. Ingen modellprediksjoner, optimizersteg eller ny VAL/TEST-vurdering.

## Hele den emitterte TRAIN-flaten, utover de 67 kandidatene

`TRAIN_REDUNDANCY_REVIEW.json` og `TRAIN_FEATURE_CORRELATIONS.json` i samme
runtime-mappe dekker alle **652 552 beslutningsrader × 242 signalfelt**,
2011-06-01 00:00 til 2025-05-30 12:55 UTC. Bare `time`, `snap` og `ctx_cont`
er lest fra TRAIN. Dette er den faktiske emitterte populasjonen, ikke den
eldre kandidat-rangeringens 994 500 kilderader eller et sample.

- 25 basis + 150 obligatoriske + 67 kandidater = 242. Alle åtte familier består.
- Ingen helt konstant kolonne gjennom hele TRAIN og ingen eksakte innbyrdes
  duplikater. Ingen eksakt negasjon, absoluttverdi eller `1-x`-relasjon mellom
  to ulike signalfelt ble funnet med de kontrollerte float32-transformasjonene.
- Alle 71 kontekstaliaser er eksakt like sine signalkopier på samtlige rader.
  De er ikke 71 ekstra uavhengige signaler. Modellen utelater allerede aliasene
  fra generell snapshot-projeksjon og bruker én eid normaliseringsstatistikk.
- `close_range_observed` er 1 på 99,9902 % av radene, men er 0 på 64 faktiske
  rader. Det er en tilgjengelighetsopplysning med semantisk rolle, ikke en
  bevist overflødig konstant. Noen kalenderår har konstant verdi; alle år er
  rapportert separat. Sjeldne cross-/retest-/divergenshendelser er også synlige.
- `rsi14_centered` og `ctx_cont.m5_rsi14_canon_v2` har Pearson omtrent 1 på
  Entry-TRAIN. Kilden beskriver den affine samme-klokke-relasjonen allerede.
  På Exit er lokal RSI fra M1, mens kontekstfeltet er siste lukkede M5-RSI;
  derfor følger ingen generell slettingsrett fra Entry-korrelasjonen.
- Andre sterke TRAIN-assosiasjoner inkluderer lokal EMA50-helning mot MACD
  (0,9782), signert candle body mot lokal close-endring (0,9680), og
  Parkinson-volatilitet mot ATR (0,9643). Alle 29 161 par og årsvise
  korrelasjoner er lagret uten automatisk terskel eller valg av features.

Pearson og sparsomhet er beskrivelser av dette TRAIN-settet. De beviser
verken prediktiv nytte, kausalitet, stabilitet senere eller at en feature kan
fjernes uten å påvirke Exit. Rapporten har ingen seleksjonsmyndighet.

## Overlapp mellom tidsrammer — eksisterende bevis gjenbrukt

Den fullførte `ENTRY_CROSS_SURFACE_INPUT_OVERLAP_20260929T010051565758Z.json`
binder lokalflate og aktive MTF-ruter med tidsstemplet float32-hash over
1 152 798 Entry-kilderader og 5 570 522 Exit-kilderader. Den fant ingen
uventede eksakte aktive duplikatpar eller manglende erklærte aliaser.
Det finnes 28 erklærte context/MTF-aliaspar for Entry og 32 for Exit,
inkludert både signal- og kontekstkopier. Lokal M5 gjentas ikke som aktiv
MTF-rute for Entry; Exit bruker M5/M15/H1/H4/D1.

Dette er byggets fullhistoriske strukturelle bevis, gjenbrukt som metadata.
Det er ikke en ny TRAIN-avgrenset MTF-ablasjon eller bevis på merverdi fra
alle tidsrammer. Ingen kildedata ble lest om for denne sammenfatningen.

## Parameterfordeling — kildealgebra, ingen modellkjøring

`PARAMETER_COMPLEXITY_REVIEW_EXACT_ROUTING.json` evaluerer de eksakte
parameterdeklarasjonene fra modellkonstruktørens AST på meta-tensorer.
Normaliseringsverdier, syntetiske markedsinputs og native modellkonstruksjon
brukes ikke. Formlene for Transformer og GRU er kontrollert mot PyTorchs
faktiske parametertelling. Det er en strukturtelling, ikke gjennomstrømning,
minnebenchmark, funksjonsparitet eller oppstartsgodkjenning.

Med den eksisterende hardware-smoke-kildens referansekapasitet
**lokale spesialister Ls=1, MTF-spesialister Lm=2**, bredde 128:

| Del | Parametre |
|---|---:|
| Hovedsekvensens tre Transformer-lag | 594 816 |
| Åtte lokale spesialistencodere | 1 586 176 |
| Åtte delte MTF-spesialistencodere, to lag | 3 172 352 |
| Sju attention-lag for samspill mellom familier/tidsrammer | 1 387 904 |
| Atten Exit-GRU-er | 1 783 296 |
| Øvrige projeksjoner, embeddings, heads og lærte skalarer | 1 108 511 |
| **Totalt** | **9 633 055** |

De åtte hjelpeprojeksjonene er **6 450 parametre / 0,067 %** av totalen.
Alle ti oppgaver har dessuten hver sin lærte log-varians. Å fjerne hjelpeheads
alene sparer lite parameterplass; eventuell gevinst gjelder læringsmål og
mål-/gradientarbeid. Andre dybder er ikke gjettet:

`parametre = 4 874 527 + 1 586 176 × Ls + 1 586 176 × Lm`.

Dette er ikke en ny autorisert v37-treningsrecipe. Bredde, feltsett og alle
andre nåværende konstruktørvalg er holdt faste i formelen. Første telling tok
feilaktig med to tomme historiske klassifikasjonsgrupper i rapportscriptet;
den er bevart og eksplisitt erstattet av eksakt åtte-familietelling.
Produksjonskode og modell ble ikke endret av rapportrettelsen.

## Hvor beregningen ligger

Ved samme referansekapasitet gir kildealgebra omtrent **2,199 milliarder MACs**
for Transformer-lagenes lineære og attention-matrisemultiplikasjoner per Entry-
eksempel. MTF-familieenkoderne utgjør **88,7 % av denne avgrensede summen**.
Vinduer: Entry 96 M5-barer; MTF M15=64, H1=96, H4=96 og D1=252.
Antall MACs er ikke FLOPs, millisekunder eller total treningskost. Input-/head-
projeksjoner, softmax, normalisering, aktiveringer, backward og minnetrafikk
inngår ikke i summen.

Exit har en annen rute: ni lokale GRU-skanninger over 480 M1-historikkrader
utgjør alene omtrent **425 millioner MACs** per ukachet random-access-state,
før MTF, path og heads. Eksisterende frozen-eval-cache kan gjenbruke markeds-
tilstand. Lifecycle-filen inneholder 668 205 056 TRAIN-statepekere, ikke så
mange uavhengige handler eller eksekverte modellforwards. Faktisk tidsbruk
avhenger av den senere bundne populasjonen, sampling, caching og batchplan.

## Hjelpeoppgavene og deres begrunnelse

Entry velger fra sine tre rå Q-verdier. Hjelpeprediksjonene mates ikke inn i
Entry-argmax, men tapene påvirker delt representasjon. Alle ti oppgaver får
lærte log-variansvekter fra samme kontrakt. Det balanserer tap gjennom en
lært mekanisme; det beviser ikke at hjelpeoppgavene forbedrer nettoøkonomi.

| Hjelpehode | Utganger | Dokumentert rolle | Uavklart merverdi |
|---|---:|---|---|
| Side-MAE | 2 | Risiko på hver side | Overlapp med dip-/tail-risiko |
| Trendline-event | 4 | Fremtidige strukturelle hendelser | Bedre handelsvalg etter kost |
| Position-size | 1 | Størrelse etter at retningen er valgt | Kalibrering og økonomi i ny modell |
| Dip | 18 | Dybde og recovery over tre horisonter | Verdi utover Entry/Exit Q og annen risiko |
| Return-forecast | 4 | Fremtidig avkastning over flere horisonter | Samsvar med utførbar kostnadsjustert beslutning |
| Timing | 12 | Tid til dip-/gunstig ekstrem innen horisont | Bidrag utover Exit-beslutningen |
| Tail-risk | 6 | Advers hale over tre horisonter | Bidrag utover øvrige risikooppgaver |
| Vol-forecast | 3 | Fremtidig volatilitet | Verdi for representasjon/størrelse |

Rolle er bevist fra kilde; bedre senere beslutningsverdi er **ikke undersøkt**
for den nye v37-modellen. Dette er ingen fast tapsgrense eller maksimal holdetid.

## Anbefalt neste beslutning

Prioriter én forhåndsbundet sammenligning av læringsoppgaver, med uendrede
features, familier, tidsrammer, kapasitet, kronologi, kostnader og budsjett.
En policyorientert variant må bevare Entry/Exit-autoritet og nødvendig sizing;
hvilke representasjonshjelpere den skal beholde må vedtas eksplisitt før start.
Dagens kontrakt krever alle heads; ingen gate svekkes her for å få en variant
igjennom. En separat kapasitetsvariant er aktuelt hvis hovedmålet er beregning.

Senere netto utfall mot forhåndsvalgte kausale baselines avgjør, ikke TRAIN-
korrelasjon, lavere hjelpeloss eller teknisk PASS. Eksisterende
`entry_exit_feature_usefulness_v1.py` kan senere måle trent modellnytte på
utviklings-VAL; den er ikke kjørt nå og gir ingen automatisk slettingsrett.
Det bestilte forberedelsesomfanget er ferdig. Native trening, optimizer,
full VAL, TEST-utfall, handel og spending forblir stengt.
