# Entry: lærerverdier, faktisk policy og EMA — 23.09.2026

Dette følger ENTRY_REPAIR_20260923.md. Diagnosene bruker modellkilden fra
4711c518 og arkivert v8 der dette er oppgitt. Etter diagnosene ble én konkret
treningsfeil rettet, beskrevet i punkt 6. Ingen nye vekter er trent, og TEST er
ikke åpnet. Alle målinger bruker arkivert checkpoint844
(partiell Epoch2), ikke de tapte evaluerte Epoch1-vektene.

## Bundet utvalg og økonomisk omfang

64 deterministisk jevnt fordelte rader fra TRAIN og 64 fra senere VAL; 63 i hvert
utvalg har gyldig Exit-episode. Vektvariant og inputs er eksplisitt bundet;
råvekter og EMA sammenlignes på samme rader. TRAIN er brukt i originaltreningen;
juni er gjenbrukt
utviklings-VAL. Tallene er begrensede diagnoser, ikke uavhengig edge-bevis.

Økonomien nedenfor er spreadinkludert brutto med den opprinnelige 512-evalueringen,
og alle valgte utfall er med. Øvrige kostnader, årsaksriktig utførelsespris og hele
handelsforløp er ikke fullstendig bundet. Positive verdiestimater kan legitimt
ha enkelte negative utfall; én tapende handel beviser ikke et feilaktig
forventningsestimat. Utvalgene er små og gir ingen generell kalibreringsgaranti.

## 1. Entry lærer ikke når den skal være FLAT

| Måling, 63 TRAIN-rader | Resultat |
|---|---:|
| Rå Entry LONG / SHORT / FLAT | 0 / 63 / 0 |
| Lærerens foretrukne LONG / SHORT / FLAT | 0 / 63 / 0 |
| Lærerens gjennomsnittlige SHORT-verdi | +7,0287 Bps |
| Samme lærers faktiske SHORT-policy, 512-vindu | -0,6497 Bps |
| Råmodellens faktiske SHORT-policy, 512-vindu | -1,9239 Bps |
| Positive lærerverdier, begge sider | 126 / 126 |
| Negative observerte utfall, begge sider | 25 / 126 |

Alle 100 modellvalgte EXIT hos TRAIN-læreren var positive. De 25 negative
utfallene kom ved kapasitetstvungen slutt; totalt 26 nådde denne slutten.
På VAL avsluttet læreren 112 av126 selv, ingen med negativt resultat.

Det betyr at verken lærer-Q eller et ukritisk gjenbruk av kapasitetsterminalens
PnL er tilstrekkelig fasit for en strategi uten maksimal holdetid. Hele åpne
posisjoner, finansiering og kapitalbinding må vurderes.

## 2. EMA svekker ikke alle komponenter likt

Samme 63 VAL-rader, samme checkpoint:

| Måling | Råvekter | EMA |
|---|---:|---:|
| Entry LONG / SHORT / FLAT | 0 / 63 / 0 | 15 / 48 / 0 |
| Entry-valgt resultat | +10,8753 | +0,9998 |
| Alltid SHORT under modellens egen Exit | +10,8753 | +15,9502 |
| Alltid LONG under modellens egen Exit | +5,3110 | -21,5518 |
| Q-rang mot valgte observerte utfall | 0,3651 | 0,0649 |
| Modellvalgte EXIT, begge sider | 112 /126 | 9 /126 |

EMA har53127 oppdateringer og decay0,9999744735. Dette underbygger at Entry-valg
og Exit-funksjon må vurderes separat. Det autoriserer ikke automatisk EMA-avslag,
kombinasjon av to modeller eller ny full trening. Begge Entry-varianter mangler
FLAT i dette utvalget. E1s full-VAL -4,407 kan ikke sammenlignes direkte med
disse små E2-utvalgene.

## 3. Rangeringen har informasjon; en enkel korreksjon er ikke nok

Et fast skille ved medianen av rå SHORT-Q i TRAIN ga:
- TRAIN: høy gruppe32, +7,424 Bps; lav gruppe31, -11,573 Bps.
- VAL med uendret TRAIN-skille: høy gruppe57, +12,231 Bps;
  lav gruppe6, -2,008 Bps.

Dette er rangdiagnose, ikke en produksjonsterskel. En separat, ren
gjennomsnittsfeil-korreksjon per side ga bedre TRAIN, men svakere VAL enn
ujusterte valg. Den forkastes; ingen ny beslutningsregel er innført.
Disse valgdiagnosene bruker frosne Exit-utfall. Entry-Q inngår også i Exit-tokenet,
så en endret modell må alltid måles på nytt gjennom hele funksjonen.

## 4. Lang holding er en målt økonomisk utfordring

For de26 TRAIN-posisjonene som nådde512 undersøkte vi når prisene tidligst
igjen tillot ikke-negativ spreadinkludert lukking etter vinduet:
median33,86 timer fra inngang; to kom ikke tilbake før TRAIN sluttet.
Største observerte negative mark før denne første prismuligheten eller dataslutt
var -8384,15 Bps. Dette er et prisbasert mulighetsmål, ikke modellens faktiske
videreføring; modellen kan velge tap tidligere eller holde videre.

En prisberegning erstatter ikke modellreplay. Den viser hvorfor ubegrenset
holding må prises og hvorfor vinnere alene gir et misvisende bilde.


3. Bruk policyutfall til å undersøke den minste endringen i Entry-verditreningen.
   Eksisterende rangsignal skal gjenbrukes. Ingen bred modell-/terskelsøk.
4. Krev forbedring mot samme baseline på senere kronologiske data, gjennom hele
   endrede Entry/Exit-funksjonen, før større trening.

Eksisterende batchede GRU-output kan gjenbrukes som fortsettelsestilstand uten
ny modellkode. På én reell cachet sekvens var største tilstandsavvik mot512
enkeltsteg 0,000001252; initialiseringen tok0,719s mot6,235s. Dette er en
avgrenset numerisk kontroll, ikke et handelsresultat.

Bevisrot: /home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923.
Filer: ENTRY_SELECTION_TRAIN64_DIAGNOSIS.json,
ENTRY_SELECTION_VAL64_EMA_DIAGNOSIS.json, ENTRY_TRAIN_LABEL_DIAGNOSIS.json,
ENTRY_FIXED_RANK_SEPARATION.json, TRAIN_TAIL_QUOTE_OPPORTUNITIES.json,
BATCH_CARRY_DIAGNOSTIC.json. INPUTS_AND_EMA_CACHE bevarer de faktiske VAL-inputene
for senere gjenbruk. Ingen ekstra modelltrening for rapportering.

## 5. To faktiske TRAIN-policyer er videreført forbi 512

Dette er de to midterste tilfellene rangert etter prisbasert
tilbakehentingstid, ikke et representativt resultat for hele populasjonen.
Begge bruker arkivert checkpoint844-lærer og samme frosne Entry-token.
Q for de første 512 tilstandene var eksakt lik den tidligere målingen.
Eksisterende inkrementell tilstand og alle kausale tidsrammer ble videreført.

| TRAIN-rad / SHORT | Mark ved 512 | Modellvalgt EXIT | Alder ved EXIT | Tilstander |
|---|---:|---:|---:|---:|
| 208932 | -68,1298 Bps | +41,3264 Bps | 317,85 timer | 12674 |
| 29847 | -41,0909 Bps | +28,2583 Bps | 45,0167 timer | 2497 |

Den andre posisjonen ble gjenopptatt fra lagret tilstand 2276; den ble ikke
startet om. Begge er nå lukket av modellen. Beregningsbudsjettet ble aldri
brukt som handelsregel. Tallene er spreadinkludert research-brutto før
finansiering, ikke full netto eller godkjent kausal utførelse.

Eksisterende GRU-tilstand kunne hentes fra den batchede beregningen uten
produksjonsendring. På én ekte VAL-episode tok 512-batch 0,719 s mot 6,235 s
inkrementelt; største forskjell i skjult tilstand var 1,252e-6.
Dette er en avgrenset teknisk måling, ikke et generelt ytelsesløfte.

## 6. Konkret treningsfeil etter fjerning av tvungen slutt er rettet

Reparasjonen i 4711c518 skilte korrekt mellom lovlige handlinger og kjente
labels i VAL, men begge eksisterende treningsveier forventet fortsatt identiske
masker. Dette er en mangel i den reparasjonen. Feilen ble reprodusert på én
cachet ekte VAL-episode: begge stoppet med MASK_SPLIT_BRAIN.

Minste rettelse i entry_v10_ctx_train_v3.py:
- behold begge handlinger lovlige i siste tilstand;
- bruk bare kjente labels i MSE, statistikk og tapets nevner;
- krev eksakt forventet supervisjonsmaske.

Samme episode etter rettelsen: monolittisk og chunket vei PASS,
2046 superviserte celler av 2048 lovlige, raw_loss 3,2580268092.
Entry-token-gradient og alle 296 målte parametergradienter var bitvis like
mellom de to veiene. Vekthash før/etter var identisk.
Dette var kun teknisk backward-kontroll uten optimizer eller parameterendring
på gjenbrukte utviklings-VAL-inputs, ikke trening eller læringsbevis.

## 7. Hva det nåværende målet faktisk belønner

Bevist fra kontraktseierne entry_fitted_q_v1.py og unified_exit_fitted_q_v1.py:
Entry LONG/SHORT lærer maksverdien fra frossen Exit-lærer ved første tilstand;
FLAT har mål 0. Gamma er 1, og mellomliggende HOLD-belønning er 0.
Ingen faktiske swap-/finansieringskostnader eller alternativ bruk av kapital
inngår i disse målene. Dette gjør ikke hvert positivt Q-estimat feil, men
objektivet gir ikke i seg selv den ønskede økonomiske selektiviteten.

I det målte TRAIN-utvalget foretrakk lærerens mål SHORT på alle 63 rader.
Verken mer trening mot samme mål, tilfeldig FLAT-straff, fast Q-terskel eller
bare flere features er en dokumentert løsning. Den eksisterende rangeringen
har allerede informasjon som kan gjenbrukes.

## Neste avgrensede arbeid

1. Bind megler/kontotype, provisjon, faktiske swap-regler og utførelsesgrunnlag.
   Dette er etterspurt. Ingen satser eller risikotoleranser er gjettet.
2. Bruk eksisterende policy/carry og data til å måle resterende åpne
   TRAIN-forløp og senere VAL med alle åpne mark inkludert. Bevar tilstand ved
   ressursavbrudd; aldri gjør et beregningsvindu til en handelsgrense.
3. Avklar policykonsistente økonomiske TRAIN-mål og kapitalbruk før én liten
   læringssammenligning med den eksisterende Entry-representasjonen.
   Ikke tren på kapasitetstvungne 512-utfall som om de var hele handler.
4. Mål endret Entry og tilhørende Exit på nytt samlet; Entry-Q påvirker
   Exit-tokenet. Hold lærermål, TRAIN-tilpasning og senere VAL adskilt.

Full livsløps-VAL er fortsatt ikke integrert i den native treneren.
Ingen forbedret Entry eller handelsfordel er påvist. Målet er aktivt, full
trening er stoppet, og PC er ikke restartet. GX1_CURRENT er urørt.
