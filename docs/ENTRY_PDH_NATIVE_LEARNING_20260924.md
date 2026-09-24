# PDH/H4: native synlighet og kausal læringskontroll — 24.09.2026

## Beslutning og avgrensing

Oppfølging av [PDH/H4-kontrollen](ENTRY_PDH_H4_STATE_CONTROL_20260924.md).
Den tidligere positive TRAIN-hypotesen er fortsatt ubekreftet. Den enkle kausale
verdilæringen nedenfor består ikke juni-kontrollen. Det er ikke grunnlag for å
promotere en regel, endre native features eller starte en større treningsjobb.

Dette arbeidet gjør et konkret hull i tidligere måling synlig: de to native
512-raders TRAIN-kontrollene inneholdt ingen av de 299 PDH/H4-hendelsene.
De kan ikke alene avgjøre modellens læring på dette sjeldne oppsettet.
Kontrollene gjelder utvikling; ingen av resultatene er uavhengig bekreftelse.

## 1. Informasjonen finnes allerede i native input

TRAIN har 313 399 rader og 299 PDH/H4-hendelser (0,0954 %).
Native pivotavstander bruker fortegnet `(pris − nivå) / ATR`. Identiteten
`PDH = R2 + S2 − S1` gir derfor:

```
(pris − PDH) / ATR = dist_to_R2_atr + dist_to_S2_atr − dist_to_S1_atr
```

Fortegnet stemmer med forskningens PDH-avstand på 99,9971 % av radene.
Alle ni avvik har eksakt null i forskningsfeltet og en liten positiv verdi
(2,38e−7 til 1,91e−6 ATR) ved rekonstruksjon fra native flyttall.
Dette er relevant ved et strengt `> 0`-brudd. Median absolutt avstandsavvik er
6,90e−7 ATR. Forskningsavstanden klippes til ±20 ATR; feltene er derfor ikke
generelt numerisk identiske.

Forskningsbetingelsen `H4 close > EMA20 > EMA50 > EMA200` kan rekonstrueres
fra native H4-avstander som `0 < d20 < d50 < d200`. Denne kontrollen har
**100 % samsvar** på alle TRAIN-rader, med samme lukkeklokke.

To navn må ikke forveksles med denne definisjonen:

- Native `ema_stack_aligned_v2` bruker EMA20/50/100/200 og krever ikke at
  close ligger over EMA20. Samsvaret med forskningsbetingelsen er 84,798 %.
- `dist_to_d1_hi_atr` gjelder et likviditetsnivå fra et 60-dagers vindu,
  ikke nødvendigvis forrige dags topp.

Dette er ulike semantiske definisjoner, ikke påviste feil i feature-eierne.
Ingen native feature er endret eller lagt til.

Rekonstruksjon av *første* brudd fra bare eligible beslutningsrader ga 302
hendelser: 297 av de 299 riktige, fem ekstra og to manglende. Med den eksakte
forskningsavstanden gjenfinnes alle 299, men tre ekstra gjenstår. Bare eligible
rader er dermed ikke en eksakt erstatning for hendelseseierens fulle barhistorikk.
Den allerede kontrollerte hendelseslisten brukes i native-målingen.

## 2. Den tidligere native målingen traff ikke oppsettet

Både `NATIVE_FIXED512_TIME_COST_ANALYSIS.json` og
`NATIVE_NSTEP512_TIME_COST_ANALYSIS.json` har 512 unike TRAIN-rader og
**null PDH/H4-hendelser**. Dette er en begrensning i kontrollutvalget, ikke
bevis på at hele tidligere trening manglet slike hendelser.

Datasettets lagrede 19-bars knee-utfall gir LONG +7,855 og SHORT −11,643 bps
på de 299 hendelsene. LONG er positiv på 57,19 % og bedre enn SHORT på
59,20 %. Disse kolonnene er **ikke** den nåværende frosne lærerens online
fitted-Q-mål, og skal ikke brukes som bevis på hva native Entry ble lært.

## 3. Fast kontroll av om betinget verdi kan læres over tid

Før kjøring ble to små surrogater fastlagt:

- Referanse: gjennomsnittlig LONG/SHORT-verdi i tre H4-tilstander.
- Kandidat: de samme tre tilstandene krysset med første-PDH-brudd, seks grupper.

Hver gruppe lærer bare fra tidligere TRAIN. Unik høyeste verdi blant LONG,
SHORT og FLAT=0 bestemmer valget; ukjent gruppe eller eksakt likhet gir FLAT.
Ingen terskel, prosentil, modell- eller horisontjakt. Fire påfølgende årsperioder
og den allerede gjenbrukte juni-VAL ble vurdert med ekspanderende TRAIN og
49 M5-barers purge. Alle fit-utfall ligger før neste evalueringsgrense.

Verdimålet er uendret h48, neste M5 ask_open/bid_open, spread, arkiverte
2 bps per utførelse og finansiering etter faktisk tid. Dette er et markert
utfall ved et observasjonstidspunkt, ikke en ny native maksimal holdetid.

| Evaluering | Kandidat LONG / SHORT | FLAT | Netto bps per valgt inngang |
|---|---:|---:|---:|
| 2022–23 | 68 / 0 | 63 240 | −3,471 |
| 2023–24 | 51 / 0 | 64 151 | +7,729 |
| 2024–25 | 82 / 0 | 65 173 | +0,837 |
| 2025–26 | 77 / 0 | 65 218 | +14,179 |
| Juni 2026, gjenbrukt utviklings-VAL | 3 / 0 | 5 506 | **−46,057** |

H4-referansen velger alltid FLAT. Det er ikke dokumentert nyttig selektivitet.
Kandidaten lærer sjeldne innganger, men ingen SHORT. Alle valgte innganger og
tap er med; ingen manglende fills eller åpne observasjoner ved periodegrensene.
Maksimalt to observasjonsvinduer overlapper. Summene er uavhengige
inngangsutfall, ikke porteføljeavkastning.

Alle tre juni-innganger er PDH-brudd i **bearish** H4. Gruppen fikk positivt
historisk snitt (+4,352 bps, 52 TRAIN-hendelser). Juni har ingen bullish
PDH/H4-hendelse og bekrefter derfor ikke den opprinnelige hypotesen. Første
fit bygget bullish forventning i bearish-H4/PDH-gruppen på bare åtte hendelser;
dette bidro til de ekstra inngangene og tapet i 2022–23.

Det konkrete problemet i dette surrogatet er at et positivt, usikkert gruppesnitt
brukes direkte som forventning. Det er ikke bevis på at all situasjonsbasert
Entry er umulig. Bearish-gruppen fjernes ikke i etterkant av juni-resultatet.
Ingen ny straff, terskel eller annen modell er tilpasset denne feilen.

Ingen årsperiode har en sikkert positiv paret forbedring i den rapporterte
uke-HAC-kontrollen. Juni har bare tre hendelser og fem kalenderuker: maskinens
normalapproksimerte intervall skal **ikke** tolkes som pålitelig signifikans.
Den økonomiske kostpolicyen er et arkivert scenario, ikke bekreftet meglerfasit.

## 4. Native valg på de samme 299 hendelsene

Én CPU-kontroll fullførte med rc0 kl.13:35:02 UTC /15:35:02 Oslo på
167,19 sekunder. Eksisterende native konstruktør, datasetteier,
sekvensrekonstruksjon og kausale MTF-vinduer ble gjenbrukt. Ingen fit eller Exit
ble kjørt. Modellen ble målt på alle 299 hendelser og 299 H4-bull-kontroller,
matchet på kalendermåned og UTC-beslutningstime. Kontrollene ble valgt med en
fast rad-hash uten tilgang til utfall; de erstatter ikke forrige fullvektede
referanse. Åtte tidligere TRAIN-rader er beholdt som paritetsankere.

| Lagret modelltilstand | PDH/H4 LONG / SHORT / FLAT | Matchet kontroll LONG / SHORT / FLAT |
|---|---:|---:|
| Original v8-funksjon og checkpoint | 0 / 299 / 0 | 0 / 299 / 0 |
| Gjeldende v10-funksjon, samme originalvekter | 218 / 81 / 0 | 217 / 82 / 0 |
| Etter det fullførte Entry-joint-retningsforsøket | 57 / 242 / 0 | 43 / 256 / 0 |

Gjeldende funksjon med gamle vekter er en målt funksjonsbaseline, ikke en
godkjent modell. Tabellen viser hvorfor lik vekthash ikke betyr lik atferd.
Ingen av tilstandene avstår på disse 598 punktene. Originalmodellen velger
konstant SHORT; den etterfølgende retningslæringen dokumenterer heller ikke
nyttig seleksjon. Et historisk positivt LONG-snitt betyr ikke at hvert SHORT-valg
er feil. Native Q gjelder dessuten Exit-politikken, ikke nødvendigvis h48-målet.
Denne kontrollen alene kan derfor ikke beregne modellens økonomiske forbedring.

Det siste retningsforsøkets **4096 TRAIN-rader inneholdt fem** PDH/H4-hendelser.
Forsøkets egne **512 evalueringsrader inneholdt to**. Dette er et annet utvalg
enn de to eldre økonomikontrollene med null, og tallene skal ikke blandes.

Den tidligere retningsoppdateringen beskyttet Q-headens parametere mens den
delte representasjonen endret seg. Det kan endre Q-valgene uten at Q-headen
har blitt trent mot observerte netto Entry-utfall; det er en kjent mekanisme,
ikke dokumentasjon på at én ny tapsfunksjon automatisk gir prediksjonsevne.

Alle tre tilstandenes vektdigester matchet den fullførte kjøringen før og etter
måling. De åtte ankerpunktenes største CPU/GPU-avvik i Q var henholdsvis
0,000820/0,000910/0,000379 bps, under forhåndsgrensen 0,001; valgene var like.
Modellkilden er byteidentisk med kilden fra det fullførte retningsforsøket.

## Neste nødvendige avklaring

Mål den faktiske LONG/SHORT/FLAT-headen når læringen endrer delt representasjon.
Gjenbruk både den opprinnelige brede kontrollen og denne faste hendelsesgruppen;
ikke erstatt hele markedet med et utvalg historiske vinnere. Eventuell ny
fit må binde treningsdekning, observerte nettomål og referanse før den kjøres.

Det foreligger ennå ikke et dokumentert mål-/samplingstiltak som forbedrer
senere Entry-økonomi. Juni-tapet skal ikke repareres med ettervalgt gruppesletting.
Den frosne bullish PDH/H4-hypotesen trenger uavhengig senere evidens; juni
inneholdt ingen slike hendelser. Ingen ny fit eller større trening er autorisert
av disse resultatene alene. TEST forblir forseglet.

## Evidens og kilde

Alle nye kontroller starter fra clean GX1_ENGINE,
`audit/v9-premiere-20260905`, `2a4ee20e6c6a124e8a586d30e13518e13016f578`.
GX1_CURRENT, native kode, vekter, checkpoints og TEST er bevart.

Evidensrot: `/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/`.

- `PDH_NATIVE_VISIBILITY_20260924`: PLAN/RESULT/TERMINAL og
  `RECONSTRUCTION_DETAIL.json`; første kontroll rc0 kl.13:16:40 UTC.
- `PDH_CAUSAL_STATE_VALUES_20260924`: PLAN/RESULT/TERMINAL, `metrics.csv` og
  privat beslutningscache; rc0 kl.13:21:23 UTC.
- `PDH_NATIVE_COHORT_20260924`: PLAN/COHORT_PLAN/START/RESULT/TERMINAL,
  kildebundet program og private Q-cacher; rc0 kl.13:35:02 UTC.

De to første brukte audit-vakten (4 GiB, 512 MiB swap, én numerisk tråd).
Native-kontrollen brukte eksisterende producer-vakt på CPU med 10 GiB,
512 MiB swap og én numerisk tråd. CUDA ble ikke initialisert. Den eksisterende
native-loaderen materialiserer PRETEST-features til og med juni 2026; bare
TRAIN-rader ble sendt gjennom modellen. Ingen VAL-beslutning eller TEST-rad
ble evaluert i native-kontrollen. Den separate surrogatkontrollen ovenfor
brukte uttrykkelig gjenbrukt utviklings-VAL.

Programmer og aggregater bevares utenfor kanonisk kilde. Mac-kopiene er
SHA-kontrollert. Ingen modellfit eller native optimizersteg er lagt til av
native-kontrollene; surrogatet har bare de erklærte gruppeestimatene.
Ingen offentlig ENGINE-push: stående push-autorisasjon gjelder CURRENT-grenen.
