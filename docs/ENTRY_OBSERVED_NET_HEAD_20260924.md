# Entry-head med observert nettofeedback — 24.09.2026

## Resultat og beslutning

Én forhåndsavgrenset analytisk tilpasning av et beslutningshode er fullført.
Den bruker den eksisterende, frosne native Entry-representasjonen og observerte
netto prisutfall. Ingen encoder, Exit, kanonisk modellkode eller produksjonsvekt
er endret. Forsøket er en oppfølging av
[PDH/native-kontrollen](ENTRY_PDH_NATIVE_LEARNING_20260924.md).

På samme 512 senere juni-punkter forbedres punktresultatet fra −17,705 til
+0,168 netto bps per mulighet, med 192 LONG /176 SHORT /144 FLAT. Referansen
her er **gjeldende v10-funksjon med originalvektene**, ikke original v8-funksjon.
Alltid SHORT ga +3,866 bps på samme observasjoner. Den nye headen består derfor
ikke den samlede forhåndsfastsatte porten. Positiv forventet nettoavkastning og
bedre native livsløpsøkonomi er fortsatt ikke dokumentert.

Koeffisientene beholdes som forskningsartefakt. Ingen etterjustering av terskel,
gruppe, lambda eller horisont; ingen promotering eller automatisk større fit.

## Hva som ble endret i forskningsforsøket

Den vanlige Entry-Q-treningen bruker frossen Exit-policyverdi i spreadinkluderte
brutto bps. De tidligere direkte retningsforsøkene brukte et hjelpehode, med
frosne Q-head-parametere og endret delt representasjon. Her læres i stedet
selve handlingsverdiene fra observerte netto utfall:

- Frosne 128 Entry-hidden-verdier fra samme v10-funksjon/checkpoint som før.
- LONG og SHORT: neste M5 ask_open/bid_open, mark ved det allerede fastlagte h48,
  spread, arkiverte 2 bps per utførelse og faktisk finansiering. FLAT er eksakt0.
- Én lineær head med upenalisert intercept. Samme automatiske Ledoit–Wolf-
  regulariseringsmetode som tidligere, beregnet bare på det brede TRAIN-utvalget.
  Ingen parameter- eller modelljakt. Lambda ble0,0014383744443006148.
- Unik argmax LONG/SHORT/FLAT; eksakt likhet gir FLAT. Ingen prosentilfilter.
- 1024 eksisterende brede TRAIN-rader, hvorav2 er PDH/H4, supplert med de øvrige
  hendelsene: **1321 unike fit-rader, inkludert alle299 hendelser**.
- Hendelser får vekt1; de1022 øvrige fit-radene får vekt306,360078 hver.
  Vektsummen313399 bevarer de to gruppenes TRAIN-frekvens. Dette er en vektet
  kontroll fra et deterministisk utvalg, ikke et randomisert forsøksdesign.
- Alle fit-utfall og49-bars purge er før VAL. Koeffisientene ble lagret og
  SHA-bundet før senere VAL-mål ble materialisert. Juni er gjenbrukt utvikling.

Dette er eksplisitt **verdi ved et observasjonstidspunkt**, ikke full livsløps-Q
under native Exit. Headens nye betydning er ikke skrevet inn i native-kontrakten.
Det er heller ikke innført tvungen EXIT, maksimal holdetid eller tapsgrense.

## Før/etter på identiske observasjoner

Tallene inkluderer alle valgte innganger og tap. De er netto under den arkiverte
kostpolicyen, ikke bekreftede meglerbetingelser eller porteføljeavkastning.

| Utvalg / head | LONG / SHORT / FLAT | Bps per mulighet | Bps per valgt inngang |
|---|---:|---:|---:|
| Bred TRAIN, uendret v10-head |747 /277 /0|−5,029|−5,029|
| Bred TRAIN, netto-head |353 /302 /369|+6,667|+10,423|
| Hendelses-TRAIN, uendret v10-head |218 /81 /0|+1,140|+1,140|
| Hendelses-TRAIN, netto-head |108 /103 /88|+4,218|+5,977|
| Juni-VAL, uendret v10-head |345 /167 /0|−17,705|−17,705|
| **Juni-VAL, netto-head** |**192 /176 /144**|**+0,168**|**+0,234**|

TRAIN er tilpasning, ikke validering. På hendelses-TRAIN ga alltid LONG
+5,266 bps per mulighet, bedre enn netto-headens+4,218. Alle299 inngår i fit.

Juni-referanser på de samme512 punktene: alltid LONG−15,319, alltid SHORT+3,866,
alltid FLAT0. TRAIN-tilpasset konstant head velger også alltid FLAT.
Ingen valgt inngang manglet fill eller hadde uferdig h48-observasjon ved grensen.
Kontrollen kunne inkludere slike observasjoner med grensemark, men målte ingen.
Maksimalt5 samtidige observasjonsvinduer i juni; netto-headens samlede observerte
eksponering var1840 notional-timer mot referansens2589. Ingen native Exit ble kjørt.

Den forhåndsfastsatte dagblokk-bootstrapen (27 dager,2000 resamplinger,seed0)
ga følgende intervaller for forbedring per mulighet, justert for fire referanser:

| Referanse | Punktforskjell, bps | Justert intervall, bps |
|---|---:|---:|
| Uendret v10-head |+17,873|[+8,483; +29,448]|
| Alltid LONG |+15,486|[−2,061; +33,277]|
| Alltid SHORT |−3,698|[−22,188; +16,299]|
| Alltid FLAT |+0,168|[−7,663; +8,332]|

Dette er utviklingsdiagnostikk på én gjenbrukt måned. Dagblokker løser ikke all
avhengighet mellom overlappende vinduer og markedsregimer. Intervallene er ikke
uavhengig bekreftelse eller garanti for senere resultat.

## Hva forbedringen består av, og hva som fortsatt svikter

En ren etterberegning på de frosne valgene, uten ny kandidat eller fit, gir:

1. Bytt bare retning og krev fortsatt inngang på alle punkter:−1,998 bps,
   forbedring+15,707 mot uendret v10-head.
2. Bruk deretter netto-headens FLAT-valg:+0,168 bps, ytterligere+2,166.
3. De144 avståtte inngangene ville med headens foretrukne retning gitt
   −7,702 bps i snitt. Dette viser en nyttig forskjell i det observerte utvalget.

Svakheten er fortsatt verdiestimatene: på de368 valgte inngangene forventer
headen i snitt**+13,574 bps**, men observerer**+0,234**. Senere feil mot det
faste vindusmålet er også høyere enn konstantmodellen (MSE3607,39 mot3424,98).
TRAIN-forbedringen alene dokumenterer dermed ikke generalisering.

Tvunget LONG/SHORT fra netto-headen velger den bedre executable siden med
balansert treff52,58 %, mot44,83 % fra uendret v10-head. Dette sammenligner
de to kostnadsjusterte sideutfallene, ikke et separat mål for rå close-retning.
Det er en beskrivende etterkontroll, uten ny retningsport eller horisontvalg.

Neste avklaring gjelder hvorfor verdiestimatene ikke holder fra TRAIN til senere
data. Gjenbruk de låste prediksjonene/representasjonene til dette; ikke tilpass
en ny korreksjon til juni. Bedre nettokalibrering og senere økonomisk seleksjon
gjenstår før større trening er begrunnet.

Brukeren har etter resultatet bedt om bredere dekning av featurekombinasjoner.
Det lineære hodet kan bare lese ut samspill som allerede finnes i den frosne
representasjonen. PDH/H4 beholdes som kontrolltilfelle. Den eksisterende bredere
læringen må gjennomgås før flere enkeltoppsett: Claude prøvde blant annet HGB
på316 snapshotfelt og1076 snapshot/MTF-felt. Regelanalysen var36 oppsett ved
fire horisonter (144 celler), ikke144 uavhengige mønstre. Den senere parede
målereparasjonen gjaldt ridge; HGBs rettede fit-/målekjede er ikke tilsvarende
etterkontrollert. Dette er en konkret avklaring, ikke autorisasjon for et bredt søk.

## Pris-/feature-kontrollen er også fullført

Kontrollen fra23.09 hadde stoppet på clean-repo-assert før datalesing. Den ble
bundet på nytt til gjeldende kilde og eksakt TRAIN-grense. Alle313399 rader ga
bitlike `ret_1`, `ret_20` og `rvol_20` både mot native kildefelt og uavhengig
gjenberegning fra source-close. Null fortegnsavvik og ingen manglende historikk.
Dette avviser en pris-/formel-/radforskyvning for disse tre feltene; det er ikke
en full sertifisering av alle features, sekvenser eller tidsrammer.

## Bevis og drift

Kjørekilde: clean GX1_ENGINE/audit-v9-grenen,
`b23ebba8f4b91708baefdc4951c066c8d1a080f3`. GX1_CURRENT er urørt.

- `ENTRY_OBSERVED_NET_HEAD_20260924`: PLAN, FIT_BINDING, EXTRACTION,
  FIT, RESULT, DECOMPOSITION, begge START/TERMINAL, aggregater og private cacher.
  Ekstraksjon rc0 kl.13:46:21 UTC,71,62 s; fit rc0 kl.13:48:25 UTC,2,98 s.
- `ENTRY_CORE_FEATURE_PRICE_ALIGNMENT_20260924`: PLAN/START/RESULT/TERMINAL,
  instrument og logg; rc0 kl.13:52:29 UTC,16,20 s.

Alle ligger under `/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/`.
Åtte native ankerpunkter hadde bitlike hidden/Q-verdier. Ingen modellvekt endret
seg under ekstraksjonen. FP32-headen avvek maksimalt0,000071 bps fra løsningen
beregnet i dobbel presisjon. Native-loaderen materialiserte eksisterende PRETEST-
features inkludert juni, men bare TRAIN-beslutninger ble kjørt gjennom modellen.
Eksisterende juni-representasjoner ble gjenbrukt i evalueringen.

Én tung jobb av gangen, eksisterende CPU-caps (ekstraksjon10 GiB, øvrig4 GiB,
swap512 MiB, én numerisk tråd), ingen CUDA eller nye native optimizersteg.
Ingen TEST, live/papirhandel, spending eller full epoch/VAL. Ingen ny testsuite;
de beskrevne kontrollene ble gjort i selve avgrensede målingen.
Mac-kopier av kode, planer og aggregater er SHA-kontrollert; ingen offentlig
ENGINE-push. Råcacher og koeffisienter blir i den private evidensroten.
