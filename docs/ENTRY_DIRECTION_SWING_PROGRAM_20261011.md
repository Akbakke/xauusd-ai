# Entry-retning på swing — program og tanke, 11.10.2026

Dette dokumentet bevarer resonnementet bak retningsvalget 11.10.2026, slik at
det ikke må rekonstrueres. Status og bindinger eies av NEXT_RUN_POLICY.json;
hvert steg forhåndsregistreres i configs/research før kjøring.

## Operatørvedtak 11.10.2026

- Swing på D1-klokke med 5 og 20 dagers horisont er primærspor («Ja, 1»).
- Regel 1 er åpnet for navngitte instrumenter som forskningsinputs og
  forskningsmål. Native Entry/Exit og serving forblir XAUUSD til eget vedtak.
- Én GPU-jobb og én CPU-forskningsjobb kan gå samtidig.
- OANDA-API brukes som manifestbundet lesetilgang til historiske candles for
  navngitte instrumenter. Ingen ordre, ingen kontoendring.
- Candlestick-/mønsterfelt slettes ikke. De holdes utenfor D1-forskningens
  inputsett. Native beholder alle åtte familier; treningstiden der bestemmes
  av sekvenslengde, MTF-vinduer og modellstørrelse, ikke av disse feltene.
- Målet, i operatørens ord: «når modellen er sikker skal den ta, usikker =
  flat … Det er retning = vi traff riktig som teller.»

## Hvorfor dette og ikke mer av det samme

Målt (klasse: målt på ekte deklarerte data):

- Pris alene på 5 min til 8 t gir retningskorrelasjon 0,01–0,03 for ridge, HGB,
  modellfrie regler og den native transformeren. Rundtur-kost er 5,8 bps.
  Under argmax(Q_L, Q_S, 0) kreves korrelasjon over 0,1 for at mer enn 2 % av
  radene handles. FLAT er riktig svar ved 95 minutter.
- Ved 1 dag til 4 uker er en korrelasjon på 0,03–0,05 økonomisk synlig, men
  ett instrument over 15 år gir MDE rundt 5 %/år mot kjøp-og-hold. Styrke må
  komme fra pooling over flere instrumenter og fra forhåndsvalgt evidensklasse.
- Scalping lykkes hos andre på futures-kost (ca. 0,4 bps rundtur, ikke
  verifisert av prosjektet), med ordrestrøm og sekunders latens. Prosjektets
  målte latens var 15–19 minutter. Dette er et annet prosjekt.
- Transformeren er aldri trent en full epoke på dagens kilde. Innvendingen
  «for lite trent» lukkes med en ekte kjøring på multihorisontmål, ikke med
  flere 256-stegs smoker.

## Endepunkt

Den forhåndsregistrerte selektive protokollen: treffprosent og netto bps på
handlene som tas ved dekning 100/50/25/10/5/2/1 %, mot myntkast-null og
sirkulær-shift-null, pluss rank-IC, netto mot kjøp-og-hold med tidsvarierende
finansiering, MDE og INKONKLUSIV-klasse. Ingen terskel velges etter resultatet.

## Modell i lag (D1)

1. Driverlag: elastisk nett og HGB med monotone begrensninger der mekanismen er
   kjent (gull ned når realrente opp, gull ned når USD opp), kvantiloutput.
2. Bayesiansk lag: hierarkisk pooling av trend-, carry- og makrosensitiviteter
   over panelet gull, sølv, platina, kobber, olje; gull som eget nivå krympet
   mot panelet.
3. Tilstandslag: Kalman-trend med usikkerhet og regime-HMM.
4. Handlingsregel: ensemble med krymping og konform kalibrering. Handle når
   E[netto] > kost med margin og P(riktig fortegn) over terskel valgt på TRAIN
   via dekningsgrid; «usikker = flat» er en del av modellen, ikke et veto.
5. Senere: sekvensmodell på D1 i kapasitetsstige, distillasjon fra ensemblet.

## Inputs (alle D1, alle manifestbundne før henting)

- Drivere: DFII10, DTWEXBGS, T10YIE, DGS2, DGS10-DGS2, VIXCLS; endringer 1/5/20 d.
- Posisjonering/aktivitet: COT netto spek og OI, GLD-beholdning, daglig
  GC-volum og ΔOI (CME; kilde må kvalifiseres av probe).
- Panel: XAG, XPT, XCU, olje, SPX500, US10Y, EURUSD, USDJPY; avkastning
  1/5/20 d, gull/sølv-ratio.
- Pristilstand fra dagens flate på D1: TSMOM 21/63/126/252 i ATR, 252-dagers
  range-posisjon, ATR-regime, nivåavstander, sweep-/retest-felt på D1/W1.
- Kalender: avstand til FOMC/CPI/NFP, sesong.

Provenanskrav: urevidert serie pluss dokumentert publikasjonsklokke og
konservativt lag oppfyller regel 1 for COT, GLD, VIX og CME-aktivitet.
ALFRED-vintages kreves bare for serier som faktisk revideres.

## Rekkefølge

0. HEAD-rettelser (modellfunksjon v10, objective v11, checkpoint-policy v6–8),
   regel 1, parallellitet, OANDA-lesetilgang og pausevedtak bundet.
1. Kildeaudit for drivere, posisjonering, aktivitet og panel.
2. Forhåndsregistrering ENTRY_DIRECTION_SWING_D1 og kjøring på CPU.
3. Re-score de 61 intradagcellene fra 27.09 under futures-kost. GO gir eget
   vedtak om futuresbroker, L2-data og latens.
4. GPU parallelt: multihorisontmål og full trening av transformer v10 med
   CONTROL4096 rank-IC per epoke, 3 seeds, forhåndsregistrert stoppregel.
5. Signal → ensemble og kalibrering → skyggejournal uten ordrer i måneder →
   distillasjon. Første handel etter skyggejournal, ikke etter backtest.

## Opprydding

Importgraf-scan 11.10 fant åtte moduler (980 linjer) uten importører eller
referanser utenfor samtaletranskripter; de slettes i egen commit. Git er
gjenopprettingskilden.
