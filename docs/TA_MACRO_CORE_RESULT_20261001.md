# MACRO_CORE: fullført måling, inkonklusiv effekt — 01.10.2026

Pris pluss realrente, bred USD og breakeven har ikke dokumentert robust merverdi.
Begge forhåndsregistrerte modeller fikk INKONKLUSIV. HGB ble bedre enn sin
matchede prisbaseline, men var nesten lik alltid-LONG. Ridge ble svakere.
Ingen native innføring eller videre tuning godkjennes av dette resultatet.

## Målt på ekte, gjenbrukt utviklingshistorikk

1761 felles inputrader fra mars 2019. Årlige fits 2020–2025 bruker samme
kausale TRAIN-/hold-rader i begge armer, med purging av framtidsutfall.
1529 prognoser gir 1528 sammenhengende porteføljeintervaller fra
01.01.2020 23:00 UTC til 01.12.2025 23:00 UTC. Siste 20 D1-rader er utelatt
fra begge horisonters prognosepopulasjon for tilgjengelige utfall.
Dette er kronologisk walk-forward i gjenbrukt utviklingshistorikk, ikke urørt OOS.

Primær horisont: 20 observerte D1-rader. Tallene nedenfor er total nettoendring
i startkapital over hele perioden, etter executable BID/ASK-spread, 2 bp slippage
per utførelse og historisk finansieringsproxy. Samme kausale risikostørrelse
og kapitalbudsjett brukes; dette er ikke årlig avkastning.

| Modell | Pris alene | Pris + makro |
| --- | ---: | ---: |
| Ridge | +57,50 % | +50,17 % |
| HGB | +56,56 % | +66,77 % |

Samme-risiko alltid-LONG og konstant lært fra samme TRAIN gir begge +66,42 %.
HGB med makro ligger dermed bare 0,35 prosentpoeng over LONG totalt.
Rå kjøp-og-hold gir +154,49 %, men høyere realisert årlig volatilitet
(16,70 % mot 9,16 % for HGB med makro); denne er diagnostisk, ikke risikomatchet.
Alle åpne posisjoner er sluttlikvidert og medregnet.

HGB med makro: 1469 LONG og 59 SHORT av 1528 intervaller (96,14 % LONG).
Det er ikke dokumentert selvstendig selektiv handelsverdi av positiv samlet PnL.

## Usikkerhet og beslutning

120 forhåndsbestemte endepunkter i én felles familie; 1999 paired stationary
bootstrap-trekk, middelblokk 60 D1-rader, eksisterende effektgrenser.
Ingen utestbare endepunkter.

- HGB mot matchet A: +0,386 bp per porteføljeintervall,
  simultant intervall [-2,983; +3,755].
- HGB mot LONG: +0,014 bp, simultant intervall [-1,454; +1,481].
- Ridge mot matchet A: -0,312 bp, simultant intervall [-2,996; +2,371].

HGBs estimerte minste detekterbare forskjell mot LONG ved deklarert styrke er
1,85 bp/intervall, over den primære relevante effekten på 1 bp.
Både ridge og HGB er INKONKLUSIV. Dette er verken en GO eller et bevis på at
makro aldri kan være nyttig. Det gir ikke grunnlag for å endre terskler,
velge nye perioder eller prøve mange alternative felt på de samme utfallene.

## Kilder og verifikasjon

DFII10/T10YIE nivå + 21-D1-endring, DTWEXBGS loggnivå + 21-D1-endring.
DTWEXBGS er bred USD, ikke ICE DXY. Alle historiske versjoner er hash-bundet.
Konservativt lag: publikasjonsdagens slutt i New York, så en hel faktisk
kanonisk XAU-sesjon. USD-arkivets publisering er omtrent ukentlig;
median observasjonsalder etter dette laget er 7,92 kalenderdager.

En separat råarkiv-orakelberegning verifiserte alle seks felt på alle 4518
klokker, uten prisutfall. En separat kontantbokberegning fra utførte BID/ASK-
endringer og eksplisitt integrert finansiering verifiserte 32 porteføljeløp,
alle 1528 intervaller, risikostørrelse, 40 gjennomsnittskontraster og 12
kronologiske fold-/målbindinger. Ingen modell ble refittet for verifikasjonen.
Bootstrap-usikkerheten ble ikke beregnet på nytt av en uavhengig eier.
49 fokuserte mekanikk-/integrasjonstester og commit-vakten består.

Kildecommit 3a1ef65b; målecommit 7c2915a8.
Forhåndsregistrering: configs/research/TA_MACRO_CORE_PREREG_20261001.json.
Aggregerte verdier, fulle hasher og interne artefaktstier:
[maskinrapport](TA_MACRO_CORE_RESULT_20261001.json).
Originale inputs, resultater og receipts er bevart; TEST er ikke brukt.

## Videre aktivt mål

Den separate makroarmen er ferdig målt og skal ikke relanseres.
Full B har fortsatt sine opprinnelige VIX/GLD/COT-blokkeringer.
Neste arbeid er native v38: bind faktisk læringsmål, sammenligningsrader og
hvilke uendrede cacher som lovlig kan gjenbrukes før ny dataset-/normalisering.
Den gjeldende Entry-Q-kontrakten bruker første tilstandsverdi fra frosset Exit;
den er ikke identisk med D1-forsøkets observerte prisutfall. Det må tas eksplisitt
hensyn til i native målingen. Ingen modell-/tapendring eller tung nybygging
startes som om denne forskjellen var løst.

Native læringsverdi, senere uavhengig generalisering, ny train/serve-paritet
og offline ordre-/gjenstart-/avstemmingskvalifisering er fortsatt ubevist.
Målet forblir aktivt, og eksisterende trenings-/TEST-/handelsvakter består.
