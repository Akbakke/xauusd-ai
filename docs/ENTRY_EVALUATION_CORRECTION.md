# Entry-evaluering før P0 — 17.10.2026

Baseline, native targets/Exit, produksjonsmodell og TEST er uendret. Separate
forskningsvarianter og kontroller er gjennomført på eksisterende data. Ingen
modelltilpasning på markedsdata, strategisimulering, datakjøp eller nye
markedsdatakall. Arbeidsstatus eies bare av NEXT_RUN_POLICY.json.

## A. Terminalkvote og kalender

Den forhåndsbundne forskningsvarianten krever **0 s forsinkelse i M1-stempelet**:
ingen senere terminalrad erstatter det avtalte tidspunktet. Dette er en streng
observasjonsregel, ikke en estimert optimal terskel eller en maksimal holdetid.
M1-candles dokumenterer ikke faktisk tick-/filltid inne i minuttet.

Alle 652 552 TRAIN-rader er kontrollert fra original råpriskilde. Antall gyldige
rad–horisontobservasjoner (hver inneholder LONG/SHORT/FLAT):

| Observerte sesjoner | Baseline | Eksakt terminal | Ekstra sensurert |
| --- | ---: | ---: | ---: |
| 1 | 651 300 | 648 691 | 2 609 |
| 5 | 650 059 | 646 797 | 3 262 |
| 20 | 646 543 | 642 214 | 4 329 |

Beholdte labels er numerisk uendret. Maksimal tidligere forsinkelse var
50 340 / 48 840 / 50 040 s. Separate Parquet-filer bevarer native_row,
ønsket/faktisk terminaltid, forsinkelse, gyldighet og korrigerte nettoverdier;
sensurerte verdier er NaN, også FLAT. Det er ikke uavhengige handler.

VAL: 1 792 prefiksrader kontrollert; 1/5/20 har 1 792/1 792/1 763 gyldige,
ingen ny sensur. VAL-halen er ikke rekonstruert: original M1-row-group 5
krysser TEST og er ikke dekodet. Sikkert råprefiks: 13.06.2010 22:04 til
09.07.2025 05:33 UTC.

Eksisterende `xau_tape_provenance_v1.scheduled_weekly_closure_mask` gjenbrukes:
2 310 840 av 2 683 330 manglende M1-minutter i råprefikset ligger i ordinær
ukentlig stengning. Ingen gyldige baseline-innganger eller terminaler ligger
i denne stengningen. De øvrige 372 490 minuttene er ikke fullt fordelt mellom
historisk dagpause, helligdag og manglende prisdekning. Fravær alene beviser
verken stengt marked eller tapt feed.

En separat New York-/DST-kandidat teller planlagte ukedagssesjoner selv når
priser mangler, og bruker samme lokale klokkeoffset. Blant felles gyldige
TRAIN-rader endrer den 5 211 / 51 138 / 190 447 brutto-labels ved 1/5/20;
8 284 / 7 571 / 10 772 tidligere gyldige blir ugyldige. Dette er **ikke en
innrømmet historisk kalender**. Dagens [OANDA-tider](https://www.oanda.com/bvi-en/cfds/hours-of-operation/)
og [eldre publiserte tider](https://www.oanda.com/assets/documents/471/Hours_of_operation_6.pdf)
har ulike åpningstider uten en komplett datert regimehistorikk. 20 631 faktiske
M1-rader ligger utenfor dagens template. Å bruke denne bakover som fasit ville
være en ny udokumentert antagelse. Eksakt-terminalvarianten beholder foreløpig
den gamle observerte sesjonsklokken; full kalenderkorrigering gjenstår.

## B. Versjonert økonomi

Den gamle v1-ruten og publiserte resultater er bevart. Ny
`execution_for_horizon_v2` bruker samme native prisformel på begge sider:

```
LONG  = 10000 * (exit_bid - entry_ask) / entry_ask
SHORT = 10000 * (entry_bid - exit_ask) / entry_bid
```

17 565 eksisterende GC/M5-observasjoner er kontrollert ved 30/120/360 minutter.
Klokker og midprislabels er bitlike. LONG-avviket er høyst 1,12e-12 bps fra
regnerekkefølge. Gammel SHORT overvurderte hypotetisk brutto med gjennomsnitt
0,02570 / 0,11532 / 0,40157 bps, maksimum 8,56906 / 6,35849 / 19,21996 bps.

Finansieringskontrollen avdekket at v1/v2 brukte NY-rollover med SHORT-kreditt,
mens den bundne native kostautoriteten bruker veggklokke og klippet kreditt.
**Alle åtte forberedte P0-armer binder derfor v3-kostprojeksjonen**, gjennom
`entry_observed_market_v1.build_entry_observed_market_targets`:

```
netto(side) = inngangsnotional-brutto - 4 bps
             - faktisk elapsed_seconds * annual_cost_bps(side) / 31557600
annual_cost_bps = LONG 569, SHORT 0
```

Spread inngår én gang i bid/ask; 4 bps er to utførelser à 2 bps, provisjon 0.
På de samme 17 565 observasjonene er v3 bitlik native kostprodusent og en
uavhengig formel, uten endret pris-/retningstarget. Mot v2 endres gjennomsnittlig
hypotetisk netto LONG/SHORT med −0,03245/0,00000 bps (30m),
−0,10308/−0,01576 (120m), −0,25280/−0,07775 (360m).
Dette er overlappende muligheter, **ikke strategi-PnL eller Sharpe**.

Projeksjonen er kildebundet, men ikke historisk brokerfasit. Faktiske
[finansieringsvilkår](https://www.oanda.com/bvi-en/cfds/financing-costs/) avhenger
av satser, oppgjør og helligdager; komplette historiske vilkår mangler.

## C. Retning og inngangskvalitet

Terminal netto markout, MFE og MAE er separate felt. Extrema måles fra observerte
M1 bid/ask-OHLC i [inngang, forhåndsbestemt terminal); ingen beste-exit velges.
Separat ex-post-diagnostikk er `netto/(observert MAE + utførelseskostnad)`.
Den er ikke et godkjent risikonyttemål eller et treningsmål; extrema er ufullstendige
ved uklassifiserte datagap. Ingen framtidig MFE/MAE legges i features.

Ny targetbruk krever egen forhåndsregistrert hypotese. Native primært h20-nettomål
og GC/M5s kostnadsfrie mid-log-retning beholdes som forskjellige definisjoner.
Ingen lært Exit er brukt i beregningene.

## D. L1 og åtte P0-armer

Full råpassering: 66 filer, 15 kontrakter, 892 100 712 rader. Alle eksisterende
Trade-metrikker på valgt serie matcher 004 eksakt. Valgt serie har 133 987
gjentatte T-signaturer; alle er identiske i samtlige 20 felt, ingen kollisjoner
med forskjellig BBO/depth/ts_in_delta. Identiske records beviser ikke alene
identiske underliggende handler. Ingen deduplisering eller maskesvekkelse.

Separat valgt L1: 6 527 ugyldige BBO-records, 557 bad-recv og 557 snapshots,
2 resets, 153 190 gapflaggede records og 29 filsømmer. Tallene overlapper.

| Kandidatstøtte på 141 806 faktiske native M5-vinduer | Vinduer |
| --- | ---: |
| Rene observerte BBO-punkter i alle fem minutter | 140 559 |
| Strengere OFI-/tidsintegralkandidat med ren forankring i forrige minutt | 83 776 |

Begge har støtte i alle 26 måneder. Kjent aggressorside kreves ikke. OFI og
BBO har separate maskekolonner; reset, gap, ugyldig BBO, filsøm og rull får
ikke låne forankring fra ugyldig historikk. Dette er kandidat-inputs, ikke
beregnede/kvalifiserte features eller bevis på komplett markedstape.

Manifestet `P0_FACTORIAL_PREPARATION_20261017.json` deklarerer kontroll, F, L,
V, FL, FV, LV og FLV. F gjenbruker signert GC-flow under gamle masker; L
beholder eksisterende GC VWAP/POC/SMC-felt; V gjenbruker 16 eksisterende
volatilitetsfelt. Ingen ny ATR/RV. Forskningskontroll har 272 native felt etter
planlagt pause av 27 mønsterfelt og eksplisitt V-ablasjon; original 315-flate er
urørt. Core beholder ATR-normalisering og prisnivåinformasjon, så V måler
inkrementell eksplisitt tilstand, ikke fravær av all volatilitet.

## Gjenstående blokkeringer og bevis

- Historiske dagtider/helligdager og full kalenderkorrigering; urekonstruert VAL-hale.
- Historisk finansiering og faktisk tick-/filltid; v3 er en deklarert kostprojeksjon.
- Faktiske OFI-/tidsvektede BBO-verdier, reset-/prefixparitet og endelig feature-admission.
- Ekte dynamiske GC session/ankret VWAP og rolling POC/VAH/VAL. Kvantiler er ikke Value Area.
- Eksakt regel-/liten XGBoost-recipe, ferdige inputs/matched kohort og urørt bekreftelsesperiode.

Ingen omfattende trening åpnes. Netto strategiresultat, Sharpe, drawdown,
eksponering og handler er ikke målt i denne bølgen. Nye fokuserte tester dekker
quote-sensur, DST, manglende sesjon, extrema, fullrecord-duplikater, BBO-masker,
notional-/kostsymmetri og alle åtte armdefinisjoner; eksakte sluttall og hasher
ligger i completion review.

Resultater i GC_ORDER_FLOW_RESEARCH_001: `ENTRY_EVALUATION_CORRECTION_20261017_011/RESULTS`
med separat `WEEKLY_CLOSURE_REVIEW`, `GC_SELECTED_L1_QUALITY_20261017_012/RESULTS`
og `GC_M5_CASH_PROJECTION_CHECK_20261017_013/RESULTS`. Alle tre terminaler: exit 0,
kilde uendret under kjøring, TEST=false. Råpriser/labels/masker ligger i RUNS,
ikke Git. Baseline-/kildekontrakter og historiske resultater er hashkontrollert.
