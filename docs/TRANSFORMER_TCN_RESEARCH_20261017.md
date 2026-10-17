# GX1: full featureflate og 26 måneders Transformer/TCN train–smoke

## Resultat og beslutning

Teknisk PASS. Lærings- og økonomikvalifisering: NO-GO for den frosne 128-stegsoppskriften. Alle fem kronologiske fold og 20 fits er fullført én gang, inkludert ti nevrale fits og totalt 1 280 optimizersteg. Terminalmålt kjøretid var 16,105 sekunder under CPU-audit med 4G RAM, 512M swap, én numerisk tråd og prosjektlås. Ingen GPU-/native trening, TEST, broker, handel eller spending. Ingen automatisk videreføring.

Begge nettverk reduserte TRAIN-feilen fra fersk initialisering, men ingen slo TRAIN-gjennomsnittet i noen av de fem foldene. En teknisk kjøring eller fallende initialtap er derfor ikke et bestått læringssignal. 128 steg er en smoke, ikke tilstrekkelig konvergens-/kapasitetsevaluering. Resultatet beviser verken at GX1 må avsluttes, at informasjonen er verdiløs eller at native Transformer bør erstattes.

## Features og produsentkontroll

Oppskriften beholder alle 254 native markedsfelt fra alle åtte genuine familier. 238 av disse manglet i den tidligere 37-felts GC-sekvensoppskriften, inkludert de 27 tidligere pausede forskningsmønsterfeltene. Det nye valget følger brukerens krav om full informativ flate før modellutfall, ikke resultatbasert featureseleksjon. I tillegg inngår eksisterende F12, L8, GC/XAU-basis, faktisk GC-kontraktsvolum og handelsteller, samt native sesjonskategori. 278 råfelt blir 285 kanaler gjennom eksplisitte one-hot-domener for SMC-tilstand og sesjon; kategorier blir ikke behandlet som ordnede tall.

F/L-verdier, GC-volum, handelsteller og basis ble kontrollert mot lagrede ekte M1-data på alle 23 807 fysiske inputrader som sekvensene bruker. OFI/refill/depletion/quoteaktivitet/BBO-integraler og microprice/respons matcher M1-aggregasjonen eksakt. VWAP/POC/VAH/VAL-avstander og reaksjon mot forrige nivå matcher nivå-/close-algebraen eksakt. GC-volum og handelsteller matcher M1-summer; basis matcher GC-close og rå XAU bid/ask-mid. Trades/300-intensiteten er algebraisk duplikat av handelstelleren og dupliseres ikke.

Alle native kontekstaliaser er bitlike sine deklarerte signalfelt. Native råformler gjenbruker den eksisterende feltledgeren og korrigert generasjon; alle 254 råformler ble ikke rekonstruert fra hele råhistorikken på nytt. 285/285 kanaler har ikke-null inputderivert gjennom hver av de to modellene på ekte input, uten vektendring i kontrollen. Dette beviser kobling, ikke informativ eller lønnsom bruk etter trening.

Det eneste konstante feltet i den forbrukte kohorten er `ctx_cont.close_range_observed`: en tilgjengelighetsmarkør som er sann på de valgte observerte radene. Feltet beholdes, men gir ingen variasjon her. 61 makrofelt med uavklart historisk publisering forblir utelatt. Bare 178 sekvenser har streng kjent aggressorside hele veien; signert delta/CVD brukes derfor ikke som komplett modellhistorikk og fylles aldri kunstig.

| Native familie | Råfelt |
| --- | ---: |
| chart_geometry_encoder | 32 |
| momentum_flow_encoder | 35 |
| price_action_candle_encoder | 23 |
| session_regime_encoder | 9 |
| smc_liquidity_encoder | 69 |
| structure_swing_encoder | 30 |
| trend_ema_encoder | 40 |
| vol_compression_encoder | 16 |

## Data, klokker og budsjett

141 806 native studievinduer gir 37 181 kvalifiserte enkeltvinduer og 12 482 sekvenser à seks sammenhengende M5-vinduer × 285 kanaler. Alle 26 måneder har støtte. Ingen imputering, hullkomprimering eller kontraktsrullering inne i en sekvens. Frosne receive-clock/reset/gap/søm-/roll-masker gjenbrukes. Dette er tidssekvenser av MBP-1-avledede M5-metrikker, ikke rå subminutt-eventtensorkjøring.

Target er samme forhåndsvalgte 30-minutters mid-markout som P0. Evaluering og økonomi bruker identiske rader mellom modellene. TRAIN-utfall må lukke før neste fold, med 24 timers embargo. Normalisering tilpasses én gang på hver folds unike fysiske TRAIN-inputrader; ingen evalueringsrefit. Alle 26 måneder er gjenbrukt utvikling. Ny splitting gjør dem ikke til urørt OOS.

| Fold | TRAIN-sekvenser | Evalueringsrader | TRAIN-inputrader |
| --- | ---: | ---: | ---: |
| dev_01 | 7,016 | 646 | 13,391 |
| dev_02 | 7,663 | 435 | 14,828 |
| dev_03 | 8,112 | 732 | 15,857 |
| dev_04 | 8,851 | 2,163 | 17,261 |
| dev_05 | 10,988 | 1,468 | 21,008 |

Liten Transformer: 17 921 parametre, én encoder-layer, bredde32, fire heads, FF64. TCN: 13 473 parametre, bredde32, kernel2, dilasjoner1/2 og gyldige kausale konvolusjoner uten nullfylt markedshistorikk. Begge bruker eksisterende mean-plus-last-readout. Seed20261015, AdamW med tidligere native rate/decay/clip, 128 steg per nettverk og fold. Batch86 = ceil(10 988/128), slik at alle TRAIN-rader faktisk observeres. Ingen HPO, tidligstopp, nye horisonter, terskler, seeds eller ekstra fits ble valgt etter resultater.

To faste ridge-referanser bruker henholdsvis siste snapshot og hele seks-vindu-sekvensen. LightGBM CORE/FLV er originale låste prognoser, uten refit. De har sine opprinnelige 211/247 felt; dette er en låst referanse, ikke lik-input arkitekturattribusjon mot de nye nettverkene. Transformer versus TCN har nøyaktig samme inputs, rader, target og budsjett.

## Prediksjon og økonomi

5 444 felles evalueringsrader. Lavere MSE er bedre. Kalibrering er regresjon av realisert markout på prediksjon, ikke sannsynlighetskalibrering. Pooled TRAIN-konstant kan variere mellom fold; dens pooled rank-IC er derfor ikke en innen-fold retningsprognose.

| Modell | MSE bps² | Rank-IC | Kalibreringshelning | Netto bps | Sharpe | M1-drawdown bps | Handler | Eksponering | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 453.625 | -0.0113 | -3.0643 | 0.00 | — | 0.00 | 0 | 0.00% | 0.00 |
| tree_CORE | 455.261 | -0.0098 | -0.1584 | -109.77 | -0.387 | 287.81 | 12 | 0.21% | 23.99 |
| tree_FLV | 456.367 | -0.0106 | -0.3832 | -129.67 | -0.666 | 221.71 | 3 | 0.18% | 5.99 |
| linear_snapshot | 516.401 | 0.0863 | 0.2759 | -1237.40 | -0.537 | 3115.45 | 496 | 29.88% | 991.97 |
| linear_sequence | 643.928 | 0.0541 | 0.0769 | -5536.03 | -2.360 | 6453.95 | 966 | 44.08% | 1931.95 |
| transformer | 588.528 | -0.0530 | -0.2052 | -2399.26 | -1.085 | 2907.68 | 506 | 25.87% | 1011.97 |
| tcn | 496.953 | 0.0733 | 0.2287 | -4293.03 | -2.336 | 4437.06 | 1051 | 35.98% | 2101.18 |

Økonomien er fast 1× initialnotional uten kompondering, én posisjon, ekte XAU M1 bid/ask og v3-kostprojeksjon. Ingen ny fast holdetidsregel: posisjoner bæres over inputhull og foldgrenser. Terminal mark-to-market og full utgangsavgiftsreserve inkluderer åpne posisjoner. TCN har én åpen posisjon ved slutt; Transformer har null. Samme-risiko alltid-LONG: +1 684,996 bps, Sharpe0,372, M1-drawdown5 319,994 bps; FLAT:0 bps. M1-verdier kan ikke avgrense intraminutt-/intragap-tap.

Sentralt scenario er 4bps round-trip og finansiering LONG569/SHORT0bps årlig, faktisk elapsed wall time. Dette er en deklarert kildebundet kostprojeksjon, ikke historisk brokerfasit eller realisert profitt. I det billigste forhåndsvalgte scenarioet (2bps, ingen finansiering) er Transformer −1 358,342bps og TCN −2 038,282bps. Ingen av de nye lærte modellene er positiv der heller.

Åtte forhåndsbundne pooled kontraster bruker parvis kalenderuke-bootstrap med 4 000 trekk og Bonferroni-korrigering innen prediksjons- og økonomifamiliene separat. TCNs MSE-forbedring mot Transformer er91,575bps², samtidig intervall[−31,620;337,260]: ingen statistisk kvalifisert arkitekturvinner. Begge er dårligere enn låst FLV i MSE (TCN forbedring−40,586,[−78,110;−4,718]; Transformer−132,161,[−359,736;−26,014]). Disse intervallene er betinget på frosne utviklingsfits, ikke uavhengig bekreftelse.

## TRAIN-fit og smoke

| Fold | TRAIN-konstant MSE | Transformer slutt-MSE | TCN slutt-MSE |
| --- | ---: | ---: | ---: |
| dev_01 | 109.873 | 116.554 | 136.498 |
| dev_02 | 113.578 | 121.323 | 141.630 |
| dev_03 | 111.645 | 119.675 | 139.827 |
| dev_04 | 130.101 | 141.110 | 161.464 |
| dev_05 | 246.701 | 260.599 | 305.929 |

Alle ti nettverk forbedrer fersk initial-MSE, men null slår TRAIN-konstanten. Det finnes dermed ikke et kvalifisert TRAIN-læringssignal i denne smoke. Alle fem fold, endelige prediksjoner, alle deklarerte TRAIN-rader og lagring/strict-load/batch-paritet bestod teknisk. Dette er ikke et argument for automatisk flere steg eller en stor GPU-kampanje.

## Native Transformer og h20

Siste faktiske native trening var15.10.2026,09:48:48–10:55:27UTC,4 071 Entry-steg på65 136 unike TRAIN-rader, seed20261016. Den brukte allerede Exit-uavhengig observert h20. 150 Exit-statefelt var uendret. Ny kontroll bekrefter at original-checkpointets SHA256 fremdeles er6bf92cccd258e561b5dad96fe24da981f027a78603401c6a4eb40b1d244886ea. Native har315 signal-/132 kontinuerlige/én kategorisk kontekst, lokal M5-sekvens96, delte familiespesialister og kontekst-/token-gates. Entry bruker M15/H1/H4/D1-kontekst; Exit bruker også M5, M1-path/side/episode og samme bundle. Ingen native modellfunksjon, weights eller Exit er endret.

Eksakt-M1-terminal h20-kandidat er separat kontrollert:642 214 av646 543 baseline-gyldige rader beholdes;4 329 sensureres uten senere quote-erstatning. Historisk sesjons-/kalendersemantikk og COT-publisering er fortsatt uavklart. Dagens [OANDA-tider](https://www.oanda.com/bvi-en/cfds/hours-of-operation/) og [publisert time-PDF](https://www.oanda.com/assets/documents/471/Hours_of_operation_6.pdf) gir ingen komplett datert regime-/helligdagshistorikk. Den korrigerte native h20-modellen er ikke trent; den krever kvalifisering, successor-datasett/normalisering/initialbaseline og campaign-binding. Denne GC/M5-studien er en separat liten encoder-studie, ikke retrening av full native96-bar/MTF-Transformer.

## GO/NO-GO og ubeviste grenser

- GO: full kvalifisert markedsflate, produsent-/klokke-/domene-/alias-/koblingskontroller og teknisk CPU train/smoke.
- NO-GO: økt treningsomfang eller økonomisk/serving-admisjon fra denne 128-stegsoppskriften; læring og edge er ikke kvalifisert.
- Native h20: fortsatt blokkert og ikke nytrent med kalenderkorrigering.
- Arkitektur: behold Transformer som hovedkandidat; ingen målt begrunnelse for å erstatte den med TCN.
- P0s opprinnelige NO-GO beholdes. GX1 er ikke avsluttet. En ny metodeundersøkelse må diagnostiseres og bindes eksplisitt før nye fits.

Merverdi fra nye features er ikke dokumentert av denne studien. TRAIN-fit, utviklingsgeneralisering, referanseverdi, projisert økonomi og framtidig launch er separate grenser. Ingen urørt OOS, faktiske fills, komplett historisk finansiering, native ny h20-trening eller live-/paper-resultat er undersøkt.

## Verifisering og artefakter

42 fokuserte feature-/produsent-/P0-kontroller og16 endelige feltkontraktkontroller bestod; overlappende tester summeres ikke. Native pre-commit-vakter bestod. Etterkontrollen validerte3 035 handler/åpne markeringer mot eksakte rå XAU bid/ask, alle kostscenarioer og siste dagsequity, med eksplisitt FP64-summeringsfeilgrense på hele ledgeren. Alle fem purging-/embargogrenser og originalt native-checkpoint er kontrollert. TEST=false, source_unchanged=true, alle manifest-/artefakthasher matcher. Ingen originalkvitteringer eller checkpoints er omskrevet.

Kvalifisering: `/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_SEQUENCE_ALL_FEATURES_QUALIFICATION_20261017_027/RESULTS`. Train/smoke: `/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_SEQUENCE_ALL_FEATURES_TRAIN_SMOKE_20261017_028/RESULTS`. Atomisk completion review: `GC_SEQUENCE_ALL_FEATURES_TRAIN_SMOKE_20261017_028/COMPLETION_REVIEW/REVIEW.json`.

Implementasjonen gjenbruker eksisterende PyTorch/native readout, økonomi- og publiseringseiere. TCN-prinsipp: [Bai, Kolter og Koltun](https://arxiv.org/abs/1803.01271); encoderprimitive: [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html). Installerte versjoner og eksakt feltsortering er frosset i027-planen.
