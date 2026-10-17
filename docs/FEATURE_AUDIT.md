# XAUUSD: strategisk feature- og gaprapport — 17.10.2026

**Beslutning:** M5 intradag med eksisterende MTF. P0 (dynamiske nivåer × GC-flow ×
volatilitet) er hovedhypotesen; P1–P5 bygges og ablateres i den bestilte rekkefølgen.
Denne bølgen er analyse og baselinefrysing. Ingen ny featurekode, fit, økonomikjøring,
SDK-installasjon, datanedlasting, sletting eller TEST-lesing er gjennomført.
Arbeidsstatus og kjøregrenser eies fortsatt bare av NEXT_RUN_POLICY.json.

## 1. Hva vi allerede har

Inventar: 707 sporede filer, 583 Python-filer syntaksanalysert uten feil,
325 testfiler; alle 23 Markdown-filer indeksert (16 under docs). Import-/definisjons-
indeks og ordnede felt ligger i [feltinventaret](FEATURE_AUDIT_INVENTORY_20261017.json).
Gjennomgangen følger de relevante beregnings-, dataset-, modell-, beslutnings- og
forskningsrutene. Dette er ikke et nytt linjevis matematisk bevis for hele repoet
eller en ny full gjenberegning av historiske markedsdata. Uendrede ekte bevis fra
featureauditen 10.10 og GC-kjøringene 004/006/007 gjenbrukes med tydelig omfang.

| Flate | Faktisk dekning og bruk |
| --- | --- |
| Native signal | 315 ordnede felt, 96 M5-steg, 132 kontinuerlige kontekstaliaser og session_id. Aliasene er allerede blant de 315; de er ikke 132 nye signaler. Feltorden matcher den faktiske TRAIN-manifesten. |
| Lærte familier | Struktur/swing 30, SMC/liquidity 69, EMA/trend 40, volatilitet 16, momentum/aktivitet 35, session/regime 70, geometri 32, rå candlegeometri 23. Summerer til 315. |
| MTF | 190 ordnede felt per native klokke; Entry bruker M15/H1/H4/D1, Exit inkluderer M5. Separate sekvenser beregnes på lukkede candles. |
| Nivåer/SMC | Pivot-/nivå- og trendlinjeregistre, touch/reaction/retest, BOS/CHoCH, pris-sweeps, historiske ekstremumavstander og sweep-ankret aktivitets-VWAP. Dette siste bruker OANDA-prisoppdateringer, ikke GC-handelsvolum. |
| Volatilitet | Wilder ATR, pris-/returvolatilitet, Parkinson, squeeze/kompresjon, spread og tid/session. Sizing-/risikokontrakter finnes; de har ikke retningsautoritet. Ikke bygg duplikat-ATR. |
| Makro | 61 eksisterende felt: 4 COT, 44 kryssaktiva, 13 Treasury/VIX/GLD-prisdrivere. De rutes gjennom session_regime_encoder. «Niende kontekstfamilie» i dokumenter betyr ingen niende selvstendig encoder i dagens kode. GLD-pris er ikke ETF-strøm. |
| GC-sidefil | 141 806 native M5-koordinater, 255 kolonner, 11,8 MB. Ekte volum, VWAP/POC/volumkvantiler, delta, relativ CVD og prisnivåbaserte footprint-sammendrag; kausale SMC-pris-/order-block-proksier. Fullt prisnivåinventar er bevart i 15 sparse laddere. Ikke konsumert av native Transformer/Exit ennå. |

Transformer bruker samme deklarerte familier, lærte familie-/TF-vekter, normalisering
og delte Entry/Exit-encodere. Offline fitted-Q-kontraktene tilhører samme bundle;
det finnes ingen separat aktiv RL-agent. Siste native minimumsfase trente bare
Entry på observerte D1-utfall, og kvalifiserte ikke videre læring. Det er ikke en
trent M5+GC-modell. XGBoost er ikke installert og har ingen call-site i beholdt
kilde. Tidligere tre-referanser er LightGBM og sklearn HistGradientBoosting.

Aktive sperrer gjelder kildeidentitet, klokke, warmup, gyldighet, splitcontainment
og entydig modellhandling. Lærte specialist_gate/tf_gate er modellkomponenter.
Ingen håndskrevet trend-/session-/konfidensterskel får veto over modellens unike
Q-argmax. Live er stengt. Retnings-Q i bps er heller ikke en kalibrert sannsynlighet.

## 2. Datadekning, kvalitet og mangler

**36 måneder er operatøroppgitt; lokalt dokumenteres 26:** nedlastingsplan,
79 fullførte batchjobber og kvitteringer gjelder mai 2024–juni 2026. Definition/
dagsvolum har støtte fra april 2024. Under den bundne GC-kilderoten finnes 66 MBP-1-
gzipfiler, 20 038 896 092 bytes, 15 outright-kontrakter. Til sammen finnes 112
kanoniske gzipfiler med kvitteringer når støttefilene tas med. Ingen separat
36-månedersleveranse ble funnet i dette avgrensede inventaret. Dette er et
avstemmingsgap, ikke tillatelse til å kjøpe eller hente på nytt.

004/006 har avstemt 892 100 712 kilderader, hvorav 62 783 740 Trade-rader.
Bokhendelser og passive F-fills er ikke ekstra aggressorhandler. B/A er kjøps-/
salgsaggressor på T; N forblir ukjent. 004/006s fulle råhashkontroll gjenbrukes;
20 GB rådata er ikke skannet på nytt i denne analysen.

| På de 141 806 faktiske native M5-radene | Kvalifisert støtte |
| --- | ---: |
| M5 volum/profil | 88 834 (62,64 %), alle 26 måneder |
| M5 full kjent aggressorside / footprint / delta / relativ CVD | 16 745 (11,81 %), alle 26 måneder |
| M15 volum/profil | 47 679 M5-rader / 15 974 unike M15-barer |
| H1 volum/profil | 6 987 M5-rader / 607 unike H1-barer |
| H1 signert flow | 36 M5-rader / bare 3 unike H1-barer |
| H4 volum / D1 volum | 1 unik H4-bar / 0 D1-barer |

Dette er maskert observasjonsdekning, ikke bevist komplett markedstape. Det
valgte M1-universet har 95 028 171 observerte kontrakter, hvorav 2 254 209 med
ukjent aggressorside. 153 190 gapflaggede records og 133 987 gjentatte T-signaturer
inngår i kvalitetsdiagnostikken; dette er recordtall, ikke antall dårlige minutter.
En flagget record kan stenge en hel M1-blokk og dermed en komplett høyere TF.
Årsakene må dekomponeres før maskene vurderes på nytt. Ingen krav er svekket her.

Rullering er kausal: fullført foregående UTC-døgns observerte volum velger blant
15 nedlastede kontrakter; 60 sekunders tilgjengelighetsforsinkelse, ingen retur
til tidligere utløp eller bytte ved manglende data samme dag. Ekte prefixkontroll
består. Kontraktsutvalget ble likevel fastlagt før denne rulleringen; det beviser
ikke at alle mulige historiske kontrakter var tilgjengelige eller at rullen er
optimal. Native tidsrammer og GC-vinduer må ha samme avtalte tilgjengelighetsklokke.

GC-priser beholdes i egne kontrakter/ticks. Ingen backjustering av handelsnivåer,
volum eller footprint. Basisfeltet bruker GC-siste handel mot XAU-mid fra samme
sist lukkede M1-vindu; det er ingen synkron tick-/utførelsespris. P2-nivåer bør
først uttrykkes som relative avstander i GC, eller oversettes med en dokumentert
kausal basis. Absolutte GC-nivåer skal ikke brukes direkte som XAU-ordrepriser.
CME-historiske helligdager/sessionfullstendighet er fortsatt ukvalifisert.

Makro-as-of er kodet med COT-lag 7 dager, lukkede paneldøgn og drivere etter
D+1 New York. Dette er deklarerte lag, ikke et bevis for hver faktisk publisering
eller opprinnelig vintage. Avvikende COT-utgivelser og senere revisjoner må
kontrolleres før P5 kan hevdes å oppfylle faktisk informasjonstilgjengelighet.
Se [CFTC-publisering](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm)
og [ALFRED-vintages](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html).

## 3. Hva MBP-1 bør utnyttes bedre til

[MBP-1](https://databento.com/docs/schemas-and-data-formats/mbp-1) har BBO-priser,
størrelser og antall ordre ved beste nivå, samt action/side, event-/receive-klokke,
sekvens, ts_in_delta og kvalitetsflagg. Den eksisterende structural audit leser
BBO-priser/størrelser for gyldighet, men GC-featurebyggeren bruker Trade-feltene.
Følgende mangler som kvalifiserte features på den faktisk valgte kontraktserien:

- **P1:** hendelsesbasert L1-OFI, tidsvektet spread/ubalanse, vektet BBO-mid,
  beste-nivå-størrelse/ordreantall, quote-/handelsintensitet, depletion/refill-
  proksier og prisrespons. Aggressor delta/CVD finnes og gjenbrukes.
- **P2:** ekte GC session-/ankret VWAP, rolling profil/POC og forhåndsdefinert
  VAH/VAL, foregående sessionnivåer/opening range og volum-/flowbekreftet reaksjon.
  Nåværende q25/q75 er volumkvantiler, ikke VAH/VAL. Helt sessionprofil/CVD kan
  ikke påstås når masker bryter kontinuiteten.
- **P3:** gjenbruk ATR/RV/squeeze/session; undersøk GC-spread/tilgjengelig L1-
  likviditet som ny tilstand. Signalverdi måles først ved fast risikobudsjett;
  dynamisk sizing er en separat sammenligning.
- **P4:** gjenbruk laddere til sided imbalance, delta-divergens og kontinuerlige
  prisrespons-/absorpsjonsproksier ved nivåene. Ikke tell samme delta/CVD to ganger.
- **P5:** gjenbruk eksisterende 61 felt. Faktiske ETF-holdings-/strømendringer og
  bred USD-driver mangler; dataversjon og tilgjengelighet må kvalifiseres først.

OFI skal bygge på BBO-endringer, ikke bare summen av A/C/M-hendelser. Fullført
instrument-event, klokke, reset/snapshot, tom/krysset bok, gap og rull må ha egne
regler. Trade-maskenes ADMIT kvalifiserer ikke automatisk book-OFI. Hver familie trenger
sin egen maske: BBO-OFI krever ikke kjent aggressorside, og usignert profil skal
ikke miste gyldige N-handler. Rapporter både felles sammenligningsutvalg og
bred tilgjengelighet; ikke tving alle P1/P2-felt ned på footprintets 11,81 %.
[Cont–Kukanov–Stoikov](https://arxiv.org/abs/1011.6402) er en relevant L1-metode,
ikke bevis for gevinst i gull på M5. Et størrelsesvektet BBO-mid må skilles fra
en estimert betinget microprice-modell. MBP-1 viser ikke L2/L3, skjult likviditet,
individuell ordrehistorikk eller vår faktiske køposisjon. Absorpsjon forblir proxy.

## 4. Gratis ressurser: konkret merverdi og kostnad

Kontrollert 17.10 via primærdokumentasjon, GitHub-kode, lisens og repo-/tree-
metadata. Test-/CI-tilstedeværelse er inspisert; eksterne testpakker er ikke kjørt.
Kode-lisens gir ikke i seg selv rett til å redistribuere leverandørdata.

| Ressurs | Dekning, lisens og observert kvalitet | Beslutning / integrasjonskostnad |
| --- | --- | --- |
| [dukascopy-node](https://github.com/Leo4815162342/dukascopy-node) | MIT; TypeScript, testfiler og CI; siste push juli 2026. Instrumentdokumentasjonen oppgir XAU tick/M1 fra mai 2003, D1 fra 1999. Dette er annen spot-feed, ikke GC-aggressorvolum. Historisk faktisk kompletthet og databruksrett må sjekkes separat. | Nyttig for en liten uavhengig spot-/basis-/quote-sjekk senere. Middels kostnad: annen feed og Node-adapter. Ingen stor nedlasting nå. |
| [fredapi](https://github.com/mortada/fredapi) | Apache-2.0; FRED/ALFRED, vintage/revisjoner, tester og CI; siste push januar 2026. Serieavhengig historikk, gratis API-nøkkel og serievilkår. | Lav/middels: punkt-i-tid USD/makro er nyttig; realrenter finnes allerede fra Treasury. Erstatter ikke faktiske publiseringstidspunkter. Nåværende WSL-FRED-transport har dokumenterte timeouts; bibliotek alene løser ikke dette. |
| [databento-python](https://github.com/databento/databento-python) | Apache-2.0; offisiell klient, typer, tester/CI, push oktober 2026. Ingen nye data følger SDK-en; datarettigheter til kjøpet gjelder separat. | Lav ekstra informasjonsverdi for eksisterende CSV/gzip-parser. Ikke installer eller kall API bare for å bygge OFI fra lokale bytes. |
| [cftc-cot](https://github.com/victorKariuki/cftc-cot) | MIT; seks rapportsett, testsuite og publish-workflow; lite prosjekt, push juni 2026. Gold 088691 finnes allerede hos oss. Legacy COT-arkiv har historikk fra 1986; disaggregated arkiv tilbyr 2006–2016 og senere årssett. | Lav ekstra informasjonsverdi som wrapper. Middels kostnad ved nytt rapportskjema; faktiske release-/revisjonsregler er viktigere enn enda et SDK. |
| [hftbacktest](https://github.com/nkaz001/hftbacktest) | MIT; Rust/Numba, latens-/kømodeller; primært L2/L3-replay og cryptoeksempler, push desember 2025. Ingen GC/XAU-datakilde. | Høy kostnad og feil første prioritet. L1 kan ikke dokumentere full kø eller faktiske XAU-fills. Bruk metodeideer, ikke nytt utførelsesrammeverk nå. |
| [py-market-profile](https://github.com/bfolkens/py-market-profile) | BSD med to klausuler (selve LICENSE kontrollert; GitHub SPDX var NOASSERTION), tre testfiler, eldre Travis-oppsett, push oktober 2023. OHLCV-gruppering av volum på Close er ikke ekte intrabar volumfordeling. | Kun formel-/tie-referanse. Utvid vår eksisterende tick-ladder-eier; ikke bruk M5-close-volum som om det var reelle prisnivåhandler. Lav kostnad. |
| [Stoikov microprice](https://github.com/sstoikov/microprice) | Forskningsnotebooks/data, push 2021; ingen lisens registrert i repo-metadata. | Metodereferanse. Ingen kodekopiering nå. Modellen krever TRAIN-estimering og gir ingen garantert M5-verdi. Middels kompleksitet. |
| [VPIN](https://github.com/yt-feng/VPIN) | AGPL-3.0, notebooks, ingen testfiler/CI funnet i tree-inventaret; push april 2026. Ikke en GC-datakilde. | Lav prioritet etter P0. Lisens og manglende testbevis gjør innkopiering lite attraktivt. Bruk kjente aggressorsider; ikke erstatt N med prisbasert klassifisering. Test merverdi utover volatilitet. |

[scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
er tilgjengelig i eksisterende miljø. Familieablasjon med ny TRAIN-fit er primær;
korrelerte aliaser og tidsavhengighet gjør individuell tilfeldig permutasjon/gain-
importance misvisende. Gruppediagnostikk må bevare tidsstruktur; SHAP/gain er
forklaring, ikke økonomisk merverdi. [XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
(Apache-2.0) er en mulig avgrenset CPU-utvidelse av eksisterende forskningsdriver,
ikke en ny arkitektur eller et krav om å erstatte den fungerende referansen.

Gratis primærkilder utover bibliotekene: [Treasury XML](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed)
(real yield fra 2003; eksisterende realrentekilde), [CFTC-arkiv](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm)
og [SPDR historisk data](https://www.spdrgoldshares.com/usa/gld/).
CFTC/Treasury tilfører mest provenans, ikke nye dupliserte felt. SPDR-holdings kan
bli ny informasjon, men vår tidligere hentesti returnerte PDF/redirect. Stabil
historisk holdingsfil og publiseringstid er ikke kvalifisert nå. Ingen offentlig
LBMA-/datapakke antas fritt lisensiert bare fordi en nettside viser den.

## 5. Behold, pause og utvikle

Behold original native baseline og alle filer/familier. Behold EMA/trend, SMC,
riktige nivåer og rå aktivitet som kontroller. Ingen features er slettet.

I **nye forskningsviews** settes 21 candlefelt og seks geomchan-felt på pause
(inkludert converging/apex/triangle-avledet kanalgeometri). Eksakt 27-feltsliste
ligger i baselinebindingen. Full historisk 315-feltsbaseline forblir uendret;
den planlagte snapshotvisningen har 288 felt før andre eksplisitte ablasjoner.
Dette er et deklarert senere view, ikke en allerede materialisert datasettendring.
Rå OHLC-/retur-/nivådata er fortsatt markedskontroller. Ikke bygg H&S/doji/
engulfing-/triangle-motorer eller optimaliser mønsterterskler i nye forsøk.

## 6. Rangert utvikling og P0-test

Forventet nytte nedenfor er en forskningsprioritering, ikke målt avkastning.
Ingen nye dataavgifter inngår. Arbeids-/CPU-kostnad må bindes før kjøring.

| Rekkefølge | Avgrenset arbeid | Forventet informasjonsnytte / kostnad / kompleksitet |
| --- | --- | --- |
| 0 – utført | Fast kilde-/feltinventar og baseline; avstem 36 mot dokumenterte 26 måneder | Unngår feil sammenligning; lav kostnad |
| 1 – P1 | Egen L1-kvalitets-/dekningsbeslutning på kausal kontraktserie, så få OFI/BBO-/intensitetsfelt i eksisterende GC-eier | Størst uutnyttet ny kildeinformasjon; middels/høy kompleksitet, én begrenset råpassering |
| 2 – P2/P3 | Session/ankret/rolling GC-profiler med riktige resets og nivåreaksjoner; gjenbruk eksisterende ATR/RV/session | Nivåbetinget flow; middels. P3-gjenbruk er lav kostnad |
| 3 – P0 hovedtest | Singler → par → tre lag, enkel regelreferanse og liten CPU-XGBoost-kandidat; samme data, kost og risiko | Direkte test av samspillshypotesen; middels, ingen ny modellplattform |
| 4 – P4 | Footprint-/absorpsjons-/divergensproksier ved de samme nivåene, bare hvis de tilfører noe utover P1 | Mulig ekstra kontekst; middels kompleksitet, eksisterende laddere |
| 5 – P5 | Faktiske release-/vintagebevis; ETF-holdings og bred USD bare ved dokumentert ekstra informasjon | Langsom regimekontekst; middels provenansarbeid, ingen ny makroindikatorkatalog |

Baseline fryses i [XAU_M5_INFORMATION_BASELINE_20261017.json](../configs/research/XAU_M5_INFORMATION_BASELINE_20261017.json)
på kilde 1c5d6557: eksisterende 007-arm A, faktiske manifest/inputs, kohort,
foldgrenser, maskefelt, prediksjoner og resultat/terminal. 007 er konsumert og
skal ikke relanseres. Den negative økonomireferansen fra 15.10 bindes separat;
den har annet utvalg og er ikke en matched GC-P0-baseline.

P0 bruker L = nye dynamiske GC-volumnivåer, F = GC-flow og V = eksplisitt
volatilitetstilstand. Felles kontroller beholder trend/EMA, SMC og session.
Fordi ATR/RV og P5 allerede finnes, testes de med fjerning/gjeninnsetting i
separate forskningsviews, ikke med dupliserte nye kolonner. V-avslag fjerner
ikke all implisitt volatilitetsinformasjon fra ATR-normaliserte kontroller.
De åtte armene er core, +L, +F, +V, +LF, +LV, +FV, +LFV. Hele og pausejusterte
kontrollflater må rapporteres slik at mønsterpausen ikke forveksles med GC-effekt.
P1–P5 får i tillegg hver sin deklarerte familieablasjon før sammenslåing.

De fem scenariene (flowbekreftet brudd, mislykket brudd/absorpsjon ved profilnivå,
VWAP-rejection med avtakende aggressivt volum, pullback/fortsettelse og ugunstig
volatilitet/likviditet) er forhåndsdefinerte mekanismediagnoser. Ikke utvid dem
til et bredt søk i terskler, ankre eller horisonter. Ingen regel krever automatisk
enighet mellom alle tre lag. Vis alle muligheter, valgte/avståtte handler og
endret handelsantall, slik at mindre eksponering ikke feilaktig kalles bedre signal.

Første modelltest bruker samme M5-beslutningsmuligheter, kausale dataforsinkelser,
M1-bid/ask, purge av overlappende utfall, indre TRAIN-valg, felles risikobudsjett
og samme kostpolicy. Overlappende muligheter kan ikke telles som uavhengige
handler. Bevar én-posisjons-/kapitalregnskap, finansiering og åpne posisjoner.
Sizing og eventuell Exit-utvidelse testes etter signaltesten; ingen endring av
native M1-Exit eller innføring av maksimal holdetid følger av denne planen.

Rapporter parvis nettoresultat, tidsbasert mark-to-market Sharpe, drawdown,
eksponering, antall handler, beslutnings-/fill-latens, timing/markouts og
kalibrering av forventet bps. Brier/log-loss gjelder bare hvis en ekte
sannsynlighetsmodell defineres. Spread finnes allerede i bid/ask; ikke trekk den
to ganger. Eksisterende kostscenario er 2 bps per utførelse, med 1/2/3/4 bps
følsomhet og historisk usikker finansiering. Dette er modellkost, ikke faktisk
slippage eller observert forbedring i XAU-utførelse. Blokkusikkerhet og
multippel-testing må registreres før fit. Transformer/RL vurderes etter dokumentert
netto merverdi; native kontrakt og train/serve-paritet må fortsatt kvalifiseres.

**Urørt bekreftelse mangler innen dokumentert GC-scope.** Alle 26 månedene har
inngått i 004/006/007 og utviklingsarbeid. En ny oppdeling gjør ikke gamle data
urørte. Juni 2026 er utviklings-VAL. Native TEST forblir lukket; den særskilte
D1-forwardjournalen gir ingen GC-testautoritet. En eventuell ubrukt del av en
allerede eid 36-månedersleveranse må først finnes og dokumenteres. Hvis den ikke
finnes, kan utviklingsforsøk gjennomføres, men endelig GC-bekreftelse må vente.
Ingen datakjøp eller ny datahenting er bestilt av denne rapporten.

## 7. Faktiske testresultater og konklusjonsgrense

Gjenbrukt ekte 006: 60 uavhengige råreferansebarer, eksakt heltallsavstemming,
posthash og Parquet readback består. M5-flow finnes; H4/D1-flow kvalifiseres ikke.
Dette beviser inputmekanikk på erklærte masker, ikke komplett flow eller edge.

Gjenbrukt 007: fem kronologiske utviklingsfolder, 16 745 felles inputrader,
45 ridge-modeller. A = eksisterende snapshots/kontekst; B = ekstra GC-pris/SMC;
C = ekstra volum/profil/footprint-sammendrag/delta/CVD. De 17 565 mål-/prediksjons-
radene over tre horisonter ble uavhengig avstemt mot M1 med 0,0 bps målavvik.

| Horisont | Utviklings-MSE A / B / C, bps² |
| --- | --- |
| 30 min | 253,844 / 253,944 / 254,072 |
| 120 min | 1 156,887 / 1 157,811 / 1 158,956 |
| 360 min | 4 052,491 / 4 054,063 / 4 056,442 |

Alle seks familiejusterte B−A/C−B-intervaller dekker null: INCONCLUSIVE.
Dette tester ikke hendelses-OFI, P0-samspill, hele footprintmønsteret, native
sekvenser eller netto økonomi. Det er ingen dokumentert GC-prediksjonsgevinst.
Den tidligere M5-testen 15.10 var også negativ for de fleste ridge/LightGBM-
celler etter kost, men hadde ikke GC-feltene og kan ikke avvise P0.

Nye kontrollbevis i denne bølgen: 583/583 Python-filer kan AST-parses;
faktisk manifestfeltorden matcher kontrakteieren; sidefilskjema/readbarhet og
bundne små resultat-/inputlock-/prediksjonshasher kontrolleres. Ingen nye
Sharpe-, drawdown-, utførelses- eller P0-ablasjonsresultater er produsert.
Den første koden som bør videreutvikles er eksisterende GC-parser/feature-eier
og M5-forskningsdriver, etter L1-kvalitetsbeslutningen. Ingen ny arkitektur.

---

# Historisk featureaudit og Entry-resultat — 10.10.2026

Revisjonen er fullført med konkrete rettelser. Den eksisterende læringsstudien
fant **ingen kvalifisert Entry-edge**. Korrigert kilde er kontrollert, men gamle
datasett og vekter må ikke behandles som oppdatert modellgrunnlag.

## Hva er kontrollert

Alle254 lokale signalfelt,71 eksakte kontekstaliaser, ett sesjonsfelt og190 MTF-felt
på M5/M15/H1/H4/D1 ble navngitt enkeltvis i det genererte feltregisteret
FEATURE_AUDIT_FIELDS.json (1,98 MB; fjernet fra repoet 14.10.2026, bytes i
git-historikken før den commiten). Hver rad har indeks,
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
