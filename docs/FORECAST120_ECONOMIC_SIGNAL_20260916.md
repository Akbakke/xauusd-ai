# Lang prognose inneholder økonomisk TRAIN-signal — 2026-09-16

På de eksisterende 1 024 brede TRAIN-inngangene gir fortegnet i den frosne
120-minuttersprognosen positivt gjennomsnitt etter native BID/ASK-kostnader.
Entry-Q velger samtidig FLAT på alle. Dette støtter en konkret svikt mellom
langtidsinformasjon og verdi-/handelslæring. Det beviser ikke generalisering,
optimal Exit eller samlet lønnsomhet.

## Én fast kontroll, ingen tilpasning

Gjenbrukte prognoser fra original95 og referanse96, med uendret utvalg og uten
nye forwards. Regel bestemt før beregningen: positiv prognose velger LONG,
negativ velger SHORT, null velger FLAT. Ingen terskel-/modell-/horizontsøk.
Bare den allerede eksisterende K24-prognosen ble undersøkt.

Alle 1 024 cachede K24-mål ble reprodusert bit-for-bit fra den bundne M5-
sluttkursserien. Inngang fylles ved første native M1-open etter Entry-baren;
utgang måles ved slutt på den nøyaktige M5-målbaren. 24 M5-barer er nominelt
120 minutter. Elleve forløp varer lenger i klokketid, opptil 3 060 minutter.
Markedspauser og finansiering er med; dette er ikke en ren intradagtest.

Priser og kostnader kommer fra LazyUnifiedExitEconomicStepProviderV1 og
compose_economic_step: gjennomførbar BID/ASK, begge kommisjoner, begge
slippagekostnader og all finansiering. Kontant-PnL holdes atskilt fra ikke-
kontant risiko/discount. Skalar lukkeverdi og native treningsprojeksjon stemmer
innen 1e-9 Bps. Ingen lærer-bootstrap brukes som observert kontantfortjeneste.

## Resultat på de 1 022 komplette forløpene

| Forhåndsbestemt regel | Gjennomsnittlig netto Bps per mulighet |
|---|---:|
| FLAT / faktisk Entry95 og96 | 0,00000 |
| Alltid LONG | -5,62348 |
| Alltid SHORT | -6,16904 |
| Prognose95, fortegn | +7,71344 |
| Prognose96, fortegn | +6,12993 |

Referanse96 velger 683 LONG og339 SHORT blant disse: sidemidlene er +3,31609
og +11,79912 Bps. Korrelasjonen mellom prognosen og LONG-netto er0,39510;
SHORT-netto har motsatt fortegn. De separate sidene og alle tolv måneder
ligger i maskinrapporten. Dette er TRAIN som modellen har sett, ikke holdout.
Etter32-stegsreferansen er prognosens økonomiske resultat svakere enn før;
vi skal ikke fremstille funnet som ny dokumentert læringsgevinst.

## To uavklarte forløp er bevart

Legacy-successors stopper før måltidspunktet for rad251654 (2025-06-19) og
rad312427 (2026-05-25). Begge er høyresensurerte, ikke økonomisk terminale.
Ingen successor er utvidet og ingen ukjent framtidsverdi er fylt inn.
Siste tilgjengelige likvidasjonsverdi er rapportert for begge åpne muligheter.

Med disse to siste markeringene blir den foreløpige gjennomsnittsverdien for
prognose96 +6,10699 Bps over1 024. Dette er IKKE komplett sluttids-PnL.
Maskinrapporten setter full_endpoint_net_cash_bps=null og
full_endpoint_economics_complete=false. Ingen samlet lønnsomhet erklæres.
Alle kjente tap er med; gjennomsnittet er per uavhengig likt notional-mulighet,
ikke porteføljeavkastning. Mulighetene kan overlappe.

Månedsmidlene med disse siste markeringene er positive i9/12 måneder for
begge prognoser. For96 er juni2025, august2025 og november2025 negative.
Siste markeringer i juni og mai gjør disse to månedsregnskapene ufullstendige.

## Konsekvens for neste rettelse

Manglende langtidsinformasjon i alle200 features/MTF er ikke lenger en god
standardforklaring. Den frosne prognosen inneholder kostnadsjustert TRAIN-
informasjon som det nesten nullstilte videreverdiestimatet ikke gir Entry.
Vi må rette verdiundervisningen, ikke starte mer av samme trening.

Neste én kandidat skal rette den svake videreverdisupervisjonen med et
kausalt, observerbart referanseforløp og bootstrap. Før kodeendring må dens
forventede handlingsverdi defineres eksplisitt og sammenholdes med gjeldende
Bellman-kontrakt. Tvungen HOLD gjennom et langt beregningsvindu estimerer
verdien av utsatt exit, ikke automatisk verdien av optimal fri exit; dette
skal ikke snikes inn som samme Q-mål. Etterpåvalg av beste pris er heller
ikke en kausal lærer. Prognosens fortegnsregel er en diagnosebaseline, ikke
en ny produksjonspolicy eller en holdetidsgrense. Ingen slik endring er valgt
for native kjøring ennå. Modell, targets, lærer og optimizer er uendret.

Læringsporten står fortsatt: bedre tilstandsavhengig Entry/Exit mot frosne
sammenlignbare mål og baselines før kronologisk native utviklings-VAL.
TEST forblir forseglet. Ingen større trening åpnes av dette TRAIN-resultatet.

## Bevis og ressursbruk

Kilde ed31b255e35821abee9a63f18ea52a71ed1b40e3, GX1_CURRENT/work/gx1-current.
Artifactrot: BASE/NATIVE_EXIT_PRIVATE_CLIP_20260916_REFERENCE/
FORECAST120_EXECUTABLE_OUTCOMES_20260916_V2. PLAN, operator, eksakt klokke-
kontroll, boundaries, alle1 024 rader og RESULT er bevart. RESULT-SHA256:
9fa60ae867bedfaeffd3ba58661870582a0d34c2f2fe01be5786be193cc9ee11.

Fullført CPU-kontroll:118,67s, topp-RSS1 255 448KiB, audit/4GiB/512MiB-swap.
Første forsøk stoppet korrekt ved de to grensene før resultatpublisering;
prisprojeksjonene var ikke lagret da. V2 bevarer grensene og skriver hver
ferdig rad fortløpende. Første operator og feil-logg er beholdt. Ingen GPU,
modellforward, backward, trening, nye modellinputs, VAL, TEST eller ordre.
