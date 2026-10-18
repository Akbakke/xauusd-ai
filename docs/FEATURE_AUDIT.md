# XAUUSD M5/MTF/MBP — individuell kilde- og dekningsrevisjon

Dette dokumentet følger det varige målet i GX1_RULES.md. Arbeidsstatus,
kjøregrenser og terminalbindinger eies av NEXT_RUN_POLICY.json og siste
CURRENT_HANDOVER.md. Tidligere rapportversjoner finnes i Git. Ingen tidligere
kort regel-/CPU-/smoketest avgjør verdien av en ny korrekt MBP-Transformer.

## Revidert flate og bevisklasse

Inventaret på73aa04f0 omfatter770 sporede filer og600 Python-filer.
Alle600 kan AST-parses. Import-, CLI-, test- og eierskapsreferanser er indeksert;
dette er et statisk inventar, ikke et matematisk korrekthetsbevis for alle linjer.
Hvert navngitt signal-, kontekst- og MTF-felt har formelkilde, eier, kildeklokke,
TF-dekning og manglende ruter i det komplette maskinlesbare feltinventaret.

Inventarer og hashmanifest:
`/home/andre2/GX1_RUNS/NATIVE_GC_MTF_DERIVATION_20261018_018/AUDIT/MANIFEST.json`.
Filene heter gx1_repo_inventory_20261018.json,
gx1_feature_audit_20261018.json, gx1_mbp_audit_20261018.json og
gx1_model_repo_audit_20261018.json. De beskriver utgangspunktet før rettelsene;
ny funksjon/datasett må kvalifiseres separat.

| Utgangspunkt | Antall og semantikk |
| --- | --- |
| Native signal |735felt:25base +162mandatory +548candidate;96M5-steg |
| Kontinuerlig kontekst |552aliaser i signalflaten, ingen552nye uavhengige signaler |
| Kategorisk kontekst |1sessionfelt |
| GC |210verdier +210egne observasjonsmasker;102kilde-/TF-felt ikke innrømmet |
| D1/makro |61felt i eksisterende session/regime-spesialist; ingen niende encoder |
| MTF |190felt beregnes selvstendig på lukkede M5/M15/H1/H4/D1-candles |

| Lært familie | Lokale signalfelt | MTF-felt per klokke |
| --- | ---: | ---: |
| Struktur/swing |30|20|
| SMC/likviditet |299|50|
| Trend/EMA |40|29|
| Volatilitet/kompresjon |18|5|
| Momentum/flow |179|22|
| Session/regime |80|4|
| Chartgeometri |32|38|
| Rå candlegeometri |57|22|

Antall felt er ikke antall selvstendige indikatorer. Masker, sammeformelaliaser,
kildeklokkebundet kontekst og lærte representasjoner telles eksplisitt hver for
seg. Omtrent150 er et brukeranslag; de eksekverbare eierne bestemmer faktisk flate.

## Påviste koblings- og målgap

1. Entry konsumerte bare M15/H1/H4/D1-MTF-sekvenser. M5s190-feltscache gikk
   til Exit;27MTF-formler manglet en direkte lokal M5-motpart. Rettelsen bruker
   eksisterende M5-rute/moduler også for Entry. Entry lukkes fortsatt på+5min,
   Exit på+1min. Samme-clock-aliaser er kildeavledede og må fortsatt være
   bitlike på ekte beslutningsrader. Endret funksjon krever fersk baseline.
2.190perTF er ikke den komplette unionen av alle lokale genuine formler.
   Blant gapene er basic/KAMA/TEMA/BB/returprimitiver, sweep-ankret aktivitet,
   enkelte foundation-/mikro-/spreadfelt og clockspesifikke kontekstformler.
   Feltinventaret navngir hvert gap; komplett all-TF-konsum er ikke bevist.
   Makro publiseres as-of; den skal ikke resamples til fiktive intradagnyheter.
3. H20-terminal nettoavkastning er aktivt Entry-target i den tidligere fysiske
   generasjonen. Dette måler ikke direkte liten MAE før MFE på M5. Et vedtatt
   target må være eksplisitt før ny trening. Operatøren valgte18.10 M5-path og
   MAE før MFE primært, med lengre horisonter som støtte; auxiliary MAE/MFE-hoder
   alene oppfyller ikke dette vedtaket.
4. Fysiske016-manifester beholder legacy fitted-Q-basismetadata og et separat
   observed-h20-avsnitt. Ny recipe/bundle skal binde faktisk fase og uavhengighet
   eksplisitt; de immutable016-manifestene omskrives ikke.
5. M30 er ikke aktivert, og tilført verdi er ikke målt.

## MBP-1: hva de eksisterende bytene faktisk gir

Rå001 omfatter66MBP-1-filer og892100712records. Den fullførte rebuilden bevarer
9173893M1-prisnivårader med eksakt buy/sell/unknown-konservering. GC velges
kausalt fra forrige fullførte UTC-døgns observerte volum med00:01-tilgjengelighet,
ingen tilbakegang i utløp og ingen substitusjon ved dagens datamangel.
GC-priser er ujusterte outright-priser. Kilden er en futuresinput, ikke vår
XAUUSD-utførelse. GC/spot-basis må bruke samme observerte kildevindu.

Kildens B betyr buy-aggressor og A sell-aggressor på Trade; N forblir ukjent.
F/passive fills er ingen ekstra aggressorhandler. BBO order counts er aggregert
antall viste ordre, ingen individuell ordreidentitet, køposisjon eller dypere bok.
[Leverandørens MBP-1-skjema](https://www.databento.tech/docs/schemas-and-data-formats/mbp-1)
og [kontraktsymbologi](https://www.databento.tech/docs/standards-and-conventions/symbology)
beskriver de observerbare feltene og ujusterte kontraktene.

| Strict komplett bøtte i rå001 | M5 | M15 | H1 | H4 | D1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Volum/profil |98594|18545|862|2|0|
| Full kjent aggressorside |19091|933|3|0|0|

Årsaken er kildebevist: hver M1 må ha handler, passere kvalitetsmasken og ha
samme kontrakt. D1 krever dermed aktivitet også i stengte minutter. Dette er
ikke bevis for at D1-voluminformasjon mangler i rådata. Nytt observed-domene
summerer bare aksepterte minutter og følger faktisk dekning, kjent/ukjent side,
kontrakt, første/siste observasjon og uendrede strict-masker. Histogrampasset
bruker de fullførte M1-/prisgridbytene og avstemmer hver side, uten ny råskanning.

Observed-SMC er eksplisitt en proxy på observerte candles; en intern manglende
prisbane eller institusjonell ordreblokk blir ikke rekonstruert. L1/ordreantall
aggregeres med egne masker og faktisk observert tid. Tidsvektet mean kan ikke
rekonstruere alle intraminutt spread-shocks. Full neural price-grid-ingestion,
slike eventstatistikker og faktisk alle-TF native-konsum må oppgis som egne
uferdige deler til de er implementert og målt.

## Beholdt korrekt eierskap og verifikasjonskrav

GC-normalisering bruker bare observed==1 på fysisk TRAIN. Lagringsnull bak
observed==0 er ingen markedsverdi og maskes etter normalisering før projeksjon.
Entry-only-trening skal ikke bruke Exit-teacher eller Exit-gradienter. Ny smoke
må også bevise uendrede Exit-vekter, ekte batchgjennomløp, familie-/TF-påvirkning
og samme bundne train/serve-beregning. Kildelesing erstatter ikke disse målingene.

Alle26månedene er gjenbrukt utvikling. Kronologisk TRAIN/VAL/kontroll og censur
ved utfallsgrensene beholdes; en ny oppdeling gjør ingen gammel periode urørt.
TEST fra juli2026 brukes ikke til tilpasning. Senere økonomi må inkludere alle
valgte handler, åpne posisjoner, spread, finansiering og øvrige deklarerte
kostnader mot samme-risiko referanse. Ingen profittpåstand følger av datakvalitet.

## Opprydding

Ingen Python-fil var bevist uten repo-referanser når import, CLI, tester og
beholdt eierskap tas med. v12_trade_state er bare testimportert, men SYSTEM_MAP
beholder offline persistenseieren eksplisitt; dette er ikke slettebevis.
Gamle aktive rapportfortellinger er erstattet av denne någjeldende revisjonen;
Git og originale maskinkvitteringer bevarer faktiske målinger. Rådata/RUNS og
checkpoints slettes bare gjennom retention-eieren. Ikke opprett nye historikkmapper.

## Grenser som gjenstår å bevise

Full individuell numerisk rekonstruksjon av hver formel på den nye fysiske
populasjonen; alle genuine formler på alleTF; full GC native-innrømmelse;
hel-TRAIN-normalisering; nye faktiske initial-/paritets-/smokebevis; målrettet
Transformer-læring, senere generalisering og nettoøkonomi. Nåstatus og nyeste
konkrete terminal leses fra den ene overleveringen og policyen.
