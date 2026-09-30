# Aktivt mål: full B og videre kvalifisert modelløp — 30.09.2026

Brukeren har uttrykkelig bestilt full implementasjon, deretter build/trening og
til slutt VAL, endelig TEST og backtest over alle år med gyldige data. Målet er
aktivt. Tidligere A/B/C-plan var avsluttet; dette er en ny videreføring.
En kildebegrensning alene fullfører ikke dette målet.

## Rekkefølge og ferdigkriterier

1. **Kvalifiser alle seks B-kilder.** Gjenbruk de fire kontrollerte ALFRED-arkivene.
   Lukk dokumenterbar GLD-tonnasje og COT 088691 Legacy Futures Only med faktisk
   tilgjengelighet og historiske versjoner. Ingen kildebytte eller tilbakefylling
   av reviderte sluttserier.
2. **Implementer hele B hos eksisterende eiere.** De tolv avtalte makrofeltene
   legges til de sju A-feltene. Bevis kausalitet, manglendeverdibehandling,
   publikasjonslag og identisk A/B TRAIN-/eval-populasjon. Bevar native familier.
3. **Mål den registrerte B-minus-A-kontrakten.** Gjenbruk modell-/kostnads-,
   baseline- og inferenseiere. Frys faktisk populasjon, årsfolds, konstanter,
   hashbindinger og beslutningsregel før utfall undersøkes eller fits kjøres.
   A/C gjentas ikke blindt; en nødvendig matchet A-referanse til B er eget
   eksplisitt sammenligningsgrunnlag.
4. **Ferdigstill bygge- og treningskontrakten.** Krev målt verdi før omfattende
   trening. Bind mål/horisont, native innføring, datasett, datadeling, ressursprofil,
   kausalitet, paritet og alle nødvendige kontroller. Oppgi eksplisitt dersom
   økonomiske porter ikke gir grunnlag for denne overgangen.
5. **Kjør kvalifisert build og trening.** Bruk eksisterende eiere, prosjektlås,
   capped wrapper og vakter. Ingen blind utvidelse eller svekkelse av porter.
6. **VAL, endelig TEST og backtest.** Frys modell og evalueringsprotokoll før
   TEST åpnes. TEST brukes én gang til endelig vurdering, aldri tuning.
   Backtest alle år med gyldig, deklarert dekning, med kostnader og åpne posisjoner.
   Historiske utviklingsår skilles fra kronologisk OOS og forseglet TEST;
   resultatene får ikke alle merkelappen urørt OOS.

Brukerens bestilling autoriserer nødvendig arbeid og disse trinnene i denne
rekkefølgen. Den er ikke bevis for læring, data eller klarhet. Native trening og
TEST er fortsatt stengt i dagens maskinpolicy fordi foregående porter ikke er
bestått. Kjøringspolicy kan først flyttes sammen med dokumentert oppfylt kontrakt.
Ingen live/paper, spending eller eksterne leverandørmeldinger er bestilt.

## Verifisert utgangspunkt

CURRENT 74ba5f64, rent tre og ingen native prosess ved overtakelse.
Fire makroarkiver er kontrollert; full B er umålt. Første dollarvintage er
04.02.2019, så alle år betyr alle faktisk gyldige år, ikke syntetisk dekning av
hele prisarkivet. Ingen modellgjennomføring eller bygge-/treningsklarhet påstås.

Et nytt avgrenset metadataoppslag er bundet i
configs/research/TA_B_FULL_GOAL_SOURCE_20260930.json etter nesten tre timer uten
arkivkall. Resten av batchen stoppes ved første HTTP429; fullførte råfiler hentes
ikke på nytt. Metadata er ikke predictor-admission.

## Første videreføring

Arkivtjenesten ga HTTP429 på første forsøk 10:51:16 UTC; to gjenstående
forespørsler ble hoppet over av den kontrollerte eieren. Ingen nye rådata ble
hentet. Ingen ny arkivprøve gjentas automatisk. Den frosne forespørselens felt
prior_last_request_utc oppga 07:51:22Z; tidligere RESULT.json viser eksakt
07:51:28.141925Z. Feltet var bakgrunnsmetadata, ikke tilgjengelighetsinput.
Korrekt målt opphold før den nye forespørselen var 10 788,14 sekunder.

Makroklokken implementeres og kontrolleres uavhengig av den blokkerte
GLD/COT-tilgangen. Dette er åtte komponentfelt, ikke en redusert B-modell.
Ingen XAU-priser eller utfall leses i denne komponentforberedelsen.
