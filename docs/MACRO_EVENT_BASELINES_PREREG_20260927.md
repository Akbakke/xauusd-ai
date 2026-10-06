# Makrohendelser — forhåndsregistrering 27.09.2026

Committet før noen pris rundt hendelsene er lest. Formål: teste om planlagte amerikanske
makrohendelser gir en handelbar retning i XAUUSD etter kost, på scalp-/intradaghorisont — ny
informasjon (tidspunkt), ikke flere prisfeatures. Forskningsarm, aldri Entry-input (regel 1).
Følger den avsluttede NO-GO-vurderingen for de modellfrie grunnlinjene.
Dette er beholdt kodebundet spesifikasjon, ikke ny kjøretillatelse.
Avsluttede rapporter finnes i Git; gjeldende scope eies av NEXT_RUN_POLICY.json.

## Kalender (kilde og manifest)

`GX1_DATA/research/macro_calendar_20260927/` (bygd av `gx1/scripts/build_macro_event_calendar_v1.py`
fra lagrede rå filer, hentet 2026-09-27 12:03 UTC, sha256 per fil i `MANIFEST.json`):
- FOMC: federalreserve.gov `fomchistorical2013–2020.htm` + `fomccalendars.htm`; planlagte møter,
  uttalelse siste møtedag 14:00 New York (lik praksis fra 2013). Datoene på kalendersiden er
  kryssjekket mot uttalelseslenkene. 8 per år (2020: 7).
- Sysselsettingsrapporten (NFP) og KPI: ALFRED publiseringsdatoer (rid 50 / 10); hovedpublisering =
  første dato i måneden (NFP) / siste dato i måneden (KPI, sesongfaktorrevisjonen i februar kommer
  først); 08:30 New York. Sommertid fra tidssonedatabasen.
- Ingen konsensusprognoser (betalte data): testen kan ikke vite om tallet overrasker, bare *når*.

## Celler (18) — tape, fylling og kost som de modellfrie grunnlinjene

Tape `XAU_M5_NATIVE_2009_20260701_PAIR_20260927`, lest til 2025-05-31; bundet kostpolicy a48f8e56…
(scenario A). T = publiseringstidspunkt; «bar s» = M5-baren som starter ved s; fylling ved barens close.
Hendelser der en påkrevd bar mangler, hoppes over og telles.

Per hendelsestype (FOMC, NFP, KPI):
- PRE_LONG / PRE_SHORT: inngang første bar i [T−24 t, T−23 t), utgang bar T−5 min. 2 celler.
- POST_MOM(h) / POST_REV(h): signal = fortegn(mid bar T+10 min − mid bar T−5 min) (første 15 minutter
  inkludert publiseringsbaren); inngang bar T+10 min; hold h ∈ {12, 48} barer (1 t, 4 t); MOM i
  signalets retning, REV motsatt. 4 celler.

Vindu: NFP og KPI 2011-06-01 → 2025-05-31 (fulle år 2012–2024); FOMC 2013-01-01 → 2025-05-31 (fulle år
2013–2024).

## Beslutningsregel (låst)

- Per celle: n, snitt netto bps per hendelse, sd, t = snitt/(sd/√n); brutto snitt og treffrate.
- 18 celler; Bonferroni ensidig 0,05/18 → **t ≥ 2,77**.
- **GO**: t ≥ 2,77 og ≥ 60 % fulle år med positivt snitt. **LOVENDE**: t ≥ 2,0 og samme årskrav.
- GO → hendelsessporet bekreftes på urørt VAL (2025-06 → 2026-06) før noe bygges. Ingen GO →
  tidspunktet for planlagte hendelser gir ikke handelbar retning; scalp på gull-pris + kalender
  avsluttes som retningskilde.

## Styrke

~168 NFP-/KPI- og ~99 FOMC-hendelser per celle. Med sd ≈ 50 bps på 4 t blir detekterbar gevinst
≈ 2,77 × 50/√168 ≈ 11 bps (NFP/KPI) og ≈ 14 bps (FOMC); på 1 t mindre.
