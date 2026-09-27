# Makrohendelser — resultat 27.09.2026: NO-GO

Forhåndsregistrert i [MACRO_EVENT_BASELINES_PREREG_20260927.md](MACRO_EVENT_BASELINES_PREREG_20260927.md)
(commit 59ee26ff, før kjøring); kjørt på ren kilde 59ee26ff. Kalender
`GX1_DATA/research/macro_calendar_20260927/MANIFEST.json` (sha256 8ccc7264…; FOMC 119, NFP 207, KPI 207
hendelser 2009-06 → 2027). Evidens: `/home/andre2/GX1_RUNS/MACRO_EVENT_BASELINES_20260927/results.json`.
Målt på ekte deklarerte bytes; VAL og TEST ikke lest.

**Ingen av 18 celler er GO eller LOVENDE.** Hendelser: KPI 168, NFP 168 (2011-06 → 2025-05), FOMC 98
(2013 → 2025-05); 2–4 hoppet over per celle for manglende barer.

| hendelse | celle | n | brutto bps | treff | netto bps | sd | t | positive år |
|---|---|---|---|---|---|---|---|---|
| KPI | post_rev_h12 | 165 | +3,50 | 0,55 | −0,52 | 35 | −0,19 | 38 % |
| KPI | post_rev_h48 | 165 | +3,57 | 0,47 | −0,52 | 58 | −0,12 | 62 % |
| KPI | pre_long | 166 | +3,50 | 0,56 | −1,98 | 82 | −0,31 | 54 % |
| FOMC | post_mom_h48 | 98 | +5,48 | 0,57 | +1,34 | 66 | +0,20 | 58 % |
| FOMC | pre_long | 98 | +3,66 | 0,46 | −1,82 | 62 | −0,29 | 33 % |
| NFP | post_mom_h48 | 164 | −0,45 | 0,52 | −4,57 | 70 | −0,84 | 38 % |
| (alle øvrige 12 celler) | | | | | < −1,4 | | ≤ −1,59 | ≤ 42 % |

- Beste netto: FOMC-momentum over 4 t, +1,3 bps (t 0,20) — støy.
- KPI: den første 15-minuttersreaksjonen reverserer ~5,8 bps (mid) i timene etter, men spread og
  slippage (~6 bps per rundtur) spiser det.
- Før hendelser: gull stiger 7–9 bps de siste 24 t før KPI og FOMC, omtrent gulls vanlige drift;
  netto long ≈ −2 bps etter kost og finansiering.

## Hva dette betyr

Tidspunktet for planlagte makrohendelser gir ingen handelbar retning i gull etter kost på
2011–2025, verken før eller etter publisering. Etter registreringens regel avsluttes scalp på
gull-pris + kalender som retningskilde.

## Ikke undersøkt

Overraskelser mot konsensus (krever betalte prognosedata); andre hendelser (PCE, detaljhandel,
ECB); sekund-/tick-oppløsning rett etter publisering; limit-ordrer.
