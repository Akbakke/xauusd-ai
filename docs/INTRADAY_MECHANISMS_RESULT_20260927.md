# Intradag-mekanismer (bølge 1) — resultat 27.09.2026: NO-GO 0/61

Forhåndsregistrert i [INTRADAY_MECHANISMS_PREREG_20260927.md](INTRADAY_MECHANISMS_PREREG_20260927.md)
(commit 29672457, før kjøring); kjørt på ren kilde 29672457 under `gx1_capped_run.sh --class producer
--mem 10G`. Tape `XAU_M5_NATIVE_2009_20260701_PAIR_20260927` (1 140 232 M5-barer, 2009-06-01 →
2025-05-30 20:55); innganger 2011-06-01 → 2025-05. Evidens:
`/home/andre2/GX1_RUNS/INTRADAY_MECHANISMS_20260927/results.json` (sha256 96927820…). Målt på ekte
deklarerte bytes; VAL og TEST ikke lest.

**Ingen av 61 celler er GO eller LOVENDE, og ingen har positivt netto snitt selv ved low-slippage
(1 bps per utførelse).** Beste netto: `orb_london_local` −1,95 bps (t −1,73). Brutto (etter spread,
før slippage) ligger nesten overalt på −0,2 … −3,4 bps, altså omtrent spreaden.

| familie | celler | beste netto ved low (bps) | beste brutto (bps) | celle med beste brutto |
|---|---|---|---|---|
| A rundtall | 8 | −2,84 | −0,81 | `rn50_cross_follow_h12` (n 3 767) |
| B oppsett (speilede par) | 36 | −2,24 | −0,21 | `setup_pdh_break_trend_H4_pair_h12` (n 1 088) |
| C COMEX-momentum | 3 | −3,79 | −1,67 | `im_first_hold_to_close` (n 3 412; beste netto er `im_overnight_to_last`) |
| D LBMA | 6 | −3,60 | −1,60 | `lbma_am_pre_short` (n 3 610) |
| E lokale klokker | 8 | −1,95 | +0,28 | `orb_london_local` (n 3 605; UTC-versjonen ga +0,56) |

- Rundtall: avvisning ved nivå (Oslers take-profit-reversering) gir ingen reversering. Kryss gir
  svak fortsettelse (se etterpåanalysen), men under spreaden.
- Oppsett: alle 36 negative netto; bare én celle har ikke-negativ bjørnefold
  (`ob_bull_retest_H1_trend_H4_pair_h12`), og den er negativ på long-siden.
- COMEX-klokka: første time forutsier ikke siste time; LBMA: ingen drift rundt auksjonene.
- Sesjoner på lokal klokke: negative begge veier; Asia long −1,39 brutto mot −1,70 med UTC-inngang
  ved rollover (liten forskjell).

## Etterpåanalyse (ikke del av beslutningen): retning før kost

Mid-til-mid-avkastning i signalets retning for de samme handlene, uten spread, slippage eller
finansiering (`GX1_RUNS/INTRADAY_MECHANISMS_20260927_LOGS/mid_diagnostic_posthoc.json`, sha fd189253…;
skript `wave1_mid_diagnostic.py` samme sted). t = min(iid, dagsklynget). Valgt etter at resultatet var
kjent — evidensklasse: målt, post hoc, ubekreftet.

Over de 61 cellene er snitt-t +0,48 (sd 1,33); 7 celler har t > 2 (uavhengige celler under null: ~1,4)
og 1 har t < −2 (speilingen av `lbma_am_pre_short`). Fem av de sju er **fortsettelse etter brudd**; de to
andre er `lbma_am_pre_short` (fast short timen før AM-auksjonen, +0,69, t 2,48) og
`fvg_bull_retest_H1_trend_H4_pair_h48` (+1,88, t 2,12). Ingen reverseringsregel er blant dem:

| celle | mid bps | t | treff | positive år | rundtur-spread på de tidspunktene |
|---|---|---|---|---|---|
| `rn50_cross_follow_h12` (kryss av $50-nivå, hold 1 t) | +1,91 | 3,80 | 0,514 | 11/13 | 2,72 |
| `setup_pdh_break_trend_H4_pair_h12` | +2,48 | 2,77 | 0,503 | 9/13 | 2,68 |
| `setup_momentum_confluence_long_pair_h12` | +0,99 | 2,68 | 0,480 | 10/13 | 2,51 |
| `setup_range_break_up_H1_trend_H4_pair_h12` | +1,50 | 2,48 | 0,501 | 8/13 | 2,64 |
| `orb_london_local` (hold til 17:00 London) | +2,58 | 2,29 | 0,511 | 11/13 | 2,30 |

Reverseringsreglene er null eller negative (`rn50_reject_fade_h12` −0,65, `eql_sweep_fade_M5_pair_h12`
−0,31). $50-kryss-effekten er fullt på plass etter 1 t (4 t: +1,93, t 1,56).

Tolkning: det finnes et svakt, konsistent fortsettelsessignal etter brudd, det stop-ordre-mekanismen
forutsier, på 1–2,5 bps mid per handel. Det er mindre enn rundtur-spreaden i de samme øyeblikkene
(2,3–2,7 bps, bredere enn snittet fordi brudd skjer i raske markeder), før noen slippage. Teknisk
analyse i gull på intradag har altså ekte, men for lite, informasjon for en som betaler spreaden med
markedsordre. Med 2026-kost (0,60 bps halv spread i 258 fyllinger) ville marginen vært < 1 bps før
latens-slippage, som i et bruddøyeblikk går mot deg.

## Hva dette betyr

- Etter registreringens regel er pris-alene-mekanismene A–E lukket for intradag-retning i XAUUSD med
  markedsordre på 2011–2025.
- Gjenstående intradagkilde med ny informasjon: OANDAs ordre- og posisjonsbok for XAU_USD (bølge 2,
  krever operatørvedtak før noe hentes).
- Fortsettelsessignalet kan bare bli handelbart hvis kost per rundtur er godt under ~2 bps i
  bruddøyeblikket. Det kan ikke avgjøres på historikk alene: latens fra beslutning til fylling er
  umålt. En forhåndsregistrert bekreftelse av de fem fortsettelsescellene på mid på urørt VAL
  (2025-06 → 2026-06) er mulig, men avgjør ikke kostspørsmålet.

## Ikke undersøkt

Tick-oppløsning og limit-ordre; andre gitre ($25, $100) og holdetider; samspill med volatilitetsregime;
bid/ask-baserte nivåer; OANDA-spreaden på 2026-nivå brukt på historikken.
