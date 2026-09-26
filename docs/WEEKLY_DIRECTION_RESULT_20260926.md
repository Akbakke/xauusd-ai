# Ukesmåling av retning — resultat 26.09.2026: NO-GO

Forhåndsregistrert i [WEEKLY_DIRECTION_PREREG_20260926.md](WEEKLY_DIRECTION_PREREG_20260926.md)
(commit f54a2275, før kjøring). Evidens: `/home/andre2/GX1_RUNS/WEEKLY_DIRECTION_20260926/`
(`h4_3folds/`, `d1_3folds/`: `report.json`, `nonoverlap.json`, `summary.md`, logger). TRAIN-only;
ingen VAL/TEST-rad. Evidensklasse: målt på ekte deklarerte bytes.

## Avvik fra registreringen (dokumentert, ikke til fordel for resultatet)

- **Fold 0 (2022-06..2023-05) kunne ikke evalueres.** Med ~4 ukers purge starter fold 0s indre
  modellvalg 24.02.2022, før den tidlige kalibreringen slutter 28.03.2022; kronologivakten
  stoppet kjøringen (`WALKFORWARD_FEATURE_FIT_AFTER_EVALUATION_START`), som den skal. Kjørt på de
  tre gyldige årene 2023-06..2026-05 med **strengere** GO-regel: Δ > 0 i 3 av 3 år (ikke 3 av 4).
- Siste fold-grense er datasettets eksakte TRAIN-slutt 2026-05-31 23:55 (registreringen skrev
  2026-06-01, fem minutter inn i VAL).

## Resultat

48 celler (H4/D1 × 2 armer × ridge/HGB × 1/2/4 uker × 2 regler), ikke-overlappende blokker,
netto etter den bundne kostpolicyen, parvis mot alltid-LONG:

- **Ingen celle er bedre enn alltid-LONG i noe år** (0/3 overalt). GO- og LOVENDE-kravet er ikke
  oppfylt i noen celle.
- Beste mulige utfall var Δ = 0: ridge valgte konstanten (konstant-alternativet) i alle fold, og
  med «alltid handle»-regelen ble modellen identisk med alltid-LONG.
- Når modellene avviker fra long, taper de: HGB 4 uker H4 −215 bps per beslutning (t −2,4),
  HGB 1 uke D1 −39 bps (t −2,2), HGB 2 uker H4 −72 bps (t −2,0). Ridge med FLAT tapte −14 til
  −36 bps i år der den netto konstante prediksjonen falt under null og den sto utenfor.
- Blokker per celle: H4 144/72/36 og D1 127/69/36 for 1/2/4 uker over tre år.

## Hva dette betyr

På 2021–26, som er ett oksemarked, gjenkjenner den kausale featureflaten (alle tidsrammer, 1 311
felt) ikke ukesretning utover driften. Det som kan læres her er «vær long», og det er
retningsfritt. For en bot som skal gå short i fallende marked og long i stigende, må treningsdata
inneholde fallende markeder; lengre historikk er nå den avgjørende forutsetningen, ikke flere
modellvarianter.

## Ikke undersøkt

- Sekvensmodellen (transformeren) på ukeshorisont; bare ridge og HGB på øyeblikksbilder er målt.
- Fold 0 (2022-06..2023-05), det flate/fallende året, fordi kalibreringen ikke er tidlig nok for
  lange purger.
- Historikk før 2019 og andre regimer.
