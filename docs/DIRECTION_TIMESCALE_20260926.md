# Retning og tidsskala — målt 26.09.2026

Svar på operatørens spørsmål: *«Gull gikk fra ~2 000 til ~5 000 og M5 går opp og ned
mange ganger om dagen — hvorfor finner ikke modellen retningen?»*

Evidensklasser (regel 2d): **[M]** målt på ekte deklarerte bytes, **[S]** bevist fra
kilde/algebra, **[N]** ikke undersøkt. Alle målinger er TRAIN-only, lesende, kjørt under
`scripts/gx1_capped_run.sh --class audit --mem 4G`; ingen modell, ingen VAL/TEST-byte i
noen statistikk. Skript, JSON og logg:
`/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/DIRECT_TARGET_IDENTITY_CLAUDE_20260926/`.

## 1. Retningen finnes — på uker og måneder, ikke på 95 minutter [M]

Native M5-tape (`XAU_M5_NATIVE_2019_20260701_PRETEST_20260829`, kuttet ved 01.06.2026),
TRAIN 01.06.2021–31.05.2026: mid 1 907,8 → 4 539,7 (topp 5 587,8 den 28.01.2026),
+8 669 log-bps; kjøp-og-hold én gang +13 791 bps (enkel, BID/ASK).

| Holdetid (ikke-overlappende) | Snitt | Std | Snitt/std | Andel opp | Antall |
|---|---:|---:|---:|---:|---:|
| 5 min | +0,02 | 6,6 | 0,004 | 0,503 | 354 569 |
| 95 min (dagens mål) | +0,46 | 28,6 | 0,016 | 0,516 | 18 661 |
| 1 handelsdag (288 M5) | +7,1 | 111 | 0,064 | 0,537 | 1 231 |
| 1 handelsuke (1 440 M5) | +35 | 244 | 0,14 | 0,549 | 246 |
| ~1 måned (6 048 M5) | +156 | 396 | 0,39 | 0,672 | 58 |
| ~1 kvartal (18 144 M5) | +481 | 720 | 0,67 | 0,684 | 19 |

Snitt/std vokser som √(horisont): trendens andel av bevegelsen er ~1,6 % per 95-minutters
vindu og blir først dominerende på måneder. Alltid-LONG i 95-minutters handler etter
hverandre i det samme oksemarkedet: −1,34 bps per handel etter spread (spread 1,66 bps mot
+0,32 bps trend per vindu), ~16 500 handler, sum −22 500 bps (ukompundert). Samme retning
som kjøp-og-hold; forskjellen er hvor ofte spreaden betales per trend-bps.

Forbehold: de lange horisontene har få uavhengige observasjoner (58 måneder, 19 kvartaler),
og «andel opp» der er oksemarkedets drift i denne perioden, ikke en bevist betinget retning.

## 2. Svingningene kan ikke handles i det øyeblikket de sees [M]

M1 TRAIN child view (1 764 986 rader, 1 555 handelsdager). Zigzag på mid-close med deklarert
stige X; ved hver bekreftet svingning: gå i svingningens retning med symmetrisk mål/stopp
±X (første passasje på M1 high/low). Baseline: samme test fra faste timeinnganger.

| X (bps) | Svingninger/dag | I ettertid: sum svingning bps/dag | P(fortsetter) | Drift-baseline L/S | Netto bps/handel |
|---:|---:|---:|---:|---:|---:|
| 10 | 39,0 | 968 | 0,492 | 0,509/0,491 | −2,00 |
| 20 | 13,5 | 619 | 0,501 | 0,510/0,490 | −1,92 |
| 40 | 4,1 | 365 | 0,514 | 0,515/0,485 | −1,03 |
| 80 | 1,19 | 205 | 0,515 | 0,535/0,465 | +0,14 |
| 160 | 0,34 | 114 | 0,506 | 0,570/0,430 | −0,59 |
| 320 | 0,085 | 61 | 0,550 (n=131, se 0,044) | 0,632/0,368 | +29 (= drift) |

Grafen inneholder i ettertid hundrevis av bps svingning per dag, men sannsynligheten for at
en bekreftet svingning fortsetter er en mynt pluss drift på alle skalaer til og med 160 bps.
Konsistent med tidligere målt trend-persistens 0,488–0,500 på 5 min–1 d i alle år.

## 3. Modellen har aldri fått lov til å lære på skalaen der retningen er [S]

`ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS = MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS = 96`
(`gx1/contracts/entry_direction_target_policy_v1.py:39`): retningsmålets knee-søk
(`gx1/contracts/entry_causal_m1_target_policy_v1.py:_selected_chord_knee`) kan maksimalt
velge 96 M5-barer = 8 timer; det valgte 19 (95 min). Exit-livsløpets referansepolicy holder
med sannsynlighet 119/120 per M1-steg (forventet ~120 min). Inputene ser uker og år
(H4/D1-sekvenser), men spørsmålet modellen trenes på er de neste 95 minuttene.

## 4. Det direkte M1-målet i ENTRY_DIRECT_OUTCOME_HYPOTHESIS er ikke nytt [M+S]

`y_{long,short}_final_pnl_at_direction_horizon_bps` i V9-datasettet hypotesen binder er
bit-identiske med V12 på alle 313 399 TRAIN-rader (maks avvik 0,0). Begge produseres av
`causal_m1_terminal_outcomes_at_horizon` (h=19): eksakt M1 ask/bid-open ved beslutning,
exit på M1-open 95 min senere — samme definisjon. Dette er knee-målet som walk-forward-
instrumentet (arkivgrenen, se `docs/ENTRY_DIRECTION_SNR_DIAGNOSIS_20260923.md`) testet
lærer-fritt med ridge/HGB på opptil 1 311 felt uten robust resultat; direkte HGB-
fortegnsklassifikator (h12, tidlig kalibrerte felt): AUC 0,497–0,500 per år.

## 5. Vent-målet Y_wait er fasit-i-ettertid [M+S]

`Y_wait(t) = max(0, R_long_net(t+kadens), R_short_net(t+kadens))` (641a463a):
snitt +13,69 bps (M5) / +13,85 (M15) mot Y_long −5,34 og Y_short −5,97.
Jensen-gapet E[max] − max(E) = +13,7 bps er verdien av å velge side etter at utfallet er
kjent — en opsjon som ikke finnes i virkeligheten. Den frosne regelen gir FLAT 57,9 % selv
med perfekt kunnskap om alle tre målene (25,5 % med FLAT = 0) og FLAT 100 % med en
prediktor uten informasjon. corr(Y_wait, siste 95-min |bevegelse|) = 0,25 mot 0,028 for
Y_long: hodet ville lært volatilitet. Riktig vent-verdi er max over *forventede* verdier
ved neste beslutning, som er ≈ 0 uten edge; FLAT = 0 med netto kost er allerede korrekt.
Status: hypotesen er ikke kjørt videre; se VEIEN_VIDERE.md for ny retning.

## 6. Første måling av ukesretning [M]

`weekly_direction_first_look.py` (samme evidensrot): én beslutning per lukket D1-bar
(2019-01..2026-05), inputs = nettopp lukket D1-rad + siste lukkede H4-rad fra den tidlig
kalibrerte forskningspakken (380 felt; kalibrering slutter 28.03.2022), utførbare BID/ASK-
etiketter med den bundne kostpolicyen (sha a48f8e56…: 2 bps per utførelse, provisjon 0,
LONG-finansiering 5,4 %/år på veggklokketid ≈ 10 bps per uke, SHORT 0). Fire ekspanderende
årsholdouts 2022-06..2026-05; ingen etikett bruker juni 2026.

| Holdout-år | Alltid-LONG netto per 1-ukes handel | 2 uker | 4 uker | p(opp) 1 uke |
|---|---:|---:|---:|---:|
| 2022-06..2023-05 | −4,5 | −0,7 | +3,6 | 0,488 |
| 2023-06..2024-05 | +18,7 | +43,6 | +97,4 | 0,552 |
| 2024-06..2025-05 | +55,4 | +119,5 | +245,8 | 0,658 |
| 2025-06..2026-05 | +46,6 | +103,8 | +238,0 | 0,627 |

- Ridge på alle 380 D1/H4-felt valgte maksimal regularisering (alpha 1e5, øvre kant av
  deklarert grid) i alle fold og horisonter, med indre MSE lik konstanten: modellen
  kollapser til driftkonstanten = alltid-LONG. Uten drift (driftnøytral variant) taper den
  68–146 bps per beslutning mot alltid-LONG.
- Ikke-overlappende IC for ridge-rangeringen: 1 uke ~−0,02 i snitt; 2 uker
  +0,09/+0,25/+0,06/+0,13 (n ≈ 25 per fold, SE ≈ 0,2 — et svakt hint, ikke et funn);
  4 uker ~+0,02.
- Tidsserie-momentum (20/60/120/250 D1-barer, long/short og long/flat): ingen slår
  alltid-LONG samlet; 60-dagers hjalp bare i det flate året 2022–23.

Lesning: på ukeshorisont blir trenden fangbar etter kost (alltid-LONG tjente i 3 av 4 år, i
motsetning til på 95 min), men TRAIN 2021–26 inneholder bare ett regime — et oksemarked.
En ukesmodell trent på disse dataene lærer «vær long». Hvorvidt featurene vet *når* man ikke
skal være long, kan ikke måles uten historikk som inneholder bjørnemarkeder (f.eks.
2011–2015, 2008). Med ~50 uavhengige uker per år er 95 %-intervallene ±10–30 bps per
beslutning.

## 7. Konsekvens for designet

Retningsbeslutningen må stilles på skalaen der den finnes (dager–uker), med få beslutninger
og lang holdetid, og måles mot alltid-LONG / kjøp-og-hold valgt før perioden — ikke mot
myntkast. M5/M1 brukes til timing av inngang i trendens retning (lav MAE), ikke til å gjette
neste 95 minutter. Den første ukesmålingen (§6) viser at det som kan læres av 2021–26 alene
er driften; for å lære når man *ikke* skal være long trengs historikk med andre regimer.
Neste steg står i VEIEN_VIDERE.md.

## Ikke undersøkt [N]

- Ukesretning utover drift er bare sett på med en lineær første måling (§6); HGB/sekvens-
  modeller og M5-snapshot-feltene er ikke prøvd på ukeshorisont.
- Regime-/bjørnemarked-oppførsel: kun 2019–2026 (ett oksemarked) finnes på disk; lengre
  gullhistorikk (samme instrument) krever navngitt kilde og manifest før henting.
- Semantisk fidelitet for SMC/geometri/divergens-felt (kun liveness/duplikater sjekket).
- Faktisk utførelseskost (2 bps per utførelse er et valgt scenario, ikke målt).
