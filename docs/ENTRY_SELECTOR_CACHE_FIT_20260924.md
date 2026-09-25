# Policy-konsistent Entry-selector: én fit, STOP — 24.09.2026

## Beslutning

STOP_SELECTOR_HYPOTHESIS. Den ene forhåndsbestemte Ledoit–Wolf-readouten
forbedret ikke Entry-verdiene på check-utvalget og ga negativ netto. Denne
selector-hypotesen er lukket. Ingen terskel-, feature-, split- eller lambda-
tuning, og ingen ny fit av samme hypotese.

## Hva som ble målt

Den native representasjonsuttakingen var allerede fullført 19.09.2026; den
måtte ikke relanseres. Den brukte 16 Entry-forwards på 256 TRAIN-rader, null
Exit-rollouts og null optimizersteg. Original Entry-Q, Exit-kontekst og
rollout-kontrakt hadde eksakt paritet. Cache SHA-256:
790a285521c5c9d51e467ab29983be183902a3ebfef3fc2add0222e92adfe806.

Fitten er én fryst lineær readout over 128 representasjonskoordinater.
Ledoit–Wolf-kovarians og felles lambda ble beregnet kun fra 127 rader
(juni–september 2025). Interceptet er upenalisert. Koeffisientene ble lagret
før check-avlesningen. De 129 check-radene er oktober 2025–februar 2026; alle
fit-handler sluttet før første check-entry. Dette er gjenbrukt TRAIN, ikke
uavhengig VAL eller TEST.

| Størrelse | LONG | SHORT |
|---|---:|---:|
| Check MSE, kandidat (bps²) | 813.133 | 153.648 |
| Check MSE, original512 (bps²) | 650.644 | 116.131 |
| Check MSE, fit-konstant (bps²) | 655.252 | 112.116 |
| Sentrert MSE, kandidat (bps²) | 802.094 | 147.011 |
| Sentrert MSE, original512 (bps²) | 648.335 | 110.160 |
| Sentrert MSE, fit-konstant (bps²) | 651.607 | 111.042 |
| Pearson korrelasjon | 0.016 | -0.311 |

Kandidaten tapte mot begge referanser på MSE og sentrert MSE for begge sider.
Uendret argmax valgte 62 LONG, 0 SHORT og
67 FLAT.

## Netto på check

| Måned | Muligheter | Valgt | Netto sum (bps) | Netto per mulighet (bps) |
|---|---:|---:|---:|---:|
| 2025-10 | 31 | 15 | -5.332 | -0.172 |
| 2025-11 | 24 | 15 | -61.629 | -2.568 |
| 2025-12 | 33 | 10 | -210.686 | -6.384 |
| 2026-01 | 27 | 13 | -57.970 | -2.147 |
| 2026-02 | 14 | 9 | 43.374 | 3.098 |

Over alle 129 uavhengige muligheter ble netto summen
-292.244 bps
(-2.265 bps per mulighet, 62 valgte).
I eksisterende én-posisjonsreplay var netto -239.172 bps,
med 59 utførte og
3 hoppet over mens posisjon var åpen.
Begge økonomikrav feilet.

Største valgte vinner var 167.932 bps og utgjorde
41.3% av positiv gevinstsum.
Uten den handelen var netto -460.176 bps.
Nettoen er negativ i fire av fem måneder.

## Fit og avgrensning

Ledoit–Wolf shrinkage delta=0.029352,
lambda=0.00060936, effektiv frihetsgrad
49.47. Check-data ble ikke brukt
til fit eller valg av lambda.

Den nødvendige kildetesten bestod med 19/19 under
gx1_capped_run.sh --class audit (4 GiB RAM, 512 MiB swap).
Én innledende planversjon stoppet før data ble lest fordi den manglet tre
operator-kontraktsfelt; ingen RESULTS-mappe eller fit ble startet. Den planen
er bevart som PLAN_ATTEMPT_001.json; revidert plan SHA-256:
e8adc89c5d6737f3519a73db7f45063388dc4ad72a88fd0e069e52e20a3cb526. Revidert operator kjørte på CPU i
0.077 sekunder.

Ny fit: 1. Nye modellforwards: 0. Optimizersteg: 0. Exit-rollouts: 0.
Originale modellvekter og checkpoint er uendret. TEST, CONTROL, full epoch,
full VAL, live og papirhandel ble ikke brukt eller startet.
training_enabled forblir false.

## Bevisbindinger

- Plan: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_ENTRY_POLICY_REPRESENTATIONS_20260919/ENTRY_SELECTOR_FIT/PLAN.json — SHA-256 e8adc89c5d6737f3519a73db7f45063388dc4ad72a88fd0e069e52e20a3cb526
- Resultat: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_ENTRY_POLICY_REPRESENTATIONS_20260919/ENTRY_SELECTOR_FIT/RESULTS/RESULT.json — SHA-256 ea574902368cb4a812c8eb32c801cbb9c8708a88922ba4f384d2c0b93ff79690
- Frosset readout: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_ENTRY_POLICY_REPRESENTATIONS_20260919/ENTRY_SELECTOR_FIT/RESULTS/FROZEN_FIT.json — SHA-256 12aadfea3b7997b60ad767bbc03aa652a2a35987e5c63aafab7e437ee57a4456
- Selector binding: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_ENTRY_POLICY_REPRESENTATIONS_20260919/ENTRY_SELECTOR_FIT/RESULTS/SELECTOR_BINDING.json — SHA-256 98f42b1699cba876dbfa3e5458dd1fc418bf7a7ac8f6055591f194dc11155670
- Native observation: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_ENTRY_POLICY_REPRESENTATIONS_20260919/frozen_entry_representations/OBSERVATION.json — SHA-256 768caa5a99a4696a819aa1832cfd4f7c272306b5b1a9998d8149c61cdb68d729
- Preflight-test: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_ENTRY_POLICY_REPRESENTATIONS_20260919/ENTRY_SELECTOR_FIT/PREFLIGHT_TEST_20260924.json — SHA-256 a2ad42122015beaf765d541a6e580ce19339a7208ded2314d175d2edfe48cc3b
- Kildecommit: 25a24acc81976966750f1f00b27489fc10bd306e (work/gx1-current)

En eventuell senere studie må være en ny, separat hypotese med sin egen
forhåndsbestemte target, modell og kronologiske utviklingsmåling. Resultatet
her gir ikke grunnlag for å utvide trening eller påstå en edge.
