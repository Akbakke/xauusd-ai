# Matchet familiebaseline: full TRAIN og senere utvikling

Baseline er gjennomført, men familiesammenligningen er ikke ferdig. Ni egne
familiekvalifiseringer/refits og de forhåndsregistrerte parrede kontrastene
gjenstår. Ingen indikatorfamilie erklæres nyttig eller unyttig av denne baselinen.

## Målt på faktiske data

Frossen kilde: 68d63d9210931663510f0a9cf490387e9c8b29be.
Baseline brukte samme initial, rekkefølge og numeriske oppskrift som den bundne
sign-referansen: ett pass, 47 602 TRAIN-foreldre og 2 976 oppdateringer.
Boot588 trente; boot589 skåret hele TRAIN og 17 288 senere Q1-foreldre uten
nye backward-/optimizersteg. Q1 er gjenbrukt utvikling, ikke urørt TEST.

Fem minutter er forhåndsvalgt hovedhorisont. Tallene gjelder observerte,
ikke-null utfall som modnes strengt før delens cutoff.

| Måling | TRAIN | Senere Q1 |
|---|---:|---:|
| Foreldre | 47 602 | 17 288 |
| Gyldige femminuttersutfall | 47 282 | 17 192 |
| Treff | 49,5347 % | 48,6156 % |
| Treff, alltid opp | 50,9052 % | 51,5821 % |
| AUC | 0,5023466974 | 0,4969610280 |
| Logloss | 0,6935680652 | 0,6943814277 |
| Logloss, fast TRAIN-prior | 0,6929832916 | 0,6927381622 |
| Brier | 0,2502103864 | 0,2506164220 |
| Brier, fast TRAIN-prior | 0,2499180600 | 0,2497955087 |

Modellen predikerte opp på 8 722/47 282 TRAIN-utfall og 3 226/17 192 senere
utfall. Hovedhorisonten er svak også på TRAIN; målingen identifiserer ikke
årsaken og beviser ikke at indikatorene mangler informasjon.

Alle60horisonter, kun deskriptivt: TRAIN hadde AUC over0,5 på59 og bedre
logloss/Brier enn fast TRAIN-prior på55. Senere hadde AUC over0,5 på54,
men logloss/Brier var dårligere enn prioren på alle60. Rangering og
sannsynlighetskvalitet må derfor holdes atskilt. Ingen nye intervaller eller
simultan signifikanspåstand følger av disse tellingene.

## Bevist konsistent innen kontrollens omfang

Kanonisk kvitteringskjede og faktiske checkpoint-/prediksjonshasher bestod
capped kontroll. Begge native vakter bestod. Treningen brukte2 993forwards;
skåringen4 057=2 976TRAIN+1 081Q1forwards, null backward/optimizersteg.
Modell/buffere/RNG/modi er bevart gjennom evalueringene. De to terminalene
rapporterer samme semantiske full-state-segl; etterkontrollen deserialiserte
ikke modell-/optimizertilstanden.

Separat4GiB-kontroll leste initial256, fullTRAIN og Q1-prediksjonene,
kontrollerte dimensjoner/domener og kanoniske rad-/beslutningstidsdigester
fra den fullførte inputauditen med samme manifestlineage, og gjenskapte alle60
modell-/priorpaneler eksakt med eksisterende metrikk-eier. Fast prior ble
gjenskapt fra hele TRAIN og brukt uendret senere. Dette er ny avspilling av
lagrede utdata, ikke uavhengig metrikkalgebra eller ny råfasit-revisjon.
Initial256 og fullTRAIN er forskjellige populasjoner, ikke en matchet
før/etter-læringskurve.

Engangstasken er deaktivert;250W-grensen er beholdt og faktisk ettertrekk
var26,83W ved0%GPU. Treningsporter er stengt. TEST, Exit og handel er urørt.

## Avvik og ubeviste grenser

En foreldet protokollhash i current_work-sammendraget er rettet.
Overleveringskontrollen krevde eksplisitt exit_code i den nye oppsummerte
terminalen; et korrigert dokument binder de faktiske nullkodene og bevarer
originalen uten overskriving.
De to eksakte program-/stagebindingene var allerede riktige; native kilde,
input og numerikk er uendret.

Forbindelsen kom tilbake før Windows-innlogging ved boot588 og589, men var
forsinket ved589. Tjenesten startet automatisk; årsaken til forsinkelsen er
ukjent. Ingen varig nettverksretting hevdes. Alle nye kontrollerte
prebootkvitteringer overlevde begge omstarter.

Inputauditens målte dekning, sjeldne H1-flowfelt og øvrige kunnskapshull
består: se DIRECTION_FAMILY_INPUT_FULL_RESULT_20261030.md.
Familieverdi, bekreftet generalisering og netto handelsøkonomi er ubevist.

## Bundet evidens

Runrot: /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_BASELINE_001

- VERIFIED_TERMINAL_002.json binder original native terminal, capped hovedkontroll,
  separat lagret-prediksjonskontroll og POWER_RETAIN.json.
- ACTUAL_ROOT_REVIEW_001.json binder begge native vinduers kjede og full-state.
- SAVED_PREDICTION_VERIFY_001/RESULT.json og TERMINAL.json binder
  faktisk avspilling, kilde-/inputbookends og eksplisitte evidensgrenser.
- RUNTIME/prefix-fit/window-0002/FINAL_TRAIN.json og OOF.json binder
  sine komplette prediksjonsarkiver.
