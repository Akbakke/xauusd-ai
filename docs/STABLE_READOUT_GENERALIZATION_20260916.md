# Frosset kandidat — lærbarhet er ikke varig generalisering

Forsøket er nå **frosset før kronologisk evaluering**. Eksisterende verdilag
viser lærbarhet på gjenbrukt TRAIN, men generalisering og økonomisk verdi er
ikke bevist. Brukerens krav er en varig løsning som tåler ulike markeder;
videre tilpasning mot det samme kontrollutvalget er stoppet.

En ustabil Entry-tilpasning ble stabilisert med én regel beregnet kun fra
TRAIN512. Separat TRAIN128: Entry LONG-MSE 32,26→25,40, SHORT 80,44→27,25;
begge slår konstantbaseline. LONG bedres i 10/12 måneder, SHORT i 8/12;
mars-LONG er fortsatt klart verre. Frosne Entry/Exit-lag er kontrollert sammen
med eksakt cacheparitet, og Exit-forbedringen består. Ingen checkpoint er promotert.

Neste er en på forhånd bundet kontroll på 256 Entries fra juni2026, valgt med
eksisterende seed uten modellutfall. Juni er gjenbrukt utviklings-VAL, ikke
urørt holdout. Vekter, mål og utvalg er frosset. Ingen TEST, ny trening eller
full VAL er åpnet. Se docs/STABLE_READOUT_GENERALIZATION_20260916.md.

## Hvorfor stabilisering var begrunnet

Den uregulerte tilpasningen ga LONG +48,50 Bps mot lærerens -16,27 på én
separat rad (parent301063). Radens regresjonsleverage var 57,96 mot maks0,999
på TRAIN. Feilens sentrerte MSE var 51,04 og biasleddet bare0,72; en biasrettelse
ville ikke løst problemet. Ingen rad ble fjernet eller særbehandlet.

Én Ledoit–Wolf-regel brukte kun de512 trente representasjonene til å bestemme
regularisering: delta0,0111847, lambda0,00284540. Samme lineære lag, alle tre
handlinger og upåvirket backbone. Ingen valideringssøk eller terskelklipping.
Koeffisientene ble frosset før det separate utvalget ble lastet for evaluering.
Formelen er dokumentert i
[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html).
Denne regelen stabiliserer regresjonen; den garanterer ikke fremtidig handelsfordel.

## Målt og fortsatt uavklart

| Separate TRAIN128 | Gjeldende96 | Frosset kandidat | Trent konstantbaseline |
|---|---:|---:|---:|
| Entry LONG MSE | 32,2571 | 25,4049 | 26,4586 |
| Entry SHORT MSE | 80,4376 | 27,2471 | 61,1691 |
| Exit LONG MSE | 1070,5441 | 1006,7374 | 1071,2324 |
| Exit SHORT MSE | 1069,4183 | 1052,8265 | 1070,1125 |

Entry velger43 LONG/37 SHORT/48 FLAT. Lærerregret3,2933→1,4171 Bps.
LONG-korrelasjon0,1534→0,3849 og SHORT0,3473→0,7640. LONGs sentrerte feil
er likevel litt verre enn dagens nesten konstante output (23,8885→24,7242).
Mars-LONG-MSE er138,48 mot24,55. Ekstremraden er fortsatt feil: +15,17 mot-16,27.
Dette skal ikke skjules av samlet gjennomsnitt eller omtales som robusthet.

Begge lag er kontrollert sammen gjennom eksisterende token-/fuse-eiere.
Gamle mellomresultater og nytt Entry-output ble reprodusert eksakt. Bare de
siste lagene ble evaluert fra cache, ingen nye backbone-forwards eller tilpasninger.
Ankerreferansenytte på67 separate Entries er+1,0661 Bps over ALLE Entries,
men under Q_mu-referansen, ikke kontant-PnL under modellens Exit-policy.

## Neste avgjørende kontroll

Vekter og 256 juni-Entries er frosset i planen lenket fra maskinrapporten.
Utvalget bruker eksisterende seed20260911/split_salt1 og bare radidentiteter.
Rapporter retninger/uker, usikkerhet, kostnader og alle valgte handler/åpne
posisjoner. Juni har tidligere vært utviklings-VAL. Ingen senere tuning skal
presenteres som uavhengig bekreftelse på samme utvalg. TEST forblir forseglet.

Konkret kjøreblokkering: eksisterende native eier krever hele5508 VAL-Entries
og en epoch-EMA (eller den eksakte gamle32-stegskalibreringen). Denne ONLINE-
kandidaten er ingen av delene. Ikke omskriv EMA/epochhistorikk, kjør nye
treningssteg for å komme rundt porten, eller bruk en separat VAL-runner.
Minste neste kodearbeid er eksplisitt, skrivebeskyttet binding av kandidaten
og det avgrensede utvalget gjennom eksisterende native eiere og vakter.
Planen er ikke starttillatelse; training_enabled er fortsatt false.

[Eksakte bindinger og målinger](../handover_snapshot/STABLE_READOUT_REVIEW_20260916.json).
Alle gamle resultater og originale checkpoints er bevart.

## Evaluatorstøtte kontrollert

Evaluatordelen er nå kontrollert for det frosne utvalget: 12 nye og40 eksisterende
VAL-tester består. Originale rad-ID-er, pause/gjenopptak, åpne posisjoner/kostnader
og uendrede modellparametere utenom de frosne verdilagene er verifisert. Kandidaten
identifiseres som ONLINE, aldri som epoch-EMA. En delvis VAL kan ikke åpne full-VAL-porten.

Native kampanjekobling og paret Exit-måling mot de frosne Q_mu-målene gjenstår.
Ingen juni-evaluering eller nye treningssteg er kjørt. Dette er tekniske bevis,
ikke generalisering eller lønnsomhet. Se handover_snapshot/BOUNDED_VAL_CORE_REVIEW_20260916.json.

## Native evaluering bundet, ikke startet

Den skrivebeskyttede native-koblingen er nå kontrollert. 207 målrettede tester
består, inkludert eksakt reward-/klokkeparitet mot TRAIN, helgegap, sensur,
bevart bootstrap, uendret treningscursor og sperre mot nye optimizersteg.
Begge faktiske modeller bruker samme frosne Entry-lærer og samme opprinnelige
Exit-boundary-lærer. Exit-feil måles på state0 per valgt Entry; dette er ikke
feildekning av alle mulige holdetilstander. Native økonomi følger hele forløpet.

NEXT_RUN_POLICY åpner bare én eksisterende native evalueringsinvokasjon per
frosset variant, med256 forhåndsvalgte juni-Entries og null treningssteg.
Riktig kildebinding, fysisk omstart og alle eksisterende vakter kreves fortsatt.
Ingen senere VAL er kjørt ennå; full epoch/full5508 VAL og TEST forblir stengt.
Se handover_snapshot/NATIVE_FROZEN_READOUT_REVIEW_20260916.json.

## Oppdatering2026-09-17: baseline ferdig, numerisk preflight-rettelse

Baseline er ferdig med256FLAT og null treningssteg. Kandidaten stoppet før
rollout på batch256/16-avvik0,00048828125Bps, med identiske handlinger. Bare
assert-grensen/loggen endres til absolutt0,001Bps; relativ toleranse0 og eksakt
handlingslikhet består.12 målrettede tester består. Modeller, vekter, mål,
utvalg, prediction/rollout/økonomi er uendret. Baseline gjenbrukes med original
89c-proveniens; AST er kontrollert lik utenom vaktfunksjonen. Én erstatnings-
invokasjon for kandidaten er avgrenset i NEXT_RUN_POLICY. Ingen retuning.
Se CURRENT_HANDOVER.md og handover_snapshot/FROZEN_VAL_NUMERIC_PREFLIGHT_20260917.json.
