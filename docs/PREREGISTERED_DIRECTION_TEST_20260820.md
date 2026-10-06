# Kodebundet selektiv retningsprotokoll

Filnavnet beholdes fordi evaluate_entry_candidate_selective_edge_v1.py binder
denne spesifikasjonen. Avsluttede kandidat-/checkpoint-/substrathistorier er
fjernet. Dette er ingen ny kjøretillatelse; NEXT_RUN_POLICY.json eier nåscope.

## Spørsmål og populasjon

Mål realisert bps etter kostnader på samme deklarerte populasjon, beslutnings-
masker og coverage, ikke gjennomsnittsaccuracy alene. Filtrering kan ikke
skjule tap, ikke-fylte muligheter eller åpne posisjoner i samlet økonomi.
Bundne modellprediksjoner, priser, mål og masks må gjelde samme ekte bytes.

## Fryst grid, null og beslutningsregel

Coverage-grid: {100, 50, 25, 10, 5, 2, 1}% fra den eksisterende eieren.

- Coin-flip null: tilfeldig side på samme mask/populasjon og med samme kostnader,
  beregnet på de bundne bytene. Gamle rapporterte null-/oracle-tall er ikke fasit.
- Autokorrelasjonsbevarende null: circular shift, minst 200 trekk; ikke iid
  permutasjon. Bruk den forhåndsbundne øvre halen.
- PASS(c) krever mean_bps(c) minus mean_bps_coinflip(c) > 2 × SE(c), og
  mean_bps(c) over 95-persentilen til circular-shift-null.
- Protokollens samlede PASS krever én c ≤ 25% på VAL og samme c på eventuelt
  senere autorisert, fryst TEST. Ingen TEST åpnes av dette dokumentet.
- Ingen bestått c på VAL er FAIL, ikke grunn til å velge et nytt grid etter resultatet.

Dette gridet og nullene er beholdt diagnosekontrakt, ikke en post-modell
terskel-/vetoautoritet. Native Entry bruker bare modellens unike argmax.
På lange horisonter må forhåndsvalgt alltid-LONG/kjøp-og-hold også vurderes;
coin-flip alene er ikke en tilstrekkelig økonomisk referanse.

## Porter før resultatpåstand

Minst fem seeds med samme bundne recipe kreves for protokollens
seedstabilitet. Kvalitativ kollapsuenighet gjør resultatpåstanden ugyldig.
Mål faktisk overlap-/tidsblokkstøtte og oppløsning før utfallene tolkes.
Et gammelt månedstall eller antall uavhengige vinduer gjelder ikke nye data.
Coverage og minstepopulasjon må valideres av eksisterende evidenseier.
Ingen terskel flyttes etter resultatet.

Inputintegritet krever actual aktiv routing, kausalitet, riktige klokker,
feltorden, deklarerte aliaser og frosne artefakter. Fysisk M5-overlapp som ikke
brukes i Entry-ruten er ikke automatisk en aktiv duplikatfeature.
Teknisk PASS, nullstegsmetadata og delvise checkpoints er ikke en kandidatmåling.

## Ubeviste grenser

En enkelt gjenbrukt utviklings-VAL-split kan ikke skille ikke-stasjonært signal
fra manglende edge, og viser ikke walk-forward-robusthet eller live-atferd.
Ny v38-modell er ikke målt her. Train/serve krever egen faktisk bundle-paritet.
TEST forblir forseglet; ingen native trening, broker, handel eller spending åpnes.
