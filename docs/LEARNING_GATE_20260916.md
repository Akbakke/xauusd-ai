# Beslutning før mer omfattende GX1-trening — 2026-09-16

Gjeldende2026-09-17: fast256 er ferdig og forkastet for utvidelse. Entry-valg
og senere Exit-overføring består ikke kravene. Se
[NATIVE_PREFIX_FIXED256_REVIEW_20260917.md](NATIVE_PREFIX_FIXED256_REVIEW_20260917.md).
Ingen økonomi, gjentakelse, større trening eller TEST er åpnet.

Brukerens prioritet: **Krev målbar læring før mer omfattende trening.**
Målet er positiv kostnadsjustert netto Bps, ikke flest mulige tekniske PASS.
Denne beslutningen endrer ikke økonomi, modell, tapsvekter eller handelsgrenser.

## Hva vi faktisk vet

Den brede parete TRAIN-kontrollen95→134 brukte 1 024 ulike Entries og 4 096
native overganger over tolv måneder, mot samme frosne lærer91. Entry valgte FLAT
på alle 1 024 i begge checkpoints. Entry-MSE gikk12.5654→12.6256. Exit-HOLD-MSE
bedret seg heller ikke samlet; LONG18.2180→18.2337 og SHORT18.0657→18.1232.
Prognoser25–120min bedret seg på TRAIN; dette er ikke senere VAL eller profitt.
1 021 av1 024 lærerbeslutninger var like med/uten fremtidig fortsettelsesverdi.
Se handover_snapshot/BROAD_TRAIN_LEARNING_REVIEW_20260916.json og tilhørende
BROAD_TRAIN95_134-resultater. Hele målingen og alle caches skal gjenbrukes.

Den stoppede134-kjøringens ufullstendige VAL ga bare HOLD i64 785 368 evaluerte
sidebeslutninger. Dette er gjentatte hypotetiske forløp, ikke like mange valgte
handler. Ingen full-policy profitt kan utledes fra denne delkjøringen.

V4-økonomien retter tidligere HOLD-regnskapskonflikt. Ny opt-in femstegs target
følger observerte successors under frossen kausal lærerpolicy. Den sprer reward
over flere beregningssteg og beholder bootstrap. Læreren kan fortsatt ta feil;
dette er én testbar hypotese, ingen dokumentert læringsgevinst. Ingen fast
holdetid eller tapsgrense er innført. Minnerettelsen deler bare frossen forward.

## Oppdatering95→96

Punkt1 og2 nedenfor er ferdige og skal ikke gjentas. Målingen åpner ikke
større trening: lokal Entry-fit bedres, bred Entry blir litt verre og allFLAT.
LONG Exit blir allHOLD, hovedsakelig gjennom en konstant verdiøkning.
Se CURRENT_HANDOVER.md og handover_snapshot/FROZEN_TRACE_LEARNING95_96_REVIEW_20260916.json.
Årsaksarbeidet under «Hvis resultatet uteblir» er nå gjennomført og dokumentert i
[årsaksrapporten](VALUE_LEARNING_CAUSE_20260916.md). Svak frossen videreverdi,
sideavhengig lærerpolicy og verdihode-/representasjonsforskyvning er målt.
Den etterfølgende klippehypotesen er nå prøvd og forkastet: sterkere Exit-
oppdateringer bedrer faktisk TRAIN512-fit litt, men forverrer separat TRAIN128
med samme femstegsmål. Entry forblir FLAT; Exit kollapser til sideavhengige
konstantvalg. Se [ferdig forsøk](EXIT_PRIVATE_CLIP_LEARNING_20260916.md).
Klippekoden tilbakeføres. Alle referanser, resultater og nye32-batchcacher
bevares og gjenbrukes. Målkomponentene er målt i TARGET_COMPONENT_CAUSE_20260916.md;
ankerutfall er deretter kontrollert på 275/512 Entries. Se
ENTRY_ANCHOR_OBSERVED_OUTCOMES_20260916.md. Ingen læringsport er bestått. Ingen
full epoch/full VAL, repetert32-kontroll eller nye tapssøk åpnes.

## Måling og beslutning, i denne rekkefølgen

1. Fullfør referanse32 mot16+16 og sammenlign faktisk modell, frossen lærer,
   optimizer, EMA, RNG og progresjon. Bare checkpointnummer og sessionidentitet
   kan avvike. Teknisk kontroll alene åpner ikke full epoch/VAL.
2. Bruk de lagrede faktiske TRAIN-inputene og femstegsmålene til en paret ONLINE-
   vurdering95 mot reference96. Samme input, mål, normalisering og eval-modus.
   Dokumenter Entry/Exit-feil, baselinefeil, verdi-/handlingsfordeling og eksponering
   for treningsbatchene. Førstebatchloggen før oppdatering er ikke et etter-resultat.
   Det cachede femstegsutvalget er16 Entries; en gevinst der beviser bare lokal
   lærbarhet. Gjenbruk også den bredere1024-kohorten for uavhengige prognoser og
   Entry/Exit-outputdiagnostikk; ikke forveksle gamle ettstegsmål med nye femstegsmål.
3. Før en videre avgrenset native læringskjøring: fastlegg sammenligningsutvalg,
   frosne mål/lærer, budsjett/sluttpunkt og beslutningskriterier i eksisterende
   plan/policy. Bruk ekte TRAIN-rekkefølge, ikke fixed64-replay. Bevar original95
   og begge tekniske armer. Ingen ny modell, target-refresh eller vektnullstilling
   uten konkret evidens for at det er nødvendig.
4. Krev reduksjon av Entry- og Exit-feil mot konsistente mål og relevante enkle
   baselines, med resultater fordelt på retning og tidsperioder. Skill handlingene
   FLAT/LONG/SHORT og HOLD/EXIT; klassesammenbrudd er ikke kvalitet. Rapporter
   faktisk markedsretning etter Entry, kostnader og usikkerhet. Lærerimitasjon
   alene er ikke økonomisk fasit. Bedre hjelpeprognoser alene er heller ikke nok.
5. Før større omfang må kvalitet også undersøkes kronologisk utenfor batchene
   som nettopp ble trent. Juni er utviklings-VAL, ikke urørt holdout. Full økonomi
   må bruke native evaluering, alle valgte handler, faktisk posisjonsbruk, cash
   pluss åpen verdi og kostnader. Positive lukkede handler alene teller ikke.
   Påkrevd GPU256-paritet, gjennomstrømning og resume må ha riktig kildescope.

Ingen vilkårlig prosentgrense for læring innføres her. Før/etter-effekter må
sammenholdes med baseline, målevariasjon, utvalgsstørrelse og tidsperioder, og
beslutningen må begrunnes. Et uklart utfall er ikke PASS og ikke grunnlag for
automatisk større trening. TEST skal aldri brukes for disse beslutningene.

## Hvis resultatet uteblir

Ved fortsatt flat Entry/Exit-læring: stopp utvidelsen og lokaliser den konkrete
mål-/gradient-/credit-assignment-svikten i eksisterende løsning. Ikke kjør en
ny epoch i håp om at problemet forsvinner. Hvis selvstendige prognoser gir signal
mens verdi-/handlingslæringen svikter, vurder en enklere Entry/Exit-oppdeling som
et eksplisitt neste forslag. Hvis heller ikke signalet holder på senere data,
revurder datagrunnlag og hypotese før mer arkitektur. Ingen lønnsomhet loves.

## Gjenbruk og håndover

Les CURRENT_HANDOVER.md for eksakte artifact-/checkpointreferanser og aktuell
terminalstatus. RUNNING_NATIVE_CALIBRATION.json er operatørstatus, ikke launch-
autoritet. NEXT_RUN_POLICY.json har training_enabled=false. Historiske planer,
COMPLETED_RUN.json og tidligere håndoverinstrukser skal ikke startes på nytt.

Oppdatering: eksisterende120-minuttersprognose har kostnadsjustert TRAIN-signal
samtidig som Entry er allFLAT. Se docs/FORECAST120_ECONOMIC_SIGNAL_20260916.md
(for filer under docs: FORECAST120_ECONOMIC_SIGNAL_20260916.md). To sensurerte
forløp gjør samlet sluttidsregnskap ufullstendig. Ingen læringsport er åpnet.
Neste arbeid er én kausalt og matematisk begrunnet rettelse av videreverdien;
frosne prognoser og alle ferdige målinger skal gjenbrukes.

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
