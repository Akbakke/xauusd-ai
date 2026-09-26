# Entry: målereparasjon og avgrenset neste beslutning — 24.09.2026

## Nåstatus

Claude er ferdig ifølge operatøren. Kanonisk kilde ved overtakelse var
GX1_ENGINE / audit/v9-premiere-20260905, HEAD
243d79d8357bdc9138877e351c86d192256c3d7a, clean. Ingen forskningsjobb eller
tungjobblås var aktiv ved kontroll. Codex la inn den tidligere bestilte patchen
12:22:29 UTC etter kontroll av alle fire før-hasher og eksklusiv lås.

Evidensrot:
`/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923/ENTRY_MEASUREMENT_REPAIR_20260924`.
APPLIED.json og PREPARATION.json binder patch og før/etter-kilde.
De to berørte testfilene fullførte med returkode 0 og 37 beståtte tester bak
gx1_capped_run.sh (audit, 4 GiB, 512 MiB swap, én numerisk arbeider).
Se linux_targeted_checks.log. Ingen full suite eller native trening.

Fem observerte feil er rettet: tape-grense før materialisering i begge lesere;
purge ved indre tidsdeling; indre sentrering/skalering for ridge; HGB-refit på
full tillatt fold; kilde-/konfigurasjons-/inputbinding for gjenbruk av cache.
Gamle resultater og checkpoints er bevart. Teknisk PASS er ikke bedre Entry.

## Hva Claudes rapport faktisk kan si

- De prøvde oppsettene har ikke dokumentert robust, senere bekreftet Entry-edge.
  Det er ikke et bevis mot all prisinformasjon, teknisk analyse eller sekvenslæring.
- Lav marginal Pearson-korrelasjon utelukker verken lekkasje eller nyttige
  samspill. Påstanden «Lekkasje: ingen» følger ikke av denne revisjonen.
  49 referansesjekker verifiserer heller ikke semantikken i alle 1 072 kolonner.
- Juni er gjenbrukt utviklings-VAL. Den er ikke en ny, uberørt bekreftelse.
- Å unngå en kjent tapsmåned i ettertid dokumenterer ingen gjennomførbar
  abstensjonspolicy. ATR-regimefilteret som faktisk ble prøvd, manglet støtte.
- Trefferate alene avgjør ikke lønnsomhet eller nødvendig horisont; gevinst- og
  tapsstørrelser, utvalg, spread, øvrige kostnader og eksponering må med.
- «Beste konstante side» valgt fra samme periodes utfall er en diagnostisk
  etterpåklok referanse. Den må skilles fra en side valgt før perioden.
- Pooled PDH/H4-funnet er et svakt forskningsspor, valgt fra 144 celler og uten
  uavhengig bekreftelse. Ingen implementering eller edge-godkjenning følger av det.

Tidsangivelsene i rapportens pooled-oppsettavsnitt har en enhetsfeil:
tapemanifestet er M5, og målet bruker posisjon + horizon_bars. h12/24/48/96 er
1/2/4/8 timer med femminuttersbarer; markedspauser kan forlenge veggklokketiden.
Maskinrapporten setup_edge_pooled_v1/report.json gir PDH/H4, n=299:
+4,646/+7,179/+9,562/+13,279 bps på disse fire horisontene. Dette er spread-
inkluderte close-fill-forskningsutfall, ikke netto porteføljeavkastning.
Den er positiv i samlet TRAIN, men ikke uavhengig bekreftet.

## Én fast før/etter-kontroll

Reproduser bare run4s snapshot_mtf / exec_close_h12 / atr / ridge, seed 0,
med samme fire TRAIN-folds og gjenbrukt juni. Behold den opprinnelige
horisontlisten 12/24/48/96/288 slik at ytre purge fortsatt er 289 tape-barer;
bare h12 fittes. Samme alpha-grid, beslutningsregler og rapportdekninger.
Primærlesning er argmax_flat ved full dekning og ved den allerede utpekte
1 %-halen; 5 % er diagnostikk. Ikke velg en ny vinner etterpå.

Den eldre run4-kilden predaterer også Claudes rettelse av siste TRAIN-folds
utfallsgrense. Sammenlign derfor gamle og nye prediksjoner på identiske
tidsstempler og identiske realiserte utfall, og rapporter utelatte gamle rader.
Dette er en samlet før/etter-kontroll av reparert instrument, ikke en isolert
kausal effekt av én kodeendring. Kontroller fit-antall og inputmanifestene.

Ikke gi ny statistisk PASS ut fra instrumentets eksisterende overlappstest.
Rapporter punktestimater, sidefordeling, FLAT og enkle referanser; eventuell
edge krever horisonttilpasset usikkerhet, senere uavhengige data og full økonomi.
Ingen ny modelklasse, featurejakt, terskelsøk, full epoch eller TEST.

## Retning videre

Bevar prosjektets data, kausalitet og infrastruktur. Neste strategiske spørsmål
er om en på forhånd definert situasjon gir bedre senere utfall enn en relevant
referanse med tilsvarende trend/side og eksponering. Brudd over forrige dags
topp med H4-opptrend er en eksisterende hypotese, ikke en ny kjøpsregel eller
en bekreftet løsning. Gjenbruk ferdige analyser før mer beregning.

## Fullført før/etter-kontroll — 24.09 kl. 12:28:58 UTC

PAIRED_H12_REPLAY fullførte med rc=0 på kilde ee42853573b59247139428f897de4afab115d80a.
Fem konfigurasjoner (to sider per fold/stadium), 129,67 s instrumenttid.
PAIRED_PLAN.json ble skrevet før kjøring; PAIRED_TERMINAL.json binder plan,
kilde og logg. Sammenligningen fullførte bak audit-vakten med rc=0.

Alle 263 557 gamle/nye evalueringsrader hadde identiske tidsstempler og eksakt
like realiserte LONG/SHORT-utfall. Ingen gammel grenseoverskridende rad falt ut
for akkurat h12. Fit-antall og TRAIN/tape/MTF-manifestene matchet.
Hver indre fit purget 266–269 beslutningsrader ved 289 tape-barers purge.
Valgt alpha var fortsatt 10 000 på begge sider i alle fem stadier.
Det forklarer hvorfor den korrigerte indre modellseleksjonen ikke endret
full-refittens valg mer enn små numeriske forskjeller: 24 av 263 557 argmax-valg
endret seg, ingen i juni. Dette begrunner ikke et nytt alpha-søk.

| Periode | Høyest rangerte 1 %, gammel → rettet, bps | Antall |
|---|---:|---:|
| 2022–23 | +6,826 → +6,826 | 633 |
| 2023–24 | +4,310 → +4,310 | 643 |
| 2024–25 | +1,389 → +1,389 | 653 |
| 2025–26 | +17,774 → +17,774 | 653 |
| Juni 2026, utviklings-VAL | −42,478 → −42,478 | 56 |

Juni ved full dekning: 4 964 handler, 3 702 LONG / 1 262 SHORT og 545 FLAT,
−5,308 bps per valgt handel; uendret. Ved topp 5 %: −22,414 bps; uendret.
Den høyest rangerte 1 %-halen er fortsatt 56 LONG og ingen SHORT.
Rapportens generelle utsagn om at argmax aldri velger FLAT gjelder derfor
ikke denne ridge-målingen.

**Konklusjon:** de fem reparasjonene er nødvendige for korrekt måling, men
de forklarer ikke juni-tapet i denne avgrensede ridge-kontrollen. Ingen bedre
Entry eller bestått læringsport er påvist. Dette resultatet gjelder ikke
automatisk alle andre mål eller den rettede HGB-refitten.

Topp-prosentilene rangeres med hele evalueringsperiodens scorefordeling.
De er retrospektiv seleksjonsdiagnostikk, ikke dokumentasjon av en kausal
inngangsterskel som kunne vært brukt på hvert daværende tidspunkt.
Utfallsmeanene er spread-inkluderte close-fill-målinger før øvrige kostnader;
overlappende handler er ikke en gjennomførbar portefølje.

Neste avklaring er avgrenset situasjonsvalg: om den eksisterende PDH/H4-hypotesen
tilfører noe utover en kjent bullish H4-tilstand med tilsvarende eksponering.
Den må vurderes mot en referanse som kan velges før utfallet, og deretter
bekreftes på senere uavhengige data. De gjenbrukte TRAIN/juni-resultatene kan
ikke alene gi slik bekreftelse. Ingen ny større treningsjobb er startet.
