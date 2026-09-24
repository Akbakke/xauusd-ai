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
