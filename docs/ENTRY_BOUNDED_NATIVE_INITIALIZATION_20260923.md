# Avgrenset native Entry-læring fra checkpoint844 — 23.09.2026

## Konkret blokkering og minste utvidelse

Den eksisterende canonical smoke-ruten kunne bare starte med tilfeldige vekter.
Det ville ikke måle n-step-endringen på den trente funksjonen vi undersøker.
To valgfrie, recipe-bundne argumenter binder nå råvekter fra et bevart
candidate-checkpoint: initial_checkpoint_path og initial_checkpoint_sha256.
De tillates bare for én bounded canonical FP32 smoke. Candidate og legacy
recipe får ikke bruke dem.

Initialisering kontrollerer checkpoint-hash, hashbundet parent-sessionkontrakt,
samme TRAIN/VAL/M5/lifecycle-artefakter, dataset-ID, eksakte parameterformer
og samme normalisering. Parent-checkpoint, lærer, optimizer, EMA og cursor
bevares. Ny smoke har ny optimizer, ny EMA og den nåværende v9-funksjonen.
Dette er en ny initialisering, ikke eksakt resume eller funksjonsparitet med v8.

## Før/etter-bevis i eksisterende trener

Før første optimizersteg måles nåværende funksjon på samme deterministiske
TRAIN- og senere utviklings-VAL-utvalg som måles etterpå. En separat,
frossen kopi av den initialiserte v9-funksjonen er referanselærer i begge
målinger. Den ordinære treningens target-refresh beholdes.
Tidligere v8-lærer og historiske sammenligninger er fortsatt bevart.

Målingen kaller eksisterende native validate og samler allerede beregnede
Entry-Q, valgte utfall og begge siders modellvalgte EXIT-indeks/åpen-status.
Ingen ekstra modellpass brukes for å samle disse observasjonene.
Før/etter-evalueringene selv er nødvendige nye pass. RNG og treningssampler
påvirkes ikke. Parameterdigest for både modell og referanselærer må være
uendret gjennom målingen.

Rapport og råvekter etter fit lagres ved siden av smoke-bundle i
*.learning_comparison. Dette er forskningsbevis uten promoteringsautoritet;
ordinære checkpointkrav gjelder fortsatt. Åpne posisjoner markeres, og ingen
maksimal holdetid innføres. Tidsbruk og det arkiverte kostnadsscenariet
etterberegnes fra de lagrede radene og de allerede bundne historiske prisene.
Treningsreward er fortsatt spreadinkludert brutto.

## Kontroller og neste ene kjøring

- Helper er kontrollert mot ekte checkpoint844 og parent-kontrakt: eksakte
  råvekter og normalisering, feil checkpoint-hash avvist, null optimizersteg.
- 8 recipe-tester og 10 launcher-tester bestod. Tester med fixtures beviser
  bare argumentbinding og avvisning; checkpoint-kontrollen brukte ekte bytes.
- Før trening må en fersk immutable recipe binde ferdig committet kilde,
  checkpoint og alle eksisterende dataartefakter. Ingen gammel recipe gjenbrukes
  som oppstartsmyndighet, og ingen gammel kandidat resumeres.

Neste forhåndsvalgte beregningsbudsjett: 512 deterministisk samplede TRAIN-rader
og 512 senere utviklings-VAL-rader, batch8, én passering/64 optimizersteg.
Dette er et eksplisitt smoke-budsjett for én læringsmåling, ikke en ny grense
for handelens varighet, en full epoch eller en terskel-/modellrunde.
Alle øvrige hyperparametre kommer fra den eksisterende V12 smoke-recipen.
Avslutt vurderingen med faktiske LONG/SHORT/FLAT-valg, markerte åpne tap,
kostnadsscenario, tid/kapitalbinding og før/etter-forskjell. Teknisk PASS
alene er fortsatt ikke en bestått læringsport.

Status ved skriving: ingen ny trening startet. TEST og GX1_CURRENT er urørt.
