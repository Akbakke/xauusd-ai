# Native-forberedelse og full repo-gjennomgang — 27.09.2026

Brukeren har etter NO-GO-målingen presisert at det konsoliderte native-oppsettet
ikke er epoch-trent, og bedt om å ferdigstille gjenstående arbeid, deretter rydde,
feilsøke og kartlegge hele repoet før eventuell trening. Dette åpner data- og
kodeforberedelse. `training_enabled=false` består; ingen optimizersteg, full epoch,
VAL-evaluering eller automatisk treningsstart er autorisert.

Ridge/HGB-resultatet og dets forhåndsregistrerte NO-GO bevares. Den avgrensede
snapshot-/ukemålingen er ikke et kapasitetsbevis for native sekvenslæring og
samspillet mellom Entry og Exit. Kravet om ny informasjon/kostendring fra den
forrige arbeidsregelen er ikke et absolutt forbud mot brukerens nye forberedelse.

## Første mål: komplette og konsistente native inputs

Bruk full-mode-kjeden som allerede eier M1/M5-par, signalmanifest, separate
Entry/Exit-flater, sekvenser og forseglet TEST. De opprinnelige tidsvinduene beholdes:
historikk fra juni 2010, TRAIN juni 2011–mai 2025, utviklings-VAL juni 2025–juni
2026, mekanisk forseglet TEST juli–august 2026. Ingen TEST-resultater beregnes
eller brukes til valg. Native modellvekter oppdateres ikke.

Ferdig tidlig M5 C0-cache fra `HISTORY2009W_EARLY_DECISION_20260927` gir de
frosne parameterne til parbyggeren. Kalibreringen beholdes fra 2009-06-01 til
2013-01-01 22:00 UTC, indre registergrense 2012-04-11 22:00 UTC. Nytt par og
dataset-run-id krever nye eksakte lineage-bindinger; eksisterende produsenter
lager disse. Ingen manuell endring av artefaktmanifester eller gjenbruk av
avbrutt C0-output som om det var ferdig.

Konkret kodeblokkering: kjeden bandt squeeze-perioden ubetinget til modellens
TRAIN-periode, og registerfit hadde ingen separat start. Eksplisitte fit-vinduer
føres nå gjennom den samme kjeden til hver eksisterende eier. Utelatte grenser
bruker samme TRAIN-verdi som før; oppgitte grenser kontrolleres for UTC, rekkefølge
og slutt senest ved TRAIN-slutt. Artefaktenes faktiske vinduer og parbinding må
stemme eksakt. Ni fokuserte kjedetester består, med eksekverbar kontroll av tidlig,
framtidig og ugyldig kalibrering; dette er mekanikk, ikke læring.

Scope og kvitteringer:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_PREPARATION_20260927/PLAN.json`.
Nye byggartefakter:
`/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_NATIVE_EARLY_20260927`.
Én CPU-produsent om gangen gjennom eksisterende capped-run-vakter. Ferdige
kalibrerings- og historikkbevis bevares. Post-rebuild- og lifecycle-bindinger er
ikke erklært klare før de faktisk er kontrollert.

## Deretter: hele repoet før en treningsbeslutning

Kartlegg alle sporede kode-, konfigurasjons-, test- og dokumentfiler, inngangene,
kontrakteierne og deres avhengigheter. Ta med oversikt over ignorerte runtime-stier
uten å eksponere hemmeligheter. Gjenbruk tidligere revisjoner som utgangspunkt,
men kontroller nåværende kode. Skill aktiv kode, historiske bevis og dokumentert
frakoblede filer. Fjern bare dokumentert overflødig innhold; dataopprydding må
fortsatt gå gjennom retention-eieren og ha logg.

Kontroller særlig de åpne trenerfunnene (M3), input-/mål-/normaliseringsbindinger,
kausalitet, Entry/Exit-læring, beslutningsautoritet, fail-closed-vakter, faktiske
oppstartsruter, gamle feilede tester og stale status-/artefaktreferanser. Én samlet
teststatus kan måles som del av denne bestilte revisjonen; gjenta deretter bare
nødvendige kontroller. Lever funn med evidensklasse, rettelse, test og gjenværende
usikkerhet. Teknisk ferdigstilling betyr ikke dokumentert edge eller treningstillatelse.
