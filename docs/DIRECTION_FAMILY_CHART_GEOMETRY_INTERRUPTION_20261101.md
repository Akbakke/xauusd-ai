# Chart geometry: trening bevart, evaluering avbrutt

W1 fullførte 47 602 TRAIN-foreldre og 2 976 optimizeroppdateringer på frosset kilde
3ac1e688a54d0df7384d6aa6212513b264a63466. Original kvittering er RESUMABLE,
vakt PASS og begge sluttkoder 0. Den lagrede tilstanden er klar for scoring,
men full TRAIN og senere Q1 er ikke ferdig evaluert.

W2 startet 1. november 2026 kl. 10:03:14 UTC på samme fysiske Windows-boot 590,
etter nye bundne Windows-/WSL-tomgangskontroller. Ingen operatøromstart ble gjort.
Windows registrerte feilkontroll 0x00020001 og automatisk oppstart 11:01:54.5 UTC,
boot 591. Microsoft betegner koden HYPERVISOR_ERROR; den underliggende årsaken
er fortsatt UNKNOWN. Den må ikke likestilles med forrige session-krasj 0x1e.

Ingen W2-resultat, progress eller native terminalkvittering finnes. Original
native sluttkode er derfor null/ukjent. Task Scheduler-resultat 1 fra forsøket
etter oppstart er en annen prosess og erstatter ikke den manglende native koden.
Original ACTIVE, CLAIM og signert logg er bevart, inkludert nullbytes i logghalen.

Siste gyldige vaktlinje kl. 11:00:56 UTC viste GPU 56 C, minne 66 C,
121,99 W, grense 250 W og 856 MiB. Ingen vaktbrudd eller normal exit er registrert.
Disse utvalgte målingene beviser ikke generell maskin- eller driverstabilitet.

En capped CPU-integritetskontroll etter krasjet bekreftet eksakte bytes for
W1-modell, egen initialtilstand, cursor og resultat. Ingen checkpoint ble
deserialisert, ingen nye modellkall eller oppdateringer utført. Uavhengig
metadata-/arbeiderreview bestod. Ferdig trening skal ikke gjentas.

Den eksakte chart-oppgaven er deaktivert, ingen native prosesser ble observert,
begge maskinvarekeepere kjører og konfigurert/fysisk effektgrense er 250 W.
Kampanjen er stengt med 2 claimede, 1 fullført og 1 avbrutt invokasjon;
vinduer 3–4 er tilbakekalt. Alle native trenings-/kjøreporter er stengt.

Neste steg er avgrenset offline-diagnose av det konkrete Windows-krasjet.
Eksisterende recovery-validator er bundet til session-familien, originalkilden
og dens avslutningskvittering; nye chart-metadata alene gir ingen kjøreautoritet.
En eventuell separat chart-scoring må kvalifisere sine egne bindinger og
bevare opprinnelig trening. Ingen original relansering er tillatt.

Syv av ti komplette varianter er ferdige. Chart geometry-scoring, price
action/candle, macro og alle 54 parrede kontraster gjenstår. Avbruddet gir
ingen nye retningsmålinger, familieeffekt eller økonomisk dokumentasjon.
TEST, Exit og handel er urørt.

Evidensrot:
 /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_CHART_GEOMETRY_001

- WINDOW_0001_VERIFIED_REVIEW.json: 6432b92abf5c60508039d58dc40bf527c229d7170e3eb67757f201e7973162cd
- CONNECTION_LOSS_OBSERVATION_001.json: b399a59dc7af1e8b0af84c8c135d5b4efe96ac3eaa74aa4f7fb54ee8a3ce9882
- INTERRUPTION_TASK_DISABLED_001.json: 0620da7f545767727cea2fe113e71eef6b3a49f3af4032839077ff98e07e0b8c
- UNEXPECTED_INTERRUPTION_INTEGRITY_001.json: 81a76902bed520721e770cd9665f3d35afa3fd1f7c8232d6d7b5526633897655
- UNEXPECTED_INTERRUPTION_PEER_REVIEW_001.json: 3d799937287a0cc5baa74fde3c99b641779628ecc0bdc684d5bf2ea918634074
- POWER_RETAIN.json: ccdecdcd458e16bf1b0e1fc35b22184501c756a2e58077f2e9699b8e7d76ae93

Feilkodens betydning: https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x20001--hypervisor-error
