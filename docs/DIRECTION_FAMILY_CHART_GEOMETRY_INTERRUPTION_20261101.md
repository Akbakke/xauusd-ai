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

To avgrensede KD-diagnoser er fullført og konsumert. Den første bekreftet
feilkoden, men manglet korrekte kjernesymboler. Den andre hentet offentlige
symbolfiler fra Microsoft og fullførte på 31,15 sekunder, med ca. 88,7 MB
committed memory. Faktiske Windows-grenser ble verifisert før debuggeren
startet: 2 GiB minne, 25 prosent CPU, åtte tillatte CPU-er og én prosess.
Begge avsluttet med kode 0, uten gjenstående prosesser eller oppryddingsfeil;
krasjfilens bytes og metadata er uendret. Ingen krasjfil ble lastet opp.

Den symbolsatte analysen viser System-prosessen og
intelppm!AcpiCStateIdleExecute / intelppm!HvRequestIdle /
nt!HvlRequestProcessorHalt i en Hyper-V hypercall-/NMI-krasjsti.
Failure bucket er 0x20001_11_209b3a_intelppm!HvRequestIdle.
Dette identifiserer CPU-hvile-/virtualiseringskonteksten, men beviser ikke
at intelppm-driveren, CPU-en eller BIOS er årsaken. Triage-dumpen inneholder
begrenset register-/stackinformasjon; 28 varsler om tomme extension-blokker
er bevart. Underliggende årsak og dokumentert retting er fortsatt UNKNOWN.

Fersk lesekontroll: Intel i9-10900K, ASUS PRIME Z490M-PLUS, BIOS1621,
Windows11 Pro25H2 build26200.9457 og intelppm10.0.26100.9278.
Aktiv strømplan er Høy ytelse. Den første skjulte idle-innstillingsspørringen
returnerte ingen eksplisitt innstillingsverdi; den verdien er ikke antatt.
ASUS lister1806beta fra15.01.2026 med oppdatert Intel-mikrokode og
stabilitets-/kompatibilitets-/sikkerhetsendringer. Release notes dokumenterer
ingen spesifikk rettelse for dette krasjet. Ingen BIOS-/driver-/OS-/strømplan-
endring eller ny omstart er utført. Beta-fastvare er ikke automatisk valgt.

Neste steg er å vurdere et målrettet stabilitetstiltak fra denne konkrete
krasjkonteksten før ny modelljobb.
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

Etterkontroll av vertsdiagnose:
- HOST_DIAGNOSTIC_002/ACTUAL_REVIEW_001.json: 0e1e7d13dcecf160ae2ee71c6c6e939e61550471d532cb3ff0657ea0d7fa40f4
- HOST_DIAGNOSTIC_002/ACTUAL_OUTPUT_PEER_REVIEW_001.json: 36b9fbc1e6a996910c153edf9b8455519277ef426d882c00af187e154db09829
- HOST_DIAGNOSTIC_002/HOST_IDENTITY_READONLY_001.json: b70aadb4c11c9fcd00dbaadc43a2506cbd451e7be4a33d0972e89b1324805598

Microsofts symbolserver:
https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/microsoft-public-symbols
ASUS BIOS-versjoner:
https://www.asus.com/us/motherboards-components/motherboards/prime/prime-z490m-plus/helpdesk_bios/?model2Name=PRIME-Z490M-PLUS
