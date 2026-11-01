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
Aktiv strømplan er Høy ytelse. En korrekt skjult innstillingsspørring
bekrefter IdleDisable AC/DC=0: CPU-hvile er aktivert. Ingen verdi er endret.
ASUS lister1806beta fra15.01.2026 med oppdatert Intel-mikrokode og
stabilitets-/kompatibilitets-/sikkerhetsendringer. Release notes dokumenterer
ingen spesifikk rettelse for dette krasjet. Ingen BIOS-/driver-/OS-/strømplan-
endring eller ny omstart er utført. Beta-fastvare er ikke automatisk valgt.

To lagrede WHEA-hendelser fra treningstiden (08:05:49 og 09:17:02 UTC)
er dekodet mot standardfeltenes gyldighetsbiter. Begge rapporterer korrigert
cachefeil ved instruksjonsutførelse på logisk APIC-ID 18, CPUVersion 0xA0655.
Minneseksjonen har gyldighetsbiter 0; ingen DIMM eller fysisk defekt komponent
kan identifiseres. En separat utvidelse bekrefter Microsofts MCA-seksjon:
begge poster har bank 0, APIC-ID 18 og status 0x9400004000040150. Status
er gyldig og korrigert, uten overflow eller corrupt-context-markering. Bare
det dokumenterte 72-byte-prefikset er tolket; resten er fortsatt ukjent.
Disse tidligere hendelsene beviser ikke årsaken til hypervisorkrasjet.
Fersk observasjon kl. 12:00:26 UTC viste samme boot, ingen nye WHEA-hendelser
siden 11:02 og ingen kjørende kjente tuningprogrammer. Brukeren kjenner ikke
BIOS-innstillingene; overklokking, undervolting og XMP står som ukjent.

Den konkrete session-only-sperren i gjenopprettingskoden er rettet for akkurat
den bevarte chart-kjøringen. 291 fokuserte syntetiske tester bestod med
sluttkode 0. Faktisk kildekontroll mot original chart-kilde bekrefter nøyaktig
tre endrede native kontrollfiler blant 65 eiere; øvrige 62 og numeriske deler
er uendret. Sammenligningskoden beholder originale treningskvitteringer og
separate nye scoringkvitteringer. Dette kvalifiserer kilde og syntetiske
kontroller, ikke modellutførelse. Den første rapportgeneratoren forventet en
pytest-oppsummering som quiet-modus undertrykte; bare rapporteringen ble
korrigert og gjentatt, ikke testene. Originalforsøket er bevart.

Vertsberedskap og full binding av den nye scoringplanen gjenstår. Ingen
modellkjøring, innstillingsendring, ny trening eller omstart er gjort i denne
rettelsen. Eventuell separat scoring må bevare opprinnelig trening; den
avbrutte originalkampanjen kan ikke relanseres.

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

Ny lesende host- og kildekvalifisering:
- FIT_CHART_GEOMETRY_001/HOST_DIAGNOSTIC_002/WHEA_CPER_DECODE_RESULT_001.json: 149db58eea2b8e82b22ae8eea4aa9a515653ffebad2807e7497b7ad2e9475035
- FIT_CHART_GEOMETRY_001/HOST_DIAGNOSTIC_002/PROCESSOR_IDLE_READONLY_001.json: 99b320286e2b4a435cf579ef48754d51613b46e1546d7052ceff4078bd4462dd
- CHART_SCORING_RECOVERY_SOURCE_001/RESULT.json: b2a3ff71f794efbca01ef4ae5aabcf185f8c96a4acad17c2f3c50d8ecb1c9ef1
- CHART_SCORING_RECOVERY_SOURCE_001/TEST_EXECUTION.json: aac101de58a3690b204997b0210776b15d9e49b89d755a6816f52173c9882745

Chart-gjenopprettingens eksakte oppskrift og gjenbrukte kontrollarbeidere er
klargjort. Uavhengig metadatareview fanget en manglende newline i den første
oppskriftshashen; dette ble rettet med den eksisterende kanoniske hasheieren
før utførelse. Bare én capped metadata-/filhashforkontroll er åpnet.
- FIT_CHART_GEOMETRY_001/HOST_DIAGNOSTIC_002/WHEA_MCA_EXTENSION_RESULT_001.json: be53c2a007c858271ba4d35b72eef2526ba70d7054cb2534b868f59c6faa186c
- FIT_CHART_GEOMETRY_001/HOST_DIAGNOSTIC_002/HOST_RUNTIME_READONLY_001.json: 955a6a0e06736979c43dc87989ad70a439a556d1931240070777c2f6580aeb32
- FIT_CHART_GEOMETRY_SCORING_RECOVERY_001/RECIPE_PREPARATION_PEER_REVIEW_001.json: 58d3b9989c3024a5dd42f85048eabdb6a7e390e42941406ca1882489094615b2
