# Prefix-fit for base/context/MTF — 2026-09-17

De eksisterende eierne for normaliseringspopulasjon og base-fit støtter nå
samme eksplisitte fit Entry-ID-fil og cutoff som lifetime-delen. Entry-context
og M5-historikk velges fra disse inngangene. Exit-current-populasjonen slutter
ved siste observerte M1-barstenging senest ved cutoff. Opprinnelig datakilde-
slutt, historikk og livsløp beholdes; tidsgrensen gjelder bare fit-observasjoner.

Uendret MTF-eier tar unionen av kausale Entry+5min- og Exit+1min-vinduer.
Ingen dobbelt tidsforskyvning, nye featuregrenser eller transformasjoner.
Prefix-modusen krever eksplisitte argumenter og samsvarende, hashbundne
populasjonsbevis. Faktiske M5/M1-tider kontrolleres før statistikk beregnes.
Gamle kall uten opt-in følger tidligere populasjon og fit.

12 unike syntetiske CPU-kontroller består. To prefix-Entries av tre gir97
unike M5-rader og6 Exit-current-rader. Alle kontinuerlige signal-/context-/MTF-
features etter cutoff økes med5000 uten at noen fitted surface eller kategori-
kontrakt endres. Scope-/hashavvik og resealede M5/M1-intervaller over grensen
avvises før fit. Eksakt close og cutoff−1ns er kontrollert. Eksisterende
sekvens-/populasjons-/audit-tester består.

Første forsøk hadde11 PASS og én fixturefeil. Den nye fixturen måtte få
varierende contextfelt, korrekt H4/D1-sessiongrid via eksisterende grid-eier,
og minst to prefix-Entries. Tre feillogger er bevart. Produksjonskoden ble ikke
endret etter første forsøk. Bare feilet test ble gjentatt, og til slutt også
seks avvisningstester som delte endret fixture. Capped CPU-audit:4GiB,
512MiB swap,åtte kjerner,én numerisk tråd. MTF-diskloaderen erstattes av
syntetiske verifiserte frames; faktisk populasjon/MTF-seleksjon/statistikk og
normaliseringskontrakter kjøres. Dette er ikke en produksjonscache-audit.

Ingen faktisk markedsdata-fit, modell-forward, GPU, optimizer eller TEST.
De nødvendige fit-eierne er rettet; faktiske prefix-artefakter finnes ikke ennå.
Neste er eksakt targeteligibilitet for alle hoder og Exit samt sample-/probe-ID-er,
deretter fersk native initialisering og kontrollbindinger. Frosset DESIGN og
CONTROL256 er uendret. Ingen læring eller generalisering hevdes.

Bevis:handover_snapshot/PREFIX_BASE_SYNTHETIC_20260917.json. Originale logger,
JUNIT-filer og RESULT.json under GX1_DATA/data/data/prebuilt/
LIFECYCLE_V2_FULL_TRAIN_20260912/PREFIX_BASE_SYNTHETIC_20260917.
RESULT SHA256:abd5325f5c612a698fc753598aed0727221ef1f603be1d738cf12d931ab37235. Parentkilde:dd68dffcabadaa02cfc08dafe1868386a3157d12.
