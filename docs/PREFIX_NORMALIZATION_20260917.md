# Tidsavgrenset lifetime-normalisering — 2026-09-17

Tidlige inngangsrader var ikke nok: eksisterende normalisering av posisjons-
historikk velger tilstander gjennom hele hvert forløp fram til TRAIN-slutt.
For det kronologiske forsøket kunne mars–mai dermed påvirke transformasjonen.

Eksisterende physical-summary-eier og materializer har nå eksplisitt opt-in
med fit-entry-rows (bundet NPY-fil) og fit-cutoff-time-ns. Et separat vektor-
utvalg setter null fit-tilstander for ikke valgte Entries og begrenser andre
til observerte M1-barstenginger senest ved cutoff. Originale Entry-koordinater
og fulle successor-counts bevares i manifest og original counts-fil; en egen
fit_state_stop_exclusive_by_entry.npy lagres. Ingen livsløp eller handler kuttes.

Hashvalget innen eksisterende varighetsbøtter bruker i prefix-modusen bare
valgte fysiske identiteter og fit-grenser. Hashene av hele pristapen beholdes
som provenance, men kan ikke endre hvilke tidlige normaliseringsrader som
velges når senere priser endres. Den gamle modusen er uendret.

24 syntetiske CPU-tester består ved første kjøring. Framtidskontrollen øker
alle priser etter cutoff med5000: kildehashene endres, men sample-stream,
fit-verdi-hash og hele fitted surface er identiske. Originale successor-counts
[29,25,14] forblir like mens fit-grensene er[10,6,0]. Eksakt barstenging og
cutoff−1ns, helgegap, feilaktige utvalg og gamle normaliseringskontrakter er
kontrollert. Capped audit:4GiB/512MiB swap/åtte kjerner/én numerisk tråd.

Dette dekker lifetime-summary-delen alene. Base/context/MTF-fits, full target-
eligibilitet, sample-/probeplan, fersk native initialisering og riktig senere
kontrollbinding gjenstår. Ingen faktisk normaliseringsfit, modell-forward,
optimizersteg, GPU eller TEST er brukt. Ingen læringsport åpnes. Frosset
kronologisk DESIGN og CONTROL256 er uendret; ingen terskler eller kriterier
justeres. Neste minste rettelse er prefix-fit-bindingen i eksisterende base-
normalisering, ikke en ny modell eller separat runner.

Bevis:handover_snapshot/PREFIX_SUMMARY_SYNTHETIC_20260917.json.
Original RESULT.json, TEST_ATTEMPT1.log/xml og operator ligger under GX1_DATA:
data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/PREFIX_SUMMARY_SYNTHETIC_20260917.
Resultat SHA256:c9b3b88a712f9fca46ea5bda45ead57a3592d796fbecd86f8a78a15101cce9db.
Parentkilde74ffc57bcc1ee3b215326c08ddd60cf7f0ca5be3; eksakte endrede filhasher
og alle testnavn står i resultatet.
