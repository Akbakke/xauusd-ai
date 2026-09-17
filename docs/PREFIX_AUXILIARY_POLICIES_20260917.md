# Frosne prefix-policyer og treningsutvalg — 2026-09-17

De to eksponerte fasitpolicyene har nye artefakter tilpasset én gang kun på
2025-06-01..2026-03-01. Eksisterende eiere og algoritmer er uendret. Input ble
filtrert ved lesing og faktisk maksimum barstenging kontrollert før cutoff.
M5-projeksjonen inneholder bare tid; M1-projeksjonen inneholder tidligere
BID/ASK. Projeksjoner, kildefiler, eiere, plan og resultater er hashbundet.
Ingen senere kontroll-/juni-priser eller TEST inngikk i fit.

Direction-policy brukte34030 komplette maksimale forløp og valgteK18 gjennom
sin eksisterende deterministiske estimator. Størrelses-ECDF brukte42644
tradable rader ved denne horisonten. Dette er fasitberegning, ikke en regel
om maksimal holdetid, ny modell eller utført læring. Valget er frosset og skal
ikke endres etter kontrollresultatet. Den gamle K19-policyen brukes ikke i
den nye forsøksfasiten. Originale policyer og data er bevart.

Klokkemålingen gir47814 lovlige TRAIN-Entries vedK18. Hele native epoch0-
rekkefølgen filtreres til disse uten ny sampling. Første4096 er låst for de
planlagte256 TRAIN16-stegene. TRAIN256-proben bruker eksisterende selector,
seed20260911/salt0, på akkurat disse4096. Begge dekker juni2025–februar2026.
Alle fire opprinnelige Exit-samples beholdes for hver Entry. Physical counts,
markedets livsløp og frosset CONTROL256/DESIGN er uendret.

Dette beviser prefix-preprosessering og lovlig tidsstøtte, ikke modellkvalitet.
Aktive size/sideMAE/trendline-labels og masker i det gamle datasettet er ennå
ikke erstattet.37 rå hjelpefasiter har uendrede formler og må beholde sin
kontrollerte tidsstøtte. Faktiske prefix-normaliseringer og ferske native
vekter/lærer/optimizer/kontrollbindinger gjenstår.200 features,åtte familier og
alle tidsrammer beholdes. Ingen modell-forward, optimizer eller GPU er brukt.

Begge eksisterende fit-eiere ble kalt én gang under audit-cgroup4GiB/512MiB,
åtte kjerner/én numerisk tråd og600sekunders øvre kjøretid. Ingen ny testserie
eller produksjonskodeendring. Engangstillatelsen er stengt; ikke gjenta fit.

Bevis:handover_snapshot/PREFIX_AUXILIARY_POLICIES_20260917.json.
Original PLAN,INPUT-projeksjoner,policyer,ECDF,utvalg,RESULT,VERIFICATION og
FIT.log ligger i GX1_DATA/.../PREFIX_AUXILIARY_POLICIES_20260917.
