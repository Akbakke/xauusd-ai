# CURRENT HANDOVER — 17.10.2026

Autoritet: /home/andre2/src/GX1_CURRENT, work/gx1-current. NEXT_RUN_POLICY.json
er eneste arbeidsstatus; prosess/terminal overstyrer prosa. Historikk ligger i Git.
Én agent, én capped tung CURRENT-jobb; aldri endre fryst kilde under kjøring.

## Nå: GC-inputgrunnlag ferdig, P0-fit fortsatt stengt

Samlet rapport: docs/OPENBB_GC_FOUNDATION_REVIEW_20261017.md.
26 måneder GC MBP-1, bare eksisterende rådata; ingen nye Databento-/betalte kall.
014/017/019/020 COMPLETE én gang, exit 0, source_unchanged=true, TEST=false.
015/016/018 er feilkontrollert/konsumert, aldri relanseringsordre. Originaler bevart.

- 014: isolert OpenBB V5 (core 2.0.1, FRED/CFTC 2.0.0), to gratis CFTC-kall.
  113 legacy-rapporter matcher eksisterende OI og fire posisjonsfelt; Managed Money
  er forskjellig fra legacy noncommercial. FRED-nøkkel mangler i miljøet, 0 kall.
  Ingen nye makrofelt registrert; eksisterende 61 og innhenting urørt.
- 017: faktiske OFI-/tidsvektede BBO-/likviditets-/prisresponsverdier kvalifisert
  i 83 776 M5, alle 26 måneder. Tolv ekte kontraktprefikser/60 M1 består.
- 019: full GC-historikk, uavhengig av spot-eligible clock. Alle 765 839 gamle
  M1-kontraktvalg og 11 Trade-metrikker eksakte. Rolling15 48 164, rolling60
  7 330, segmentprofiler 89 203; 2 263 komplette sesjonsprefikser, alle 26 måneder.
  Sesjonsprefiks median 25 / maks 185 minutter; ingen hel ren sluttført GC-dag.
  Faktisk 70 % Value Area, ikke q25/q75. Ingen svakere kvalitetskrav/deduplisering.
  Kanoniske publiserte cachebaner: 019/PUBLISHED_ARTIFACTS.json; opprinnelig
  prepublication-kvittering er bevart. Ingen absolutt GC-pris som XAU-ordrenivå.
- 020: uavhengig M1-prefiksmaske/fysiske grenser/integerparitet består.
  37 181 felles F12/L8-inputrader. Eksakt M1-horisontbane gir 37 181/36 264/
  31 544 ved 30/120/360 minutter; dette er observasjoner, ikke handler.
  V16/ATR/RV gjenbrukes. Åtte armer: kontroll/F/L/V/FL/FV/LV/FLV.
  332 220 manglende minutter er ordinær ukestengning, 40 916 øvrige er uklassifisert.
  M1-stempel er ikke dokumentert tick-/filltid. Ingen modell/strategiøkonomi kjørt.

## Konkret neste gate

CFTC-rapport 30.09.2025 var først annonsert publisert 19.11.2025; dagens
native +7-døgnsregel er kjent utrygg under nedstengningen. 18 462 observasjoner
flagges mot annonsert etterslepsplan, som ikke er en komplett faktisk vintage-/
utgivelseskalender. Foreløpig COT-utestenging fjerner hele nov/des 2025; den er
ikke en ferdig PIT-reparasjon. Behold fryst baseline. Lag eksplisitt korrigert
forskningskontroll eller forhåndsregistrert variant uten ukvalifiserte makroinputs.
Deretter må eksakt begrenset regel-/XGBoost-recipe og egen kjøreautoritet bindes.
Ingen automatisk fit, ny epoke, full VAL eller gjenbruk av konsumerte godkjenninger.

Felles v3-kostprojeksjon: korrekt SHORT-inngangsnotional, spread i bid/ask,
execution sentralt 4 bps, LONG 569 / SHORT 0 årlige bps, 11 hypotetiske
kost-/funding-scenarioer inkl. ugunstig SHORT. Ikke historisk brokerkostsannhet.
Beregningshorisonter er ikke maksimal holdetid eller en ny handelsregel.
Alle 26 måneder er utvikling. Urørt GC-bekreftelse finnes ikke; ingen edge/profitt
bevist. Dette sperrer uavhengig bekreftelse/promotion; P0 kan kun være utvikling.
Native TEST, produksjon, live/paper, broker/ordre og spending forblir stengt.

## Uendret native-grunnlag og retensjonsgrense

Fryst baseline: configs/research/XAU_M5_INFORMATION_BASELINE_20261017.json;
docs/FEATURE_AUDIT.md og inventar. Native 315 genuine felt / åtte spesialister,
61 makrofelt; ingen ny modellarkitektur. 27 patternfelt er bare pauset i planlagt
forskningsview, ingen native features/kode slettet. Baseline, Entry/Exit-eiere,
historiske resultater/økonomifunksjoner er bevist uendret i avsluttende review.
Native D1/h20 er separat fra M5/GC. Dagens Entry-implementering/korreksjoner:
docs/ENTRY_TARGET_AUDIT.md og docs/ENTRY_EVALUATION_CORRECTION.md. Tidligere
native læringskvalifisering er REJECT; 180d-fasen er ikke en aktuell kjøreordre.

DATA/RUNS ryddes bare med retention-eieren etter GX1_RULES regel 9. Tidligere
hard-sletting/VHDX-komprimering er fullført, ikke en nåværende blokkering.
Bevar kanoniske GC-gzipfiler/kvitteringer, gjeldende D1-inputs, modellvekter,
checkpoints, terminaler og de nye hash-bundne forskningsinputene. Ingen ny
slette-/komprimerings-/omstartsordre i denne bølgen. Git er kodehistorikken.
