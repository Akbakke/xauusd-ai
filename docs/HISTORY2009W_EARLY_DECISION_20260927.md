# Tidlig kalibrering og én beslutningsmåling — 27.09.2026

Operatøren har overført arbeidet til Codex og autorisert tidlig kalibrering og én
forhåndsbundet sammenligning. Eksakt oppsett og beslutningsregel:
[PREREGISTRATION](HISTORY2009W_EARLY_DECISION_PREREG_20260927.json).

## Målt stoppunkt

V2-innhentingen fullførte 16:43:42 UTC med rc0. M5/M1 er bitlike med v1 etter
utelukkelse av nøyaktig 1 410/1 535 planlagte helgebarer. Senkalibrert squeeze
fullførte 16:50:04 UTC; C0 ble deretter avbrutt av operatøren. Ingen tung prosess
eller tung lås ved overtakelsen. C0 har ikke terminalt ferdigmanifest. Bevar alt;
verken den avbrutte C0 eller 2025-kalibreringen brukes som tidlige OOS-inputs.

## Avgrensning og konkret rettelse

Kalibrer eksisterende squeeze- og registereiere på 2009-06-01 til 2013-01-01
22:00 UTC (halvåpent; ferdig lukket felles dagsgrense). Registerets indre grense
følger eksisterende 80/20-konvensjon, avledet av tidsstempler før noen utfall.
Bygg M5 C0 og alle 241 signalfelt + eksisterende kontekst og MTF gjennom de
kanoniske eierne. Den eksisterende walk-forward-eieren får en eksplisitt inngang
for den eksisterende M5-featurebasen. Ingen oppdiktede knee-mål eller falskt
native split-manifest; ingen ny featureimplementasjon.

Én D1-beslutningsklokke, én horisont på 1 440 native M5-barer, én full featurearm,
to eksisterende lærere (ridge/HGB). Ti årlige holdouts fra juni 2015 til juni 2025.
Inner-modelvalg og standardisering bruker bare tidligere, purgede rader. Både
indre og ytre kontroll avvises hvis feature-fit ikke er ferdig før perioden.
Full seq513-/M1-treningsmaterialisering utsettes til beslutningsverdi er påvist;
den tidligere historikkinnhentingen er allerede fullført. Native trening forblir stengt.

Kostpolicyen brukes som navngitt prospektivt scenario, ikke historiske eller
nykontrollerte brokerkostnader. Alle utfall bruker BID/ASK og begge utførelser.
Lange horisonter vurderes på ikke-overlappende blokker og årsaggregering;
overlappende daglige prediksjoner teller ikke som uavhengige uker.

Kravet er positiv netto og positiv parvis forbedring mot alltid-LONG og en
LONG/SHORT/FLAT-konstant valgt fra tidligere fit-data. Konfidensgrensene beregnes
på årsverdier, med korreksjon for de to modellene; minst 75 prosent positive
årsfordeler og støtte i både positive og negative alltid-LONG-regimer kreves.
Ingen retuning etter utfallet. Et eventuelt GO er forskningsbevis, ikke native
policyprofitt eller automatisk autorisasjon til trening.

## Begrensninger

Dette er tidligere inspiserte utviklingsår, ikke et prosjektomfattende urørt
holdout. M5 close-fill er eksisterende forskningskonvensjon, ikke dokumentert
M1-utførelse. Kosthistorikk og quote-to-fill-latens er ikke kalibrert. Forskningen
undersøker snapshot+MTF, ikke transformerens sekvenskapasitet. TEST-rader og
markedsutfall fra juni 2025 eller senere brukes ikke i sammenligningen.

## Kontroll av faktisk kjøring

Kalibrering, C0 og M5-featureflate fullførte med rc0. Den første sammenligningen
fullførte 20 konfigurasjoner, men en separat kontroll fant en eldre grensefeil:
D1-klokken flyttet foldstart/slutt til neste valgte beslutningsrad. Åtte fit-sett
inneholdt én rad med utfall etter deklarert start; sju holdouts hadde én for sen
utfallsrad, hvorav fem inngikk i blokkstatistikken. `FOLD_BOUNDARY_AUDIT.json`
bevarer målingen. Første `walkforward/` og `DECISION_GATE.json` er derfor erstattet
som beslutningsgrunnlag og skal ikke brukes som endelig resultat.

Minste rettelse bruker råtapens tidsgrenser for både fit-purge og holdout-utfall.
54 målrettede tester består, inkludert en regresjonstest for grove klokker.
Kun modellsammenligningen beregnes på nytt i `walkforward_strict/`; ferdig squeeze,
registerkalibrering, C0 og featureflate gjenbrukes uendret. Alle forhåndsbundne
parametere og akseptkrav står fast. Ny kjøring bindes separat i
`CORRECTION_BINDING.json`; sluttstatus følger `STRICT_TERMINAL.json`,
`DECISION_GATE_STRICT.json` og `VERIFICATION.json`. Ingen native trening.
