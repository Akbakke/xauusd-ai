# C — forhåndsregistrert kombinasjon, 29.09.2026

Én frossen kombinasjon av fem tidligere post-hoc valgte celler. Ingen ny celle-,
terskel- eller parametersøk. Denne registreringen og kildeimplementasjonen
committes før nye gullutfall beregnes. [Kjørbar kontrakt](../configs/research/TA_C_PREREG_20260929.json).

## Populasjon og tidligere bruk

Evaluering: signal kjent fra 01.06.2025 00:00 UTC til før 01.07.2026 00:00 UTC.
Les fra 01.01.2024 for oppvarming. Hele evalueringen klassifiseres som gjenbrukt
utviklingsevidens. Tidligere weekly_direction_first_look.json bekrefter fold fra
01.06.2025 og siste beslutning 29.05.2026; filhash bindes i JSON-kontrakten.
LEARNING_GATE_20260916.md erklærer juni2026 brukt som utviklingsdata, og
EDGE_ROOT_CAUSE_REVIEW_20260929.md beskriver gjenbrukt juni-VAL. A har dessuten
nå brukt juni–desember2025. Dette er ingen urørt bekreftelse.

Samme native M5-manifest som A. Bare 2024–2026-filene besøkes, og dekodede
prisrader filtreres til barstart før juli2026. Hele råfiler hashes for identitet;
påstanden er at TEST-utfall ikke evalueres, ikke at ingen byte fra den blandede
2026-filen leses. Ingen native TEST-split åpnes.

## Signal, kombinasjon og måleklokke

Prioritet ved like samtidige signaler, i denne uendrede rekkefølgen:

1. rn50_cross_follow_h12
2. setup_pdh_break_trend_H4_pair_h12
3. setup_momentum_confluence_long_pair_h12
4. setup_range_break_up_H1_trend_H4_pair_h12
5. orb_london_local

Eksisterende primitiveier, parametere og speilede LONG/SHORT-par gjenbrukes.
Signalet er kjent ved M5-barstart + fem minutter. Første fire celler beholder
krav om sammenhengende foregående bar, men filtreres aldri på framtidig
gapfrihet. London følger IANA/sommertid: åpning08:00, første12 M5-close danner
range, første senere close utenfor rangen før17:00 gir ORB-signal.
Manglende åpningsbar gir ingen kjent komplett range.

Motsatte samtidige retninger gir FLAT. Like retninger gir én kandidat med
første celles målevindu. Én reservert posisjonsplass; nye signaler ignoreres
fram til forrige mål er gjort opp ved en faktisk quote. Det samme utvalget
brukes for aktiv inngang, alltid-LONG på de samme mulighetene og passiv simulering.
En uteblitt passiv berøring åpner ikke ekstra handler.

Inngang bruker første faktiske M5-open ved/etter kjent signal. Første fire
målvinduer slutter60 minutter etter kjent signal; ORB slutter17:00 London.
Utgang bruker første faktiske open ved/etter dette tidspunktet. Signaler som
ikke kan få inngang før måltidspunktet beholdes som uutførte med null utfall.
Ved datagrensen inngår fortsatt eksponering til siste tilgjengelige bar-close,
med markert sensurering og utførbare likvidasjonskostnader. Disse vinduene er
forskningsmål, ingen nye native regler for maksimal holdetid.

Dette reparerer klokke/utførelsessemantikken fra den eldre close-proxy-målingen;
nye tall er derfor ingen eksakt gjentakelse av tidligere 61-cellers statistikk.

## Passiv modell og kostnader

Ordren tenkes plassert ved beslutnings-open: kjøp på BID, salg på ASK.
Hele den neste faktiske femminuttersbaren må få plass før planlagt måltid.
Kjøp krever ASK-low <= kjøpslimit, salg BID-high >= salgslimit. Det brukes
ingen bar før ordren er plassert. Berøring gir antatt pris lik limit og antatt
fyllingstid ved barens slutt. Faktisk intrabar-tid er ukjent innen fem minutter.

Ingen inngangsslippage utenfor limit. Alle markedsutførelser belastes deklarerte
0 /0,5 /1 /2 bps hver; aktiv inngang har to slike kostnader, passiv bare utgang.
Provisjon0 følger eksisterende kostkontrakt. Mid-bevegelse fra beslutnings-open,
bid/ask-effekt (kan være prisforbedring), slippage og finansiering rapporteres
hver for seg. Historisk EFFR +1,29 prosentpoeng og null finansiering vises begge;
SHORT-kreditt beholder fortegnet. Ingen påstand om historiske brokerswaps.

Rapporter alle kandidater, plasseringer, berøringer og manglende fyllinger.
Aktiv PnL på akkurat berøringsutvalget sammenlignes med resten og med passiv
PnL på samme utvalg. Hovedgjennomsnittet deler på alle valgte muligheter,
inkludert null ved uteblitt berøring. Dermed skjules ikke fyllingsutvalget.
Berøring beviser ikke køplass, faktisk fill, volum, latency eller markedspåvirkning.

## Risiko, økonomi og inferens

Samme kausale risikobudsjett som A: siste fullt lukkede D1,21 returperioder,
10prosent årlig volmål, leverage-tak1 og fast initialkapital100. Antall enheter
fryses ved beslutnings-open for alle tre armer. Ingen realisert framtidsvol brukes.
ATR14 fra samme lukkede D1 normaliserer rå bps. Realisert risiko kan være ulik.

Kontantregnskap følger faktiske open/close-quotes; sammenfallende tidsstempler
bruker open. Alle handler og terminalt åpne posisjoner avstemmes. Daglig Sharpe
mot null kontantrente og daglig/quote-basert drawdown oppgis. Kalenderdager,
inkludert stengte dager med videreført verdi, annualiseres med365,25.
Dette er bokføring av kjent verdi, ikke syntetiske beslutningsquotes.

Fryst inferensfamilie:72 endepunkter. Aktiv minus samme utvalgs LONG, passiv minus
LONG og passiv mot FLAT, hver med fire slippagenivåer, to finansieringsscenarier
og tre mål: rå bps per valgt mulighet, ATR-normalisert bps og Sharpe-forskjell.
Mot FLAT er Sharpe-målet armens egen Sharpe mot null kontantavkastning, ikke en
beregnet Sharpe for en konstant nullserie. Individuelle celler får bare signaltelling.

Paret stasjonær bootstrap på felles kalenderdager:1999 trekk, forventet
blokklengde20 dager, seed0. Ingen blokkoptimalisering. Eksisterende max-|t|-eier
gir simultane95prosent grenser for hele den testbare familien; udefinerte mål
blir eksplisitt INKONKLUSIV. Oppgi styrke ved1/2/5 rå bps,0,01/0,02/0,05
normaliserte enheter og0,1/0,2/0,3 Sharpe; MDE ved80prosent styrke.
Dette korrigerer den nye familien, ikke tidligere post-hoc valg eller VAL-gjenbruk.

Primær beslutning gjelder passiv arm ved1 bps per markedsutførelse. GO krever
at alle fire nedre simultane grenser (mot FLAT og LONG, begge finansieringer)
er over1 rå bps per valgt mulighet, og positiv solvent risikojustert økonomi.
En nødvendig øvre grense under1 gir NO_GO; ellers INKONKLUSIV. Normaliserte mål,
Sharpe og øvrige kostnivåer er diagnostikk i samme korreksjonsfamilie.

Selv GO gir bare forslag om egen prospektiv utførelses-/ordrebokforskning.
Ingen native trening, handelsåpning eller lønnsomhetsløfte.

## Kjøring og teknisk evidens

Kjør én gang fra ren, committet kilde med eksakt manifesthash gjennom
scripts/gx1_capped_run.sh --class producer --mem 8G --swap 512M.
Output: /home/andre2/GX1_RUNS/TA_RESEARCH_20260929/MEASUREMENT_C_001.
STARTED, SIGNALS, COHORT, utfall, dagregnskap, parvise dager, bootstrap-statistikk,
RESULT og TERMINAL bevares. Detaljerte pris-/signaldata publiseres ikke.

Fokuserte syntetiske tester kontrollerer signaltilgjengelighet, framtidsmutasjon,
konflikter/duplikater, faktisk neste quote over gap, korrekt berøringsside,
kontantavstemming, signed funding, sensurering, kildefilfilter og hele
inferens-/artefaktkjeden. Dette beviser mekanikk, ikke indikatorverdi.
