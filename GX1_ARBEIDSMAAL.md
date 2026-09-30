# GX1 arbeidsmål — oppdatert 29.09.2026

Målet er en ærlig XAUUSD-bot som tar retning på den tidsskalaen der retningen faktisk
finnes, og som slår relevante baselines etter kostnad. Målet er aktivt og ikke oppnådd.

## Oppfølging 30.09.2026: undersøk B-kildene

Første kildeprøve er fullført og bevart: arkivmetadata bekrefter to historiske
kopier, men CDX og ALFRED-POST fikk timeout. Begge COT-datoforespørslene
returnerte samme kopi 07.04.2019, etter gullrevisjonen. Oppfølgingen bindes i
configs/research/TA_B_SOURCE_SNAPSHOT_PROBE_20260930.json: bare disse to
arkivkopiene og samme lille ALFRED-prøve med skjemaets manglende submit-felt
rettet. Den rettelsen er ikke bevis for årsaken til timeouten.

Brukeren ba «Ja undersøk B». Dette åpner en avgrenset kildeundersøkelse:
GLD/COT-versjoner, arkivmetadata og ALFRED-transport. Fullført A/C og tidligere
B-resultat bevares. Ingen ny fit, native trening, TEST, handel eller spending.

Før nye kildebytes hentes bindes
configs/research/TA_B_SOURCE_REOPEN_PROBE_20260930.json: fem forespørsler om
arkivmetadata og én liten DFII10-prøve for juni 2025. En byte-lik kopi av
CURRENTs rene HTTP-hjelpere kjøres som transport på Mac fordi samme skjema
tidligere svarte der og fikk timeout på WSL. Rå svar og hasher føres tilbake til
CURRENTs runtime; Mac er ikke en forskningskodebase. Ingen arkivsnapshot eller
modellinput godkjennes av denne transportprøven.


## Gjeldende arbeidsomfang

Den avtalte [A/B/C-planen](docs/TA_RESEARCH_PLAN_20260929.md) er avsluttet:
A/C INKONKLUSIV, B umålt med dokumentert kildebegrensning. Instrumentrettelser,
forhåndsregistreringer, autoriserte målinger og resultatkontroller er fullført.
[Samlet beslutning](docs/TA_RESEARCH_DECISION_20260929.md) binder neste
operatørgrense; planen er ferdig, botens økonomiske mål er ikke oppnådd.

V37-inputforberedelse og repo-/kompleksitetsgjennomgang bevares.
Ingen native optimizersteg, trening eller utførelsesforskning følger av resultatene.
Brede indikator-, modell- og terskelsøk er stengt. Eventuell gjenåpning av full B
krever dokumenterbar historisk GLD/COT-tilgjengelighet og nye manifestbindinger.
Se CURRENT_HANDOVER.md og VEIEN_VIDERE.md. Ingen ny kjøring er autorisert.

## Suksesskriterier

Senere LONG/SHORT/FLAT-valg må slå relevante kausale baselines etter kost, gjennom
flere markedsperioder. Økonomi inkluderer alle valgte handler og åpne posisjoner,
utførbare BID/ASK-priser og kostnader. TRAIN-fit, senere generalisering og samlet
økonomi rapporteres hver for seg. Konstant bias, all-FLAT/all-HOLD, teknisk PASS
og bedre hjelpeprognoser alene er utilstrekkelig. Tidligere tidsskalamålinger
beskriver de undersøkte oppsettene; de beviser ikke at en hel markedstype er ulærbar.

## Bevares

Alle features, alle åtte familier, alle tidsrammer og kausale inputs. Gjennomførbar
BID/ASK-økonomi og kostnader. Ingen fast tapsgrense eller maksimal holdetid; en
beregningshorisont er ikke en handelsregel. TEST er forseglet; ingen live/paper eller
spending. Ingen modell loves å være lønnsom «evig».

Én agent og én tung jobb om gangen innen CURRENT, gjennom `scripts/gx1_capped_run.sh` og eksisterende
vakter. Ingen blind trening, søk eller forebyggende refaktorering; mål før du bygger.
Stående publiseringsautorisasjon gjelder ferdig kode, dokumentasjon og aggregater; rådata,
vekter og hemmeligheter publiseres aldri.
