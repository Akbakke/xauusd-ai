# GX1 arbeidsmål — oppdatert 30.09.2026

Målet er en ærlig XAUUSD-bot som tar retning på den tidsskalaen der retningen faktisk
finnes, og som slår relevante baselines etter kostnad. Målet er aktivt og ikke oppnådd.

## Aktiv oppfølging: neste datasteg for B

ALFRED avviste samlet DFII10 med eksplisitt grense450 vintager per forespørsel.
Den prøven bevares. Samme fire serier og fulle datoomfang hentes nå i fortløpende
blokker på høyst450 etter configs/research/TA_B_ALFRED_BROWSER_CHUNKS_20260930.json.
Ingen vintager utelates; duplikater og samlet dekning skal avstemmes.

ALFRED-prøven gjennom vanlig nettleser lyktes; ZIP/CRC og historiske datoer er
kontrollert. Fire komplette makroarkiver bindes nå i
configs/research/TA_B_ALFRED_BROWSER_20260930.json. Internet Archive ga HTTP429;
nye arkivforespørsler er stoppet. Henteeieren stopper nå resten av en batch ved429.
Ingen historisk dekning utledes fra feilresponsene.

Brukeren ba «Kjør neste steg». Kun den nye manifestbundne kildeprøven
configs/research/TA_B_NEXT_SOURCE_STEP_20260930.json åpnes: ALFREDs vanlige
nettleserskjema, årlige GLD-arkivoppslag og arkiv for den daterte COT-rapporten
26.03.2019. Tidligere prøver gjenbrukes og relanseres ikke. Ingen fit, kontakt med
leverandør, kjøp, TEST eller handel er åpnet.

## Tidligere kildeundersøkelse

Brukeren ba 30.09.2026 «Ja undersøk B».
[Undersøkelsen](docs/TA_B_SOURCE_INVESTIGATION_20260930.md) hentet og kontrollerte
ekte historiske GLD- og COT-filer. Full B er fortsatt umålt: to enkeltkopier
dokumenterer ikke sammenhengende publiserings-/versjonsdekning. Den kontrollerte
COT-adressen har bare én arkivkopi 01.03–07.04.2019; GLD-indeksen fikk timeout.
ALFRED-skjemaet virker på Mac, men POST fikk fortsatt timeout etter rettet
submit-felt og 60 s grense. To fokuserte tester besto.

Tre kildeprøver er avsluttet og hashkontrollert; ingen relansering er nødvendig.
Neste datagrense er dokumenterte GLD/COT-versjoner og fungerende makrotilgang,
eventuelt via registrert FRED/ALFRED-API-nøkkel. Leverandørens generelle
vintagefunksjon alene er ikke godkjent dekning. Ingen B-fit eller redusert
kildevariant er åpnet; A/C-resultatene bevares. Native trening, TEST, handel og
spending forblir stengt.

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
