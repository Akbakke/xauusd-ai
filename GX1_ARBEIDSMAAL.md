# GX1 arbeidsmål — oppdatert 28.09.2026

Målet er en ærlig XAUUSD-bot som tar retning på den tidsskalaen der retningen faktisk
finnes, og som slår relevante baselines etter kostnad. Målet er aktivt og ikke oppnådd.

## Gjeldende arbeidsomfang

Brukeren har autorisert feilretting, ferdigstilling av native inputs og full
repo-gjennomgang/opprydding før eventuell trening. Se [CURRENT_HANDOVER.md](CURRENT_HANDOVER.md)
og [VEIEN_VIDERE.md](VEIEN_VIDERE.md) for målt tilstand og neste steg.
Den tidlig kalibrerte ridge/HGB-kontrollen er fullført med NO-GO. Det er ikke
et målt tak for den nye native modellen, som ikke er epoch-trent med disse inputene.

Modell- og featurekompleksitet skal begrunnes med målbar beslutningsverdi. Antall
features alene forklarer ikke manglende edge. Rett først konkrete feil; vurder
senere én begrunnet forenkling om gangen med samme kausale data og kostmodell.
Ingen optimizersteg eller native trening er autorisert av forberedelsesarbeidet.

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
