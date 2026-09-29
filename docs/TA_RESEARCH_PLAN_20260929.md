# Forskningsplan for tekniske indikatorer — vedtatt 29.09.2026

Målet er å måle om en begrenset, kausal indikatorrepresentasjon gir bedre
XAUUSD-beslutninger enn relevante baselines etter kostnader. Målet er aktivt.
Dette dokumentet binder arbeidsomfanget; det er ikke en kjørbar forhåndsregistrering.
A er fullført med INKONKLUSIV for begge modeller; B/C gjenstår.

## 0. Dokumentasjon og autoritet

Publiser den eksisterende rotårsaksrapporten med kontrollerbare presiseringer.
Originalrapport, tidligere resultater og checkpoints bevares. Synkroniser regel 1,
arbeidsmål, policy og overlevering. Dokumentasjonsbevis ligger under
/home/andre2/GX1_RUNS/TA_RESEARCH_20260929/DOCUMENTATION_VERIFICATION.json.
Den avsluttede v37-inputforberedelsen skal ikke relanseres.

Eneste kodebase er /home/andre2/src/GX1_CURRENT, work/gx1-current. Én agent og én
tung jobb innen CURRENT. Alle tunge CPU-kontroller og forskningsfits går gjennom
scripts/gx1_capped_run.sh med eksisterende audit/producer-profiler og vakter.
training_enabled=false gjelder native trening. Ingen native optimizer, full
native VAL, TEST-utfall, live/paper eller spending. Native featurefamilier bevares.

## Fremdrift 29.09.2026

Dokumentasjonsvedtak og rotårsaksrapport er publisert i 52c8761e.
Ridge-instrumentet har nå 19 alphaer fra 0,01 til 1e7, eksplisitt ren-ridge-modus
via eksisterende --ridge-constant-alternative off og rapportering av konstantvalg
og nedre/øvre søkegrense. Konstanten beholdes også når ren ridge måles.
Fokuserte eiertester med uavhengig sklearn-referanse besto gjennom capped audit.
Bevis: /home/andre2/GX1_RUNS/TA_RESEARCH_20260929/RIDGE_REPAIR_VERIFICATION.json.
De utvidede [økonomi-/inferensfunksjonene](TA_RESEARCH_INSTRUMENTS_20260929.md)
er også mekanisk kontrollert: sammenhengende kontantregnskap, signert
finansieringskurve, kausal risikostyring, porteføljestatistikk, paret stasjonær
bootstrap og felles max-t med styrke/MDE. [A er nå målt](TA_A_RESULT_20260929.md) med fullstendig kildebinding og
96 erklærte endepunkter. Begge primære modelldommer er INKONKLUSIV, uten GO.
B/C-kildekontroll, registreringer og målinger gjenstår. A skal ikke gjentas.

## 1. Reparer målte instrumentfeil hos eksisterende eiere

- Utvid ridge-regularisering til minst 1e7 eller deklarert Gram-relativ skala.
  Rapporter valgt alpha, grensevalg og andel konstant-fits. Ren ridge må kunne
  måles eksplisitt; den kausalt lærte konstanten beholdes som separat baseline.
- Beregn sammenhengende kjøp-og-hold med faktisk inn-/utgang, markering av åpne
  posisjoner og historisk finansiering. Bind historisk USD overnight-serie og
  foreslått 1,29 prosentpoeng brokerpåslag til kontrollert kostkontrakt.
  SHORT-kreditt skal ikke klippes bort. Null finansiering er ko-primær diagnostikk.
- Sammenlign modeller og baselines med samme kausale risikostyring. Rapporter
  vol-normalisert merverdi, rå bps, netto porteføljeøkonomi, Sharpe-forskjell og
  drawdown. Overlappende utfall er ikke uavhengige handler.
- Bruk paret stasjonær blokk-bootstrap og max-t/Romano-Wolf-korreksjon over hele
  den deklarerte hypotesefamilien. Bind blokkvalg og resamplingsregler før kjøring.
- Beregn styrke ved tre økonomisk meningsfulle effekter og MDE ved deklarert
  signifikansnivå og ønsket styrke. Beslutningen skal ha GO, NO-GO og INKONKLUSIV.
  Utilstrekkelig presisjon alene er ikke bevis for manglende signal.

Ingen brede søk eller nye rammeverk. Gjenbruk eksisterende data, cacher og beståtte
kontroller. Rett bare dokumenterte blokkeringer, med fokuserte tester.

## 2. Commit kjørbare forhåndsregistreringer før nye eksperimenter

Hver registrering binder kode, inputmanifester, as-of-regler, perioder, folds,
targets, baselines, risikostyring, kostnader, hypotesefamilie, økonomiske effekter,
styrke/MDE, beslutningskriterier, kapasitet, seed og output-identitet.
Hash-bind mottatte data før kjøring. Ikke velg regler ut fra nye forsøksutfall.

### A — kompakt D1-representasjon

Nøyaktig sju felt: (C - C.shift(L)) / ATR14 for L = 21, 63, 126 og 252
handelsdager; 252-dagers range-posisjon; ATR14 / ATR252; og
(C - EMA200) / ATR14. Bind ATR-/EMA-/range- og stengingssemantikk eksplisitt.

Mål ren ridge og én begrenset HGB-arm mot enkel trendfølging, kausalt lært
konstant, sammenhengende kjøp-og-hold og LONG med sammenlignbar risiko.
Primær foreslått utfallshorisont er 20 handelsdager, sekundær fem. Kontroller
eksisterende utfalls-/markedsstengingssemantikk før tallene fryses i registreringen.
En beregningshorisont innfører ingen maksimal holdetid i native handel.

Bruk gyldige kronologiske årsfolds innen 2011–2025, tilstrekkelig oppvarming,
purge og kausal normalisering. Historikk som allerede er inspisert er
utviklings-/gjenbrukt evidens, aldri et påstått urørt holdout.

### B — begrenset makrotillegg til A

Høyst 15 på forhånd navngitte felt fra DFII10 realrente, USD/DTWEXBGS,
breakeven-inflasjon, GLD-beholdning, COT netto spekulativ posisjon og VIX.
Bind originalkilde, hentemetode og immutabelt kildemanifest før henting.
Bruk faktisk publikasjonstilgjengelighet og historiske dataversjoner, med minst
én handelsdag etter tilgjengelighet. Reviderte sluttserier er ikke uten videre
historiske as-of-inputs. Kan en kilde ikke dokumenteres, lukk den armen tydelig.

Mål B minus A på samme tilgjengelige populasjon og de samme folds, targets,
kostnader og risikobaselines. Manglende data skal ikke gi skjult utvalgsforskjell.
Ingen innhenting eller eksponering utenfor det navngitte forskningsomfanget.

### C — avgrenset bekreftelse og kostnadsdekomponering

Frys en kombinasjon av de fem post-hoc fortsettelsescellene før bekreftelsen:
rn50_cross_follow_h12, setup_pdh_break_trend_H4_pair_h12,
setup_momentum_confluence_long_pair_h12,
setup_range_break_up_H1_trend_H4_pair_h12 og orb_london_local.
Bind samtidig eksakt kollisjon/duplikatbehandling og beslutningstidspunkt.

Kartlegg faktisk tidligere bruk av VAL før periodevalget. Juni 2026 er allerede
gjenbrukt utviklings-VAL. Ingen del kalles urørt uten dokumentert grunnlag.
Skill mid-signal, spread, slippage 0 / 0,5 / 1 / 2 bps og finansiering.
Registreringen må si om bps belastes per utførelse og håndtere begge sider.

En enkel passiv modell bruker korrekt beslutningsquote og berøring i neste bar.
Rapporter fyllingsandel, ikke-fylte signaler, utvalget som faktisk fylles og
etterfølgende PnL. Ingen tilbakevirkende limitplassering eller dobbelttelling.
Barberøring beviser ikke køplass, faktisk fill eller lønnsom live utførelse.

## 3. Kjør og dokumenter uten blind utvidelse

Start først etter commit av relevant registrering og beståtte nødvendige
instrumentkontroller. Bind én entydig run-ID; bevar feil og delresultater.
Krev terminalkvittering, aggregate/per-fold resultater og bevis for at TEST ikke
ble åpnet. Stabil langkjøring kontrolleres etter prosjektets intervaller, ikke
med minuttvise modellrunder. Reparér rapportering uten kostbar omkjøring når
beregningene allerede er gyldige.

## 4. Beslutning og ferdigkriterium

- GO på A/B: foreslå én ny mål-/horisontkontrakt til operatøren. Ingen automatisk
  native bygging, trening eller endring av beslutningsautoritet.
- GO på C: foreslå separat forhåndsregistrert utførelses-/ordrebokforskning.
- NO-GO eller INKONKLUSIV: oppgi nøyaktig populasjon, kostnader, effekt og
  presisjon. Ikke generaliser til at all teknisk analyse er umulig.

Sluttrapporten skiller mellom målt på deklarerte data, bevist teknisk konsistent
og ikke undersøkt. Målet er ferdig når A/B/C er avsluttet med dokumenterte
beslutninger eller eksplisitt kilde-/målebegrensning, og overleveringen er
synkronisert. Ferdig kode, dokumentasjon og aggregerte bevis commit/pushes innen
stående autorisasjon; rådata, modellvekter og hemmeligheter publiseres aldri.
