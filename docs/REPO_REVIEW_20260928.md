# Repo-gjennomgang og konkret opprydding — 28.09.2026

Autoritet: `/home/andre2/src/GX1_CURRENT`, `work/gx1-current`. Kildegrunnlag
`10c78d7090b33a6926614cd9c13e4d02d93c626b` med rettelsene i denne commit.
Ingen native trening, handelskjøring eller TEST-utfall er åpnet.

## Dekning

Alle 850 sporede filer før denne rapporten er hashbundet. Det omfatter 586 Python-
filer, 105 JSON og 114 Markdown. Alle Python-/JSON-filer består strukturkontroll;
alle Markdown er indeksert. Filmanifest:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_PREPARATION_20260927/REPO_REVIEW_INVENTORY_20260928.json`.
Manifestets HEAD/diff-hash angir det presise tidspunktet for inventeringen;
den siste diffen og nye rapporter tilkommer og er gjennomgått separat.

Dette er full filinventering, samlet testkontroll og manuell gjennomgang av
kritiske kjøreveier og endringer. Det er ikke en påstand om linjevis semantisk
verifikasjon av alle 430 000+ linjer. Tidligere kilde-/dokumentlesekopier og
importgraf er gjenbrukt, med kontroll mot gjeldende kilde. Historiske fortellinger
og fullførte planer er bevis, aldri alternative oppstartsveier.

Manuell gjennomgang dekker kildeautoritet, kausal kalibrering, native-par/M5/M1,
quotegeometri, minne/produsentkjede, featureeierskap, Entry/Exit-targets, masker,
modellens beslutningsautoritet, diagnostikk, bundlet metadata og launch-vakter.
Ignorerte runtime-stier kontrolleres av handoverens eksisterende eier. Rådata,
vekter, hemmeligheter og eksterne datasett er ikke blanket-auditert eller publisert.

## Rettet etter observerte feil

| Funn | Rettelse og bevis |
|---|---|
| Ugyldig tidlig fit-/kilde-/oppvarmingsbinding | Eksplisitte tidlige fit-vinduer, eksakte descriptorer og målt 252-D1-oppvarming. Tidligere fokuserte tester og faktiske bytekontroller gjenbrukes. |
| ASK=BID avvist ulikt av lesere | Samme ikke-kryssede quotegeometri som native-kilden. Råpriser/kost uendret; 189 tester og faktisk flatekontroll. |
| M1 toppminne | Kolonnevis validering og frigjøring av ubrukte Arrow-buffere. Fullkjøring kom gjennom alle 1 382 Group-A-chunks. |
| Kunstig positiv gate-entropi | Eksakt xlogy; one-hot-rader gir null selv om alle ruter brukes gjennom epoken. |
| Manglende maskebevis fikk all-true fallback | Krev lagrede masker. Testfixtures erklærer reell tett supervisjon eksplisitt. |
| Små variasjoner ble kalt konstante | Eksakt nullspenn for konstans; strukturelt FLAT=0 beholdes. |
| Stille target-clamps | Ugyldig MAE/binary label avvises; eventtap/diagnostikk deler masken; BCE ser bare observerte celler. |
| Avvikende sesjonsklokke | Samme ASIA/EU/OVERLAP/US-eier som modellkonteksten. |
| Gamma-metadata oppga 1.0 | Eksport og bundle-leser bindes til validert fitted-Q-kontrakt: exp(-rho*delta_wall_clock). Beregningen endres ikke. |
| Foreldede tester | Oppdatert til kausal 119/120-referanse uten hindsight-clamp, vedvarende fullpopulasjon, komplette testfixtures og avvisning av pensjonerte oppstartsruter. |

Ingen sikkerhetsgrenser, kostsatser, featurefamilier eller modellhoder er fjernet.
Dette er retting av målte feil, ikke ny arkitektur eller dokumentert edge.

## Samlet testkontroll og avgrenset ny kontroll

- Fullsuite: 5 628 bestått, 18 feil, 3 hoppet over; 14 subtester bestått, 26m09s.
  `REPO_FULL_SUITE_20260928.log` og tilhørende terminal i kjøringsmappen.
- 17 feil er samme testgjeld som ved konsolideringen. Én tilleggstest krevde
  unreduced BCE. Tapet beholder nå unreduced BCE på maskerte celler før eksakt
  gjennomsnitt; testen og numerisk maskerings-/gradientkontroll består.
- Etter triage: 133 tester i berørte filer består, inkludert alle 18 feilede
  tilfeller. `REPO_TRIAGE_TESTS_20260928.log`.
- M3 før samlet suite: 140 fokuserte tester besto. Metadataendringen har egen
  kontroll med 150 bestått i `M3_BUNDLE_METADATA_TESTS.log`. Overlappende tester summeres ikke.
- Fullsuite er ikke kjørt om. Testresultatet er derfor ikke omtalt som en ny
  grønn fullsuite. Tre skips gjelder historiske targets som krever TRAIN-scope;
  omfanget ble ikke utvidet for å fjerne disse skipene.

## Uavklart inputkontrakt — må løses før ny rebuild

`smc_pivot_envelope_position` er udefinert på sju M1-TRAIN-rader fordi de fire
bekreftede pivotprisene er like. NaN er tilsiktet hos feature-eieren og testet;
modellinput krever alle verdier endelige. M1-parquet og manifest er ikke ferdige.
Diagnose: `M1_SMC_ENVELOPE_DIAGNOSIS.json`. Ingen ny rebuild er startet.

To sammenhengende alternativer: eksplisitt tilgjengelighetsindikator med definert
numerisk representasjon når verdien mangler, eller en begrunnet revisjon/fjerning
av målingen. Begge krever konsekvenskontroll i lokal/MTF-flate, normalisering,
signal-/inputschema, liveness og lineage. Nullfylling uten indikator, sletting
av de sju radene eller uannonsert pivotendring skjuler konflikten.

Ferdige Group-A-chunks er bevart. Nytt skjema kan gjøre dem uforenlige;
gjenbruk må bevises med eksisterende input-/formel-/kildebindinger. Den midlertidige
M1-registry-fitten ble ikke publisert og kan ikke kalles ferdig gjenbruksartefakt.
Post-rebuild/readiness/lifecycle-bindinger gjenstår. Ingen treningsklar-erklæring.

## Forenkling og diskopprydding

[Feature- og kompleksitetsvurderingen](FEATURE_COMPLEXITY_REVIEW_20260928.md)
skiller 241 signaler fra åtte hjelpeoppgaver og encoderkapasitet. Første foreslåtte
sammenligning er færre hjelpeoppgaver med samme inputs/økonomi. Dette er en hypotese,
ikke gjennomført ablasjon eller tillatelse til å trene. Samlet parameterfordeling
og redundans utenfor de 67 kandidatfeltene er ikke ferdig målt.

Oppryddingen her fjerner duplisert event-targetbygging, den fabrikerte maskefallbacken
og tester som krevde gjenåpning av pensjonerte launch-ruter. Sikkerhetsatferden er
nå testet ved faktisk avvisning. Ingen eksterne data, gamle checkpoints eller unike
resultater er slettet. Den tidligere loggførte slettingen av 264 repo-filer er
bevart som historikk; den telles ikke på nytt. Ekstern sletting krever bevis på
redundans og kontroll av aktive avhengigheter gjennom retention-eieren.
