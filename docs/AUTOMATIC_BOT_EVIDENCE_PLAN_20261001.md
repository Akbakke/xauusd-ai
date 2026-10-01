# Aktivt mål: dokumentert automatisk XAUUSD-bot — 01.10.2026

Operatøren ba om å gjøre den foreslåtte veien til et aktivt mål og arbeide videre
gjennom alle punktene. Målet er ikke oppnådd. Dette er gjeldende arbeidsrekkefølge;
historiske forsøk er bevis og skal ikke relanseres.

## Avgrenset første måling: MACRO_CORE

En egen navngitt forskningsarm bruker prisfeltene fra A og seks makrofelt:
DFII10 (nivå og 21 observerte D1-raders endring), DTWEXBGS (loggnivå og endring)
og T10YIE (nivå og endring). DTWEXBGS er bred handelsvektet USD, ikke ICE DXY.
Dette er et eksplisitt eget forsøk, ikke en omdefinering eller fullføring av B.
VIX, GLD og COT legges ikke til denne armen etter at resultater er kjent.
Seks-kilde B beholder opprinnelig mål og dokumenterte kildeblokkeringer.

1. Kvalifiser de allerede mottatte ALFRED-versjonene med
   configs/research/TA_MACRO_CORE_COMPONENTS_20261001.json.
   Bruk bare hash-bundne arkivbytes og de eksisterende XAU-beslutningsklokkene.
   Arkivdatoens slutt i New York er konservativ publiseringsgrense; deretter
   kreves en hel observert kanonisk handelssesjon før første beslutning.
   Ingen bakoverfylling eller historiske sluttversjoner. Bevar både observasjonsdato,
   versjonsdato og første tillatte beslutning. Dette trinnet leser ingen prisutfall.
2. Frys målemanifest etter faktisk målt inputdekning, før fits/utfallsinspeksjon.
   Bruk A's allerede deklarerte mål: observert fremtidig mid-endring delt på
   kausal ATR14 over 20 observerte D1-rader (primært), 5 (diagnostisk).
   Dette er prognosemål, ikke fasit fra en lærer eller en maksimal holdetid.
3. Pris alene og pris+makro bruker identiske kausale ytre og indre TRAIN-rader,
   samme hold-rader, purging, ridge/HGB-konfigurasjon og felles porteføljekapasitet.
   Gjenbruk A's kostnader, finansieringsproxy/zero-sensitivitet og statistikkeier.
   Konstant lært fra samme TRAIN, kausal trend, samme-risiko LONG og kjøp-og-hold
   rapporteres. Matchet A inngår i korrigert familie; ingen sammenligning mot
   gammel A på en lengre eller annen periode. Ingen parameterjakt.
4. Bekreft cached inputidentitet og kausalitet separat. Etter målingen:
   kontroller utfall og porteføljeregnskap fra lagrede bytes, rapporter alle
   år/folds, handler og sluttlikvidering. Gjenbrukt utviklingshistorikk skal
   aldri kalles urørt OOS. Uavklart effekt betyr ikke GO.

## Separat native v38 og senere innføring

Native v38 har 254 felt, men tilgjengelig fullført datasett har v37/242.
Inputkontrollen fra 01.10 er bevart; den beviser ikke læring. Før native fits
må eksisterende kontrakteiere binde nye feature-/datasett-/normaliseringsbytes,
ekte mål og sammenligningspopulasjon. Ny initialbaseline kreves ved endret
ONLINE-funksjon. En avgrenset sammenligning må skille TRAIN-tilpasning fra
senere generalisering og observerte kostnadsjusterte utfall. Gjenbruk først
input- og outcome-cacher der kontraktene tillater det. Ingen full epoch/full VAL.

Makro kan innføres i den samme delte Entry/Exit-modellen når den deklarerte
evidensporten er bestått og native mål-/inputkontrakt er bundet. Ingen separat
Exit-modell eller håndskrevet makroveto. Native training_enabled er fortsatt
false mens kontrakt og måleplan mangler; denne planen er ingen launch-oppskrift.

## Godkjenningsbevis før handelsklarhet

- Kronologisk senere, uavhengig generalisering etter fryst modellvalg.
- Positiv kostnadsjustert økonomi og relevant baselinefordel med usikkerhet.
- Eksakte features, normalisering, klokker og handlinger fra samme bundle i
  reell train/serve-paritet; vekthash alene er utilstrekkelig.
- Offline kontroll av ordretilstand, idempotens, gjenstart og brokeravstemming
  med observerbar tilstand og feil-lukket oppførsel. Dette gir ikke brokeradgang.
- TEST forblir forseglet. Live/paper, spending og automatisk promotion er stengt.
  Eventuell endret operativ autorisasjon krever særskilt vedtak.

Én agent og én tung jobb innen CURRENT; alle tunge steg går gjennom capped-eieren.
Målet holdes aktivt ved negative/inkonklusive resultater, men slike armer utvides
ikke med flere forsøk uten ny konkret hypotese. Arbeid videre på uavhengige
punkter og registrer presise blokkeringer.

## Status ved forhåndsregistrering

MACRO_CORE kilde- og sammenligningsmekanikk er implementert i eksisterende
research_ta_campaign_v1-eier. 48 fokuserte syntetiske tester består, inkludert
hele eksisterende A/B/C-testfilen, eksakt matchede rader og fortsatt stengt B.
Kildekontroll på ekte bytes og læringsmåling er ennå ikke kjørt.
Testlogg: /home/andre2/GX1_RUNS/TA_MACRO_CORE_20261001/CODE_REVIEW_001/TESTS.log.

## Kildekontroll fullført, separat måling fryst

Tre kilder er kvalifisert på ekte arkivbytes. En separat verifikator har
kontrollert nivå, endring og valgte historiske versjoner på alle 4518
beslutningsklokker per kilde. 1761 rader har alle seks felt, fra
06.03.2019 til 30.12.2025. USD er begrensningen; ingen eldre verdier bakoverfylles.
Median observasjonsalder for USD er 7,92 kalenderdager etter det konservative laget.
Dette er langsom kontekst. Cachebyggerens daily_panel og gjenværende HTF-definisjoner
har identisk AST med den fullførte A-byggingen; indikator- og klokkeeier er bundet.

configs/research/TA_MACRO_CORE_PREREG_20261001.json fryser den separate målingen.
Årlige folds 2020–2025, felles input- og prognoserader, samme mål/hyperparametre,
kostnader og 120 sammenligningsendepunkter er bundet før fits.
49 fokuserte tester består, inkludert komplett syntetisk kjøring med
receipt-/cache-/populasjonskontroll. Ingen markedslæring er målt ennå.
Kilde- og klokkebevis: docs/TA_MACRO_CORE_RESULT_20261001.json.

## Måling fullført — videre arbeid

MACRO_CORE MEASUREMENT_001 er komplett og kontrollert. Begge learnerne er
INKONKLUSIV, uten grunnlag for native makroinnføring. Se
[full resultatrapport](TA_MACRO_CORE_RESULT_20261001.md) og JSON-bindingsrapporten.
Kildekontroll og den matchede målingen er ferdige delmål; ikke gjenta dem.
Neste aktive delmål er å binde native v38s faktiske læringsmål og uendrede
gjenbrukbare inputdeler, deretter en konkret avgrenset læringskontrakt.
Entry-Qs frosne Exit-verdi og observerte D1-mål må ikke behandles som samme fasit.
training_enabled forblir false til den konkrete kontrakten og evidensporten er løst.
Senere generalisering, paritet og operativ kvalifisering er ikke undersøkt her.
