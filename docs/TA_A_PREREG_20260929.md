# Måling A — kjørbar forhåndsregistrering 29.09.2026

Registreringen committes før henting av finansieringsserien og før nye XAU-utfall
beregnes. Eksekverbar autoritet: configs/research/TA_A_PREREG_20260929.json.
Orkestrator: gx1/scripts/research_ta_campaign_v1.py. Den gjenbruker eksisterende
indikator-, klokke-, ridge/HGB-, økonomi- og inferenseiere. Dette er den avgrensede
A-armen; B og C gjenstår og inngår ikke som skjulte utvidelser av denne kjøringen.

## Data, klokke og horisont

Bind native M5-kilden XAU_M5_NATIVE_2009_20260701_PAIR_20260927 med manifesthash
83af4819543c638369eaef4cc23e3805e73d7f853c89fc94e58b5233d8befb4b.
Les bare årspartisjoner 2009–2025, filtrert fra 2009-06-01 til før 2026-01-01.
Verifiser filhashene fra manifestet før lesing. Ingen 2026-årsfil eller TEST
åpnes. 2009–2010 gir oppvarming og de første kausale treningsradene.

Årsfolds 2011–2025, kronologisk ekspanderende TRAIN. Denne historikken er allerede
inspisert utviklingshistorikk. Fra juni 2025 overlapper den tidligere utviklings-VAL.
Resultatet er ny, forhåndsbundet walk-forward-evidens på gjenbrukt historie, ikke
et urørt holdout eller endelig bekreftet handelsfordel.

M5-stempel er baråpning; baren blir kjent fem minutter senere. D1 bruker samme
22:00 UTC-handelsdøgn som gx1.time.session_detector og HTF-eieren.
Bare ikke-tomme D1-biner som er lukket før datagrensen inngår. Source absence
bevares; ingen syntetiske barer eller flytting av klokke for sommertid.
Fem og tjue handelsdøgn betyr fem og tjue observerte D1-biner på denne klokken,
inkludert observerte delvise handelsdøgn. Det er ikke kalenderdager eller 288
ganger et M5-antall. Kildens faktiske helge-/ferie-/pausehull følger med.

Beslutningen har D1-lukketid. Fylling bruker første observerte M5-åpningsquote
på eller etter denne tiden: LONG kjøper ASK, SHORT selger BID.
Ingen bruk av siste quote før markedspausen som om ordren kunne utføres da.
Dette er en deterministisk utførelsesmodell med deklarert slippage, ikke målt
ordreutførelse. Perioder uten ny quote gir senere fill og ekte veggklokketid.

Primær målhorisont er 20 observerte D1-døgn; sekundær er fem.
Target = (framtidig fill-mid minus dagens fill-mid) / dagens lukkede ATR14.
Tjue døgn brukes som felles purge både før årsgrensen og ved indre modellvalg.
Ingen TRAIN-target får strekke seg inn i den aktuelle evalueringsperioden.
Felles evalueringspopulasjon krever tilgjengelige sju felt, alle prognoser og
begge horisonters framtid innen kildegrensen. Dermed utgår siste målhorisonts
beslutninger fra evalueringen. Faktisk første/siste fill og radtall rapporteres.

## Sju felt og to modeller

Fire momentumfelt: (C - C.shift(L)) / ATR14 for L = 21/63/126/252.
Range-posisjon: (C - laveste low over 252) / (høyeste high over 252 - laveste low).
ATR14/ATR252 og (C - EMA200)/ATR14 er de to siste feltene.
Bruk canonical Wilder ATR og SMA-seedet EMA fra technical_indicators_v1.
Oppvarming er utilgjengelig; ingen nøytralutfylling eller framtidig normalisering.

Ren ridge bruker det kontrollerte intervallet 0,01–1e7. Indre kronologisk hale
på 20 % velger alpha ved MSE, med egen purge og egen skalerings-fit. Den valgte
modellen refittes på hele kausale ytre TRAIN. Konstanten får ikke erstatte
ridge-armen; den rapporteres som separat referanse og MSE-diagnose.

Én HGB-arm: maksimalt 100 iterasjoner, learning rate 0,1, minimum 20 bladrader,
seed 0; samme indre hale/purge velger iterasjon. Eksisterende HGB-eier kan velge
den kausale konstanten; dette rapporteres eksplisitt. Ingen nye søk.
Minimum 100 ytre TRAIN-rader og 20 indre rader. Fold uten tilstrekkelige rader
registreres som utilgjengelig; det oppfinnes ingen erstatningsfold.

Prognosens fortegn bestemmer LONG/SHORT; eksakt null gir FLAT.
Retningen oppdateres daglig. Horisonten er et prognosespørsmål, ikke tvungen
lukking etter fem eller tjue dager.

## Referanser og sammenhengende økonomi

Referanser på samme populasjon: kausalt lært TRAIN-konstant; enkel trend
(fortegn av gjennomsnittet av de fire momentum-fortegnene); alltid-LONG med
samme kausale risiko; og sammenhengende kjøp-og-hold.

Modell, konstant, trend og risiko-LONG bruker 21 tidligere fill-mid-avkastninger
til volatilitetsestimatet, annualisering 252, målvolatilitet 10 % og maksimalt
1,0 ganger initialkapital i mid-notional. Dette er eksplisitte forskningsvalg,
ikke tunede terskler eller brokergrenser. Quantity beregnes med samme regel
for alle og et fast initialt kapitalbudsjett på 100. Kjøp-og-hold beholder
100 / første fill-mid enheter gjennom hele evalueringsperioden.

Ingen kunstig lukking ved fold- eller horisontgrense. Alle faktiske endringer
i beholdning betaler spread, 2 bps slippage per utførelse og 0 provisjon
(videreført sentralt kostscenario). Sluttposisjonen lukkes på siste
evalueringsquote som regnskapsgrense. Åpne posisjoner underveis inngår i
utførbar likvidasjonsverdi. Følg den kontrollerte kontantregnskapseieren.

To ko-primære finansieringsscenarioer: null og historisk benchmark pluss
1,29 prosentpoeng. [FRED DFF](https://fred.stlouisfed.org/series/DFF) er en daglig
USD overnight-serie i prosent per år; kilde er Federal Reserves H.15.
Hentemetode/vindu er bundet separat i TA_FUNDING_SOURCE_RETRY_20260929.json før henting.
Mottatte bytes og full kalenderdagsdekning bindes i RECEIPT.json.

DFF er kun en etterfølgende kostnadsmodell, aldri et prognoseinput.
Den fryste reviderte historien utgis ikke for å være vintage-riktige features
eller faktisk historisk brokerfinansiering. LONG-kost = benchmark + påslag;
SHORT-kost = påslag - benchmark. Negativ kost er kreditt. Basis er signert sides
åpningsnotional, veggklokke og 31 557 600 sekunder per år. Det konstante påslaget
er utledet fra eksisterende brokersnapshot, ikke verifisert historisk konstant.

## Hypotesefamilie, effektstørrelser og beslutning

Hele familien er 96 endepunkter: 2 horisonter × 2 modeller × 2 finansieringer
× 4 referanser × 3 statistikker. Ingen valg av vinnende dekning, år eller seed.
Årsresultater og samlet brutto/nettoøkonomi, kostnader, Sharpe, realisert risiko
og drawdown rapporteres. Alle økonomiske perioder inngår.

De tre statistikkene er gjennomsnittlig parvis nettoavkastningsdifferanse i bps,
samme differanse delt på kjent underliggende daglig volatilitet i bps,
og differanse i annualisert Sharpe mot null kontantavkastning.
Avkastningsserien kommer fra samme eier som porteføljerapportens Sharpe.

Paret stasjonær bootstrap: 1 999 felles trekk, seed 0 og forventet blokklengde
60 observerte D1-intervaller, tre ganger maksimal målhorisont.
Felles max-|t| gir tosidige simultane intervaller ved alpha 0,05.
Dette bygger på at blokkresamplingen tilstrekkelig beskriver tidsavhengigheten;
regimeskifter kan svekke inferensen, og årstabellene må vurderes sammen med den.

Tre på forhånd valgte relevante effekter:
- Netto meravkastning: 1, 2 og 5 bps per dag; første verdi tilsvarer omtrent
  2,5 prosentpoeng årlig før renters rente og er minste praktiske forbedring.
- Vol-normalisert differanse: 0,01, 0,02 og 0,05 av underliggende dagsvolatilitet.
- Sharpe-forskjell: 0,10, 0,20 og 0,30.

Første verdi er minste relevante effekt. Rapporter betinget styrke ved alle
tre, Monte Carlo-presisjon og MDE mot null ved 80 % styrke.
GO/NO_GO/INKONKLUSIV for et endepunkt følger samtidige grenser som beskrevet i
TA_RESEARCH_INSTRUMENTS_20260929.md.

Udefinert Sharpe eller null bootstrap-variasjon forblir et navngitt
INKONKLUSIV-medlem i hele familien; ingen p-verdi eller intervall oppfinnes.
Max-t beregnes på de sammen resamplede, statistisk definerte medlemmene.
Hele familien og den eksakte delmengden rapporteres; ingen mislykket kandidat
fjernes. Et utilgjengelig påkrevd endepunkt hindrer GO for den modellen.

En modell får GO på den primære 20-døgnsarmen bare dersom alle tre endepunkter
slår minste relevante effekt mot risiko-LONG, konstant og trend i begge
finansieringsscenarioer, og samlet netto er positiv i begge scenarioer.
Ethvert påkrevd NO_GO gir NO_GO; ellers INKONKLUSIV når GO ikke er oppfylt.
Fem-døgnsarmen og sammenligningen med ujustert kjøp-og-hold er diagnostikk,
men er inkludert i korreksjonen. GO åpner bare forslag om ny mål-/horisontkontrakt.

## Kjøring og artefakter

Kjør med ren work/gx1-current og SHA-256 for den committede JSON-registreringen.
Alle kildeeiere er hash-bundet. Ingen automatisk gjenkjøring hvis output finnes.

1. Capped audit: research_ta_campaign_v1 fetch-funding med finansieringsmanifestet.
2. Capped producer: research_ta_campaign_v1 run-a med A-registreringen.

Begge kalles med .venv/bin/python -m gx1.scripts.research_ta_campaign_v1,
--spec og --spec-sha256 gjennom scripts/gx1_capped_run.sh.
JSON-filen binder endelige outputstier. STARTED, FITS, dagspanel, prognoser,
parvise avkastninger, bootstrap-statistikker, RESULT og TERMINAL bevares.
Terminalkvitteringen og resultatets filinventar må bekreftes før tolkning.
Bare aggregerte bevis og dokumentasjon publiseres, aldri rådata.

## Transportrecovery før første markedsmåling

Første manifest i c385c316 fikk lesetimeout uten mottatte bytes. Originalt
manifest og FUNDING_DFF_001/FAILED.json er bevart. Et nytt manifest
TA_FUNDING_SOURCE_RETRY_20260929.json binder samme URL, kilde og datovindu,
60 sekunders svartid og separat FUNDING_DFF_002-output. A-registreringens
transport-/kildebinding oppdateres før første markedskjøring. Ingen numeriske
forsøksvalg eller beslutningsregler endres.
