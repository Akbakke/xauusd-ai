# Instrumentreparasjon for A/B/C — 29.09.2026

Dette er implementasjons- og testbevis, ikke et markedsresultat eller en
forhåndsregistrering. Eieren er gx1/scripts/research_model_free_baselines_v1.py.
Tidligere 27.09-resultater og native kostkontrakter er bevart. Den gamle CLI-en
følger sin gamle registrering; nye A/B/C-kall må binde disse forskningsfunksjonene
gjennom en ny, committet registrering.

## Målt blokkering og rettelse

Den tidligere alltid-LONG-referansen lukket og åpnet på nytt per måleblokk.
Den representerte dermed ikke sammenhengende kjøp-og-hold. Statisk finansiering
fra et 2026-snapshot var heller ikke historiske renteforhold.

portfolio_path holder eksplisitte antall enheter mellom deklarerte quotetider.
Den handler bare endringen i antall: kjøp på ASK og salg på BID. Slippage og
provisjon i bps belastes utført notional per utførelse. En konstant beholdning
betaler inn- og utgang én gang. Kontantregnskapet skiller mid-PnL, spread,
slippage, provisjon og finansiering. Åpne posisjoner verdsettes på utførbar
likvidasjonsquote med estimert utgangskost; denne reserven skilles fra betalte
kostnader. En sluttdato er en evalueringsgrense, ikke en maksimal holdetid.

ResearchFinancingCurve integrerer benchmarkrenter over hele veggklokketiden,
inkludert helger og renteendringer. Positive tall er kostnader:
LONG = benchmark + påslag; SHORT = påslag - benchmark. Kreditt klippes ikke.
Påslaget og dagtelleren må gis eksplisitt. Testen 4,11 % benchmark og 1,29
prosentpoeng påslag gjenskaper +5,40 % LONG-kostnad og -2,82 % SHORT-kostnad
fra det eksisterende snapshotets signerte satser; den beviser ikke et konstant
historisk brokerpåslag. Den finansierte basisen er åpningens BID/ASK-notional,
vektet ved tillegg og forholdsmessig redusert ved delvis lukking.
Null finansiering er et eksplisitt separat scenario. Manglende rentedekning
feiler lukket. Ingen historisk renteserie er hentet ennå.

Dette er en kontinuerlig finansieringsmodell for forskning. Faktiske historiske
rollover-dager, tidspunkt, multiplikatorer og brokerbelastninger er ikke bevist.
Native kostkontrakter er ikke endret.

## Rettferdig risiko og porteføljeøkonomi

causal_risk_units bruker bare avkastning kjent ved den aktuelle quoten.
Modell og LONG får samme volatilitetsskala, risikobudsjett og eksplisitte
eksponeringstak. Oppvarming og null volatilitet gir utilgjengelig input, aldri
en skjult handelsregel. Registreringen må velge en felles gyldig populasjon.
Dette er lik kausal risikostyring; realisert risiko rapporteres separat.

portfolio_summary inkluderer første inngangskostnad, alle senere
posisjonsendringer og åpne beholdninger i nettoøkonomi og drawdown.
Sharpe måles mot null kontantavkastning med deklarert annualisering.
En insolvent bane rapporteres med tap og drawdown; Sharpe utelates.

## Inferens, styrke og presisjon

stationary_bootstrap_indices implementerer sirkulær blokkresampling med
geometrisk blokklengde etter [Politis og Romano (1994)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870).
Samme indeks skal brukes på modell, referanse, normalisering og alle endepunkter.
Blokklengde, antall trekk, seed og hele hypotesefamilien fryses før kjøring.

max_t_inference bruker felles max-|t| over den deklarerte familien, med
bootstrap-estimert standardfeil holdt fast i de sentrerte bootstrap-røttene.
Dette er en single-step max-t-korreksjon, ikke en implementasjon av hele
Romano-Wolf stepdown-algoritmen. Metodegrunnlag for felles bootstrap-inferens:
[Romano og Wolf (2005)](https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0262.2005.00615.x).
Det brukes tosidige simultane intervaller. Den endelige ordensstatistikken
samsvarer med p-verdioppløsningen (antall overskridelser + 1)/(trekk + 1).

Hvert endepunkt får en eksplisitt minste relevant effekt, tre positive,
økonomisk deklarerte effektstørrelser, betinget styrke og MDE mot null ved ønsket
styrke. Styrken beregnes ved lokasjonsskift av bootstrap-feilene under familiens
felles kritiske verdi. Det er en betinget presisjonsdiagnose, ikke observert edge.

- GO for effekten: simultan nedre grense er over minste relevante effekt.
- NO_GO for effekten: simultan øvre grense er under minste relevante effekt.
- INKONKLUSIV: intervallet krysser denne grensen.

Dette er bare effektens beslutning. Økonomi, datatilgang og den overordnede
A/B/C-beslutningen må også bestå sine forhåndsregistrerte krav. Manglende eller
degenererte familiemedlemmer kan ikke stille fjernes; instrumentet feiler
lukket. Bootstrapens forutsetninger må vurderes mot faktisk tidsavhengighet og
regimeskifter; tekniske tester beviser ikke statistisk dekning i markedet.

## Kontrollert og gjenstående

Eierens tester består, inkludert uavhengig to-fill-kontantregnskap,
renteintegrasjon over helg, fortegn på kreditt, endring av posisjonsbasis,
åpne posisjoner, insolvens, kausalitet, perfekt paring, skalainvarians,
familiekorreksjon og konsistens mellom intervall og korrigert p-verdi.
Eksisterende intradag- og makrohendelsestester besto etter økonomiendringen.
Alle tester ble kjørt gjennom capped audit. Ingen ny fullsuite.

Maskinbevis:
/home/andre2/GX1_RUNS/TA_RESEARCH_20260929/ECONOMICS_INFERENCE_VERIFICATION.json.

Gjenstår før markedskjøring: bind faktisk D1-/utfallsklokke, finansieringskilde og
kildemanifester; integrer funksjonene i A/B/C med vol-normalisert delta og paret
Sharpe-forskjell; commit kjørbare forhåndsregistreringer med tallfestede effekter,
familier, kostscenarioer og styrkekrav. Historiske data og TEST er ikke lest i
denne reparasjonsbølgen. Ingen forbedret prognose eller lønnsomhet er målt.
