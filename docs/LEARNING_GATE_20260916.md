# Læringsport for GX1 — gjeldende 19. september 2026

Brukerens prioritet er målbar læring før mer omfattende trening. Målet er
robuste Entry/Exit-beslutninger og positiv kostnadsjustert økonomi.

## Nåstatus

Entry fuse256 er ferdig analysert: begge Entry-sider har bedre samlet verdiestimering enn tidligere modeller og TRAIN-konstanter, men fortsatt FLAT256/256 og uendret sidefast Exit. Samlet læringsport er ikke bestått. Brukt256-scope er stengt. Én separat256→512-videreføring er bundet, ikke startet; se CURRENT_HANDOVER.md.
Siste residual256 ga identiske, sidekonstante handlinger. Den nye ONLINE-startbaselinen
er målt selv med identiske vekter; læreren er eksplisitt bevart.

CURRENT_HANDOVER.md og VEIEN_VIDERE.md angir aktuelle filbindinger og handling.
Historiske læringsnotater er bevart i Git og handover_snapshot; de er ikke
nye kjøreordre. Ingen full epoch, full VAL, CONTROL/TEST eller trading er åpnet.

## Kriterier brukt i den fullførte vurderingen — bevares videre

1. Bruk sammenlignbare frosne targets. Nåværende native initialmåling inneholder
   allerede kausal Entry-fasit. DERIVED_TRAIN_BASELINE brukes bare til historiske
   modeller med gammel etterpåklok Entry-fasit, aldri som ny ONLINE-startbaseline. Exit-targets,
   lærer, utvalg, tidsgrenser og masks skal være de samme. Gjenbruk lagrede
   historiske prediksjoner. Ny ONLINE-funksjon må få egen nullstegs baseline;
   initialvektenes hash alene beviser ikke samme prediksjonsfunksjon.
2. Rapporter begge sider og alle ni TRAIN-måneder: MSE, sentrert feil,
   korrelasjon og verdi-/målfordeling. Skill fellesverdi fra LONG−SHORT-
   kontrast. Lavere bias alene er ikke bedre tilstandsavhengig læring.
3. Sammenlign Entry LONG/SHORT/FLAT og Exit HOLD/EXIT mot relevante konstante
   TRAIN-baselines. FLAT gir0. Ta med referanseverdi/regret og handlingenes
   fordeling. All-FLAT er ikke dokumentert selektivitet; all-HOLD er ikke
   dokumentert tålmodighet. Entry alene består ikke samlet Entry/Exit-port.
4. Tolk prøven som gjenbrukt TRAIN med fitted-overlapp. God TRAIN-fit er ikke
   generalisering. Referanse-policyverdier er ikke realisert profitt fra
   modellens egen handelsstrategi. Rapportér disse størrelsene hver for seg.

Ingen vilkårlig prosentgrense skal optimaliseres på resultatet. Vurder effekt,
baselines, variasjon og tidsperioder. Et uklart utfall er ikke PASS og gir
ikke automatisk grunnlag for større trening.

## Før eventuelt større omfang

Ved klar forbedring må neste kronologiske vurdering bindes før utførelse med
modell, mål, utvalg, budsjett og kriterier. Mars–mai og juni2026 er allerede
utviklingsdata og må ikke omtales som urørt holdout. TEST forblir forseglet.

Økonomivurdering bruker gjennomførbare priser, realistiske kostnader, alle
valgte handler og åpne posisjoner. Lukkede vinnere alene er aldri samlet
lønnsomhet. Påkrevde native paritets-/resume-/ytelsesporter må passe kildescope.
Ingen teknisk PASS starter automatisk full epoch eller full VAL.

Ved svakt resultat: lokaliser én konkret årsak i bevarte mål-, gradient- eller
outputbevis. Ikke blind ekstra trening, brede tapsvekt-/terskel-/modellsøk,
forebyggende refaktorering eller gjentakelse av beståtte kontroller. En ny
nødvendig rettelse begrunnes i målinger før koden endres.

Bevar kausalitet, alle features/familier/tidsrammer, kostnader, successors,
bootstrap, originale checkpoints og native vakter. Ingen fast tapsgrense eller
maksimal holdetid. Én agent, én tung jobb; ingen live/paper eller spending.
