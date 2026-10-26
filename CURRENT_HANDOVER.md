# CURRENT HANDOVER — XAUUSD, retning først

Autoritet er `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current` på
`gx1-3090-lan`. Les GX1_RULES helt ved første arbeid. Start overtakelse med
`bash scripts/gx1_handover.sh --check`; NEXT_RUN_POLICY er eneste arbeidsstatus.
Root eier integrasjon, policy og launch. Prosess og terminal overstyrer prosa.
Historikken og originale kvitteringer bevares; ingen gamle planer relanseres.

## Aktuell status

276, 277, 279 og 280 er COMPLETE/consumed: alle fire faste nativeprefixer er
kjørt med uendret 275-protokoll, egen fersk 0U, én full B16-pass og samme
første 256 TRAIN før/etter. Rapportert senere kohort er 70,647 foreldre.
Selve identiske firefold-joinsammenligningen, R1/252/259, rang og 48 blokk-CIs
står fortsatt igjen. Alle 30 programpunkter er uferdige i ledger004.
GLOBAL training/native/epoch/launch er false. Ingen aktiv tung jobb.

## Siste faktiske kjøring — 280

Fryst d72ef7bb / policy da3bd836 gjennom begge vinduer, inkludert pausen.
Alle 50 nativekilder er fysisk SHA-kontrollert uendret. Prefix4-normalisering
267, view268, direction260 og AdamW/fullstate270 er gjenbrukt uten refit.
Boot540 ga RESUMABLE: 61,472 foreldre / 3,842 oppdateringer. Etter kontrollert
maskinledighet og fysisk boot541 fortsatte samme pass med 2,440 oppdateringer.
Terminal er COMPLETE: 100,499 TRAIN / 6,282 oppdateringer / 17,750 OOF.
7,425 forwardkall/returer, én resume, null tomme batcher eller legacy/forbudte
åpninger. Siste batch har tre foreldre. Ingen ny epoke eller checkpointvalg.
Worker-tider 12007.302 / 10518.978 sekunder og receipt-UTC er ulike tidsbaser.

| Horisont | Balansert OOF-treff | Modell minus egen prior, logloss |
| --- | ---: | ---: |
| 1 min | 50.0010 % | -0.0000391008 |
| 5 min | 50.0000 % | +0.0000713974 |
| 15 min | 50.0000 % | -0.0001404696 |

Lavere logloss er bedre. De små positive proper-punktene på 1/15 min beholdes;
5 min er svakere. Modellen velger OPP i 17,710/17,720, 17,662/17,662 og
17,536/17,536 kvalifiserte celler. Dette beviser ingen nyttig tosidig retning.
På samme TRAIN256 forbedres 5 min fra egen0U og egenprior; 1/15 min forverres
fra egen0U. Mot egenprior er final5/15 litt bedre, final1 svakere. Alle tre
finalTRAIN-klasser er OPP. Dette er ikke hel-TRAIN-kapasitet eller glemselbevis.

Alle ni kontekster / 75 kategorier og masketellere er bevart, også ukjent,
observert null, begge sider og tomme kategorier. Ikke-null head-/GC-gradienter
beviser ikke individuell featurelæring eller nyttig samspill. Bare INITIAL
har faktisk grad/no-grad-paritet; FINAL/OOF true-flagg er standardbokføring.
Root har verifisert 0U/terminalPT og tre NPZ med opake SHA-er, uten numerisk
replay. Uavhengig fysisk arrayakse-/scoreparitet hører til neste B-admisjon.

Begge guards og observers exit0; peak57C core / 66C minne, 136.94/135.06W,
2186/2231MiB VRAM. Maskinen er prosjektledig, taskReady/restart0 og GPU160Widle.
OUTER_TERMINAL, ACTUAL_ROOT_CHECKS, ACTUAL_NATIVE_SOURCE_CHECKS, tre ACTUAL_*
PEER_REVIEW, ACTUAL_QUALIFICATION_REVIEW og postterminalidle under280 binder
faktiske resultater og grenser. Ubrukte vinduer3/4 er revokert. Ingen relansering.

## Neste arbeid — faktisk B, deretter C/D og begrunnede kontraster

Neste er ny eksakt, capped read-only B-admisjon: alle 70,647 senereforeldre,
identiske prior/R1/252/259-celler, innenfold råskårrang/ties, bias, alle ni
kontekster og opprinnelig 48 dag/uke-proper-CIs. Ingen ny fit, terskel eller
kalibreringsutvelgelse. Rangkontrakt57syntetiskePASS og worker41helpergrupper
samt påkrevde10+61kontroller er kvalifisert; faktisk arkiv-/rapportparitet og
hele CI-kallstien gjenstår. Original mislykket plan er consumed/bevart.

B må kjøres før C-ruteintegrasjon: den forventer de 50 historiske nativebytes.
C-komponentene har34unike syntetiskePASS, men genuine C er ikke innrømmet.
Den konkrete ruteblokkeringen står i C_GUARDED_LAUNCH_ROUTE_GAP_001. Minimal
integrasjon etter B berører fire eksisterende transporteiere og de to C-filene;
C må binde eksakte gamle276ref til nye transportref, øvrige46eiere uendret.
Ingen vaktomgåelse eller endring av modell/trener/Windows-vakter. Deretter egen
kvalifikasjon og nøyaktig194forward /6reverse /0optimizersteg på fire276states.

D er fortsatt forberedelse: gjenbruk024 M1_GC_INPUTS/kvalitetsmasker/roll og245
mål. Skill samtidig GC-respons, senere GC og senere XAU; faste1/5/15min og
-1/0/+1min kontroller, der ikke-kausal arm aldri blir kandidat. E/F-kontraster
krever konkret målt begrunnelse og egen avgrenset admisjon, én mekanisme av gangen.

## Indikatorbetydning og aktivt utvalg

ACTIVE_FEATURE_RELATION_SOURCE_REVIEW_001 binder kildefunnet fra brukerens
spørsmål: pris/EMA-avstander og EMA-helninger i ATR-enheter, stack, kryss og
alder er allerede kodet og aktive. Modellen mottar tall og lærbare familie-/TF-
strukturer, ingen språklig indikatorundervisning. Prediktiv betydning må læres.
Bestemte ikke-GC MTF-candlefelt og XAU-tickvolumrelasjoner er parkert i compact;
GC-kontraktsvolum/flow og61D1/makrofelt er aktive. Fullt bibliotek er derfor
ikke identisk med aktivt konsum. Kildemekanisme beviser ikke lært samspill,
og parkeringen er ingen bevist årsak. Dette skal inngå i begrunnede E/F-vurderinger.

## Varige grenser og gjenbruk

275s177syntetiskePASS, 273sprivateCUDA4U/bitexactresume, 271fullstate og267/268
fysiske inputs/normalisering/view gjenbrukes. Ingen kvalifiseringsvekter i fit.
M5 er beslutningsklokken. Åtte familieeiere og D1/makro beholdes; kompakt aktivt
utvalg er eksplisitt avgrenset. Ingen fast tapsgrense eller maksimal holdetid.
Horisonter er beregningsmål. Retning først; ingen Exit denne bølgen.

TRAIN mai2024–desember2025. CONTROL januar–mars2026 er lukket; juni2026 er
gjenbrukt utviklingsVAL. TEST fra juli2026 er forseglet; bare vedtatt månedlig
forwardjournal uten refit. Ingen live/paper/broker/ordre/spending/promotion.
Teknikk, TRAIN-fit, senere overføring, generalisering og økonomisk edge er ulike
bevis. Én capped CURRENT-tungjobb. Kilde/policy fryses gjennom hele kampanjen,
også RESUMABLE-pauser. Stabil drift kontrolleres15–30min eller sjeldnere etter
flere stabile kontroller; ingen minuttpolling eller fyllarbeid. DATA/RUNS ryddes
bare med retentioneier. Push ferdig kode/docs/interne refs/aggregater til
Akbakke/xauusd-ai work/gx1-current er stående autorisert; aldri rådata/vekter/
hemmeligheter eller forcepush. Målet er fortsatt aktivt og ikke oppnådd.
