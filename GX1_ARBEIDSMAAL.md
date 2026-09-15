## Brukerpresisering 2026-09-15

Ingen videre epoch før den målte Entry/Exit-svakheten er undersøkt med faktiske
TRAIN-targets, prediksjoner og læringssignal, og en eventuell konkret justering er
begrunnet/verifisert. Ikke fortsett bare fordi ingen kodefeil ble funnet. Native-
trening er stoppet på checkpoint85/epoch2offset1152/global5233. Ingen fast grense.

# Gjeldende GX1-mål — 2026-09-14

Få modellen til å lære selektive, retningsmessig gode Entries og Exit som
realiserer best mulig kostnadsjustert nettoverdi. Første større læringsforsøk
bruker ett år: TRAIN 2025-06-01 inklusiv til 2026-06-01 eksklusiv, med hele
juni 2026 som VAL. Ingen ny femårs-epoch nå. Eksisterende ramme er opptil
30 epocher, VAL hver epoch og early stopping med patience 5. Første komplette
epoch og VAL skal vurderes før videre anbefaling om større treningsomfang.

Ettårsutvalget har 65 295 av 313 399 opprinnelige TRAIN-rader. Full parenthistorikk,
child-ID-er, prisbaner, normalisering, alle 200 features, åtte familier og
tidsrammer beholdes. Femårsvekter er initialisering; dette er videre kalibrering,
ikke en modell som bare har sett ett år. Juni er gjenbrukt utviklings-VAL.
TEST forblir forseglet.

Intradag er ønsket stil. M5 er ingen pålagt handelsvarighet eller eneste fokus.
Kvalitet er viktigere enn antall handler. Høyere tidsrammer og familier skal
bidra målbart. Bevar et selvstendig markeds-/forecastsignal for Entry og skill
det fra Entry-handlingenes verdilærer. Ingen bred jakt på EMA-regler, terskler,
nye modeller eller rammeverk før eksisterende læring er målt.

Risiko er avklart: «Ingen fast grense». Ingen fast tapsgrense eller maksimal
holdetid. Exit sammenligner forventet videre nettoverdi med gjennomførbar lukking.
Bevar fysisk pris-/kostnadsregnskap, successor-semantikk og etterprøvbar
verdsetting av åpne posisjoner. Lukkede vinnere alene er ikke samlet profitt.
Risiko og v4-økonomimål er bundet i docs/RISK_OBJECTIVE_20260914.json.

## Bekreftet beredskap

Kilde 128c55f2 bestod faktisk native overgang og 32 sammenhengende mot 16+16
steg over omstarter: alle 14 sammenlignede tilstandskomponenter er eksakt like.
GPU256-paritet består med identiske handlinger. Absolutt samlet gjennomstrømning
er dokumentert for TRAIN32 og ett fullt VAL-vindu. Eksisterende CPU-/cachebevis
gjenbrukes. Ingen samlet relativ speedup eller fullført juniresultat hevdes.

Ny Exit-head initialiseres eksplisitt til close_now_baseline_v1 i online,
target, EMA og relevant Adam. Øvrige vekter og tilstander beholdes. Første
lærerbatch velger FLAT13/LONG1/SHORT2; studenten FLAT0/LONG4/SHORT12. Gammelt
positivt HOLD-bootstrap er borte fra startlæreren. Forecast-gradienten når de
fire undersøkte Entry-rutene, mens Entry-Q/Exit-gradientene er frakoblet der.
Dette åpner for første ettårslæring; modellen er ennå ikke dokumentert kalibrert
eller profitabel. Den delte backbone er ikke fullstendig isolert fra Exit.

De beståtte tekniske portene gjelder det avtalte ettårsutvalget. Gjeldende
NEXT_RUN_POLICY.json blokkerer nye hele epocher. Bare eksplisitt bundet kontroll
på16/32 ekstra steg kan kjøres etter verifisert optimizer-/checkpointovergang.
Se handover_snapshot/EXIT_LEARNING_ADJUSTMENT_20260915.json; eldre tekniske porter
er bevart i handover_snapshot/NATIVE_YEAR_LEARNING_READINESS_20260914.json.
Målmodellen oppdateres etter komplett epoch og VAL. Neste bevis er faktisk
læringsresultat: selektivitet, Entry-kvalitet, Exit-atferd og samlet cash pluss
åpen verdi etter kostnader. Mer adaptiv ML/RL må begrunnes med bedre resultater
på senere perioder uten læringslekkasje.

Første gamle femårs-epoch og juni-VAL er bevart. Epoch2 stoppet på checkpoint315,
offset320, totalt19 908 steg. Gammel juni analyseres bare med første epochs EMA:
2 227 lukkede og 3 281 HOLD ved månedsslutt av 5 508 valgte. Full-policy netto
Bps fra den gamle kjøringen er ikke tilgjengelig. Alle originaler bevares.

Eneste kodebase er /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Start/gjeninntreden: bash scripts/gx1_handover.sh --check, CURRENT_HANDOVER.md
og NEXT_RUN_POLICY.json. Bruk bare native campaign via gx1_capped_run.sh:
TRAIN16, VAL256, åtte CPU-arbeidere, tre timers VAL-vinduer, FP32, eksisterende
optimaliseringer og maskinvarevakter. Historiske kildekopier er avhengigheter.

Underagenter er uttrykkelig autorisert. Én tung jobb samtidig; root eier
integrasjon, verifikasjon og oppstart. Gjenbruk beståtte tester og analyser.
Kontroller stabil langkjøring omtrent hver time. Stående autorisasjon gjelder.
Oppdater handover ved vesentlig endring og commit/push ferdig arbeid. Ingen
live-/papirhandel, spending eller TEST-bruk.
