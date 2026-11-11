# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste maskinlesbare arbeidsstatus. Originale
terminalkvitteringer/prosesser overstyrer prosa. Historikken finnes i Git.

## Mål og gjeldende port

Lær framtidig OPP/NED/FLATT før Entry. M5 er prognosetakten, kausalt lukket
M15/H1/H4/D1 og genuine indikatorer beholdes; alle fem TF har egne mål.
GC er på pause; TEST, Entry/Exit, broker, handel og promotion stengt.
TA_M5_LOCKED_SHRINK_NATIVE_QUALIFICATION_ADMITTED.
Alle trenings-/launchporter false; training_enabled_scope=null.
Ingen modelljobb lever ved lukking. Konsumerte planer relanseres aldri.
Målet er fortsatt aktivt: native rettelse og fersk generalisering ubevist.

## Siste faktiske resultat

[Rangbevarende sannsynlighetsjustering](docs/TA_M5_INNER_PRIOR_SHRINK_REVIEW.json)
er COMPLETE/konsumert, exit0, uendret kilde9ae77972.
ROOT MARKET_STATE_TA_M5_INNER_PRIOR_SHRINK_20261111_002.
PLAN476eb63c, workera5c56f1e, RESULT8e127f55, SLOPE_LOCKb4bd9003.
Begge prognosemål slår INITIAL og fryst prior i Brier og logtap på både
heleTRAIN29574 og hele senere17753. Dette er den første samlede beståtte
kontrollen i denne avgrensede korrigerte-R1-readoutbølgen.

Kun to skalarer er tilpasset de allerede lagrede alpha.01-prognosene fra
tre disjunkte indre TRAIN-scorevinduer. Prefix-prior pervindu, hver kjent/moden
rad én gang, pooled logtap. Encoder/normalisering så heleTRAIN;
indre labels er nå kalibrator-fitdata, IKKE encoder-OOF.
Slope-lås ble skrevet før fullTRAIN-headprediksjoner/senere arrays ble åpnet.
Ingen nye alphas/terskler/featurevalg. FP64-transform av genuine nativeFP32
logits: z'=b+a*(z-b), der b er den frosne fullTRAIN-prioren ved anvendelse.

S-slope .7981812905830987, F-slope .6385367563238673.
Begge har strengt indre konvekst optimum; derivat0<0,derivat1>0 og
sluttderivat under3e-19. Rangering og ties er eksakt bevart på alle rader.
Ingen kollaps til prior. Forsøket forbedrer amplituden, tilfører ikke rangering.

Brier INITIAL → justert / fryst prior (lavere bedre):
TRAIN S .248473364 → .248300632 / .249651008,10010known.
TRAIN F .249327011 → .249240480 / .249876211,11909known.
Senere S .249652488 → .249638480 / .249827988,6096known.
Senere F .249855791 → .249786447 / .249833416,7165known.
Senere AUC uendret S .523043324/F .510747753.
Alle allerede fastsatte marginer består. S-gevinst mot INITIAL er liten.

To scalarfits,fire NPZ-åpninger,én privat output;0modeller/checkpoints/
encoderforwards/backward/updates. Helekommando1,433s,worker0,674s.
Uavhengig24skårceller/144felt, maksavvik1,11e-16,0,385s,ingen refit.
Konvekst optimum, lås, rangering, transform og primære marginer dobbeltsjekket.
Dette er faktisk kjøringstid, ikke benchmark av heltrening.

## Hva dette forklarer og ikke forklarer

Alpha.01 W2-hodene på R1 hadde allerede reparert TRAIN: S .248194763/F .249065517,
men full senere tapte INITIAL, og F tapte prior. Den nye TRAIN-only
skaleringen gir bedre senere scores, med noe svakereTRAIN enn uskalertW2,
men fortsatt bedreTRAIN enn INITIAL/prior. Dette isolerer en korrigerbar
amplitudefeil ved fast rangering. Unik optimizer-/encoderårsak er ikke bevist.
SterkereL2 .1 ble tidligere valgt av indre scores, men tapte INITIAL på alle
fire celler. Ny shrink er et separat design, ikke ny alpha-søking.
Alle gamle resultater og originalvekter er bevart.

Senere oktober–desember2024 er gjenbrukt utvikling, ikke urørt generalisering.
Små middelgevinster er ingen statistisk sikker edge eller handelsøkonomi.
Sannsynlighetskvalitet er forbedret, men generell kalibreringsgaranti er ubevist.
Ingen rounded-tensorimport/native end-to-end-paritet er gjort for justeringen.

001-klargjøring ble aldri innrømmet/claimet. Kilderevisjon fant manglende
AUC-nøkkel i native score og private-source-bookend-feil før fit.
002 rettet kun disse; metadata/syntetisk og uavhengig diffkontroll består.

## Neste direkte steg

Fold de to LÅSTE slopes inn i eksisterende fire M5-tensorer:
w'=a*w, b'=prior+a*(b-prior). Ingen refit.
Kontroller FP32-avrunding og faktisk head-regning på gjenbrukte
TRAIN/later-representasjoner før eventuell native import. Full senere-cache
har allerede state/future_representation; ingen ny encoderfangst behøves.
Bevar alle andre modellfelt og samme primære sammenligning.
Én separat frossen native kontroll er nå innrømmet:
MARKET_STATE_TA_M5_LOCKED_SHRINK_NATIVE_20261111_001, PLANc6ca851b.
Worker895dd461, cap2cf2abe4, preflight9,793s med0modell/fit/forward/claim.
28 fokuserte tester består; uavhengig kilde- og faktisk JSON-skjema er kontrollert.
Kun ny eksakt cap-sti, samme250W/170Wdraw/65Ccore/80Cmemory-vakter.
12G/512MiB, én numerisk tråd,900s workergrense. Modellen gjenopprettes fra R1,
de låste slopes foldes fra W2s FP32-vekter via FP64 til én FP32-avrunding.
Eksisterende importer endrer kun fire M5-tensorer.5918faktiskeCUDAheadkall
på begge fulle populasjoner; to ekte row0-forwards for fersk kandidatbaseline
og faktisk weights-only diskroundtrip. Én modell/to kapseldekodinger/én
privat weights-skriving;0fits/optimizer/backward/treningssteg.
Alle øvrige modellfelt/buffere/RNG/modi beholdes. Faktiske avrundede skårer
og fullpopulasjonens målte avrundingsavvik avgjør de uendrede marginene.
Ingen full encoderfangst, fresh constructor, optimizer-resume eller
fersk generalisering påstås. Kilde/policy fryses under kjøringen.
