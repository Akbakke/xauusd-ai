# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_FOLD2_SPAN_NORMALIZATION_001_ADMITTED_NO_NATIVE_FIT. Edge-målet er uoppnådd.
Alle native treningsflagg er false. Én eksakt normalisering er admittert:
MARKET_STATE_TA_FOLD2_SPAN_NORMALIZATION_20261113_001, audit4G/512M/180s/kill-after10.
Seks tofeltsfit på kvalifiserte TRAIN-prefix-rader; alle gamle statistikker beholdes.
Ingen modell, checkpoint, forward, native update eller TEST. Plan/SHA står i policy.

Normaliseringseieren støtter nå eksakt fold2-parent20123245 og fysisk resultat8f2acfaf,
uten fold1-enhetsunntak. Fold1-standardveien og beholdt normaliseringsartefakt er identiske.
27fokuserte metadata-/syntetiske tester og faktisk CLI-preflight PASS. Uavhengig kilde-
og populasjonsbinding gjennomgått. Hele fold2-populasjonen valideres før de seks fittene.
Etterpå gjenstår kvalifisering av aktuell feature-view og egen fersk native initialbaseline.
Ingen slik modelljobb eller native trening er åpnet av normaliseringsplanen.

## Fullført fysisk fold2-kvalifisering

SPAN_REUSE002 er COMPLETE/CONSUMED, exit0, kilde084d6c51,264,72s under4G/512M.
ROOT: /home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_REUSE_20261113_002.
RESULT8f2acfaf, TERMINAL3de8d27a, VALIDATION9313ef0d; eksakte referanser i policy.
Alle116668rader=116197gjenbrukte+471nye. Alle seks radhash-/tilgjengelighetsbevis matcher.
413overlapprader er bitlike for tid, native-posisjon og FP32spans. Tre gap: lokal181,
M5261, M1529; alle31gamle felt bitlike på nye rader (14601celler). Tre produsentkall,
15sidecar-dekodinger,6klokker; null normalisering/model/target/checkpoint/update.
Rekonstruksjon fra kildestart beregnet2473843historiebars;471 er nye outputrader.
11fokuserte subset/default-tester og sluttkontroll PASS; ingen relaunch.

Rettelsen til tidligere metadataavlesning er bindende: f5b4112c var rå aksehash, og
sammenligningen med namespacet e1a30ec1 var ugyldig. Alle seks intervaller matcher riktig
_hash_int64_indices. Original feilkvittering er bevart og overstyrt av ROW_AXIS_CORRECTION
iSPAN_REUSE001-root. Accuracyresultatene påvirkes ikke. 001 feilet før CLAIM på forventet
TRAIN-kvitteringsstatus; originaler bevart. 002 rettet status, planbookends og feilregnskap.

Siste prognoserapport er docs/TA_PERIOD_TRANSFER_REVIEW.json.

## Siste målte resultat

På1689kjente M5 F-sider:865riktige, accuracy51,2137%, balansert50,3929%, AUC0,507776.
AlltidUP gir51,9242%. UP-recall71,72%, DOWN-recall29,06%;71,34%av svarene erUP.
Treklasse-argmax på4096rader svarer alltidFLAT:58,7646%accuracy, balansert1/3.
Dette beviser ingen retningsseleksjon. Kontrollert regneverk og originalreplay stemmer.

M5 oktober INITIAL→FINAL: AUC0,534610→0,565820 mens accuracy55,1387%→53,9967%.
Dette viser at terskelaccuracy kan skjule bedre rangering i punktestimatet. Treningsgevinstens
parede intervaller inkluderer0. Selve FINAL oktober AUC er over0,5 i begge dag-/ukeintervaller
for M5 og H1. M15 oktober treningsgevinst+0,107965 utelukker0 i begge, men FINAL-nivåets
ukesintervall inkluderer0,5. Ikke bland nivå, treningsgevinst og uavhengig bekreftelse.

FINAL AUC oktober/november/desember2024, alle5TF og alle måneder beholdt:
M5 0,565820/0,511835/0,505090; M15 0,607200/0,497404/0,499639;
H1 0,647872/0,483441/0,445580; H4 0,468590/0,629468/0,483199;
D1 0,608163/0,557749/0,574633. H1 desember−oktober−0,202292 utelukker0 i begge
intervaller; M5s månedsfall gjør ikke det. M15s relative treningsendring mot oktober er
negativ i begge intervaller i november/desember. H4 motsier en universell oktoberfordel.

Brier-forverring FINAL−INITIAL utelukker0 i begge intervaller i november M5/M15/H1 og
 desember M5/M15/H1/H4. H1 desember meanPUP80,9992% mot39,8039%faktiskUP påkjentside.
M15 november: Brier+0,059128, kvadrert gjennomsnittsbias−0,003224, rest+0,062352.
Én enkel biasforklaring er utilstrekkelig; desember har stort algebraisk nivåbidrag.
Brier=kvadrert gjennomsnittsfeil+varians av feil er regnskap, ingen kausal forklaring eller
prøvd kalibreringsrettelse. D1 har4–5uker/måned og brede intervaller.

## Utført og verifisert

PERIOD_TRANSFER_READOUT002: /home/andre2/GX1_RUNS/MARKET_STATE_TA_PERIOD_TRANSFER_READOUT_20261113_002.
Kilde08fac47a; PLANca58ed5e, RESULTedaf8879, TERMINALceb904be, VALIDATION5205adbd.
To lagrede4096raders INITIAL/FINAL-arkiver;0modeller/checkpoints/forwards/fits/updates.
5,25s under4G/512M. Begge endepunkter deler foreldre/klokker/nativeFtargets/masker.
Originalreplayfeil0.40scoreceller og1440intervallpunkter kontrollert; maksregnefeil2,22e-16.
2000parede trekk, samme globale70dager/14uker i alle måneder/faser. Få bidragende uker
per måned; udefinerte trekk bevares/telles. Uavhengig kilde-/resultattolkning PASS.
001 hadde plan/helper-navnemismatch fanget av CLI-preflight før CLAIM; originaler bevart.
002 rettet kun planbindingen og er COMPLETE/CONSUMED. Ingen relaunch/promotion.

Tidligere PREPOOL_CAPTURE001/READOUT001 er ferdige/consumed; ikke gjenta dem.
Se docs/TA_PREPOOL_PREDICTIVE_REVIEW.json: alle fem PREPOOL−SHARED AUC-/accuracy-
intervaller inkluderer0. M5 senere AUC0,503209/0,501529 motnative0,507776.
H1 shared−native rangering er bedre i begge intervaller, men usikker mot0,5.
D1 har brede intervaller. Dette isolerer ikke pooling eller all tilgjengelig informasjon.
Ingen ny modell-/treningskode, feature eller terskel er endret i denne bølgen.

## Gjenbruk og ubeviste grenser

Dagens funksjon beholder317signal,132kontekst,255felt perTF, åtte familier og D1/makro.
FullTA-konsum er kontrollert via tidligere8×5bevis107c9531, span8878d17a/4476c63b,
current inputdrift2c87d24c og native paritet. Ingen gjentatt feltcensus trengs.
TRAIN29574beslutninger er mai–september2024; D1-historikk går til2023-05-09.
D1-EMA var konstant påTRAIN-beslutninger; eldre historikk varierer. COT2024-vintage er
UNKNOWN, ingen konkret lekkasje er påvist. Juni2026 er gjenbrukt utviklings-VAL.

Alle intervaller er punktvise på gjenbrukt utvikling, uten korreksjon for mange sammenligninger.
INITIAL arver eldre backbone-trening; retrospektiv splitt av eksponerte rader er ikke OOF.
Nære TF-rangeringer alene begrunner ingen arkitekturendring. Ikke gjenta poolingvarians,
separate hoder eller samme kontekstintervensjon. Eldre detaljer ligger i Git/lenkede rapporter.
Kjent-side-nevner er ikke valgbar Entry-dekning. Årsak, uavhengig generalisering, Entry/Exit-
kvalitet og økonomisk edge er ubevist. Korte forsøk dømmer ikke full historikk/kapasitet.
Full kvalifisert historikk/30–40epoker er senere retning, ikke en åpnet kjøring.
GC er på pause; M5/MTF/26månedersmålet består. TEST forseglet. Ingen broker, handel,
spending, promotion eller DATA/RUNS-opprydding.
