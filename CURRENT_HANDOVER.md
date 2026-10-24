# CURRENT HANDOVER — XAUUSD, retning først

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`,
`gx1-3090-lan`. Les GX1_RULES helt ved første arbeid; start overtakelse med
`bash scripts/gx1_handover.sh --check`. NEXT_RUN_POLICY er eneste arbeidsstatus.
Root eier integrasjon, policy og launch. Prosess/terminal overstyrer prosa.
Historikken finnes i Git og originale eksterne receipts; åpne bare ved konkret
avvik eller nødvendig ny gate. Gjenbruk beståtte kontroller og ferdige inputs.

## Mål og aktuell status

Bruker spisset24.10 til **retning først**: etterprøvbare OPP/NED-prognoser.
Trades, bevegelsesstørrelse/MFE/MAE, risiko og Exit utsettes. Målet er aktivt og
ikke oppnådd. TRAIN-fit, senere overføring, generalisering og økonomisk edge
krever forskjellige bevis. Teknisk PASS er ikke gode prognoser.

259 er COMPLETE/consumed og forkastet som retningskandidat. Ingen bedre
retning er kvalifisert. 263 har bestått syntetiske mekanikktester, mens den
fysiske 261-kontrollen feilet før første prefix. 264 er fullført og forbrukt: fire senere prefixer støttes, det første
blokkeres av uobserverte D1-felt. Ingen modell- eller statsfit er innrømmet.
Kun én normaliseringsjobb265 er innrømmet; modell- og treningsgater er false. Native fit, CONTROL,
TEST, Exit og trades er lukket. Root eier launch; se 264-status nederst.

## Siste faktiske retningsmåling — 259

Uendret239 logistisk R1 med tre felt: nåværende ret_1, ret_1 ved **eksakt
D−300s** og qualified_parent_present. Samme245 renprisfasiter, fem kronologiske
TRAIN-prefixer og frosne R1/252-kontroller. Ingen normaliserings-/kontrollrefit,
klassevekter, HPO, terskelvalg eller kalibreringsfit. Nativefeatures beholdes.

| Horisont | Senere OOF | Alltid OPP | Lagret252 | Balansert OOF |
|---|---:|---:|---:|---:|
| 1 min | 49.8831% | 50.3099% | 50.4972% | 49.7481% |
| 5 min | 50.9006% | 51.0086% | 50.7493% | 50.1875% |
| 15 min | 51.5822% | 51.6899% | 51.6819% | 50.2923% |

Historikk bidrar i alle15 modeller og30 prefix-/senerepopulasjoner, uten ekstra
modellkall. Alle tidligere-returkoeffisienter er negative og bidragene varierer.
Dette avklarer konsum for pakken, ikke generell temporalverdi eller kausal effekt.

Avgrensede proper-scoreforbedringer:5min slår lagret R1 med fire dag/uke95%-
intervaller under0;15min slår egenprefixprior tilsvarende, ukesøvre svært nær0.
1min er verre ennprior med fire intervaller over0. Alle12intervaller mot252
krysser0. Av36pooled er10negative/22krysser0/4positive. Fellesnegative folds
over alle6sammenligninger er0/1/2, underkravet4. Alle3followups/samletfalse;
hardaccuracy er underalltidOPP påalle3horisonter. Ingen kontroll/h velges somvinner.

## Faktiske input- og utføringsbevis

Åtte ferdige payloads144496754B ble SHA-verifisert av fryst worker.245cached
F32[stop,66]col65 matcher252/256ret_1, samme210prefixstats. Eksakt lag finnes
for117799/118249foreldre;450ukjente=1førdomene+449uteneksaktparent.213genuine
rånuller beholdes medpresence. Fravær betyr ukjent kvalifisert parent, ikke
manglende markedsbar. Ingen filtrering/row−1/nearest/backfill.253første1minutt-
emisjon gjør historiskparent kjent innenD−4min;051body ble ikke lest pånytt.

Globalproof før0fits; femprefitproofs før0/3/6/9/12fits. Nåværende/lagF32hashes,
F64trekolonneorden, parent/D, prefixstats og15maturity-/fitmasker matcher256.
14-arraykontrollparitet er byteidentisk256; begge18fold/pooledscores avstemt
førfit. Alle118249foreldre/88400OOF beholdes. Sju syntetiske grupper består.
Syntetisk1fit/2populasjoner er adskilt fra genuine15fits/30populasjoner; alle
seks callbacklag teller faktiske vellykkede kall. Egne11-keymodelltilstander
og15roundtrips er kvittert uten ekstra genuine inferens;30logitbidragsrapporter.

6.367sworker/6.667sytre, fryst6383d8c2/policy, beggeexit0 ogsourceunchanged.
Cap4GiB/512MiB/64pids/énnumerisktråd ogCURRENTprosjektlås er kvittert. Prosess-
RSSpeak327827456B, rchar271333626B, scratch0. RSS er ikke cgrouppeak; endelig
cgrouppeak er ikke fanget.15modeller15702B ogOOF19895338B bevares utenforGit.
0nyencode/norm/R1-/252refit/nativeF/U/CONTROL/TEST/quote/originalOBS/forbidden.

Peergrensen er JSON-/statkonsistens og rapportaritmetikk. Workerens payload-
hashes, rekonstruksjoner, modellroundtrip og numerikk er ikke uavhengig replayet.
GjenbruktTRAIN er eksplorativ utvikling; nominelle intervaller er ingen urørt
generalisering, økonomi eller promotion.259ACTUAL_REPORT_PEER/DECISION binder
refs i policy og i:
`/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_EXPLICIT_PREVIOUS_RETURN_REFERENCE_20261024_259`.
Originale receipts/checkpoints/modeller bevares. IngenDATA/RUNS-rydding.

## Tolkning og neste avgrensede spørsmål

Forrige256 brukte3345snapshotfelt+lag i fast ExtraTrees. TRAIN55–60%, senere
50–51%; balansert50.15–50.53%. Bare5/15modeller brukte lagretur.259 avklarer
at eksplisitt bruk alene ikke løser prognosene. Gapet kan være forenlig med
støytilpasning, ustabile relasjoner eller representasjonsgrenser; årsakene er
ikke skilt. Ingen generell dom om at indikatorer/nativehistorikk er uten verdi.

Native har allerede ordnet M5/MTF-historikk, men gjeldende fleroppgaveregresjon
er ikke den rene245 binære prisretningsoppgaven. Neste nødvendige spørsmål er
et rent retningsmål på denne genuine historikken mot samme kronologiske
referanser. Dette er kilde-/designkvalifikasjon, ingen nytrening, vektsøk,
featurefjerning eller launch. FullTRAINnormalisering fra native er ikke en
kvalifisert OOFprefixnormalisering. Nye fit/data/nativegater forblir false.

## Fullført260 — retningskomponent, ingen native fit

`gx1/contracts/entry_native_direction_research_v1.py` er implementert med
uendret native backbone, ett feature-only forward og separat FP32 trelogitthode
på entry_q_joint_hidden. Alle lokale/context/femTF/femGC-argumenter videreføres;
ingen detach, ny produksjonsfase eller eksport-/ordreautoritet. Loss er ren
uvektet maskert BCE; tom supervisjon gir loss=None og ingen oppdateringstillatelse.
Eksakt fasit/klokke/cutoff, null/missing/umodne nevnere og forskningsidentitet
kontrolleres. Numerisk FP32/MHA-policy må anvendes på backbone ved ekte kjøring.

Tre kildepeers og root bestod; to actualpeers bekrefter syntetisk konsistens.
To nye kode-/testfiler; native eierkode er uendret.
Én capped syntetisk kvalifikasjon er COMPLETE/consumed:45cases PASS,7forward-
spies/3backward/23lossforsøk/3BCE. Faktiske native constructor-/encoder-/forward-
kall og Python-markedsåpninger var sperret; alle tellere0. Ingen datafit,
optimizersteg eller genuine modellgradienter.4.865sytre, beggeexit0/fryst9a9bfa19,
worker/PLAN/binding/policy/statbookends uendret. Cap4GiB/512MiB/64pids/éntråd.
Alle execution/data/training/nativegater er igjen false.

Spies og syntetiske leaf-gradienter beviser komponentmekanikk, ikke ekte
native-/GC-konsum, fersk fullinitial, checkpoint/resume eller bedre prognoser.
Påkrevd precommit bestod10guardrails+61nativekontrakttester undercap.
SOURCE_REVIEW/INPUT_FEASIBILITY/PLAN/RESULT/TERMINAL/ACTUAL_QUALIFICATION_REVIEW
står i policy260 og i
`/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_NATIVE_SEQUENCE_COMPONENT_20261024_260`.

Neste konkrete datagap:210 er parentvektet snapshotnormalisering, ikke native
96M5-/ctx-/TF-/GC-normalisering. Legacy prefixfit støtter jointEntry/Exit;
dynamisk Entry-only-fit krever fortsatt wholeTRAIN. Gjenbruk eksisterende
union-/fitprimitiver i en eksplisitt Entry-only forskningsprefixkontrakt:
sju baseflater +femGC-flater på uendrede245foreldre/cutoffs, uten labelmaskestyrt
utvalg.055/223 bevares. Bundne V29-/fem squeeze-konstanter er fitted2009–2013,
før alle245cuts; disse kan beholdes. Cachemanifest40.4MB er allerede SHA-verifisert;
ikke les body pånytt uten konkret gatebehov. Ingen tett sekvenskopi trengs,
men eksisterende loader har reell SHA-/parquet-/temp-mmap-I/O som må budsjetteres.
Fysiskprefix/freshinitial/genuinegradient/resume gjenstår før genuine fit.

## Datagrunnlag og varige grenser

245 bruker F64-midtpris vedD=bar_start+300s ogT=D+1/5/15min fraeksakteM1-open.
Full9-array198kostnads-/klokkeparitet er bit-eksakt; originale kostnader/targets
bevares.118249TRAINforeldre/88400OOF, binære modnerader88102/87944/87344.
Nullbevegelser/manglende endpoints/umodne rader beholdes i nevnerne. Fit:
observed & sign!=0 & D<cutoff & terminal<cutoff; score terminal<egetholdoutend.
OPP er betinget på observert ikke-null-bevegelse, ingen ubetingetFLAT-klasse.

Femcutoffs2024-10-01/2025-01-01/2025-04-01/2025-07-01/2025-10-01, enderneste
kvartal/siste2026-01-01. Prefix/holdout29849/17753,47602/17288,64890/17537,
82427/18072,100499/17750. Bootstrap2000/seed20260911/min8 er gjenbruktutvikling,
nominale eksplorative intervaller, ikke urørt generalisering eller økonomi.

Snapshotet har historiske sammendrag; native harordnet96M5 og16/64/96/96/252
TF-historikk, men annen fleroppgaveregresjon enn renprisretningen245. Aktiv3345
harret_1/EMA-/hendelsesaldre/CVD/profiler, ikke lokalret_20/roc100. Parkertchart
utelukker ikke17chart-navngitte aktive skalarefelt. NativeTransformer, åtte
spesialister/niendeD1makro, genuinefeatures/TF og055/057/111 beholdes.19.10-
kompaktprofilhar7aktivefamilier;chart er eksplisittparkert. Konsum er ikke verdi.

Ingen native/ONLINE-endring:223inputs/227fresh/229initial/230stateaudit/
231privatequalification/232epoke/233analyse bevares.232 gav ingen stabil retning;
alle12CONTROL-directionCI krysset0 ogalle512actions varFLAT. ONLINE-endring
krever fersk initialbaseline. Langstøtte er ingen Exit-/maksimalholderegel.

Fysiskgrunnlaget er23gjenbrukteutviklingsmåneder:TRAIN mai2024–desember2025,
CONTROL januar–mars2026.26måneder er målet; april–juni er ikke fysiskbundet.
Juni er gjenbruktutviklingsVAL. TEST frajuli er forseglet for alltilpasning/utvalg;
bare vedtattmånedligforwardjournal scorer utenrefit. IngenExit/live/paper/broker/
ordre/spending/promotion. Økonomi omfatter åpneposisjoner ogallevalgthandler.
Langhorisontreferanse er alltidLONG/kjøp-og-hold.

Én tungCURRENT-jobb via gx1_capped_run.sh med lås/hardwarevakter ogfryst kilde/
policy tilterminal. Stabilkjøring kontrolleres15–30min ellersjeldnere, ingen
minuttmodellpolling/fyllarbeid. IngenDATA/RUNS-rydding utenretention-eier.
Oppdater policy/handover i sammebølge; fokuserte kontroller/AST/stale-path/
gitdiff--check og ærlige grenser. Pushrett tilwork/gx1-current gjelder kode/docs/
interne refs/aggregater, aldri rådata/vekter/hemmeligheter ellerforcepush.

## Aktuelt 264 — fysisk støtte målt, alle gater lukket

264 er COMPLETE/consumed: worker/outer exit 0, 548.742/549.297 s på fryst
18b47435. Kilde og eksterne bokender er uendret. Source/current fulldecode 0;
clock/reuse/cache/GC 1/1 hver; NPY 30/30; squeeze 150/150,12/12,1/1;
V29 1/1,11/11,11/11,33/33. Alle forbudte kall og stats-/modellfits var 0.
Cap 4 GiB/512 MiB/64 pids/én tråd/lås er verifisert. Process-RSS peak
4,335,644,672 B er prosessmåling, ikke cgrouppeak; rchar_delta 5,980,473,984 B
inkluderer ikke mmap page faults. Seks mmaps 26,534,936,912 B ble gjenbrukt,
ny scratch 0 B. Opprinnelige 261/262/263/264-receipts og inputs bevares.

Alle fem genuine prefixer er kontrollert. Fold 0 (før 2024-10-01, 29,849
foreldre) har 153 no_observed_prefix_values-blokkeringer: 51 GC D1-felt,
gjentatt i signal/ctx_cont/gc_mtf_d1. Fold 1–4 har 0 blokkeringer og fit_allowed
true i kvalifikasjonsrapporten. Dette gir ingen fitadmisjon i policy og er
ikke bevis for retningsverdi. Actual metadata/stat-peer består; ingen uavhengig numerisk replay. Neste steg er å
velge minste kausale sammenligningsscope fra faktisk støtte uten å fjerne
features, fylle ukjent som observasjon eller bruke senere stats i første fold.
Alle execution/data/fitgater er false; ingen relansering av fullførte planer.

Native eiere og 263 sine 30 beståtte syntetiske tester er uendret. 264 source-
peer kontrollerte 20 PLAN-nøkler, 48 inputstat-identiteter og 17 kildehash.
V29-lesningen er rettet med to manglende metadataref; den forrige 261-jobben
stoppet ved én observert ubundet open, ikke to observerte åpninger. En relativ
policy.path i root sin 264-binding ble fanget før kommando/claim og rettet til
kanonisk sti; originalbinding og preflight-kvittering er bevart separat.
Eksakte terminaler, resultater, prefixrapporter og grenser er bundet under
264 i NEXT_RUN_POLICY. TEST, Exit, trades og trening er lukket. Målet er aktivt.

## Aktuelt 265 — én normaliseringsjobb, ingen modellfit

Første fysisk støttede cut er 2025-01-01, originalfolds 1–4. Valget bruker
inputstøtte før modellfit, uten senere retningsutfall. Alle features, masker og
statistikkkrav beholdes. Fold 0 sine 51 uobserverte D1-felt fylles ikke eller
fittes fra fremtiden. Bear-OB har bare tre unike D1-observasjoner ved første
støttede cut, selv om de gjentas i mange M5-rader; støtte er ikke retningsverdi.
Senere sammenligning må bruke samme 70,647 foreldre/fourfold-cohort for alle
referanser, aldri firefold mot gamle femfold-aggregater.

265 worker0047a76a/PLAN8f1bb809 har root/source-peer PASS: 14 PLAN-nøkler,
17 eiere, 48 inputstat-identiteter, uendret 264-inputeier og ingen ny dekoding.
Fire komplette originale rapport-/verdi-hashpariteter må bestå før første fit.
Behold bare ekte returnerte inputspesifikasjoner og rapporter; eksisterende
native fitter rekonstruerer og kontrollerer hele populasjonen pånytt.

Én capped audit er innrømmet: 1800 s/4 GiB/512 MiB/64 pids/én tråd/lås;
8 kvalifikasjoner, 4 prefix-normaliseringer, 48 surfacefits, 4 GC-fits og
4 kategorifits. Ingen ukjent-fallback eller gamle 055/223-stats. Samme sju
base-/fem GC-flater og sann prefixlineage valideres av uendrede native eiere.
Kilde/policy holdes fryst til terminal. Root konsumerer også et mislykket
forsøk og lukker gatene. Alle modell-, native trenings- og TEST-gater er false.

Ingen faktisk normalisering eller bedre prognoser er ennå bevist. Fersk
samlet native initial, featureview, genuine gradienter og eksakt resume gjenstår
før retningsfit. Native-/modell-/Datasetkode og de 30 beståtte testene endres
ikke. Root eier policy, launch og samlet kontroll; eksakte refs står under265.
