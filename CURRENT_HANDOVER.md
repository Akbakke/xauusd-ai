# CURRENT_HANDOVER — GX1

## Autoritet og arbeidssted

- Arbeid bare i `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
- Les `GX1_RULES.md` helt; uendret SHA256
  `2c7f0fc9e564d5986db59157dee6cc947bdfd3260f742bacc464853a28683a58`.
- `NEXT_RUN_POLICY.json` er eneste arbeidsstatus. Originale terminaler,
  prosessobservasjon og hashbundne kvitteringer overstyrer prosa.
- Overtakelse: branch/log, `bash scripts/gx1_handover.sh --check`, dette
  dokumentet og aktuelle policyfelt. Åpne eldre bevis bare ved konkret behov.
- Root eier policy, launch og samlet kontroll; andre agenter har avgrenset eierskap.
- Historikk ligger i Git og originalkvitteringene, ikke i denne overleveringen.
  Siste fulle historikkversjon: `c6a8c6672:CURRENT_HANDOVER.md`.

## Gjeldende status og neste steg

NEXT_RUN_POLICY.json er eneste arbeidsstatus. Aktuelt felt er
native_v3_fit004_exact_recovery_20261128_002. Appmålet mot dokumenterbar
retningsedge er aktivt og uoppfylt. Bruker ba om videre arbeid med alle genuine
indikatorer og deres samspill; ingen generell profittpåstand følger av testen.

Den låste V3 PASS2 ved global432 er nå fullført på samme opprinnelige
FIT004-trajektorie. RECOVERY002 tok 805,35 s: 431 nye optimizeroppdateringer,
596 forwards, ingen skip eller refit. Original C1 ble gjenopprettet på nytt;
tekniske kontrollvekter var aldri initializer. Original INITIAL-referanse
er bevart. Alle seks bookends består. Resultat og FINAL.pt/.json er varige.
Originalt fysisk arbeid før FIT004-stoppet forblir UNKNOWN innen1..216;
det logiske sluttpunktet432 omskriver ikke dette.

REFERENCE002: fersk C1 matcher ekte original-C1 på alle ti semantiske bevis.
COLD002: separat kald original-C1→C2 matcher referanse-C2 på alle ti.
Alle tre 002-planer og alle tidligere planer er lukket/forbrukt med null
gjenstående invokasjoner. Alle seks trenings-/evalueringsflagg er false.
Ingen modelljobb kjører. Kjøringens fryste kilde var da04fbe4a.

### Faktisk læringsutfall

Bare forhåndslåste PASS2 er kandidat. INITIAL/PASS1 velges aldri i etterkant.
PASS2 samlet TRAIN-verdict er FAIL; learning_qualified=false.
Aktivitet slår prior på alle fem TF, men dette er ikke retningslæring.
Betinget side slår COHORT-priorens NLL og Brier under begge vektinger kun
på M5/M15. H1/H4/D1 feiler. Side-AUC:

| TF | AUC | Side Brier minus prior, radvektet | Primær sidekontroll |
|---|---:|---:|---|
| M5 | 0,521052 | -0,000265981 | PASS |
| M15 | 0,537790 | -0,000406239 | PASS |
| H1 | 0,522139 | +0,000358074 | FAIL |
| H4 | 0,500753 | +0,002628812 | FAIL |
| D1 | 0,516545 | +0,001826896 | FAIL |

H1/H4/D1 har gjennomsnittlig UP-feil +2,60/+5,02/+4,49 prosentpoeng.
M15/H1/H4/D1 gir fortsatt bare UP ved binær argmax. M5 har begge sider,
men dette er ikke alene bevist selektivitet. H4/D1 feiler proper-score-
årskontrollen2021–2024 og består2025; M5 feiler2021.
Lagret Brier-dekomponering skiller gjennomsnittlig sannsynlighetsfeil fra
sentrert rest uten nye forwards eller fittet kalibrator. H1/D1 har et
vesentlig nivåbidrag; H4 har dessuten nær-null rangering og høyere
sentrert Brier enn prior. Dette identifiserer ingen treningsårsak.

Alle åtte familier har ikke-null datagradient i de tre allerede låste
sluttpanelene. Ett panel har samlet rågradient som motarbeider M5; andre
gjør ikke det. Dette beviser kontakt, ikke nytten av hvert enkelt felt eller
at gradientkonflikt er årsaken til hele treningsutfallet.

### Neste konkrete arbeid

Kvalifiser én oppdateringsfri FINAL-kontroll på de samme faste panelene
[0,107,214]. Skill aktivitet og betinget side på hver av alle fem TF, og
skill readout-bias fra vekter/delte familieblokker. Bevar originale masker,
tapsnevnere, indikatorer og TF. Manglende sidestøtte blir UNKNOWN.
Dagens aggregat M5-mot-aux skiller ikke aux-aktivitet fra aux-retning;
det er det konkrete målehullet før eventuell endret ruting/tap vurderes.

Dette krever minste observerutvidelse, fokuserte kontroller og en ny eksakt
engangsplan med aktuell guardadmisjon. Ingen ny fit, epokeserie, kalibrator,
terskel-, loss-, modell-, prior- eller normaliseringsendring er innrømmet.
FINAL-lokale gradienter kan ikke bevise årsaken gjennom432oppdateringer,
senere generalisering eller økonomisk edge.

## Bevarte tekniske grenser

Original FIT004 og COLD001 stoppet på MoUso-tilstedeværelse. Originale
kvitteringer/journal/backuper beholdes; ingen prosesskill, reboot,
vedlikeholdssperre eller svekket vakt ble brukt. 002 fullførte under samme
grenser. Journal og diagnosebindinger finnes i policy.
Recoverykode:102 guardtester og62 ulike syntetiske leaf-kontroller;
namespace002:17 berørte guardtester og9 snapshot-kontroller PASS.
Uendrede numeriske kontroller og genuine inputs gjenbrukes.

## Kvalifisert datagrunnlag og funksjon

- Native OANDA practice/XAU_USD M5 MBA finnes fra 2009 til juni 2026.
  Alle kildeår/TF er skannet. Separat rettet provider-observasjonsreferanse
  har 542 753 høyere-TF-barer og 7 055 789 eksakte feltsammenligninger uten avvik.
- 1 410 restaurerte M5-observasjoner er utenfor vedtatt PAIR_WEEKCLOSED-domene.
  Referansen er ikke innført som modellinput eller ny featurehistorikk.
- 354 550 TRAIN-rader (2021–2025) har null tapte/ekstra rader fra lagret råkilde til
  canonical/CAPTURE. Alle 17 870 ukentlig-grid-fravær er allerede fravær i råsvar;
  ingen priser skal oppdiktes eller gap automatisk klassifiseres som stengt.
- V3 har 1 772 410 modne, kjente aktivitetsceller og 340 umodne haler
  (M5/M15/H1/H4/D1: 1/3/12/48/276). Tvetydig side blir aldri FLAT.
  Originale nominal-/kompletthets-/sourcegap-/kausale ATR-masker beholdes.
- Full fersk V3 INITIAL på 354 550 rader og egen C2-kald replay er kvalifisert.
  Låst COHORT-scoring skiller factoryreferanse, INITIAL og COHORT-prior;
  alle fem TF vurderes med rad-/målbarvekting, kalibrering og 25 tidsblokker.
- Åtte genuine spesialistfamilier og D1/makro samarbeider med M5 som hovedklokke
  og M15/H1/H4/D1-kontekst. M30 trenger egen kontrakt-/nyttevurdering.
- De 18 interne recurrence-/squeeze-prefixfunnene ble avkreftet før faktisk
  konsumhistorikk. Ingen ny inputbit eller gjentatt full replay er begrunnet.

## Fortsatt ubevist

13/3/4-statusen er ikke oppgradert av teknisk logging, dataparitet eller replay.
Historiske V1/V2 FAIL er ikke gjeldende V3-nevnere eller nye V3-læringsresultater.

- 03/07: historisk samme-feed markedskalender/quote-tilgjengelighet og relevant
  source-known/asof er uavklart. Fravær alene beviser ikke markedsstenging.
- 09: relevant publisering/revisjonsvintage for makro er fortsatt uavklart.
  44 av 61 felt er OANDA-prisderivater fra 11 instrumenter; øvrige 17 er
  COT4/Treasury9/VIX2/GLD2. Ingen aktuell OANDA44-lookahead er målt.
  May18 Treasury er kjent 2467 s før første faktiske input; Apr1 før faktisk
  input 2022-04-03T22:05Z er UNKNOWN. Deklarert 2022-04-03T04:00Z er et separat,
  sterkere klokkekrav; begge publiseringspåstander er ubevist.
  Ingen automatisk arkiv-/HTTP-retry.
- 11/12: provider-HLC-observasjon er ikke ubetinget markedskompletthet eller
  representativitet. Nominal klokke og observerte providerbarer er ulike bevis.
- 15/19: V3-læring og senere generalisering er ikke kvalifisert. TRAIN-år med
  whole-TRAIN-statistikk er ikke kronologisk OOF eller uavhengig senere evaluering.

Full kalender/vintage er ikke et universelt hinder for alle lokale tekniske
kontroller. En avgrenset fit trenger egen eksakt admission; læring og senere
generalisering må måles som utfall, ikke antas bevist før første fit.

## Varige kjøre- og oppryddingsgrenser

- Det varige M5/MTF/26-måneders-MBP-målet i `GX1_RULES.md` består.
  Målet er nyttige tosidige prognoser og senere LONG/SHORT med liten MAE før MFE;
  GC/orderbok er utsatt, ikke slettet. Metode: `docs/NATIVE_LEARNING.md`.
  Én tung CURRENT-jobb via `scripts/gx1_capped_run.sh`, lås og eksisterende
  hardware-/hostvakter; frosne kilder/policy under kjøring. Stabile lange
  kjøringer kontrolleres omtrent hvert 15.–30. minutt, ikke minuttvis.
- Ingen TEST-tilpasning, M1-retningsinput/fasit/normalisering/utvalg, GC,
  Entry/Exit, broker/live/paper/ordre/spending/promotion i denne bølgen.
  TEST er forseglet; juni 2026 er gjenbrukt utviklings-VAL.
- Ingen fast tapsgrense/maksimal holdetid eller fjerning av genuine familier/TF.
  Teknisk PASS, alltid FLAT/HOLD og TRAIN-score beviser ikke handelsnytte.
- Alle originale MAIN-/recovery-checkpoints, kvitteringer, inputs og backuper
  beholdes. Intet nytt slettevedtak foreligger.
  `full_project_retention_cleanup_20261120_001` er forbrukt med resterende 0.
  DATA/RUNS ryddes bare av `gx1.scripts.cleanup_gx1_evidence_v1` etter regel 9,
  med fersk dry-run, kanonisk prosjektlås, verifisert eierskap/referanseomfang
  og eksakt godkjent inventar.
- Denne dokumentforkortingen endrer ingen kode, policy, data eller vakter.

## Bevis som åpnes ved konkret behov

E betyr /home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001.
Aktuell policy binder eksakte SHA for alle planer, originalkvitteringer,
kildekvalifisering, admissionsnapshots, resultater og backuper.

- V3_FIT004_EXACT_RECOVERY_TRAIN_REVIEW_20261128_002.json:
  låst PASS2, alle fem TF, år/blokker, Brier-dekomponering og neste målehull.
- V3_FIT004_EXACT_RECOVERY_RECOVERY_ACCEPTANCE_20261128_002.json:
  original restore, sluttkursør/tellinger, seks bookends og finalcheckpoint.
- REFERENCE/COLD_ACCEPTANCE med samme suffiks:
  fysisk original-C1 og kald-C2-paritet på alle ti bevis.
- SOURCE/QUALIFIED/FINAL_BACKUP med samme suffiks:
  separat Mac-disk, fsync og uavhengig medlemsreadback; FINAL bare når kvittert.

002-runroots under /home/andre2/GX1_RUNS har prefiks
NATIVE_ALL_TF_PROVIDER_HLC_V3_FIT004_RECOVERY_20261128_002_ og suffiks
reference/cold/recovery. Original FIT004 ligger i sin egen uendrede root.
Nyeste originalkvitteringer overstyrer denne prosaen.
