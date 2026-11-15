# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier gjeldende status;
prosess- og terminalkvitteringer overstyrer prosa. Tidligere overleveringer
finnes i Git.

## Gjeldende status

FOLD2_S_FULL_FIT_003_COMPLETE_PRIMARY_REJECTED_EDGE_UNQUALIFIED.
Fold2 S-only full-GC 25-cellers engangsplan er fullført og konsumert.
training_enabled=false, native_training_enabled=false og
full_epoch_training_allowed=false. Windows-oppgaven
GX1RandomAccessCampaignV2 er deaktivert. Ingen relansering, ny modellfit,
TEST, Entry/Exit, broker eller handel er innrømmet.

## Gjort og kontrollert

- Kandidat001 ble forkastet før CLAIM på feil skille mellom featureklokke og
  S-fasitklokke. Privat kandidat002 besto fysisk input-/checkpointkontroll,
  men dens fire optimizersteg var bare mekanikk, ikke full læring.
- Kildepakken 16ad9095, opptaket df978893 og lukket relaunch-gate
  fa875c7d ble pushet. Engangskampanjen kjørte etter fysisk Windows-boot
  655 med 20G/512M/250W-vakter. Den besøkte alle 47 338 modne TRAIN-rader i
  2 959 B16-batcher (siste 10), utførte 2 959 optimizersteg, skrev sju
  checkpoints og skåret samme 4 096 senere utviklingsrader før og etter.
  Terminal exit 0, uendret kilde; signert vakt topp 57 °C kjerne, 66 °C minne,
  138,36 W og 2 292 MiB. TEST ble ikke lest.
- Forhåndsvalgt M5 minutt 5 på 1 392 kjente sider feilet begge referanser:
  FINAL Brier 0,328075 mot fryst prior 0,248755, FINAL AUC 0,471748 mot
  samme-TF nåfortegn 0,499836. Retnings-Brier slo prior i 0/25 celler og
  0/14 senere M5-uker. M5-modellen registrerte 0/347 sertifiserte skifter
  mellom nabopunktene. Den ga gjennomsnittlig 27,4 % UP på rader med 54,5 %
  faktisk UP og valgte DOWN i 20 285/20 480 M5-rad/tidspunktkombinasjoner.
- Ved native endepunkt følger modellens M5- og D1-UP-sannsynligheter
  hverandre nesten helt (korrelasjon 0,999731 over 4 096 senere rader).
  Blant 473 rader med kjent side på begge TF er fasitene ulike i 222,
  mens modellens 0,5-UP/NED-svar er ulike i bare 2. Dette er målt
  TF-diskrimineringssvikt i sluttoutput, ikke bevist årsak i én bestemt port.
  Lagrede vekter viser at de seks TF-/tid-kolonnene i spørringsprojeksjonen
  faktisk ble oppdatert (RMS-endring 0,0310); de var ikke fryst.
- Lesende CPU-portmåling på 256 faste senere rader: lokal åttefamilieport
  la 78,3 % snittvekt på volatilitet/kompresjon etter fit, mot 2,8 % før.
  TF-porten var ulik mellom tidsrammer, men sluttretningene ble nesten like.
  Dette er målt ruting, ikke årsaksbevis for én bestemt indikatorfamilie.
- En separat lesende CPU-replay av initial- og sluttcheckpoint på samme
  1 024 spredte TRAIN-rader fant M5 minutt 5 Brier 0,252211 → 0,291098 og
  AUC 0,517416 → 0,484980 på 366 kjente sider. Dette er en begrenset
  TRAIN-prøve, ikke full TRAIN-score. Alle fem tidsrammers siste retningspunkt
  ble dårligere på denne prøven.
- Samlet bevis og eksakte kvitteringer:
  docs/S_FOLD2_S_ONLY_FULL_FIT_RESULT_REVIEW_20261115.json.
  Tidligere funksjonsrevisjon: docs/FEATURE_AUDIT.md. 25 fokuserte rutetester
  besto før kjøringen; mekanisk PASS er ikke bedre handelsbeslutninger.

## Ikke gjort / faktisk hindring

Modellen er trent, men den nyttige retningsdelen lærte ikke en robust
fortsettelse/vending i denne testen. Nåværende tapsfunksjon optimerer
aktivitet og sertifisert side sammen; aktivitet bedres kraftig fra fersk
initial, mens side forverres. Koden har genuine familie-/tidsrammeporter og
oppmerksomhet, men dette er ikke bevis for at indikatorene samarbeider
prediktivt. Nøyaktig årsak til sidekollapsen er ennå ikke isolert.
S-fasiten er fem marginale framtidstilstander per tidsramme, ikke en
eksakt vendetid eller en selektiv, kostet Entry-beslutning. Full individuell
nytte for alle indikatorer og økonomisk Edge er ubevist.

## Neste steg

1. Behandle plan003 som konsumert og denne 25-cellers S-hypotesen som
   forkastet. Ikke velg ny epoke, terskel eller modell fra gjenbrukt senere
   utvikling.
2. En konkret, uinnrømmet 30-epokers diagnose er fryst i
   docs/S_FOLD2_S_TINY_OVERFIT_DIAGNOSTIC_DESIGN_20261115.json: 1 024 faste
   TRAIN-rader i 64 B16-vinduer, høyst 1 920 oppdateringer, samme ferske
   modell og 25-cellers tap, samme-rad M5-side før/etter. Dette er en
   memoriseringskontroll uten senere modellvalg, TEST eller økonomisk påstand.
   Ny eksakt engangsadmisjon og fersk boot kreves før trening.
3. Hvis selv liten TRAIN-side ikke kan fittes, mål gradient-/hodekonflikt
   før ny arkitektur. Hvis den kan fittes, er generalisering og en direkte
   vending-/Entry-måling neste sperre. Bruk fryst referanse og senere
   kronologisk evidens; økonomi må inkludere alle valgte handler og kostnader.

Opprydding: historiske overleveringsavsnitt er fjernet her og bevart i Git.
Ingen Python-fil er slettet; gjeldende referanse-, import-, test- og
eierskapsrevisjon beviser ingen frakoblet eier. DATA/RUNS og checkpoints
tilhører retention-eieren.
