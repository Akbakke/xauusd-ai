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
- Lagrede checkpoint, lest uten trening på de samme 256 senere radene, viser
  tidlig splitt: ved steg 512 falt M5-aktivitets-Brier 0,280493 → 0,078990,
  mens M5-side-Brier steg 0,243376 → 0,269708. Mellom steg 2560 og 2959
  steg side-Brier 0,271524 → 0,327660, og predikert UP falt 47,2 → 29,8 %.
  De siste 399 stegene bedret side-Brier på en separat prøve av 512 nylige
  TRAIN-rader (0,308953 → 0,265399), men forverret altså senere score.
  Full siste TRAIN-kohort hadde 50,4 % faktisk M5-UP på 2 161 kjente sider;
  enkel M5-fasitubalanse forklarer ikke den senere 29,8 %-prognosen.
  På hele TRAIN er M5/D1-fasiter ulike i 2 072/4 234 felles kjente tilfeller,
  M5/H4 i 3 704/7 470. Samme sluttretning på slike ulike oppgaver er en
  konkret svikt i TF-kondisjonering, ikke bevis mot MTF-data i seg selv.
- Tapet gir M5-side ved minutt 5 én av 25 sidetermene og har i tillegg
  25 aktivitetstermer. I full-GC bruker alle fem tidsrammer felles
  backbone og retningshode; TF-/tidkoordinater legges først til i
  spørringsprojeksjonen. Sluttlogitten varierer mye mer mellom rader enn
  mellom tidsrammer på samme rad. Dette lokaliserer svikten til læringsmål /
  utgangsgrense, men beviser ikke én isolert rotårsak. Eldre TA-kjøringer
  med eget M5-hode og TF-dekodere ga ingen etablert generell M5-gevinst.
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

Det målte problemet er at boten lærer aktivitet og felles retningsnivå, men
ikke nyttig TF-spesifikk retning eller sertifiserte M5-vendinger i denne
S-kjøringen. Et stort indikatorbibliotek hjelper ikke når M5-sidemålet er
ett av 25 sidemål, fasiten ofte er ukjent, og samme hode presses av alle fem
tidsrammer. Dette er en konkret mål-/utgangskobling; hvilken del som
forårsaker mest skade er ennå ikke isolert. S-fasiten er fem marginale
framtidstilstander per TF, ikke direkte vendetid eller kostet Entry.
Individuell indikatornytte, generalisering og økonomisk Edge er ubevist.

Operatørpresisering 16.11: M5 er prognosens brennpunkt og senere Entry-klokke.
M5, M15, H1, H4 og D1 skal hver ha sin egen målte retning/tilstand; andre
tidsrammer understøtter M5 med kausalt lukket kontekst. Uenighet mellom
tidsrammer er et gyldig utfall, ikke en modellfeil. Fem output-plasser er
utilstrekkelig når de i praksis gir samme retningssvar. Den nåværende
full-GC-kjøringen oppfyller ikke dette funksjonelle kravet. Dette vedtaket
åpner ikke trening, Entry, TEST eller handel.

## Neste steg

1. Plan003 er konsumert og full-GC S-hypotesen forkastet. Operatøren har
   trukket den uinnrømmede 30-epokers memoriseringsdiagnosen; ingen trening
   åpnes av dette arbeidet.
2. Bind én separat, etterprøvbar retningsprognose for hver av de fem TF,
   med M5 som brennpunkt og de fire øvrige som målt, kausal kontekst.
   Bruk lagrede checkpoint/score til å avgrense hvorfor dagens delte
   dekoder ikke skiller TF; vurder minste endring i eksisterende hode-/tapsti.
   Mål både kvalitet per TF og om konteksten faktisk hjelper M5. Ingen bred
   arkitekturendring eller gjentatt modellrunde uten en målt blokkering.
3. En senere særskilt innrømmet, kort matchet kontroll må vise både M5-side
   mot fryst referanse og faktisk vendingsdiskriminering før mer trening.
   Senere kronologisk generalisering og alle valgte/åpne posisjoners økonomi
   trengs før Edge. TEST, broker og handel er fortsatt stengt.

Opprydding: historiske overleveringsavsnitt er fjernet her og bevart i Git.
Ingen Python-fil er slettet; gjeldende referanse-, import-, test- og
eierskapsrevisjon beviser ingen frakoblet eier. DATA/RUNS og checkpoints
tilhører retention-eieren.
