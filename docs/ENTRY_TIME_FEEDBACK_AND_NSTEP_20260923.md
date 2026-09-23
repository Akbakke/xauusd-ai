# Entry: tid, kapitalbinding og policybaserte mål — 23.09.2026

Dette er gjeldende oppfølging av ENTRY_TARGET_AND_EMA_DIAGNOSIS_20260923.md.
Brukeren presiserte at eksisterende historikk skal brukes, og at tiden etter
inngang skal måles som tilbakemelding på Entry. Ny megleravklaring er ikke
en blokkering for dette avgrensede forskningsarbeidet.

## 1. Historisk kostnadsgrunnlag er gjenfunnet og kontrollert

Den prospektive policyen fra 12.09 bruker samme M1-priskilde som V12,
SHA256 a8b2b9c1ea9eacfeda3370dbe971843f6caf9a1bbfbc7026e732256048952d32.
Policyfilens SHA256 er
a48f8e56da21cfa670a80c3b4bfdf735d8ce0b29a25e269184bd5f44fb240a69;
parameterautoritetens SHA256 er
b5cfc8ebaf5b5116747266c667edbbf73f4b862f6c4259c4c0fa202fd1095e37.
Begge ble kontrollert på Linux.

Dette forskningsscenariet har 0 provisjon, 2 Bps slippage per utførelse,
5,4 prosent årlig finansieringsbelastning LONG og 0 SHORT. Gunstig SHORT-kreditt
klippes til null. Det er et eksplisitt gammelt scenario, ikke komplett historisk
kostnadssannhet eller ferskt bekreftede meglerbetingelser. Det brukes her til
etterberegning; V12s treningsbelønninger er fortsatt spreadinkludert brutto.
Ingen meglerkontakt, handel eller ny sats er brukt.

4 Bps engangskostnad alene endret ikke lærerens SHORT63/63 på TRAIN eller
LONG2/SHORT61 på VAL. Finansiering alene kan heller ikke forklare SHORT-problemet
under denne referansen. Kapitalbinding og store åpne tap gjenstår.

## 2. Målt tid og prisforløp etter Entry

ENTRY_TIME_FEEDBACK_64.json inneholder hver side for 63 gyldige TRAIN-rader og
63 utviklings-VAL-rader, både råmodell og frossen lærer. Klokkene er rekonstruert
fra samme hashbundne M1-kilde; alle cachede reward-rader ble kontrollert mot
sidekorrekte priser. Ingen nye modellkall eller fits ble brukt til tidsmålingen.

Hver rad rapporterer:
- faktisk veggklokketid fra research-fill til modellvalgt EXIT eller siste observasjon;
- om posisjonen er lukket eller fortsatt åpen;
- brutto mark og mark under den uendrede historiske kostnadsreferansen;
- verste og beste observerte mark fram til EXIT/siste observasjon;
- tid til første observerte break-even etter kostnadene;
- Bps per observert notional-time som beskrivende kapitalbindingsmål.

Åpne posisjoner er sensurerte observasjoner, aldri påståtte lukkede handler.
Uavhengige, overlappende hypotetiske handler er ikke en porteføljebacktest.
Retur per time er her måling, ikke en ny handelsregel eller tapsvekt.

| Råmodell, 63 innganger per utvalg | TRAIN | Senere utviklings-VAL |
|---|---:|---:|
| Modellvalgt lukket / fortsatt åpen | 45 / 18 | 54 / 9 |
| Median holdetid blant lukkede | 71 min | 41,5 min |
| Gjennomsnittlig mark etter historisk kostnadsscenario, per mulighet | -5,9239 Bps | +6,8753 Bps |

Dette gjelder de eksisterende 512-vinduene. Det er ikke full handelshistorikk,
ny læring, netto porteføljeavkastning eller uavhengig edge-bevis.

## 3. Faktiske videreføringer viser hvorfor tiden betyr noe

| Frossen TRAIN-policy / SHORT | Siste bruttoverdi | Faktisk alder | Status |
|---|---:|---:|---|
| Rad 29847 | +28,2583 Bps | 45,0167 timer | Modellvalgt EXIT |
| Rad 208932 | +41,3264 Bps | 317,85 timer | Modellvalgt EXIT |
| Rad 263652 | -1839,2952 Bps | 1728,8167 timer | Åpen, tilstand lagret |

Den siste kontrollen behandlet 71388 tilstander og stoppet ved 900 sekunder
beregningsbudsjett. Den tvang ikke EXIT. Verste observerte mark var
-3056,5316 Bps. Modellens siste Q var HOLD -60,8375 / EXIT -342,8009, mens
observert research-lukkeverdi var -1839,2952 Bps.
Dette er en konkret svakhet i den lærte verdien ved svært sen holding.
Disse tre målrettede forløpene er ikke et representativt populasjonsestimat.

En diagnostisk felles korreksjon av begge Q-verdier til kjent lukkeverdi
bevarte Exit-valgene, men forbedret ikke den cachede Entry-valgvurderingen.
Ingen slik korreksjon er innført som modell- eller beslutningsregel.

## 4. Minste læringsendring: bruk policyforløpet som allerede beregnes

Den gamle Entry-broen brukte bare lærer-Q i første Exit-tilstand, selv om
hele det observerte vinduet allerede var beregnet av samme frosne lærer.
På TRAIN var alle 126 sidemål positive og SHORT foretrukket på alle 63 rader.

Ny kontrakt bruker en policybasert n-step-verdi:
1. Følg lærerens kausale, entydige HOLD/EXIT-valg.
2. Ved første valgte EXIT: bruk den observerte rewarden der.
3. Hvis posisjonen fortsatt holdes ved observasjonsgrensen: bruk samme frosne
   lærers videreføringsverdi. Ingen tvungen lukking eller maksimal holdetid.
4. Ved eksakt action-tie: stopp label-videreføringen og bootstrap lærerens
   like verdi, uten å velge en handling.
5. Alle mål er stop-gradient. Framtidige utfall brukes bare som labels;
   Entry-input ved beslutningstidspunktet er uendret.

Dette bruker eksisterende beregningsvindu og null ekstra modellforwards.
Exit-tap, modellarkitektur, alle features/familier og beslutningsautoritet
er uendret. Entry-kontrakten/iterasjonstilstanden er v3; Exit-kontrakten v3
beskriver den nye Entry-broen. Gamle recipes må ikke relanseres eller resumeres.

På eksisterende cache: TRAIN får 25 negative sidemål, med lærerens målvalg
LONG16/SHORT47/FLAT0. VAL får 13 negative sidemål. Dette er labeldiagnostikk
med framtidige treningsutfall, ikke kausale Entry-prediksjoner eller en
forbedret tradingmodell. FLAT-selektivitet er fortsatt ikke dokumentert.

## 5. Kontroll og konkret videre arbeid

- Ny helper matcher en uavhengig sløyfe eksakt på alle 126 TRAIN- og
  126 VAL-sideforløp. Åpne forløp bootstrapper; lukkede bruker valgt reward.
- 11 avgrensede kontrakttester bestod, inkludert tie, observasjonsgrense
  og avvisning av hull i tilstandsprefikset.
- Begge eksisterende native treningsveier er kontrollert på samme ekte
  cachede episode: identiske tap og gradienter, uendrede modellvekter.
- Ingen optimizersteg eller ny trening er gjennomført. TEST er forseglet.

Neste trinn er én avgrenset, kildebundet læringssammenligning i eksisterende
native kjede med baseline før fit og samme senere VAL-utvalg. Bruk det
eksisterende kostnadsgrunnlaget eksplisitt og rapporter Entry-valg, hele
observerte policyutfall, tidsbruk, åpne posisjoner og ugunstig prisbevegelse.
Tiden er nå målt; ingen vilkårlig tidsstraff eller tapsvekt er lagt til.
Bedre selektivitet og bedre økonomi må vises av læringen før større kjøring.

Runtime/evidens: /home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923.
GX1_CURRENT er urørt. PC er ikke restartet.
