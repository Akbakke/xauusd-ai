# Uavhengig Entry-target-audit — 17.10.2026

## Konklusjon og avgrensning

**Nåværende native Entry trenes til tre nettoverdier i bps ved 20 observerte
sesjoner: LONG, SHORT og FLAT=0. Den aktuelle Entry-only-banen bruker ingen lært
Exit-policy.** Dette er terminal markout under deklarerte kostnader; verken beste
mulige exit, sannsynlighet for oppgang, optimal venting eller bevist lønnsom inngang.

Implementasjonen er rekonstruert uavhengig fra original M1 på alle 652 552
TRAIN-rader ved 1/5/20 sesjoner, og på 1 792 VAL-rader innen et sikkert råprefiks.
Gyldighet, brutto-labels, elapsed og netto har eksakt float32-paritet. Alle 17 565
GC/M5-probens midprislabels har float64-paritet. Det er samtidig **økonomiske og
semantiske mangler**: forsinkede native sluttkvoter, uklassifiserte kalender-/datagap,
ulik kostmodell mellom sporene og feil notional i forsknings-SHORT-regnskapet.
Ingen targets, Exit, produksjonsmodell eller forseglede tester er endret.

Autoritet: `NEXT_RUN_POLICY.json`. Kjøring 010 er fullført én gang, exit 0,
kilde uendret, ingen fit eller TEST-lesing. 008/009 er konsumerte auditorfeil
(konfigurasjonssti, deretter delvis råsesjon ved auditens leseslutt), bevart uten
relansering. Rettelsene gjaldt bare auditoren. Den fryste baseline, 007-resultater
og eksisterende targets er uendret. Denne rapporten supplerer `FEATURE_AUDIT.md`.

## 1. Faktisk kode og faktisk trente fase

| Eier | Eksakt ansvar |
| --- | --- |
| `gx1/contracts/entry_causal_m1_outcomes_v1.py:161` | `build_entry_m1_fill_surface`: beslutningsklokke og eksakt M1 bid/ask ved inngang. |
| Samme fil, linje 265 | `_entry_notional_pnl_bps`: LONG/SHORT brutto fra inngangsnotional. |
| `gx1/contracts/entry_observed_market_horizons_v1.py:162` | `causal_m1_terminal_outcomes_at_trading_days`: 1/5/20 observerte sesjoner, terminalkvote og elapsed. |
| `gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py:1013` | `_trading_day_horizon_label_frame`: float32-materialisering, NaN ved sensur/splitgrense. |
| `gx1/contracts/entry_observed_market_v1.py:134` | `build_entry_observed_market_targets`: netto LONG/SHORT, FLAT=0; ingen modellargumenter. |
| `gx1/contracts/entry_observed_market_horizons_v1.py:435,450` | Labels legges i egne batchfelt; `observed_entry_batch_targets` velger h20 til primært Entry-tap. |
| `gx1/contracts/entry_model_native_training_objective_v1.py:27,59` | Fasekontrakt og maskert MSE i rå bps. |
| `gx1/models/entry_v10/entry_v10_ctx_train_v3.py:9574,9741` | Entry-only bruker observerte targets og `unified_exit_stats=None`; Exit-teacher-banen ligger i alternativ gren. |
| `gx1/scripts/research_m5_intraday_recheck_v1.py:169,210,316` | Separat GC/M5 midprislabel, handelsregnskap og én-posisjons-simulator. |
| `gx1/scripts/audit_entry_gc_foundation_v1.py` | Denne uavhengige råpris-/maske-referansen, ikke targeteieren. |

Siste fullførte native fase er `recent_365d_s20261016`, avsluttet 15.10 kl.
10:55:27 UTC. Dens `RECENT_365D_S20261016/PHASE_REVIEW.json` har sha256
`fd2f3cd39ce38556d2da99591d227e3d79d389cf10730a0473bd20fff2d2f139` og
binder `NATIVE_RECIPE.json` med sha256
`2e005c964a0eae539da8ca1852205e30d670c55c85785359c950883e65b099c3`.
Den faktiske scopespecen er identisk med auditens, sha256
`149117700b2aa44667fe17c70a91586cb8b0baabbfac2ce1e99e662b093abd37`:
`entry_only_observed_horizon_20d`, `entry_observed_market_target_bps_h20d`,
`exit_updates_entry=false`, samme TRAIN/VAL/pris-/kostbindinger. Gjennomgangen
bekrefter loss-horisont 20, uendrede mål/rader og 150 uendrede Exit-state-felt.
Dette er kontroll av den faktisk kjørte oppskriften, ikke bare dagens config.

Policyens eldre `learning_gate.paired_review` peker fremdeles på 730-dagers
treningsvinduet i `RECENT_YEAR` (recipe-sha256
`cdf9493b3fb84054d8b7a7ce63d3d52ad38561234628871f2c5f310a8440ebc4`).
Også denne bruker samme h20-target. Fasenavnet recent_year er derfor ikke alene
bevis for vinduslengde eller siste kjøring. Auditens completion review og separat
`LATEST_PHASE_TARGET_BINDING_REVIEW.json` skiller disse bevisene. Den senere
180-dagersfasen ble avvæpnet og er ikke kjørt.

**Viktig faseforskjell:** beholdt `joint_fitted_q` lærer fortsatt Entry fra en
frosset Exit-teachers første-state-verdi. Eldre `entry_only` bruker den gamle
TRAIN-valgte 95-minutters horisonten. Ingen av dem må forveksles med siste
h20-fase. Genuine auxiliary heads består; påstanden om h20 gjelder primært
Entry-action-Q-tap, ikke at samtlige øvrige heads har samme mål.

## 2. Eksakte native formler, priser og klokke

La M5-raden ha åpningsstempel `t`. Beslutning og inngang er `d=t+300 sekunder`.
Det må finnes en M1-rad med **åpningsstempel nøyaktig d**; ingen erstatningskvote
ved manglende inngangsminutt. LONG kjøper på `A0=ask_open(d)`, SHORT selger på
`B0=bid_open(d)`. OHLC-mid brukes ikke i disse terminal-labelene.

Sesjonsindeks `s(u)=floor((u-22:00 UTC)/24 timer)`. Klokken er fast UTC, ikke
bevist historisk CME-/New York-kalender. La `S` være sorterte sesjoner som faktisk
har M1-rader. For h=1,5,20 velges `S[index(s(d))+h]`. Ønsket utgang er samme
klokkeoffset i denne sesjonen; faktisk utgang `e` er første M1-åpning på/etter
dette tidspunktet **innen den valgte sesjonen**. Ingen slik kvote gir sensur.
Det finnes ingen maksimal tillatt sluttkvoteforsinkelse i denne eieren.

Med `Be=bid_open(e)`, `Ae=ask_open(e)`:

```
G_LONG  = 10000 * (Be - A0) / A0
G_SHORT = 10000 * (B0 - Ae) / B0
T_LONG  = float32(float64(float32(G_LONG))  - 4 - 569 * elapsed_seconds / 31557600)
T_SHORT = float32(float64(float32(G_SHORT)) - 4 -   0 * elapsed_seconds / 31557600)
T_FLAT  = 0
```

Brutto og elapsed materialiseres i float32; kostnader beregnes deretter i
float64 før endelig cast. Finansieringsåret er 365,25 døgn. 4 bps er to
utførelser à 2 bps slippage, deklarert provisjon 0. Spread inngår allerede i
bid/ask og trekkes **ikke** en gang til. LONG-finansiering 5,69 % årlig, SHORT 0
skyldes deklarert klipping av gunstig kreditt. Dette er aktuell kostautoritet,
ikke rekonstruerte historiske swappriser. Matematiske fortegn/notional er
symmetriske; faktiske sidepriser og finansieringsvilkår er ulike og eksplisitte.

Primært tap er maskert MSE mot `[T_LONG,T_SHORT,T_FLAT]`. 1 og 5 sesjoner er
observasjon, ikke alternative horisonter som velges etter resultat.
Beslutningskontrakten krever unik argmax; eksakte ties feiler lukket. Negative
markouts klippes ikke. FLAT=0 representerer unngått eksponering; den verdsetter
ikke en senere inngangsmulighet. Ingen label setter maksimal faktisk holdetid.
D1 er kontekst/hovedhorisont, mens raden fortsatt har M5-beslutningsklokke.

## 3. Exit-uavhengighet, kausalitet og holdout

Råreferansen beregner targets direkte fra bid/ask og tidsindekser uten å kalle
produksjonens horisontfunksjon. Nettoprodusenten har bare brutto, elapsed og
økonomi som argumenter. Derfor kan endrede Exit-vekter/-handlinger ikke endre
labelen når råpriser, klokke og kostautoritet er faste. Trainerens aktuelle
Entry-only-gren hopper over Exit-teacher-beregningen; eksisterende native
wiring-/gradienttester dekker dette. Ingen lært Exit ble kjørt i audit 010.
Dette beviser label-uavhengighet, ikke identiske prediksjoner etter enhver mulig
endring av delt modellrepresentasjon eller en annen treningsfase.

Framtidige priser brukes til egne supervisjonsfelt. Native feltorden/315
features og årsaks-/prefixkontroller fra eksisterende featureaudit gjenbrukes;
labels legges ikke inn i featuretensoren av horisontbinder. GC006 har egne
prefix-/råavstemmingsbevis. **Denne auditen har ikke gjenberegnet hele den
historiske featureflaten på nytt**, og gir ikke et nytt altomfattende lekkasjebevis.
Makro-vintage og full GC-bokkontinuitet forblir åpne spørsmål.

Original M1-filens row-group 5 krysser native TEST-grensen. Bare footer ble
inspisert for dette valget; gruppen ble ikke dekodet. Row-group 0–4 gir råpriser
til 09.07.2025 05:33 UTC. Hele siste delvis leste målsesjon utelates fra VAL-
referansen; **VAL-halen er ikke rårekonstruert**. TRAIN/utviklings-VAL har
allerede vært brukt. Ingen ny splitting gjør dem til urørt bekreftelse.

## 4. Rårekonstruksjon, sensur og gap

| Split/h | Rader kontrollert | Gyldige | Sensurert | Forsinket sluttkvote | Største forsinkelse |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN 1 | 652 552 | 651 300 | 1 252 | 2 609 | 50 340 s |
| TRAIN 5 | 652 552 | 650 059 | 2 493 | 3 262 | 48 840 s |
| TRAIN 20 | 652 552 | 646 543 | 6 009 | 4 329 | 50 040 s |
| VAL-prefiks 1 | 1 792 | 1 792 | 0 | 0 | 0 |
| VAL-prefiks 5 | 1 792 | 1 792 | 0 | 0 | 0 |
| VAL-prefiks 20 | 1 792 | 1 763 | 29 | 0 | 0 |

Alle gyldige rader har bitlik brutto/elapsed/netto mot uavhengig referanse.
Materialisert full VAL har 70 573 / 69 535 / 65 772 endelige brutto-labels ved
1/5/20; dette er **inventar**, ikke råparitet for den urekonstruerte halen.

Manglende inngang, utilstrekkelig framtid, manglende sluttkvote i målsesjonen
og utgang etter splitgrensen gir NaN; ingen syntetisk FLAT/0-label. Flere
horisonter kan ha ulik gyldighet på samme historikkrad. Trenerens kvalifiserte
h20-populasjon er derfor ikke nødvendigvis samme antall som alle endelige labels.

**Mangel:** terminaleieren krever ikke kontinuerlig M1-path. Alle gyldige
TRAIN-paths har ikke-sammenhengende minutter, som også inkluderer normale
markedsstengninger. Auditen har ikke klassifisert disse i kalenderstengt versus
leveransegap. Sesjoner helt uten data blir hoppet over uansett årsak. Kontraktens
generelle tekst «missing_or_gapped_m1_path: label_invalid» er dermed strengere
enn den faktisk implementerte observerte sesjonseieren. Inntil dette er avklart,
bør resultatet beskrives som **observert terminal markout**, ikke verifisert
kontinuerlig utførbar posisjonsbane.

Overlappende M5-observasjoner deler framtidige priser og er ikke uavhengige
handler. Et MSE-snitt eller snitt av beste hindsight-handling er ingen portefølje.
GC007 bruker kronologiske ekspanderende folder, én dags embargo og faktisk
utgang før foldgrensen. Native splitcontainment sensurerer grenselabels. Disse
kontrollene gir ikke alene gyldige uavhengige standardfeil eller urørt generalisering.

## 5. MFE, MAE og terminal markout er forskjellige størrelser

`entry_causal_m1_outcomes_v1.py:286` beregner separat M1-OHLC-path `[inngang,utgang)`:

```
MFE_LONG  = 10000 * (max(bid_high) / A0 - 1)
MAE_LONG  = 10000 * (1 - min(bid_low) / A0)
MFE_SHORT = 10000 * (1 - min(ask_low) / B0)
MAE_SHORT = 10000 * (max(ask_high) / B0 - 1)
```

MAE er tapsmagnitude; MFE kan være negativ når også beste observerte exitpris
er dårligere enn inngangsprisen. Horisontens terminalverdi bruker siste
bid/ask-open, ikke et OHLC-ekstremum. Eldre 19 M5-barer / 95 minutter og
MFE/MAE-auxiliary er beholdt separat; de er ikke det primære h20-targetet.
Auditens eksempler viser observerte extrema uten å fylle datagap. H20-targetet inneholder ingen
MFE/MAE eller hindsight-valgt exit; auditens h20-extrema er bare diagnostikk. Slike extrema
garanterer verken tilgjengelig likviditet, korrekt intraminutt-rekkefølge eller
oppnåelig exit. Native M1-open ved beslutning har heller ingen deklarert
60-sekunders utførelseslatens slik GC/M5-proben har.

## 6. Representative, reproduserbare regneeksempler

24 faktiske native eksempler inneholder rå bid/ask, ønsket/faktisk slutt,
sesjonsgrenser, lengste sluttforsinkelse, brutto, hver kostnad og observert
MFE/MAE. Valg er første eksempel per hindsight-argmax og forhåndsbestemte
klokkegrenser, ikke modellvalgte eller lønnsomhetsvalgte handler. Avrundede,
deriverte h20-eksempler fra TRAIN:

| M5-beslutning UTC | Brutto LONG | LONG finansiering | Netto LONG | Brutto SHORT | Netto SHORT | FLAT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01.06.2011 00:05 | −221,6978 | 43,6194 | −269,3172 | 216,3648 | 212,3648 | 0 |
| 10.06.2011 12:35 | 4,2424 | 43,6194 | −43,3770 | −9,7842 | −13,7842 | 0 |
| 10.06.2011 12:45 | 59,9443 | 43,6194 | 12,3249 | −65,5064 | −69,5064 | 0 |

Alle trekker i tillegg 4 bps utførelse. Her er 20 observerte sesjoner 28
kalenderdøgn: finansiering `569*2419200/31557600=43,6194 bps`. Eksempel to
har positiv terminal brutto LONG, men FLAT er best etter kostnad. Eksempel tre
har positiv netto LONG samtidig som observert ugunstig LONG-ekskursjon er ca.
357,65 bps. Det skiller gunstig retning/endpoint fra en praktisk god inngang.
Eksakte råpriser og klokkeboundary-eksempler ligger i RUNS, ikke i Git.

## 7. GC/M5-sporet: annen label og feil SHORT-notional

GC007 lærer én skalær `y=10000*log(mid_exit/mid_entry)`, før spread, slippage
og finansiering. Det er ingen tre-actions LONG/SHORT/FLAT-label. Alle 17 565
lagrede y-verdier og utgangstider er rekonstruert bitlikt fra rå M1.

Beslutning er M5-slutt; inngang er første M1-open på/etter beslutning+60 s,
med inntil 300 s ekstra venting. Utgang er første M1-open på/etter faktisk
inngang+30/120/360 min, med inntil 4 200 s ekstra venting. Path-gap større
enn 6 timer avviser; mindre gap er tillatt. Dermed er horisontene nominelle,
ikke alltid eksakte 30/120/360 minutter. Verken markoutlabel eller simulering
kjører lært Exit. 007 gjennomførte ingen økonomisimulering.

Separate økonomiruter inkluderer spread via bid/ask, 2 bps per execution som
sentral slippage (sensitiviteter 1/2/3/4), provisjon 0, og finansiering per krysset
17:00 New York-rollover: LONG −1,5589 og SHORT +0,8849 bps. Denne siste
konvensjonen er forskjellig fra native veggklokke med klippet SHORT-kreditt.

**Regnskapsavvik:** eksisterende forskningskode bruker
`G_SHORT_old=10000*(B0/Ae-1)`, mens inngangsnotional krever
`G_SHORT=10000*(B0-Ae)/B0`. Den første bruker utgangspris som nevner.
Forskjellen er alltid ikke-negativ:

```
G_SHORT_old - G_SHORT = 10000*(B0-Ae)^2/(B0*Ae)
```

| Nominell horisont | Mål/rader | Gjennomsnittlig overvurdering | Maksimum |
| --- | ---: | ---: | ---: |
| 30 min | 5 935 | 0,025702 bps | 8,569058 bps |
| 120 min | 5 894 | 0,115321 bps | 6,358490 bps |
| 360 min | 5 736 | 0,401570 bps | 19,219958 bps |

Dette er forskjeller mellom hypotetiske SHORT-bruttoutfall på targetkohorten,
**ikke realisert strategitap eller en beregnet endring i Sharpe**. Midprislabelen
og publiserte GC007 MSE er uberørt. En framtidig P0-økonomirute må få eksplisitt,
symmetrisk inngangsnotional og samme deklarerte kostnader for alle armer før fit.
Den gamle fryste referansen må beholdes; ingen retroaktiv omskriving i denne audit.

## 8. Eksakt årsak til 11,81 % full signert GC-flow

Uavhengig avstemming av eksisterende 004 M1-masker mot alle 006 M5-koordinater
matcher begge maskene eksakt. 97,62785 % kjent aggressorvolum gjelder samlet
observert valgt M1-volum. Kravet til full signert M5 er derimot null ukjent volum
og gyldighet i **alle fem** minutter, samt én kontrakt gjennom vinduet.

| Sekvensielt frafall, i denne uttrykkelig valgte rekkefølgen | Fjerner | Gjenstår |
| --- | ---: | ---: |
| Alle native M5-rader | — | 141 806 |
| Manglende eligible M1 | 562 | 141 244 |
| Ingen valgt kontrakt | 256 | 140 988 |
| Kontraktsbytte inne i M5 | 10 | 140 978 |
| Minst ett M1 uten trusted Trade | 1 073 | 139 905 |
| Ugyldig Trade / bad receive | 0 | 139 905 |
| Minst én gapflagget record | 135 | 139 770 |
| Snapshot-Trade | 0 | 139 770 |
| Gjentatt Trade-signatur | 50 936 | 88 834 |
| Ukjent aggressorvolum | 72 089 | 16 745 |

Marginalene overlapper: duplicate-signatur finnes i 51 022 M5, ukjent side i
120 286, gapflagg i 149, ingen trusted Trade i 1 339, ingen kontrakt i 257.
Waterfalltall er rekkefølgeavhengige. Alle 29 eksakte årsakskombinasjoner ligger
i `RESULT.json`. 72 089 vinduer feiler **bare ukjent side etter usignert kvalitet**.
Det forklarer den store forskjellen mellom volumandel og full vindusdekning.

«Duplikat» betyr eksisterende konservative, konsekutive signatur på
`ts_recv,ts_event,sequence,action,side,price,size,flags`. BBO, depth og
`ts_in_delta` inngår ikke. De 50 936 er derfor **avvisning på signatur**, ikke
bevis på 50 936 fulle byte-identiske duplikatvinduer. Fullrecord-kollisjoner må
undersøkes før et eventuelt maskedesignvedtak; ingen svekkelse er gjort.

Reset er ikke en egen nåværende Trade-maskeregel. CVD brytes ved ugyldig minutt,
døgngrense og rull, og betyr relativt delsegment. Manglende vedvarende
bokkontinuitet er **ikke en direkte årsak til dagens 11,81 %**, fordi denne
masken ikke kvalifiserer book-OFI. Dette er et separat, fortsatt uavklart L1-gap:
reset/snapshot, fullført instrument-event, BBO-seams, pris-/størrelsesgyldighet
og tidsintegraler må revideres på valgt kontrakt før L1-features får egne masker.
Instrumentfiltrerte sekvenshopp er ikke alene bevis på tapte feed-records.

Ingen N-side er gjettet. Eventuell framtidig delvis-kjent familie kan rapportere
B,S,N og eksakte observerte delta-grenser `[B-S-N,B-S+N]` under egen maske;
den må ikke kalles full signert delta/CVD eller overskrive dagens strenge maske.
Dette er bare en mulig videre metode, ikke implementert eller innrømmet her.

## 9. Bevisbindinger, tester og videre gate

Run-root:
`/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/ENTRY_TARGET_AND_GC_L1_AUDIT_20261017_010/RESULTS`

- `RESULT.json`: `04259102f253c4dc1619d6436c9e9277bf234b725c596ba8f734211784a5149f`.
- `TERMINAL.json`: `e9cb8abf406acd438f47f363d8f07ce68d6116f9f4818718c41cbbdc6df58bb9`.
- `NATIVE_EXAMPLES.json`: `092038cb08d608b32e0ad046e69258a7b7898ea2cec6ddcc313057c825f2e140`.
- Source ved kjøring: `796b95fece1105b4dbef5dc4395978992ab00d41`.
- Original M1: sha256 `15c4dce883912406b06cbd4e8d91d134feaee7cc1b65ba0b7dc9e979c89778fc`.
- Faktisk `PHASE_REVIEW.json`: `872d79379de60d9bfb619a4b09856d8f68a99319481636d5c79904db847516f1`.

Fokuserte kontrakt-/horisont-/builder-/objective-tester og native wiringtester
består; auditens seks nye adversarial-tester dekker uavhengig aritmetikk,
sluttforsinkelse/sensur, overlappende GC-årsaker, kanoniske bindinger og delvis
råsesjon ved VAL-slutt, samt faktisk h20-trainer med to endrede, forbudte
Exit-teachere (identiske targets/tap/modellstate). Eksakte endelige testtall eies av completion review.
Ingen fullsuite, modellfit, strateginetto, Sharpe eller drawdown er beregnet i
revisjonen; ingen ubevist økonomiverdi oppgis.

Tre separate konklusjoner:

1. **Korrekt implementering:** råparitet PASS for faktisk native h20-markout på
   full TRAIN og avgrenset VAL; GC007 midlabel PASS. Den generelle gapteksten
   matcher ikke terminaleierens faktiske krav. GC-forsknings-SHORT-regnskap har avvik.
2. **Økonomisk meningsfull definisjon:** delvis dokumentert, ikke ferdig kvalifisert.
   Observerte sesjoner/gap, forsinket kvote, historisk finansiering og faktisk
   utførelse begrenser tolkningen. H20 er heller ikke M5-intradag-targetet.
3. **Generaliserbar prediktiv verdi:** ikke bevist. Eksisterende native fase var
   avvist; GC007 ga ingen positiv inkrementell prediksjonsstøtte. Denne auditen
   har ikke foretatt ny læring eller testet en P0-strategi.

Neste arbeid er separat L1-kvalifisering og deretter eksisterende parser/P2-
nivåer med per-familie-masker. P0-fit krever at disse er verifisert og en ny
endelig oppskrift binder symmetrisk handelsregnskap, kostnader, horisonter,
chronologiske folder og matched sammenligningsutvalg. Native TEST, produksjon
og live forblir stengt. Kjøps-/leveranseinventaret dokumenterer fortsatt 26,
ikke 36 måneder; ingen nye data eller betalte kall er gjort i denne audit.
