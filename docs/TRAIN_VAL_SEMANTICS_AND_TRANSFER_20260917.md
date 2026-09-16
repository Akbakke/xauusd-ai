# TRAIN/VAL-bindinger og retningssignal — 2026-09-17

Den undersøkte svikten kan ikke forklares med ulike normaliserings-, checkpoint-
eller lærerbindinger: alle40 faktiske TRAIN-inputcacher samsvarer med native
VAL på disse punktene. Den frosne kandidatens avvisning står ved lag. Det finnes
også målbar svak overføring i den uendrede prognosegrenen.

## Hva som er verifisert

Samtlige inputcachefiler og deres online-inputtensorer er hashkontrollert.
Metadata, base-normalisering, child-kontrakt og lifetime-normalisering er
identiske med den faktiske native-recepten. Lifetime-SHA:
edb729d5bea83f909549da48fa2dd2389329ef64712854b55c261255dec049ae.
Original checkpoint, de frosne koeffisientene og boundary-læreren samsvarer.
Koden bruker samme collate_random_access_states_v1, forward_exit_random_access_batch,
likvidasjonsrelative Q-koordinater og build_reference_policy_hold_targets.
Tidligere fullført native kontroll viste likhet mellom cache/ukachet samt
CPU-pipeline og original økonomi. Disse beståtte kontrollene er gjenbrukt.

Denne auditen påstår ikke en ny dynamisk GPU-replay av identiske TRAIN/VAL-
eksempler, og utelukker ikke enhver mulig dataflytfeil. Ingen konkret mismatch
ble funnet i det kontrollerte omfanget. Ingen modellkode er endret.

## Samlet TRAIN-gjennomsnitt skjulte en svakhet ved inngangen

Pathlengden er min(state_index+1,512) hos begge tilstandseierne; lengde1
identifiserer derfor state0 eksakt. Vi brukte de lagrede komponerte outputene,
uten nye prediksjoner, lærerberegninger eller readout-fits.

| Utvalg ved state0 | Antall Entries | LONG MSE før→etter | SHORT MSE før→etter |
|---|---:|---:|---:|
| Trente |275|995,57→751,23|997,67→759,84|
| Separate TRAIN |67|516,63→487,73|515,90→544,83|
| Senere juni-VAL |256|1408,83→1625,02|1410,80→1553,61|

State0 er altså representert i treningen. Den separate SHORT-feilen var allerede
verre enn både originalmodellen og TRAIN-konstanten på dette viktige stedet.
Samlet separat TRAIN over512 tilstander så bedre ut fordi andre holdetilstander
hadde bedre gjennomsnitt. Det gir ikke robust dokumentasjon for lærerens verdi
ved inngangen.67 separate ankre er dessuten et lite, fortsatt TRAIN-basert utvalg.

Bootstrapkomponenten i separat state0 har gjennomsnittlig absolutt størrelse
0,0287/0,0190Bps, mot observerte komponenter18,2663/18,2392Bps. Targetet i denne
målingen domineres av observerte markedsutfall, ikke stor bootstrap-amplitude.
Dette er beskrivende komponentstørrelser, ikke en generell uavhengighetsgaranti.

## Prognosegrenen svikter også på juni

De allerede lagrede prognosediagnostikkene er eksakt like i baseline og kandidat.
Readout-endringene skapte derfor ikke denne prognosesvikten. Targetene er faktiske
framtidige M5-sluttkursavkastninger før handelskostnader; Exit-læreren brukes ikke.

| Nominell horisont | MSE modell | MSE nullprognose | Korrelasjon | Riktig retning |
|---|---:|---:|---:|---:|
|5 minutter|100,5425|100,2769|0,0578|43,75%|
|25 minutter|602,4832|589,5135|0,0679|46,09%|
|60 minutter|1465,5808|1430,5336|0,0559|46,48%|
|120 minutter|3225,0460|3081,7023|0,0209|42,97%|

Alle fire har også større MAE enn nullprognosen. Ved120 minutter er forventet
retur i snitt+5,0884Bps mens faktisk middel er−9,1096Bps. Modellen varsler opp
213/256 ganger; markedet går opp109/256. Horisontene teller observerte M5-barer,
ikke garantert lik forløpt klokketid. Prognosetapet er L1/median, ikke en kalibrert
sannsynlighet for retning. Nullreferansen krever ingen tilpasning til juni.

Tidligere frosset120-prognose ga+6,1299Bps etter kostnader på1022 komplette
TRAIN-forløp, mot−5,6235 for alltidLONG og−6,1690 for alltidSHORT. Den gamle
TRAIN-gevinsten var derfor ikke bare en positiv konstant markedsretning.
Den var likevel TRAIN, ikke overføringsbevis. Dette understreker behovet for
å måle reell uavhengighet og kronologi før nye læringspåstander.

## Konsekvens

Ingen ny Exit-justering eller kobling til den gamle prognosen er begrunnet som
en allerede robust løsning. Vi har ikke dokumentert et overførbart retningssignal.
Det er ikke bevis for at alle mulige modeller/features er ubrukelige; juni er
ett gjenbrukt utviklingsutvalg. TEST er fortsatt forseglet.

Neste avgrensede diagnose: mål tids- og framtidsvindu-overlapp mellom de
allerede lagrede trente og separate TRAIN-eksemplene. Dette avgjør hvor mye
uavhengig læringsbevis den tidligere kontrollen faktisk ga. Ikke refit eller
snu fortegn etter juni-resultatet. Modellforwards, trening og nye kjøringer
forblir stengt til en konkret ny måling er begrunnet og avgrenset.

Maskinbevis: [bindings-/tilstandsaudit](../handover_snapshot/TRAIN_VAL_SEMANTICS_AUDIT_20260917.json)
og [eksisterende juni-prognoser](../handover_snapshot/FORECAST_TRANSFER_EXISTING_OUTPUTS_20260917.json).
Begge er laget med null nye modellforwards, targets, optimizersteg eller fits.
