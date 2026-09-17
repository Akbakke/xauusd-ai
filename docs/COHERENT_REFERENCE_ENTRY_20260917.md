# Samme observerte ankerverdi for Entry og Exit — 2026-09-17

Den konkrete målkjedesvikten er rettet i eksplisitt opt-in-modus. Exit brukte
observerte referansemål, mens Entry fortsatt fikk den gamle lærerens rå Q ved
inngangen. Den nye forbindelsen bruker samme state0-Q_mu for begge oppgaver.

Entry-sideverdien er første gjennomførbare likvidasjon pluss max(gyldig HOLD
Q_mu,0). FLAT er0. Den eksisterende referansepolicyen119/120, observerte
reward/kostnader, veggklokke-gamma og terminal-/bootstrap-eier gjenbrukes.
Q_mu er verdien av en deklarert referansepolicy, ikke optimal Q eller profitt.
120 observerte beregningssteg er fortsatt ingen maksimal handelsholdetid.

## Omfang og tidsgrense

reference_policy og positiv heltalls-reference_cutoff_time_ns må bindes i
fabrikk, adapter, state-view, collator og native treningsbro. Gamle oppskrifter
uten cutoff følger gammel beregning. Ingen produksjonsoppskrift er åpnet for
modusen ennå. State0-ankeret har fortsatt null Exit-lossvekt; antall samplede
online-rader, importancevekter og antall forward/backward endres ikke.

Observerte overganger og den frosne lærerens bootstrap-input må ligge senest
ved cutoff. Hele støtteområdet kontrolleres før state-inputs materialiseres;
et forløp over grensen avvises, uten trunkering, tvangslukking eller oppdiktet
terminal. Klokkebindingen verifiseres igjen ved state-validering/collation og
trening. Dette erstatter ikke manglende full targeteligibilitet for resten av
modellens hjelpeoppgaver eller TRAIN-prefix-normalisering.

## Verifikasjon

77 unike syntetiske CPU-tester består: felles Entry/Exit state0-verdi,
state4-sample med state0-anker, terminaler/sensurering/asymmetriske sider,
eksakt cutoff og avvisning før inputbygging, eksplisitte bindingskrav,
frosne gradients og native bro. Eksisterende legacy-Entry samt Exit-targets
og gradients er uendret i sammenligningen. Eksisterende adapterkjede består.

Første kjøring:75 bestått og én feil i den nye testen. deepcopy gjorde låste
NumPy-arrays skrivbare, så testen stoppet før tidsgrensekontrollen. Bare
mappingkopieringen ble rettet; produksjonskoden ble ikke endret. Den feilede
testen og eksisterende adapterkjede bestod deretter. Ingen bestått fullsuite
ble gjentatt. Begge logger bevares. Audit-cgroup:4GiB/512MiB swap, åtte CPU-
kjerner, én numerisk tråd,64 tasks. Kun små syntetiske testmoduler ble kjørt;
ingen markedsdata, GPU, faktisk modell-forward, normaliseringsfit eller
optimizersteg. TEST forblir forseglet.

Dette er teknisk kontraktbevis. Læring, generalisering og lønnsomhet er fortsatt
ubevist. Frosset kronologisk design/CONTROL256 er uendret og ikke kjørbart.
Neste konkrete blokkering er binding av fersk modell og prefix-normalisering
i eksisterende native eiere; ingen faktisk fit eller native kjøring åpnes her.

Resultat:handover_snapshot/COHERENT_REFERENCE_ENTRY_SYNTHETIC_20260917.json.
Original RESULT.json og TEST_ATTEMPT1/2.log ligger under GX1_DATA:
data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/COHERENT_REFERENCE_ENTRY_SYNTHETIC_20260917.
RESULT SHA256:2f788edc6da619a406a0b44e8f1c215e933462942a7006b30370880098300fbb.
Parentkilde:f3818a92a456a383b2b9eaa466839df3e921b9a3; eksakte endrede filhasher
står i resultatet. Ingen opprinnelige checkpoints eller målinger er endret.
