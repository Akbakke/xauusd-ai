# CURRENT HANDOVER — XAUUSD M5 Entry

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current` på
`gx1-3090-lan`. GX1_RULES er bindende; NEXT_RUN_POLICY eier arbeidsstatus.
Prosesser og terminalkvitteringer overstyrer prosa. Én integrator, én capped
CURRENT-jobb og fryst kilde/policy gjennom hver kjøring. Originalmålet står i
[docs/ORIGINAL_PROJECT_GOAL.md](docs/ORIGINAL_PROJECT_GOAL.md).

## Aktivt mål og neste steg

Målet er bedre M5 Entry LONG/SHORT/FLAT med MBP-1, SMC og lukket M15/H1/H4/D1:
retning, størrelse, liten MAE før første MFE, kostnad, kvalitet og dekning.
Brukeren prioriterte22.10 en solid featuregrunnmur og deretter generell robusthet:
ingen bittesmå tilpasninger eller overtilpassing; undersøk også motsatt retning
ved mye MAE. MAE alene beviser ikke feil retning eller invers edge.

184s avgrensede MFE-endring er implementert og185 kildekvalifisert:133 fokuserte
CPU-tester,0feil/hopp,6,863s, audit4G/512M/én tråd/pids64. Kilde-/matematikkpeer,
AST/compile og diffkontroll bestod. Native183roller/182unike kildefiler er bundet;
kun tre nativefiler er endret. Alle øvrige nativehashes er bevart.
Dette er teknisk bevis.184 er ikke trent eller målt på genuine inputs og har
ingen fersk initialbaseline.173 gjelder forrige170funksjon.172s konkrete
kildekvalifikator må tilpasses185 før en eventuell ny konstruksjon.

186s brede periode-/reverseringskontroll er COMPLETE/consumed og actualpeer-
kontrollert: fire lagrede167/177 OBS én gang hver,0F/R/U/fit,22,488s ytre tid.
Samme entry og opprinnelige horisontsett/masker/kostnader er bevart; FLAT/ties
beholdes. Ingen ny input-/Parquet-/PT-lesing eller prediksjon. Mainresultatet er
bevart. En for stor kvartalseksport feilet; korrigert kompakt eksport bestod.
Digest dekker fem brede kvartalsbånd;1/5/15 finnes i ALL/originalRESULTS, ikke
kvartalsdigest. Ingen estimator eller originalanalyse er endret.

Historisk167s faktisk valgte signaler, terminalrespons etter deklarerte kostnader:

| Periode | Valgte | Original bps | Motsatt bps | Ukeblokkintervall for forskjell |
|---|---:|---:|---:|---:|
|TRAIN|58|−6,983|−5,115|[−2,183;8,072]|
|Q1CONTROL|154|−6,091|−6,263|[−1,952;1,714]|

Forskjellen er motsatt minus original; begge sider har negativt gjennomsnitt.
CONTROL preMAE øker7,361→7,587bps og MFE faller3,185→3,002bps ved inversjon.
TRAIN-kvartalenes inversjonseffekt skifter fortegn.177s faktiske valg er fortsatt
alle512FLAT; tvungen retningsdiagnostikk er ingen valgt handel eller entrygevinst.
Ingen universell snuregel eller invers edge er kvalifisert. MAE alene avgjør
ikke motsatt sides verdi. Dette er prisbanemål, ingen realisertP&L.

TRAIN256 dekker193dager/84uker fordelt25/34/36/42/32/42/45rader i syv kvartaler;
fem av syv kvartaler har under8støttede uker for167s valgte signaler. CONTROL256
dekker68dager/14uker i Q1. Sparse prober er ikke alle118249TRAIN-rader eller
urørt generalisering. Ingen favorittmåned/-minutt eller ny terskel velges.

186/DECISION.json beholder184 som én falsifiserbar M5/MFE-målhypotese, uten
påstått retningsgevinst. Neste arbeid er187s eksterne forberedelse av ny CPU-
konstruktør, tilpasset actual185-kvalifikasjonen og med eksakt ny188/BUNDLE-
destinasjon. Gjenbruk fungerende172/173-mekanikk; ingen modell-/featureendring.
Root/peer må kontrollere før engangskjøring admits. Native initial188 og fersk
privat teknisk kvalifikasjon må senere bestå før én separat admitted, matchet
hel TRAIN-epoke. Ingen ekstraepoker/HPO eller automatisk gjenbruk av gamle planer.
Sammenlign bredt mot egen fersk initial og177 med samme data/budsjett: retning,
størrelse, MAE før MFE, kostnader, LONG/SHORT/FLAT-dekning og periodestabilitet.
Lavere pooledtap eller alltidFLAT fullfører ikke målet. All trening er stengt nå.

## Siste genuine modellresultat og tapshypotese

177s ene matchede epoke er consumed:118249unike TRAIN-rader/7391oppdateringer/
32sluttforwards.178s lagrede sammenligning er canonical-validert/consumed.
LONG0/SHORT0/FLAT256 på både TRAIN og CONTROL; alltidFLAT er ingen selektivitet.
Begge siders predikerte beste horisont er1 på alle512sluttrader.

CONTROL nettkontrast-MSE (bps²), samme rader/masker:

| Minutt | Egen173initial | 167final | 177final |
|---|---:|---:|---:|
|1|57,459|572,260|61,678|
|5|228,358|686,241|227,369|
|15|772,101|1358,138|774,629|

Alle6primærpaneler/TRAIN+CONTROL forbedres mot167, men5av6 forverres mot
173initial. CONTROL5min har liten bedring med usikkerhet over/under0.
CONTROLh1 predstd2,461 motobservasjon7,504bps, corr0,0578. Overamplituden fra167
er dempet; bedre kvalifisert retning/Entry, generalisering og edge er ubevist.

181 viste at GELU-decoderen tillater tilstand×horisont-respons, med alle480 egne
sideceller/masker/RMS.182 gjenbrukte179s taskgradienter: samlet gradient er lokalt
samrettet med terminalen i alle7 undersøkte, overlappende grupper. Ingen tvungen
horisontlikhet, axisfeil eller begrunnelse for klonet terminaldecoder er påvist.
Dette er lokale vitner, ingen AdamW-/7391stegsårsaksanalyse.

183s lagrede TRAIN256-analyse er COMPLETE/consumed,0F/R/U,0,674s ytre tid.
MFE pooled enhetsfeil1,061→0,519 skjuler tidlig1–15-feil1,360→1,877, mens tidlig
rå-MSE bedres56,173→51,891bps². PreMAE bedres både tidlig1,221→0,684 og senere
1,035→0,459. Tidlig MFE har allerede92,65% av idealisert outputgradient-L2²;
50:50-gruppemiddel gir99,988% og12,73× output-L2 ved samme prediksjoner.
Dette beviser ikke svak tidlig parametergradient eller forventet læringsgevinst.

184 er bare MFE-only-prioritet: snitt av ikke-tomme1–15-/16–480-gruppemidler.
Hver gruppe pooler inklusive+closed LONG/SHORT SSE/ekteN med original111RMS;
ingen FLAT-path. Én gruppe får fullt middel, ingen gruppe utelater MFE/+s.
Terminal170, MAE/forecast,111unitkontrakt/-primitiv, rå/pooled diagnostikk,
inputs/targets/kostnader/score og modellarkitektur er bevart. Ny MFEoptimizer-
evidens navngis separat; kontraktskjema16 krever ny baseline før nativearbeid.
15/50:50 er en eksplisitt M5-designkonvensjon, aldri målt optimum eller Exitregel.

## Featuregrunnmur og gjenstående bevis

175 avstemte aktive kilder/input/numerikk/kausalitet. Aktiv profil er
`compact_mbp_smc_ema_atr_clock_macro_entry_v1`:7lokale/5MTF-familier etter19Oct-
unntaket; hele bibliotekets8spesialister og9.D1/makro beholdes. Alle695aktiveMTF-
felt139×5TF er numerisk kvalifisert; aktive GC-verdier/masker hadde0F32avvik.
SMC, EMA/trend, swing, støtte/motstand, volumprofil og MBP-1 beholdes kausalt.
Sjeldne/manglende observasjoner forblir ukjente; de fylles ikke med syntetikk.
MBP-1 er aggregert BBO/ordretall/OFI, ikke ordreidentitet/kø/skjult dybde.

180 målte trent177s lokale inputavhengighet:1F/5VJP/0updates,16genuine TRAIN,
alle84640paritetsceller delta0bps og identiske action-/horisontties.3317kanoniske
koordinater er regnskapsført. Alle32aktive av48deklarerte familie/TF-laner og
alle8flytende inputflater har lokal ikke-null respons; makro61 beholdes som
overlapp. Dette beviser konsum, ikke featureviktighet eller optimal utnyttelse.
Parkerte felt er ikke generelt numerisk-/nytte-kvalifisert eller forkastet.
Nesten parallelle1/5/15-inputgradienter er ikke alene en arkitekturfeil.
Kronologisk OOF-merverdi og generell feature-/modellnytte gjenstår.

## Varige grenser

Aktive beslutningsinputs: TRAIN[2024-05-01,2026-01-01), Q1CONTROL[2026-01-01,
2026-04-01), altså23måneder. April–juni er senere gjenbrukt utvikling som venter
separat fysisk binding; råkildemålet er26måneder.256-raders prober er ikke hele
TRAIN118249. MTF-cache har lengre historikk. TEST fra juli er forseglet for
utvalg/tilpasning; bare vedtatt månedlig forwardjournal scorer den uten refit.
Ingen Exit/broker/live/paper/ordre/spending/promotion.

Entry beslutter på lukketM5.1–480min er prognosedomene, aldri fasttap eller
maksimal holdetid. Rapportér teknikk, TRAIN-læring, kronologisk gjenbrukt utvikling,
generalisering og økonomi separat. Lange støttemål har alltidLONG/kjøp-og-hold
som forhåndsvalgt referanse. Ingen ekstraepoch/HPO eller relansering av consumed
engangsplaner. Native initialmåling krever ny fysiskboot; CPU-konstruksjon har
ikke eget bootpredikat. Ingen nativeplan er admitted nå.

Originalkvitteringer/checkpoints bevares; DATA/RUNS-rydding krever retention-eier.
Historikk ligger i Git/originalkvitteringer og åpnes bare ved konkret avvik/gate.
Metoden er [docs/NATIVE_LEARNING.md](docs/NATIVE_LEARNING.md).
