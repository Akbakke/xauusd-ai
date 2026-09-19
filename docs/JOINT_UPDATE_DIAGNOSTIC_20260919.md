# Samlet gradient og Adam-retning — én avgrenset diagnose

Entry fuse256 forbedret verdiestimater, men alle Entry/Exit-handlinger er uendret.
Saved-state-audit bekrefter726 optimizerparametertensorer på steg256, Entry/Exit-
tapsvekter0,9753/0,9759 og endring i alle fuse-/Entry-hodeelementer. Ingen frosset
Entry eller avslått tapsvekt er dokumentert. Parameterbevegelse beviser ikke
nyttig retningslæring. Historisk routingmåler omfattet bare fire gateparametre.

Eksisterende native diagnose er utvidet med final_joint_update. Ingen modell,
treningsmatematikk eller handelsregel er endret. Samme verifiserte TRAIN16-cache,
originale targets/masker og frosne lærer. Fem forwards: ONLINE Entry inferens og
gradientmodus, original lærer Entry, native ONLINE Exit og lærer Exit. Diagnose
bruker eksisterende Exit-backward og nøyaktige Entry-/hjelpetap. Ingen optimizersteg.

Alle akkumulerte parametergradienter kontrolleres mot separat Entry/Exit/hjelpe-
beregning. Rapporten skiller Entry fellesverdi, LONG−SHORT-kontrast og FLAT,
fuse, Entry-hode, Exit-parametre og alle prediksjonsparametre. Lagret Adam-
historikk, opprinnelig klipping og weight decay brukes til en FP64 beregning av
mulig neste oppdatering. Beregnet tapseffekt er første orden, ikke utført trening.

En fast kontrast utelater bare nåværende hjelpegradient. Historiske momenter
beholdes, så dette isolerer ikke hele tidligere hjelpepåvirkning. Målingen er
på én gjenbrukt eval-batch uten dropout; den gjenskaper ikke historiske steg.
Ingen målvarians antas predikerbar, og intet resultat er profitt/generaliseringsbevis.

Seks fokuserte tester bestod, inkludert sammenligning mot virkelig native AdamW
med historikk og klipping, delt Exit-gradient, uendret modell og opprydding ved
feil. To rapporteringstester bestod etter rettelse av en konkret gammel peker:
handover viste hovedencoder som siste fullførte kjøring etter fuse256.
Inputaudit bekreftet eksakt batch/rad/target/mask-paritet og ny ONLINE-funksjon.

Plan: NATIVE_JOINT_UPDATE_DIAGNOSTIC_20260919/PLAN.json under vanlig BASE.
Én invokasjon, null optimizer, ingen CONTROL/VAL/TEST eller automatisk utvidelse.
Kilde fryses under kjøring. Gjenbruk tester og audits; vurder resultatet før kodeendring.
