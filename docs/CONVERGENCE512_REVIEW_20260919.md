#512: bedre Exit-valg på TRAIN, Entry-port ikke bestått

Native sluttførte19. september2026 kl17:08:23UTC /19:08:23Europe/Oslo.
Nøyaktig256 nye oppdateringer, sluttcheckpoint512, guard PASS. Task er deaktivert,
ingen native prosess. Original256, Adam-historikk, lærer og datarekkefølge er
bevart; targets/masker/cohort er kontrollert identiske. Ingen ny modell eller loss.

Exit tar nå tilstandsavhengige valg. På1024 samplede tilstander per side:

|Side|HOLD/EXIT512|Referanseverdi256|Referanseverdi512|Beste konstant|
|---|---:|---:|---:|---:|
|LONG|758/266|−0,1634|+1,4451|0,0000|
|SHORT|306/718|0,0000|+1,7321|+0,1071|

Ved256ankre er verdiene+2,9595/+1,1470 Bps mot+2,2174/0 på256. Begge
Exit-sider og begge måleflater har bedre MSE, sentrert feil og valgreferanseverdi
enn256 og relevante konstanter. Samplede valg slår256 i6/9 måneder og den globale
TRAIN-konstanten i7/9 på begge sider. Dette er delvis TRAIN-læring, ikke jevn
forbedring hver måned, kronologisk generalisering eller realisert strategiprofitt.

Entry velger fortsatt FLAT256/256. Beste LONG-minus-FLAT er−3,2631 Bps,
beste SHORT-minus-FLAT−5,6067. Begge sider har lavere verdiestimeringsfeil,
men kontrastkorrelasjonen innen måned faller0,2377→0,1899. Den faste diagnosen
som velger LONG/SHORT uten FLAT går fra234/22 og−1,4155 Bps til254/2 og−3,8524.
Dette er ikke en handelsregel, og støtter ikke tvungne handler eller oppskalering.
Lavere MSE er derfor ikke tilstrekkelig tegn på bedre Entry-retning.

Alle256 kontrollpunktene er TRAIN-rader som allerede inngikk i de første4096
treningsradene. De er en bevisst læringsmåling; ingen holdout-påstand. Senere
TRAIN-rader øker ikke direkte eksponering for disse256 inngangspunktene.

Forhåndsbundne krav gir REJECT_EXPANSION_CONVERGENCE512_ACTION_GATE_FAILED.
Bare Entry-handlingskravet feiler numerisk; samlet Entry/Exit-port er ikke bestått.
Brukt scope er stengt. Ingen automatisk512→1024, full epoch/VAL eller ny
normaliserings-/gradientrunde. Referanseverdiene er Q_mu under fast referansepolicy.
De evaluerer ikke automatisk den forbedrede Exit-policyens samlede økonomi.

Neste avklaring står i VEIEN_VIDERE.md. Gjenbruk lagrede målinger og eksisterende
Entry/Exit-kobling før nye forsøk. Ingen targetfeil er bevist av forskjellen
mellom Q_mu og egen policy. TEST forblir forseglet; ingen live/paper/spending.
