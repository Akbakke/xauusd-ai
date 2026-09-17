# Fast256: læringsporten er ikke bestått

Beslutning: REJECT_EXPANSION_LEARNING_GATE_FAILED. Ikke gjenta, utvid eller
tilpass mot mars–mai. Ingen økonomi eller positiv handelsfordel er målt.

## Entry først

| Senere kontroll, MSE i Bps² | Før | Fast ONLINE256 | TRAIN-konstant |
|---|---:|---:|---:|
| Entry LONG | 530.4489 | 513.0887 | 507.3162 |
| Entry SHORT | 432.7069 | 421.2408 | 429.3527 |
| Exit anker LONG | 1159.8781 | 1163.6113 | 1161.3577 |
| Exit anker SHORT | 1158.3271 | 1161.8770 | 1159.6841 |
| Exit samplet LONG | 1143.4677 | 1145.2581 | 1143.5184 |
| Exit samplet SHORT | 1140.4759 | 1142.0053 | 1140.4797 |

LONG Entry taper mot konstanten; paret95% ukeintervall for MSE-forskjell er
[0,941;11,510]. SHORT slår konstanten[-14,377;-2,540], men omtrent90% av gevinsten
mot start skyldes endret middelfeil. SHORT sentrertMSE412,588→411,433,
korrelasjon0,022. LONG sentrert feil øker litt.

Entry går fra27LONG/162SHORT/67FLAT til66/190/0 på kontrollen. TRAIN blir255/1/0
mot målets112/77/67. Kontrollens referanseregret12,851Bps taper mot
konstantvalgets12,422. Dette er frosset Q_mu, ikke faktisk handelsprofitt.

| Entry måned, MSE i Bps² | Før | ONLINE256 | TRAIN-konstant |
|---|---:|---:|---:|
| 2026-03 LONG | 875.1574 | 843.9171 | 832.2418 |
| 2026-03 SHORT | 628.9942 | 607.3849 | 622.5904 |
| 2026-04 LONG | 506.7943 | 491.6900 | 486.4693 |
| 2026-04 SHORT | 385.7495 | 379.3082 | 383.5807 |
| 2026-05 LONG | 221.6948 | 215.4485 | 214.8144 |
| 2026-05 SHORT | 290.7702 | 284.0102 | 289.1602 |

LONG taper mot konstanten alle tre måneder. SHORT slår den, men har
månedskorrelasjoner0,024/−0,005/−0,070. Robust retning er ikke dokumentert.

## Exit og TRAIN

Exit blir samlet verre mot begge baseliner på senere kontroll, både ved
ankeret og samplede states. Ukeintervallene krysser null; forbedring er ikke
dokumentert. Anker-MSE øker begge sider hver måned. Sampled SHORT varierer,
så ikke hver enkelt månedscelle blir verre. LONG blir1024/1024HOLD og SHORT
964/1024EXIT. Sideavhengige konstantvalg er ikke dokumentert selektivitet.

Alle seks TRAIN-side/surface MSE og sentrerte feil bedres mot start og
TRAIN-konstant. Små reduksjoner og nesten allLONG Entry består likevel ikke
hele beslutningskravet. TRAIN-proben er blant de4096 trente Entries og gir
ikke uavhengig generaliseringsbevis. Alle måneder/uker finnes i JSON.

## Verifikasjon

Én native kjøring,TRAIN16,256 oppdateringer,ingen full epoch/fullVAL.
Start2026-09-17T08:25:56.023113Z, terminalkvittering09:40:40.893985Z
(10:25:56–11:40:40 Europe/Oslo). Fysisk boot453. GuardPASS/trainer0/observer0.
Observerte topper59C kjerne,68C minne,163,42W og8014MiB; ingen grenseendring.

Fast siste ONLINE, ikke valgt best/EMA. Eksakt samme lærer, mål, komponenter,
masks og kohorter før/etter.256 Entry-/ankerobservasjoner og1024 samples per
rolle. Kontroll13 og TRAIN17 samplede states per side er sensurert; ingen
kunstig terminal. Absolutt bootstrap er rundt0,03–0,05Bps; observerte komponenter
dominerer. Svak overføring består også med den observerte referansemålkjeden.

TRAIN-konstanter; én paret kalenderuke-bootstrap,5000 trekk,seed20260911.
14 kontrolluker, alle fire samples i sin Entry-uke. Avhengighet mellom uker
og overlapp kan bestå. Gjenbrukt utviklingskontroll, ikke urørt TEST.
CPU-analyse under4GiB/512MiB-vakt av lagrede outputs; null forwards/fit.
MSE=sentrertMSE+bias² og parede differanser er kontrollert separat.

## Neste konkrete hypotese

V4 entry_q_joint_source.detach() stopper Entry-feilens gradient til den
lokale/MTF/globale representasjonen. Faktisk førstebatchmåling viser unused
Entry-gradient på begge routing-gater. Entry-hodet og routingvektene endres;
andre hjelpeoppgaver trener representasjonen. Dette er bevisst lærer-
isolasjon, ikke manglende inputs, og beviser ikke årsaken til sviktende overføring.

Ingen modellkode endres på grunnlag av kontrollresultatet. Neste avgrensede
arbeid er en TRAIN-only gradientkontrast på identisk lagret modell/batch,
uten optimizersteg. Faktiske modellkall må bindes til eksisterende native
vakter. Ingen brede søk eller samme-CONTROL-tuning.

## Bevis

Analyse: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_PREFIX_FIXED256_LEARNING_20260917/PAIRED_LEARNING_REVIEW.json
SHA256: f52f81ac451d1dc0413df447a3587612537ae0300b02029e28fef5e305e25c78
Kjørekilde: cf6e2b239bbf94496add7814e8062ebd554a0487
Sluttmåling: d8c845ba8eaebed24bedfd677f9f41250043492348c0cbffa1f644e0ce6024a8
Native kvittering: 92d1fe741f89b0fa84e8731abe873eed7ac2f63a238802c8d7d94ce7cf2eb6f5
Treningstilstand: dc73d0fd7887ebcb1d9e8a8f62d5d4c3b0038a5ee439b5ad79950c45d119d7b6
Analyseoperator: ab7148c7a368a6f5d48fe128c453c6f2912346ba88d7ff8357b48c74fb3212ac

Aggregerte bevis: handover_snapshot/NATIVE_PREFIX_FIXED256_REVIEW_20260917.json.
Råobservasjoner, utvalg og vekter forblir i GX1_DATA. Tasken er deaktivert;
chronological_learning_run er fjernet etter arkivering av opprinnelig policy.
training_enabled=false. Målet er fortsatt aktivt.
