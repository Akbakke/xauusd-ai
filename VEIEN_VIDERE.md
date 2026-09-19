# Veien videre — avklar hvorfor bedre verdier ikke endrer valg

## Nøyaktig stoppunkt for neste agent

Entry fuse256 er ferdig analysert. Gjenbruk review og verdict; ikke gjenta
initialmåling, trening, tester eller paret analyse. Kjørescope er stengt.
Begge Entry-sider slår sammenligningsmodellene på samlet verdiestimering,
men alle Entry/Exit-handlinger er identiske med hovedencoder256. Porten er ikke bestått.

1. Start med handover --check, CURRENT_HANDOVER.md, NEXT_RUN_POLICY.json og
   docs/ENTRY_FUSE_FIXED256_REVIEW_20260919.md. Ingen ny native jobb er bundet.
2. Les eksisterende lagrede optimizer-/modellbevis og native TRAIN16-målers
   dekning. First-batch-gradienten18,151 gjelder fire routing-gateparametre;
   den beviser ikke full nyttig oppdatering gjennom felles trunk/fuse og
   Entry/Exit-hoder. Tidligere private-head-diagnose dekket heller ikke fuse.
3. Avklar minste manglende måling som kan skille felles biaslæring fra
   tilstandsavhengig oppdatering. Gjenbruk eksisterende cache/targets og måler;
   ingen ny arkitektur, tapsvektsøk eller ekstrafit på antagelse. Statisk
   inspeksjon og lagrede bevis først. Ikke gjenta historisk diagnose uendret.
4. Dersom ny native diagnose er nødvendig, må den bindes separat med dagens
   ONLINE-funksjon og riktig initial/final/cache-paritet før utførelse. Ingen
   optimizersteg eller modellendring er nå autorisert av kjørepolicyen.
5. Rett bare målt blokkering. Bevar alle200 features, familier/tidsrammer,
   kausalitet, kostnader, opprinnelig lærer og checkpoints. Vurder begge hoder.

Targetvariasjon inneholder støy; reskalering/terskelendring for å tvinge fram
handler gir ikke dokumentert læring. Først bedre beslutninger, så separat
bundet kronologisk kvalitet og full økonomi med åpne posisjoner. TEST forseglet.
Ingen full epoch/full VAL/CONTROL, live/paper eller spending. Én agent/én tung jobb.

Artefaktrot: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_ENTRY_FUSE_NORMALIZED_FIXED256_20260919.
PAIRED_TRAIN_REVIEW.json, DECISION_GAP_AUDIT.json, VERDICT.json og
COMPLETION_REVIEWED.json er verifisert; REVIEW_OPERATOR.py og
DECISION_GAP_OPERATOR.py bevarer beregningen. Alle originaler skal beholdes.
