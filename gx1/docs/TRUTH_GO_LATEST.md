# TRUTH GO – Latest proof pointer (SSoT)

**Dato:** 2026-02-20

**Siste GO run_dir** (fra `GX1_DATA/reports/truth_e2e_sanity/LAST_GO.txt`):

```
E2E_SANITY_20260220_205954
```

Full path: `$GX1_DATA/reports/truth_e2e_sanity/E2E_SANITY_20260220_205954`

---

## PASS-kriterier

- **Entry bundle ctx:** `ctx_cont_dim` = 6, `ctx_cat_dim` = 6 (footer)
- **Exit ML:** `exit_ml_io_version` = IOV3_CLEAN, `exit_ml_input_dim` = 35
- **IMPORT_PROOF:** `forbidden_hits` = []
- **Policy snapshot:** RUN_POLICY_USED finnes i run_dir
- **Exits:** exits jsonl finnes (eller placeholder når 0 trades, med footer 6/6)

---

## Viktige filer i run_dir

| Artifact        | Path (relativt run_dir)           |
|-----------------|-----------------------------------|
| Chunk footer    | `replay/chunk_0/chunk_footer.json` |
| Import proof    | `replay/chunk_0/IMPORT_PROOF.json` |
| Policy snapshot | `replay/chunk_0/RUN_POLICY_USED.yaml` |

Sitert fra footer: `ctx_cont_dim` 6, `ctx_cat_dim` 6, `exit_ml_io_version` IOV3_CLEAN, `exit_ml_input_dim` 35, `status` ok.  
Sitert fra IMPORT_PROOF: `forbidden_hits` = [].
