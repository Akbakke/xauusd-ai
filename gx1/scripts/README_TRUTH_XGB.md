# README_TRUTH_XGB.md
## 🚨 TRUTH XGB LANE — SINGLE SOURCE OF TRUTH (SSoT)

Dette dokumentet er **KANONISK KJØREPLAN** for XGB i GX1.
Hvis du er usikker på hva som er riktig å gjøre: **les denne filen**.
Hvis noe avviker fra dette: **det er feil**.

Dette eksisterer for å stoppe:
- arkeologi i gamle scripts
- feil univers (TRIAL160 / entry_v10_ctx)
- utilsiktet bruk av utdaterte prebuilt / modeller
- “men det funket jo i fjor”-katastrofer

---

## 🧠 MENTAL MODELL (LES FØRST)

GX1 har **ÉN** gyldig XGB-lane i TRUTH/SMOKE:

> **BASE28_CANONICAL → XGB Universal Multihead v2 → Signal Bridge**

Alt annet er **legacy**, **eksperiment**, eller **arkiv**.

---

## ✅ DETTE ER TRUTH (IKKE DISKUTERBART)

### 1. Environment
```bash
export GX1_DATA_ROOT=/home/andre2/GX1_DATA