#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path

# Set working directory
workspace = Path(__file__).resolve().parent
os.chdir(str(workspace))
sys.path.insert(0, str(workspace))

# Actually execute the search
gx1_data = Path("/Users/andrekildalbakke/Desktop/GX1_DATA")

print("Searching for prebuilt parquet files...")
print(f"GX1_DATA: {gx1_data}")
print()

# Search all parquet files
all_parquets = list(gx1_data.rglob("*.parquet"))
print(f"Found {len(all_parquets)} total parquet files")
print()

# Filter for prebuilt/features
prebuilt_candidates = [
    p for p in all_parquets
    if "prebuilt" in str(p) or "features" in str(p)
]

print(f"Prebuilt/features candidates: {len(prebuilt_candidates)}")
print()

# Group by year
years = [2020, 2021, 2022, 2023, 2024, 2025]
year_files = {}

for year in years:
    year_str = str(year)
    matches = [
        p for p in prebuilt_candidates
        if year_str in p.name or year_str in str(p.parent)
    ]
    year_files[year] = matches

# Print results
for year in years:
    print(f"{year}:")
    if year_files[year]:
        for f in year_files[year]:
            rel = f.relative_to(gx1_data)
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  ✅ {rel}")
            print(f"     Size: {size_mb:.1f} MB")
    else:
        print(f"  ❌ NOT FOUND")
    print()

# Print mapping JSON
print("=" * 60)
print("SUGGESTED MAPPING (JSON):")
print("=" * 60)
mapping = {}
for year in years:
    if year_files[year]:
        # Prefer files with "v10_ctx" or "features" in name
        preferred = None
        for f in year_files[year]:
            if "v10" in f.name.lower() or "features" in f.name.lower():
                preferred = f
                break
        if preferred:
            mapping[year] = str(preferred)
        else:
            mapping[year] = str(year_files[year][0])

import json
print(json.dumps(mapping, indent=2))
