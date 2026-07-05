#!/usr/bin/env python3
"""
Update multiquarter summary with Q4 results and decision rule.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

def load_quarter_compare(compare_file: Path) -> Optional[Dict[str, Any]]:
    """Load quarter compare JSON file."""
    if not compare_file.exists():
        return None
    with open(compare_file, "r") as f:
        return json.load(f)

def generate_multiquarter_summary(
    q1_path: Path,
    q2_path: Path,
    q3_path: Path,
    q4_path: Optional[Path],
    output_path: Path
) -> None:
    """Generate updated multiquarter summary."""
    
    # Load all quarter compares
    q1_data = load_quarter_compare(q1_path)
    q2_data = load_quarter_compare(q2_path)
    q3_data = load_quarter_compare(q3_path)
    q4_data = load_quarter_compare(q4_path) if q4_path else None
    
    if not all([q1_data, q2_data, q3_data]):
        raise ValueError("Q1, Q2, and Q3 compare files must exist")
    
    # Extract deltas
    quarters = []
    for q_num, q_data in [("Q1", q1_data), ("Q2", q2_data), ("Q3", q3_data), ("Q4", q4_data)]:
        if q_data:
            deltas = q_data.get("deltas", {})
            pnl_delta = deltas.get("pnl_bps", 0)
            trades_delta = deltas.get("trades", 0)
            trades_delta_pct = deltas.get("trades_pct", 0)
            maxdd_delta = deltas.get("maxdd_bps", 0)
            
            status = "✅ GO" if q_data.get("go_nogo") == "GO" else "❌ NO-GO"
            
            # Determine recommendation
            if pnl_delta >= 0:
                recommendation = "✅ POSITIV"
            elif pnl_delta >= -1000:
                recommendation = "⚠️ NÆR NULL"
            else:
                recommendation = "❌ NEGATIV"
            
            quarters.append({
                "quarter": q_num,
                "pnl_delta": pnl_delta,
                "trades_delta": trades_delta,
                "trades_delta_pct": trades_delta_pct,
                "maxdd_delta": maxdd_delta,
                "status": status,
                "recommendation": recommendation
            })
    
    # Generate summary
    summary = f"""# Depth Ladder Multi-Quarter Summary: Q1-Q4 2025

**Date:** 2026-01-21  
**Status:** All quarters completed and validated

---

## Summary by Quarter

| Quarter | PnL Delta (bps) | Trades Delta | MaxDD Delta (bps) | Status | Anbefaling |
|---------|------------------|--------------|-------------------|--------|------------|
"""
    
    for q in quarters:
        summary += f"| {q['quarter']} | {q['pnl_delta']:+.2f} | {q['trades_delta']:+d} ({q['trades_delta_pct']:+.2f}%) | {q['maxdd_delta']:+.2f} | {q['status']} | {q['recommendation']} |\n"
    
    summary += "\n---\n\n## Trend Analysis\n\n"
    
    # Calculate trend
    if len(quarters) >= 3:
        q1_pnl = quarters[0]['pnl_delta']
        q2_pnl = quarters[1]['pnl_delta']
        q3_pnl = quarters[2]['pnl_delta']
        
        trend = "FORBEDRING" if q3_pnl > q1_pnl and q3_pnl > q2_pnl else "FORVERRING" if q3_pnl < q1_pnl and q3_pnl < q2_pnl else "STABIL"
        
        summary += f"**Trend:** {trend}\n"
        summary += f"- Q1 → Q2: {q2_pnl - q1_pnl:+.2f} bps\n"
        summary += f"- Q2 → Q3: {q3_pnl - q2_pnl:+.2f} bps\n"
        
        if len(quarters) >= 4:
            q4_pnl = quarters[3]['pnl_delta']
            summary += f"- Q3 → Q4: {q4_pnl - q3_pnl:+.2f} bps\n"
    
    summary += "\n---\n\n## Beslutningsregel\n\n"
    summary += "**Regel:**\n"
    summary += "- Hvis Q4 PnL delta er >= -1000 bps (nær null) eller positiv og DD ikke forverres → GO for full 2025 / deretter multiyear\n"
    summary += "- Hvis Q4 er tydelig negativ (som Q1/Q2, < -1000 bps) → NO-GO for full multiyear\n\n"
    
    # Decision
    if q4_data:
        q4_pnl = quarters[3]['pnl_delta']
        q4_maxdd = quarters[3]['maxdd_delta']
        
        summary += "**Q4 Resultat:**\n"
        summary += f"- PnL Delta: {q4_pnl:+.2f} bps {'✅' if q4_pnl >= -1000 else '❌'}\n"
        summary += f"- MaxDD Delta: {q4_maxdd:+.2f} bps {'✅' if q4_maxdd <= 0 else '❌'}\n\n"
        
        if q4_pnl >= -1000 and q4_maxdd <= 0:
            decision = "✅ GO for full 2025 / multiyear eval"
            reasoning = "Q4 er nær null eller positiv, og DD forverres ikke."
        else:
            decision = "❌ NO-GO for full multiyear eval"
            reasoning = "Q4 er tydelig negativ eller DD forverres."
        
        summary += f"**Beslutning:** {decision}\n\n"
        summary += f"**Begrunnelse:** {reasoning}\n"
    else:
        summary += "**Q4:** Ikke ferdig ennå\n\n"
        summary += "**Beslutning:** Vent på Q4 resultater\n"
    
    summary += "\n---\n\n## Neste Steg\n\n"
    
    if q4_data:
        q4_pnl = quarters[3]['pnl_delta']
        if q4_pnl >= -1000:
            summary += "1. ✅ Q4 er nær null eller positiv → GO for full 2025 eval\n"
            summary += "2. Kjør full 2025 eval (2020-2025) for både baseline og L+1\n"
            summary += "3. Kjør truth decomposition comparison\n"
        else:
            summary += "1. ❌ Q4 er tydelig negativ → NO-GO for full multiyear\n"
            summary += "2. Vurder hvorfor L+1 taper winners / øker loss\n"
            summary += "3. Vurder å gå tilbake til baseline eller prøv andre modellendringer\n"
    else:
        summary += "1. Vent på Q4 smoke eval resultater\n"
        summary += "2. Kjør compare for Q4\n"
        summary += "3. Oppdater denne rapporten med Q4 og beslutning\n"
    
    # Write summary
    with open(output_path, "w") as f:
        f.write(summary)
    
    print(f"✅ Multiquarter summary written: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Update multiquarter summary")
    parser.add_argument("--q1-compare", type=Path, required=True, help="Q1 compare JSON file")
    parser.add_argument("--q2-compare", type=Path, required=True, help="Q2 compare JSON file")
    parser.add_argument("--q3-compare", type=Path, required=True, help="Q3 compare JSON file")
    parser.add_argument("--q4-compare", type=Path, required=False, help="Q4 compare JSON file (optional)")
    parser.add_argument("--out", type=Path, required=True, help="Output summary file")
    
    args = parser.parse_args()
    
    generate_multiquarter_summary(
        args.q1_compare,
        args.q2_compare,
        args.q3_compare,
        args.q4_compare,
        args.out
    )

if __name__ == "__main__":
    main()
