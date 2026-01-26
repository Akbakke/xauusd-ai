#!/usr/bin/env python3
"""
Verify EntryV10Ctx Bundle

Post-train verification script to ensure bundle is loadable and valid.
"""

import argparse
import json
from pathlib import Path

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle


def main():
    parser = argparse.ArgumentParser(description="Verify EntryV10Ctx bundle")
    parser.add_argument("--bundle_dir", type=str, required=True, help="Path to bundle directory")
    parser.add_argument("--expected_hash", type=str, default=None, help="Expected feature_contract_hash (optional)")
    
    args = parser.parse_args()
    
    bundle_dir = Path(args.bundle_dir)
    
    print("="*80)
    print("[VERIFY] EntryV10Ctx Bundle Verification")
    print("="*80)
    print(f"\nBundle directory: {bundle_dir}")
    
    # Check required files
    required_files = [
        "model_state_dict.pt",
        "bundle_metadata.json",
        "feature_contract_hash.txt",
    ]
    
    missing_files = []
    for fname in required_files:
        if not (bundle_dir / fname).exists():
            missing_files.append(fname)
    
    if missing_files:
        print(f"\n❌ MISSING FILES: {missing_files}")
        return 1
    
    print("\n✅ All required files present")
    
    # Load bundle
    try:
        # Get feature_meta_path (required for ctx bundle)
        # Try to find it in bundle_dir or use default
        feature_meta_path = bundle_dir.parent.parent / "entry_v9" / "nextgen_2020_2025_clean" / "entry_v9_feature_meta.json"
        if not feature_meta_path.exists():
            # Try alternative location
            feature_meta_path = Path("gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json")
        
        if not feature_meta_path.exists():
            print(f"\n❌ FAILED: Feature metadata not found. Tried: {feature_meta_path}")
            return 1
        
        bundle = load_entry_v10_ctx_bundle(
            bundle_dir=bundle_dir,
            feature_meta_path=feature_meta_path,
            is_replay=True,
        )
        print("\n✅ Bundle loaded successfully")
    except Exception as e:
        print(f"\n❌ FAILED to load bundle: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Verify metadata
    metadata = bundle.metadata or {}
    print("\n[BUNDLE_METADATA]")
    print(f"  supports_context_features: {metadata.get('supports_context_features', False)}")
    print(f"  expected_ctx_cat_dim: {metadata.get('expected_ctx_cat_dim', 0)}")
    print(f"  expected_ctx_cont_dim: {metadata.get('expected_ctx_cont_dim', 0)}")
    print(f"  feature_contract_hash: {metadata.get('feature_contract_hash', 'N/A')}")
    print(f"  model_variant: {metadata.get('model_variant', 'N/A')}")
    
    # Get transformer config for additional info
    transformer_config = bundle.transformer_config or {}
    print(f"  seq_input_dim: {transformer_config.get('seq_input_dim', 'N/A')}")
    print(f"  snap_input_dim: {transformer_config.get('snap_input_dim', 'N/A')}")
    print(f"  max_seq_len: {transformer_config.get('max_seq_len', 'N/A')}")
    
    # Verify context features support
    supports_context_features = metadata.get("supports_context_features", False)
    if not supports_context_features:
        print("\n❌ FAILED: supports_context_features must be True")
        return 1
    
    expected_ctx_cat_dim = metadata.get("expected_ctx_cat_dim", 0)
    if expected_ctx_cat_dim != 5:
        print(f"\n❌ FAILED: expected_ctx_cat_dim must be 5, got {expected_ctx_cat_dim}")
        return 1
    
    expected_ctx_cont_dim = metadata.get("expected_ctx_cont_dim", 0)
    if expected_ctx_cont_dim != 2:
        print(f"\n❌ FAILED: expected_ctx_cont_dim must be 2, got {expected_ctx_cont_dim}")
        return 1
    
    model_variant = metadata.get("model_variant", "")
    if model_variant != "v10_ctx":
        print(f"\n❌ FAILED: model_variant must be 'v10_ctx', got {model_variant}")
        return 1
    
    print("\n✅ Context features metadata valid")
    
    # Verify feature contract hash
    feature_contract_hash = metadata.get("feature_contract_hash")
    if args.expected_hash:
        if feature_contract_hash != args.expected_hash:
            print(f"\n❌ FAILED: feature_contract_hash mismatch")
            print(f"  Expected: {args.expected_hash}")
            print(f"  Got: {feature_contract_hash}")
            return 1
        print(f"\n✅ Feature contract hash matches: {feature_contract_hash}")
    else:
        print(f"\n✅ Feature contract hash: {feature_contract_hash}")
    
    # Load metadata JSON for additional info
    metadata_path = bundle_dir / "bundle_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        print("\n[ADDITIONAL_METADATA]")
        for key, value in metadata.items():
            if key not in ["supports_context_features", "expected_ctx_cat_dim", "expected_ctx_cont_dim", 
                          "feature_contract_hash", "model_variant"]:
                print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("✅ BUNDLE VERIFICATION PASSED")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())

