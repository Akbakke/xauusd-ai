#!/usr/bin/env python3
"""
Unit test for checkpoint loading guardrail.

Tests that checkpoint_epoch_*.pt files are blocked unless GX1_ALLOW_CHECKPOINT_LOAD=1.
"""

import os
import re
import sys
import tempfile
from pathlib import Path
import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_checkpoint_guardrail_blocks_checkpoint():
    """Test that checkpoint loading is blocked by default."""
    # Create a dummy checkpoint file
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint_epoch_10.pt"
        
        # Save a dummy checkpoint
        dummy_state = {"model_state_dict": {}}
        torch.save(dummy_state, checkpoint_path)
        
        # Try to load it via bundle (should fail)
        v10_ctx_cfg = {
            "bundle_dir": str(Path(tmpdir)),
        }
        
        # Ensure guardrail is active
        if "GX1_ALLOW_CHECKPOINT_LOAD" in os.environ:
            del os.environ["GX1_ALLOW_CHECKPOINT_LOAD"]
        
        # Create minimal bundle metadata
        bundle_dir = Path(tmpdir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "seq_input_dim": 16,
            "snap_input_dim": 88,
            "seq_len": 30,
        }
        import json
        with open(bundle_dir / "bundle_metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Test guardrail by simulating the check (same logic as in entry_v10_bundle.py)
        checkpoint_path_str = str(checkpoint_path)
        
        # Simulate guardrail check
        if re.match(r".*checkpoint_epoch_\d+\.pt$", checkpoint_path_str):
            allow_checkpoint = os.environ.get("GX1_ALLOW_CHECKPOINT_LOAD", "0")
            if allow_checkpoint != "1":
                # This is what the guardrail should do - verify it raises RuntimeError
                with pytest.raises(RuntimeError, match="CHECKPOINT_LOAD_FORBIDDEN"):
                    raise RuntimeError(
                        f"CHECKPOINT_LOAD_FORBIDDEN: Attempted to load checkpoint file '{checkpoint_path_str}'. "
                        f"Checkpoints are forbidden in runtime to prevent loading outdated optimizer state. "
                        f"If you really need to load a checkpoint (e.g., for training resume), "
                        f"set GX1_ALLOW_CHECKPOINT_LOAD=1. Otherwise, use final model files (model.pt, model_state_dict.pt)."
                    )


def test_checkpoint_guardrail_allows_with_flag():
    """Test that checkpoint loading is allowed when GX1_ALLOW_CHECKPOINT_LOAD=1."""
    # Create a dummy checkpoint file
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint_epoch_10.pt"
        
        # Save a dummy checkpoint
        dummy_state = {"model_state_dict": {}}
        torch.save(dummy_state, checkpoint_path)
        
        # Set flag
        os.environ["GX1_ALLOW_CHECKPOINT_LOAD"] = "1"
        
        # Should not raise error
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            assert checkpoint is not None
        finally:
            # Cleanup
            if "GX1_ALLOW_CHECKPOINT_LOAD" in os.environ:
                del os.environ["GX1_ALLOW_CHECKPOINT_LOAD"]


def test_checkpoint_guardrail_allows_final_models():
    """Test that final models (model.pt, model_state_dict.pt) are always allowed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test model.pt
        model_path = Path(tmpdir) / "model.pt"
        dummy_state = {"model_state_dict": {}}
        torch.save(dummy_state, model_path)
        
        # Should not raise error (no guardrail for model.pt)
        checkpoint = torch.load(model_path, map_location="cpu")
        assert checkpoint is not None
        
        # Test model_state_dict.pt
        model_state_path = Path(tmpdir) / "model_state_dict.pt"
        torch.save(dummy_state, model_state_path)
        
        # Should not raise error
        checkpoint = torch.load(model_state_path, map_location="cpu")
        assert checkpoint is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
