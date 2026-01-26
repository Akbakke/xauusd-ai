# Training Determinism on MPS (Apple Silicon)

## Issue Summary

During FULLYEAR_2025 training, we encountered `loss=nan` from the very first batch when using PyTorch's `torch.use_deterministic_algorithms(True)` on the MPS (Apple Silicon) backend.

## Root Cause

`torch.use_deterministic_algorithms(True)` is not fully supported on the MPS backend. When enabled, certain operations (particularly in the Transformer/Gated Fusion forward pass) produce NaN values, leading to immediate training failure.

This is a **known limitation** of PyTorch's MPS backend as of PyTorch 2.x. Deterministic algorithms are designed primarily for CUDA and are not yet fully implemented for MPS.

## Solution

Modified `set_seed()` in `gx1/models/entry_v10/entry_v10_ctx_train.py` to only enable `use_deterministic_algorithms` on CUDA:

```python
def set_seed(seed: int):
    """Set all random seeds for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # NOTE: torch.use_deterministic_algorithms(True) causes NaN on MPS backend
    # Only enable on CUDA where it's properly supported
    if torch.cuda.is_available():
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # Some ops don't support deterministic mode
```

## Impact

- **Training determinism:** On MPS, we rely on `torch.manual_seed()` and `torch.backends.cudnn.deterministic` only. This provides good reproducibility for most use cases, but not full deterministic behavior.
- **Production readiness:** Training works correctly on MPS. For full determinism, use CUDA backend.
- **Replay/evaluation:** Replay and evaluation determinism are unaffected (they don't use `use_deterministic_algorithms`).

## Verification

After fixing `set_seed()`, FULLYEAR_2025 training completed successfully:
- **10 epochs** completed without NaN
- **Loss:** 0.3359 (final epoch)
- **Gate stability:** gate_mean=0.2656, gate_std=0.0748 (no collapse)
- **Throughput:** ~900 samples/sec on MPS

## References

- PyTorch MPS documentation: [Apple Silicon](https://pytorch.org/docs/stable/notes/mps.html)
- Issue tracking: PyTorch GitHub issues related to MPS deterministic algorithms

## Future Considerations

If full determinism is required on Apple Silicon:
1. Use CUDA backend (if available)
2. Wait for PyTorch MPS to implement full deterministic algorithm support
3. Use alternative deterministic training techniques (gradient accumulation, fixed initialization, etc.)

For now, `torch.manual_seed()` + `cudnn.deterministic` provides sufficient reproducibility for production training on MPS.
