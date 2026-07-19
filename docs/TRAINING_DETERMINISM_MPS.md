# Training determinism contract

GX1 training must record Python/library/device versions, source identity,
recipe, seed, data hashes, split manifests and all determinism settings in the
immutable run event.

Determinism is device-specific. A CPU, CUDA or MPS run is not interchangeable
merely because it uses the same seed. Unsupported deterministic kernels must
fail or be explicitly declared in a research-only event; they may not silently
switch algorithms or devices in decision-valid training.

Reproducibility is verified by rerunning a bounded deterministic fixture and
comparing outputs/state hashes within the declared exact tolerance. A seeded
run alone is not proof.
