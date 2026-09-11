"""Exact static Exit state retired by the reviewed source-state successor."""


def _retired_static_exit_state_keys() -> frozenset[str]:
    encoder_suffixes = (
        "self_attn.in_proj_weight",
        "self_attn.in_proj_bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "linear1.weight",
        "linear1.bias",
        "linear2.weight",
        "linear2.bias",
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
    )
    return frozenset(
        [
            f"exit_path_encoder.layers.{layer}.{suffix}"
            for layer in range(2)
            for suffix in encoder_suffixes
        ]
        + [
            "exit_entry_query_norm.weight",
            "exit_entry_query_norm.bias",
            "exit_entry_path_attention.in_proj_weight",
            "exit_entry_path_attention.in_proj_bias",
            "exit_entry_path_attention.out_proj.weight",
            "exit_entry_path_attention.out_proj.bias",
            "exit_fuse.0.weight",
            "exit_fuse.0.bias",
            "exit_fuse.1.weight",
            "exit_fuse.1.bias",
            "exit_fuse.4.weight",
            "exit_fuse.4.bias",
        ]
    )


RETIRED_STATIC_EXIT_STATE_KEYS = _retired_static_exit_state_keys()
