from gx1.contracts.signal_bridge_v3 import ORDERED_CTX_CONT_NAMES_V3
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _build_ordered_ctx_cont_names


def test_v10_metadata_ctx_cont_names_use_full_v3_contract() -> None:
    legacy_base_names = list(ORDERED_CTX_CONT_NAMES_V3[:21])

    got = _build_ordered_ctx_cont_names(len(ORDERED_CTX_CONT_NAMES_V3), legacy_base_names)

    assert got == list(ORDERED_CTX_CONT_NAMES_V3)
    assert len(got) == 123
