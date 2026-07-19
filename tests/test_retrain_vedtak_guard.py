import pytest

from gx1_guards.gates import GateError, require_retrain_vedtak


def test_retrain_vedtak_accepts_exact_auditable_id() -> None:
    assert (
        require_retrain_vedtak(" MODEL_NATIVE_SEQ513_REBUILD_20260717 ")
        == "MODEL_NATIVE_SEQ513_REBUILD_20260717"
    )


@pytest.mark.parametrize(
    "value",
    [None, "", "TODO", "short", "<EXPLICIT_VEDTAK_ID>", "decision with spaces"],
)
def test_retrain_vedtak_rejects_missing_placeholder_or_ambiguous_id(
    value: str | None,
) -> None:
    with pytest.raises(GateError, match="blocked"):
        require_retrain_vedtak(value)
