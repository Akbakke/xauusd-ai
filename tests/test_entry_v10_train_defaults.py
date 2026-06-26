from __future__ import annotations

import ast
from pathlib import Path


TRAINER_PATH = (
    Path(__file__).resolve().parents[1]
    / "gx1"
    / "models"
    / "entry_v10"
    / "entry_v10_ctx_train_v3.py"
)


def _trainer_ast() -> ast.Module:
    return ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))


def _canonical_env_defaults(module: ast.Module) -> dict[str, str]:
    for node in module.body:
        if not (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_CANONICAL_ENTRY_TRAIN_ENV_DEFAULTS"
            and isinstance(node.value, ast.Dict)
        ):
            continue
        defaults: dict[str, str] = {}
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                defaults[str(key.value)] = str(value.value)
        return defaults
    raise AssertionError("_CANONICAL_ENTRY_TRAIN_ENV_DEFAULTS not found")


def _env_str_defaults(module: ast.Module) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for node in ast.walk(module):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_env_str"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[1], ast.Constant)
        ):
            continue
        defaults[str(node.args[0].value)] = str(node.args[1].value)
    return defaults


def test_entry_v10_env_defaults_match_canonical_guard_contract() -> None:
    module = _trainer_ast()
    canonical = _canonical_env_defaults(module)
    env_defaults = _env_str_defaults(module)

    mismatches = {
        key: {"env_str_default": env_defaults[key], "canonical_default": canonical[key]}
        for key in sorted(canonical.keys() & env_defaults.keys())
        if env_defaults[key] != canonical[key]
    }

    assert mismatches == {}


def test_entry_v10_bad_path_aux_default_is_parked() -> None:
    env_defaults = _env_str_defaults(_trainer_ast())
    assert env_defaults["ENTRY_AUX_BAD_PATH_WEIGHT"] == "0.0"
