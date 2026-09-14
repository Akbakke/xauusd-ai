"""Focused draft tests; run capped only after the current exclusive job ends."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gx1.contracts.entry_model_native_train_launch_v1 import (
    candidate_execution_pause_reason,
    require_candidate_execution_budget,
    require_candidate_execution_pointer,
)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "budget.json"
        self.recipe_path = Path(self.temp.name) / "recipe.json"
        self.recipe = {"profile": "candidate", "trainer_cli": {"epochs": 30}}
        self.recipe_digest = "1" * 64
        self.budget = {
            "schema_version": "gx1_candidate_execution_budget_v1",
            "recipe_json": str(self.recipe_path),
            "recipe_sha256": self.recipe_digest,
            "expected_active_pointer_sha256": None,
            "stop_after_optimizer_steps": 4,
            "stop_after_completed_val_epochs": 1,
            "max_invocation_seconds": 5400,
        }

    def load(self, payload=None, raw=None, digest=None):
        self.path.write_text(
            json.dumps(self.budget if payload is None else payload)
            if raw is None
            else raw
        )
        return require_candidate_execution_budget(
            self.path,
            digest or hashlib.sha256(self.path.read_bytes()).hexdigest(),
            recipe_path=self.recipe_path,
            recipe_sha256=self.recipe_digest,
            recipe=self.recipe,
        )

    def test_budget_is_separate_from_immutable_recipe(self):
        before = copy.deepcopy(self.recipe)
        self.assertEqual(self.load(), self.budget)
        self.assertEqual(before, self.recipe)

    def test_exact_binding_and_shape(self):
        for field, value in [
            ("recipe_sha256", "2" * 64),
            ("recipe_json", "/elsewhere/recipe.json"),
            ("expected_active_pointer_sha256", "invalid"),
            ("stop_after_optimizer_steps", True),
            ("stop_after_optimizer_steps", 0),
            ("stop_after_completed_val_epochs", 31),
            ("max_invocation_seconds", 5401),
            ("max_invocation_seconds", float("nan")),
            ("extra", 1),
        ]:
            with self.subTest(field=field, value=value):
                changed = dict(self.budget)
                changed[field] = value
                with self.assertRaises(ValueError):
                    self.load(changed)
        with self.assertRaises(ValueError):
            self.load(digest="0" * 64)
        raw = json.dumps(self.budget)[:-1] + ',"max_invocation_seconds":2}'
        with self.assertRaises(ValueError):
            self.load(raw=raw)
        self.recipe["profile"] = "smoke"
        with self.assertRaises(ValueError):
            self.load()

    def test_long_window_only_for_explicit_native_recipe(self):
        self.recipe.update(schema_version="gx1_unified_exit_random_access_full_train_recipe_v1",
                           val_limits={"max_wall_seconds": 10800})
        changed = {**self.budget, "max_invocation_seconds": 12000}
        self.assertEqual(self.load(changed), changed)
        with self.assertRaises(ValueError):
            self.load({**changed, "max_invocation_seconds": 12001})
        self.recipe["val_limits"]["max_wall_seconds"] = 4200
        with self.assertRaises(ValueError):
            self.load(changed)

    def test_exact_progress_boundaries_and_continuation(self):
        def reason(steps, epochs=0, elapsed=1):
            return candidate_execution_pause_reason(
                self.budget,
                global_optimizer_steps=steps,
                completed_val_epochs=epochs,
                elapsed_seconds=elapsed,
            )

        self.assertIsNone(reason(3))
        self.assertEqual(reason(4), "optimizer_step_ceiling")
        self.assertEqual(reason(5), "optimizer_step_ceiling")
        self.assertEqual(reason(3, 1), "completed_val_epoch_ceiling")
        self.assertEqual(reason(3, elapsed=5400), "invocation_wall_limit")
        self.assertIsNone(reason(3, elapsed=5399.9))
        self.budget["stop_after_optimizer_steps"] = 8
        self.assertIsNone(reason(4))
        self.assertEqual(reason(8), "optimizer_step_ceiling")
        self.budget["stop_after_optimizer_steps"] = None
        self.budget["stop_after_completed_val_epochs"] = None
        self.assertIsNone(reason(100000, 30))
        for args in [
            (True, 0, 1),
            (-1, 0, 1),
            (0, True, 1),
            (0, 0, float("nan")),
            (0, 0, -1),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                reason(*args)

    def test_pointer_prevents_accidental_restart_or_wrong_resume(self):
        require_candidate_execution_pointer(self.budget, None)
        with self.assertRaises(ValueError):
            require_candidate_execution_pointer(self.budget, "2" * 64)
        self.budget["expected_active_pointer_sha256"] = "2" * 64
        require_candidate_execution_pointer(self.budget, "2" * 64)
        for actual in [None, "3" * 64]:
            with self.assertRaises(ValueError):
                require_candidate_execution_pointer(self.budget, actual)


if __name__ == "__main__":
    unittest.main()


def test_native_long_window_policy_is_explicit_and_hash_bound(tmp_path):
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import (
        WINDOW_SCHEMA, require_native_window_policy,
    )
    from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256
    import pytest
    base = dict(schema_version=WINDOW_SCHEMA, recipe={"path": str(tmp_path / "recipe.json"), "sha256": "1" * 64},
                invocation_number=1, max_invocation_seconds=5400,
                budget_path=str(tmp_path / "budget.json"), progress_path=str(tmp_path / "progress.json"),
                campaign_cursor_path=str(tmp_path / "cursor.json"), training_session_directory=str(tmp_path / "session"),
                test_data_used=False)
    for seconds in (5400, 12000):
        policy = {**base, "max_invocation_seconds": seconds}
        policy["policy_sha256"] = canonical_sha256(policy)
        assert require_native_window_policy(policy, verify_files=False)["max_invocation_seconds"] == seconds
        with pytest.raises(RuntimeError, match="WINDOW_POLICY_INVALID"):
            require_native_window_policy({**policy, "max_invocation_seconds": 5400 if seconds == 12000 else 12000}, verify_files=False)
    invalid = {**base, "max_invocation_seconds": 12001}
    invalid["policy_sha256"] = canonical_sha256(invalid)
    with pytest.raises(RuntimeError, match="WINDOW_POLICY_INVALID"):
        require_native_window_policy(invalid, verify_files=False)
