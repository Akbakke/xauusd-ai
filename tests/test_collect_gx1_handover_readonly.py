from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.collect_gx1_handover_readonly import (
    REQUIRED_ROLES,
    _parse_evidence,
    collect,
    native_status,
)


class HandoverCollectorTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, dict[str, Path]]:
        repo = root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
        (repo / "source.txt").write_text("source\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "source.txt"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
        evidence: dict[str, Path] = {}
        source = root / "snapshots"
        source.mkdir()
        for role in REQUIRED_ROLES:
            path = source / f"{role}.json"
            path.write_text(
                json.dumps({"schema_version": f"fixture_{role}_v1", "status": "ABSENT_OR_IDLE"}) + "\n",
                encoding="utf-8",
            )
            evidence[role] = path
        return repo.resolve(), evidence

    def test_collects_complete_hash_bound_bundle_without_mutating_repo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, evidence = self._fixture(root)
            output = (root / "bundle").resolve()
            branch = subprocess.run(
                ["git", "-C", str(repo), "branch", "--show-current"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            commit = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            manifest_path = collect(
                repo=repo,
                output=output,
                evidence=evidence,
                expected_branch=branch,
                expected_commit=commit,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "gx1_handover_bundle_v1")
            self.assertEqual(manifest["decision"], "CAPTURED_NOT_RUN_AUTHORITY")
            self.assertTrue(manifest["git"]["clean"])
            self.assertFalse(manifest["test_accessed"])
            self.assertEqual(set(manifest["evidence"]), set(REQUIRED_ROLES))
            for role, binding in manifest["evidence"].items():
                copied = output / binding["bundle_path"]
                self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), binding["sha256"], role)
            semantic = dict(manifest)
            digest = semantic.pop("bundle_sha256")
            canonical = json.dumps(semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), digest)
            self.assertEqual(
                subprocess.run(
                    ["git", "-C", str(repo), "status", "--porcelain=v1"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout,
                "",
            )

    def test_native_observation_checks_exact_binding_without_model_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, evidence = self._fixture(root)
            runtime, session = root / "runtime", root / "session"
            runtime.mkdir()
            session.mkdir()
            pointer = {"session_contract_sha256": "session-digest", "phase": "train",
                       "global_optimizer_steps": 17, "epoch_index": 0}
            (session / "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json").write_text(json.dumps(pointer))
            artifact = evidence["campaign_plan"]
            binding = {"schema_version": "gx1_native_handover_binding_v1",
                       "source_repo": str(repo),
                       "source_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                       "runtime_root": str(runtime), "training_session": str(session),
                       "session_contract_sha256": "session-digest",
                       "immutable_artifacts": {"plan": {"path": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}}}
            path = root / "binding.json"
            path.write_text(json.dumps(binding))
            with patch("scripts.collect_gx1_handover_readonly._native_processes", return_value=[]):
                observed = native_status(path)
            self.assertEqual(observed["checkpoint"], pointer)
            self.assertEqual(observed["process_observation"], "NO_NATIVE_PROCESS_OBSERVED")
            self.assertFalse(observed["state_payload_rehashed"])
            self.assertFalse(observed["test_accessed"])
            self.assertNotIn("complete", observed)
            artifact.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                native_status(path)

    def test_requires_every_role_and_rejects_test_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, evidence = self._fixture(root)
            values = [f"{role}={path}" for role, path in evidence.items() if role != "boot_status"]
            with self.assertRaisesRegex(ValueError, "missing evidence roles: boot_status"):
                _parse_evidence(values)
            forbidden = root / "TEST" / "status.json"
            forbidden.parent.mkdir()
            forbidden.write_text("{}\n", encoding="utf-8")
            values.append(f"boot_status={forbidden}")
            with self.assertRaisesRegex(ValueError, "must not access a TEST path"):
                _parse_evidence(values)

    def test_rejects_output_under_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, evidence = self._fixture(root)
            output = root / "TEST" / "bundle"
            with self.assertRaisesRegex(ValueError, "output must not access a TEST path"):
                collect(
                    repo=repo,
                    output=output,
                    evidence=evidence,
                    expected_branch="master",
                    expected_commit=subprocess.run(
                        ["git", "-C", str(repo), "rev-parse", "HEAD"],
                        check=True, capture_output=True, text=True,
                    ).stdout.strip(),
                )


if __name__ == "__main__":
    unittest.main()
