"""Real Session protocol with a deliberately isolated completed population owner.

The opaque policy proof, physical binding reads, identity guards and mutable
checkpoint save/load are real. Tiny file inputs do not claim native admission.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from gx1.contracts import entry_candidate_checkpoint_policy_v1 as policy
from gx1.contracts import local_random_access_campaign_v2 as files
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


@pytest.fixture
def session_case(tmp_path, monkeypatch):
    inputs = {}
    for role in ("train", "quote"):
        path = tmp_path / (role + ".bin")
        path.write_bytes((role + "-immutable-input\n").encode())
        inputs[role] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    from tests.test_entry_dynamic_m5_loss_scale import _synthetic_loaded_loss_scales
    from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import dynamic_m5_loss_scale_identity
    loaded_units = _synthetic_loaded_loss_scales(tmp_path)
    inputs["loss_units"] = loaded_units["binding"]
    prefix = {"entry_training_phase": trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
              "dynamic_m5_loss_scales": dynamic_m5_loss_scale_identity(loaded_units),
              "unit_input_bindings": copy.deepcopy(inputs)}
    semantic_calls = []

    def completed_population_owner(value):
        if value is None:
            return None
        assert value["entry_training_phase"] == trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        semantic_calls.append(copy.deepcopy(value))
        for role, binding in value["unit_input_bindings"].items():
            files.require_binding(binding, label="isolated completed " + role, verify_file=True)
            # The real owner may reach one dependency through multiple joins.
            files.require_binding(binding, label="rejoined " + role, verify_file=True)
        return {"train_rows": 32, "optimizer_steps": 2, "epochs": 1, "drop_last": False}

    monkeypatch.setattr(policy, "_m5_single_epoch_checkpoint_scope", completed_population_owner)
    contract = {"schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
        "authority": {"candidate_training": True, "bundle": False, "test": False,
                      "promotion": False, "paper": False, "live": False},
        "nonce": "current-session", "chronological_prefix": prefix,
        "training": {"checkpoint_policy": policy.checkpoint_policy_metadata(
            checkpoint_monitor=policy.DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR,
            prefix_training_binding=prefix)}}
    semantic_calls.clear()
    reads, mutable_hashes, preparations = Counter(), [], []
    raw = files._file_sha256_uncached
    def real_read(path, *, identity, scoped):
        reads[str(path)] += 1
        return raw(path, identity=identity, scoped=scoped)
    monkeypatch.setattr(files, "_file_sha256_uncached", real_read)
    mutable_sha = trainer._sha256_file
    def real_mutable_sha(path):
        mutable_hashes.append(str(path))
        return mutable_sha(path)
    monkeypatch.setattr(trainer, "_sha256_file", real_mutable_sha)
    prepare = trainer.prepare_verified_checkpoint_policy
    def actual_prepare(*, contract):
        preparations.append(copy.deepcopy(contract))
        return prepare(contract=contract)
    monkeypatch.setattr(trainer, "prepare_verified_checkpoint_policy", actual_prepare)
    return {"contract": contract, "inputs": inputs, "reads": reads,
            "mutable_hashes": mutable_hashes, "preparations": preparations,
            "semantic_calls": semantic_calls, "tmp": tmp_path}


def open_session(case, *, name="BUNDLE", read_only=False, contract=None):
    return trainer._CandidateTrainingSession(out_bundle_dir=case["tmp"] / name,
        contract=copy.deepcopy(case["contract"] if contract is None else contract), read_only=read_only)


def state_for(session, model, *, checkpoint_index=1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    return {"schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
        "session_contract_sha256": session.contract_sha256, "checkpoint_index": checkpoint_index,
        "phase": "train", "epoch_index": 0, "next_batch_offset": 0,
        "global_optimizer_steps": 0, "epoch_order": torch.tensor([1, 0], dtype=torch.int64),
        "model_state": model.state_dict(), "target_model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": optimizer.state_dict(), "weight_ema_state": trainer._WeightEma(model, 0.5).checkpoint_state(),
        "lr_scheduler_state": None,
        "rng_state": trainer._attended_session_rng_state(device=torch.device("cpu")),
        "training_progress": trainer._new_candidate_training_progress(
            checkpoint_monitor=session.checkpoint_monitor,
            prefix_training_binding=session._contract.get("chronological_prefix"),
            verified_context=session.verified_checkpoint_policy_context),
        "complete": False}


def test_dynamic_session_reuses_admission_across_forward_and_actual_save_load(session_case):
    case = session_case
    session = open_session(case)
    context = session.verified_checkpoint_policy_context
    assert context is session.verified_checkpoint_policy_context
    assert len(case["preparations"]) == len(case["semantic_calls"]) == 1
    assert case["reads"] == Counter({b["path"]: 1 for b in case["inputs"].values()})
    class TinyForward(torch.nn.Linear):
        def forward(self, value):
            assert files._FILE_VERIFICATION_MEMO.get() is None
            return super().forward(value)
    model = TinyForward(3, 2)
    with torch.no_grad():
        assert torch.isfinite(model(torch.ones(2, 3))).all()
    first = state_for(session, model)
    session.save_checkpoint(first)
    restored = session.load_checkpoint()
    assert restored["global_optimizer_steps"] == 0
    for key in model.state_dict():
        assert torch.equal(restored["model_state"][key], first["model_state"][key])
    session.save_checkpoint(state_for(session, model, checkpoint_index=2))
    assert session.load_checkpoint()["checkpoint_index"] == 2
    assert len(case["semantic_calls"]) == 1
    assert case["reads"] == Counter({b["path"]: 1 for b in case["inputs"].values()})
    assert any(Path(p).name in trainer._CANDIDATE_TRAINING_STATE_FILENAMES for p in case["mutable_hashes"])
    assert files._FILE_VERIFICATION_MEMO.get() is None
    assert "verified_checkpoint_policy_context" not in json.loads(session._contract_path.read_text())
    assert set(restored) == trainer._CANDIDATE_TRAINING_STATE_KEYS


def test_new_resume_and_read_only_sessions_each_admit_fresh_inputs(session_case):
    case = session_case
    writer = open_session(case)
    writer.save_checkpoint(state_for(writer, torch.nn.Linear(3, 2)))
    resumed = open_session(case)
    reader = open_session(case, read_only=True)
    assert resumed.verified_checkpoint_policy_context is not writer.verified_checkpoint_policy_context
    assert reader.verified_checkpoint_policy_context is not resumed.verified_checkpoint_policy_context
    assert resumed.load_checkpoint()["checkpoint_index"] == reader.load_checkpoint()["checkpoint_index"] == 1
    assert len(case["preparations"]) == len(case["semantic_calls"]) == 3
    assert case["reads"] == Counter({b["path"]: 3 for b in case["inputs"].values()})


@pytest.mark.parametrize("role", ["train", "quote"])
def test_session_rejects_changed_admitted_dependency_without_rehash(session_case, role):
    case = session_case
    session = open_session(case)
    model = torch.nn.Linear(3, 2)
    with torch.no_grad():
        model(torch.ones(1, 3))
    path = Path(case["inputs"][role]["path"])
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    before = case["reads"].copy()
    with pytest.raises(RuntimeError):
        session.verified_checkpoint_policy_context
    assert case["reads"] == before
    assert not session._active_path.exists()


@pytest.mark.parametrize("part", ["contract", "prefix", "monitor", "policy"])
def test_session_context_rejects_full_contract_or_policy_tamper(session_case, part):
    session = open_session(session_case)
    if part == "contract":
        session._contract["nonce"] = "other-session"
    elif part == "prefix":
        session._contract["chronological_prefix"]["unit_input_bindings"]["train"]["sha256"] = "0" * 64
    elif part == "monitor":
        session._contract["training"]["checkpoint_policy"]["checkpoint_monitor"] = policy.CHECKPOINT_MONITOR
    else:
        session._contract["training"]["checkpoint_policy"]["save_top_k"] = 1
    with pytest.raises(RuntimeError):
        session.verified_checkpoint_policy_context


def test_session_rejects_a_foreign_opaque_context(session_case):
    session = open_session(session_case)
    other_contract = copy.deepcopy(session_case["contract"])
    other_contract["nonce"] = "another-session"
    other = open_session(session_case, name="OTHER_BUNDLE", contract=other_contract)
    session._verified_checkpoint_policy_context = other.verified_checkpoint_policy_context
    with pytest.raises(RuntimeError):
        session.verified_checkpoint_policy_context


def test_verified_policy_does_not_hide_mutable_checkpoint_corruption(session_case):
    case = session_case
    session = open_session(case)
    session.save_checkpoint(state_for(session, torch.nn.Linear(3, 2)))
    assert session.load_checkpoint()["checkpoint_index"] == 1
    pointer = json.loads(session._active_path.read_text())
    state_path = session._slot_path(pointer["slot"])
    with state_path.open("ab") as handle:
        handle.write(b"corruption-after-verified-input-admission")
    hashes = len(case["mutable_hashes"])
    with pytest.raises(RuntimeError):
        session.load_checkpoint()
    assert len(case["mutable_hashes"]) > hashes
    assert case["reads"] == Counter({b["path"]: 1 for b in case["inputs"].values()})


def test_verified_session_still_rejects_progress_policy_mutation(session_case):
    session = open_session(session_case)
    state = state_for(session, torch.nn.Linear(3, 2))
    state["training_progress"]["checkpoint_selection"]["checkpoint_policy"]["save_top_k"] = 1
    with pytest.raises(RuntimeError):
        session.save_checkpoint(state)
    assert not session._active_path.exists()


def test_dynamic_session_first_admission_rejects_wrong_expected_sha(session_case):
    contract = copy.deepcopy(session_case["contract"])
    contract["chronological_prefix"]["unit_input_bindings"]["train"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        open_session(session_case, contract=contract)


def test_dynamic_session_cannot_reuse_an_ambient_hash_scope(session_case):
    with files.file_verification_scope():
        with pytest.raises(RuntimeError):
            open_session(session_case)
    assert not session_case["semantic_calls"]


def test_legacy_session_keeps_none_context_and_default_policy_path(session_case):
    contract = copy.deepcopy(session_case["contract"])
    contract.pop("chronological_prefix")
    contract["training"]["checkpoint_policy"] = policy.checkpoint_policy_metadata()
    session = open_session(session_case, contract=contract)
    assert session.verified_checkpoint_policy_context is None
    assert not session_case["preparations"]
    session.save_checkpoint(state_for(session, torch.nn.Linear(3, 2)))
    assert session.load_checkpoint()["checkpoint_index"] == 1
    assert not session_case["reads"]
