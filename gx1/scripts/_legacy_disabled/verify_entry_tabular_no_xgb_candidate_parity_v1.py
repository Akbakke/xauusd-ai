"""Verify serve parity for the tabular no-XGB Entry candidate package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gx1.runtime.entry_tabular_no_xgb_candidate import (
    assert_no_xgb_feature_names,
    build_feature_matrix,
    feature_contract_hash,
    json_default,
    predict_proba,
    score_probabilities,
    sha256_file,
)
from gx1.scripts.evaluate_entry_selective_edge_v1 import _split_files
from gx1.scripts.evaluate_entry_tabular_no_xgb_baseline_v1 import _load_tabular_x


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def _file_meta_hash(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime_ns)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _verify_artifact_hashes(package_dir: Path) -> list[dict[str, Any]]:
    csv_path = package_dir / "artifact_hashes.csv"
    if not csv_path.exists():
        raise RuntimeError(f"missing artifact hash table: {csv_path}")
    rows = pd.read_csv(csv_path)
    reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in rows.to_dict(orient="records"):
        path = Path(str(row["path"]))
        mode = str(row.get("hash_mode", "full"))
        expected = str(row["sha256"])
        if not path.exists():
            failures.append(f"missing:{row['artifact']}:{path}")
            observed = None
        elif mode == "full":
            observed = sha256_file(path)
        elif mode == "meta(path,size,mtime_ns)":
            observed = _file_meta_hash(path)
        else:
            failures.append(f"unknown_hash_mode:{row['artifact']}:{mode}")
            observed = None
        ok = observed == expected
        if not ok:
            failures.append(f"hash_mismatch:{row['artifact']}")
        reports.append({
            "artifact": row["artifact"],
            "path": str(path),
            "hash_mode": mode,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "ok": bool(ok),
        })
    if failures:
        raise RuntimeError(f"ARTIFACT_HASH_VERIFICATION_FAILED: {failures[:20]}")
    return reports


def _model_n_features(model: Any) -> int:
    for attr in ("n_features_in_", "n_features_"):
        value = getattr(model, attr, None)
        if value is not None:
            return int(value)
    booster = getattr(model, "booster_", None)
    if booster is not None:
        return int(booster.num_feature())
    raise RuntimeError("could not determine model feature count")


def _parity_for_split(
    *,
    split: str,
    parquet_path: Path,
    feature_names: list[str],
    model: Any,
    max_predict_rows: int,
) -> dict[str, Any]:
    runtime_x, runtime_names = build_feature_matrix(parquet_path, expected_feature_names=feature_names)
    research_x = _load_tabular_x(parquet_path, feature_names)
    if runtime_names != feature_names:
        raise RuntimeError(f"{split}: runtime feature names differ from manifest")
    if runtime_x.shape != research_x.shape:
        raise RuntimeError(f"{split}: runtime/research shape mismatch {runtime_x.shape} vs {research_x.shape}")
    diff = np.abs(runtime_x.astype(np.float64) - research_x.astype(np.float64))
    max_abs_diff = float(np.nanmax(diff)) if diff.size else 0.0
    if max_abs_diff != 0.0:
        raise RuntimeError(f"{split}: runtime/research feature matrix mismatch max_abs_diff={max_abs_diff}")

    if max_predict_rows > 0 and len(runtime_x) > max_predict_rows:
        idx = np.linspace(0, len(runtime_x) - 1, num=max_predict_rows, dtype=np.int64)
        pred_x = runtime_x[idx]
    else:
        pred_x = runtime_x
    probs = predict_proba(model, pred_x)
    chosen_side, chosen_prob, score = score_probabilities(probs)
    return {
        "split": split,
        "rows": int(len(runtime_x)),
        "n_features": int(runtime_x.shape[1]),
        "runtime_research_max_abs_diff": max_abs_diff,
        "prediction_rows": int(len(pred_x)),
        "mean_p_long": float(np.mean(probs[:, 0])),
        "mean_p_short": float(np.mean(probs[:, 1])),
        "mean_p_flat": float(np.mean(probs[:, 2])),
        "chosen_long_rate": float((chosen_side == 0).mean()),
        "chosen_short_rate": float((chosen_side == 1).mean()),
        "mean_chosen_prob": float(np.mean(chosen_prob)),
        "mean_score": float(np.mean(score)),
        "score_p95": float(np.percentile(score, 95)),
        "nonfinite_predictions": int((~np.isfinite(probs)).sum()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = Path(args.candidate_package_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(package_dir / "candidate_manifest.json")
    feature_manifest = _read_json(package_dir / "feature_manifest.json")
    go_no_go = _read_json(package_dir / "candidate_go_no_go.json")

    if manifest.get("package_status") != "NOT_PROMOTED_NOT_LIVE_READY":
        raise RuntimeError("candidate package is not explicitly NOT_PROMOTED_NOT_LIVE_READY")
    if bool(manifest.get("promotion_allowed")) or bool(manifest.get("live_ready")):
        raise RuntimeError("candidate package unexpectedly allows promotion/live")
    if go_no_go.get("decision") != "NO_LIVE_PIN":
        raise RuntimeError("candidate go/no-go must be NO_LIVE_PIN before shadow gate")

    feature_names = [str(item["name"]) for item in feature_manifest.get("features", [])]
    if not feature_names:
        raise RuntimeError("empty feature manifest")
    assert_no_xgb_feature_names(feature_names)
    observed_feature_hash = feature_contract_hash(feature_names)
    expected_feature_hash = str(feature_manifest.get("feature_contract_hash"))
    if observed_feature_hash != expected_feature_hash:
        raise RuntimeError("feature manifest hash mismatch")
    if manifest["feature_contract"]["feature_contract_hash"] != observed_feature_hash:
        raise RuntimeError("candidate manifest feature hash mismatch")

    artifact_reports = _verify_artifact_hashes(package_dir)

    primary_model_path = Path(manifest["model"]["primary_model_path"])
    if sha256_file(primary_model_path) != manifest["model"]["primary_model_sha256"]:
        raise RuntimeError("primary model hash mismatch")
    model = joblib.load(primary_model_path)
    n_model_features = _model_n_features(model)
    if n_model_features != len(feature_names):
        raise RuntimeError(f"primary model feature count mismatch: model={n_model_features} manifest={len(feature_names)}")

    for wf in manifest["model"].get("walkforward_models", []):
        path = Path(wf["path"])
        if sha256_file(path) != wf["sha256"]:
            raise RuntimeError(f"walk-forward model hash mismatch: {path}")
        wf_model = joblib.load(path)
        if _model_n_features(wf_model) != len(feature_names):
            raise RuntimeError(f"walk-forward model feature count mismatch: {path}")

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    files = _split_files(dataset_dir, splits)
    split_reports = [
        _parity_for_split(
            split=split,
            parquet_path=files[split],
            feature_names=feature_names,
            model=model,
            max_predict_rows=int(args.max_predict_rows),
        )
        for split in splits
    ]

    report = {
        "schema_version": "entry_tabular_no_xgb_serve_parity_report_v1",
        "candidate_id": manifest["candidate_id"],
        "candidate_package_dir": str(package_dir),
        "dataset_dir": str(dataset_dir),
        "status": "PASS",
        "promotion_allowed": False,
        "live_ready": False,
        "decision": "NO_LIVE_PIN_UNTIL_SHADOW_PAPER_GATE",
        "feature_contract_hash": observed_feature_hash,
        "n_features": len(feature_names),
        "primary_model_path": str(primary_model_path),
        "primary_model_n_features": n_model_features,
        "splits_checked": splits,
        "artifact_hashes_checked": len(artifact_reports),
        "split_reports": split_reports,
        "next_required_gate": "shadow_paper_gate_with_candidate_manifest",
    }
    report_path = out_dir / "serve_parity_report.json"
    _write_json(report_path, report)
    md_lines = [
        f"# Serve Parity Report: {manifest['candidate_id']}",
        "",
        "Status: PASS",
        "",
        f"- feature_contract_hash: `{observed_feature_hash}`",
        f"- n_features: `{len(feature_names)}`",
        f"- artifact hashes checked: `{len(artifact_reports)}`",
        f"- splits checked: `{','.join(splits)}`",
        "- decision: NO LIVE PIN UNTIL SHADOW/PAPER GATE",
        "",
    ]
    for item in split_reports:
        md_lines.append(
            f"- {item['split']}: rows={item['rows']} max_abs_diff={item['runtime_research_max_abs_diff']} "
            f"prediction_rows={item['prediction_rows']}"
        )
    (out_dir / "serve_parity_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=json_default))
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-package-dir", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--max-predict-rows", type=int, default=5000)
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
