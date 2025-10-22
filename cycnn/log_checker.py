#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import shutil
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------- Basic utilities ----------

def normalize_key(key: str) -> str:
    return key.replace("/", "_")

def is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0

def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
    except Exception:
        return None

def print_diff(title, expected: Set[Path], actual: Set[Path]):
    missing = expected - actual
    extra = actual - expected
    empty = {p for p in actual & expected if not is_nonempty_file(p)}

    print(f"\n📂 {title}")
    if missing:
        print("❌ Missing files:")
        for p in sorted(missing):
            print(f"   - {p}")
    if extra:
        print("⚠️ Extra files:")
        for p in sorted(extra):
            print(f"   - {p}")
    if empty:
        print("⚠️ Empty files (0 B):")
        for p in sorted(empty):
            print(f"   - {p}")
    if not missing and not extra and not empty:
        print("✅ All expected files are present and non-empty.")

# ---------- Regex patterns ----------

RE_TRACEBACK = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)
RE_ANY_ERROR = re.compile(
    r"(?:RuntimeError|FileNotFoundError|CUDA error|CUDA out of memory|device-side assertions|Exception:)",
    re.IGNORECASE,
)

RE_CONFIG_BLOCK = re.compile(r"^configuration:\s*(\{.*?\})\s*$", re.DOTALL | re.MULTILINE)

RE_DATA_TEST = re.compile(r"^✔️ Test data:\s*(\d+)\s*samples", re.MULTILINE)
RE_DATA_VAL  = re.compile(r"^✔️ Validation data:\s*(\d+)\s*samples", re.MULTILINE)

RE_TEST_START = re.compile(r"^===> Testing model from .+$", re.MULTILINE)
RE_MODEL_OK   = re.compile(r"^Model loaded successfully\.\s*$", re.MULTILINE)
RE_TEST_SUM   = re.compile(
    r"^Test loss:\s*([+-]?\d+(?:\.\d+)?),\s*Accuracy:\s*(\d+)\/(\d+)\s*\((\d+(?:\.\d+)?)%\)\s*$",
    re.MULTILINE,
)
RE_CM_SAVED   = re.compile(
    r"^Confusion Matrix saved as:\s*(.+confusion_matrix\.npy)\s*and\s*(.+confusion_matrix\.png)\s*$",
    re.MULTILINE,
)

RE_TRAIN_START = re.compile(r"^===> Training .+ begin$", re.MULTILINE)
RE_EPOCH_TRAIN = re.compile(r"^\[Epoch\s+(\d+)\]\s*Train Loss:\s*([+-]?\d+(?:\.\d+)?)\s*$", re.MULTILINE)
RE_EPOCH_VAL   = re.compile(
    r"^\[Epoch\s+(\d+)\]\s*Validation loss:\s*([+-]?\d+(?:\.\d+)?)\,\s*Accuracy:\s*(\d+)\/(\d+)\s*\((\d+(?:\.\d+)?)%\)\s*$",
    re.MULTILINE,
)
RE_TRAIN_DONE  = re.compile(r"^Training Done!\s*$", re.MULTILINE)

# Narrow benign warning patterns to specific known lines and DO NOT change status for them
RE_BENIGN_WARN = re.compile(
    r"^(?:.*FutureWarning: You are using `torch\.load` with `weights_only=False`.*|.*UserWarning: The verbose parameter is deprecated\..*)$",
    re.MULTILINE,
)

# ---------- Validation ----------

@dataclass
class ValidationResult:
    ok: bool
    status: str                # "ok" / "corrupted"
    reasons: List[str]
    details: List[str]
    artifacts: List[Path]

def approx_equal_percent(calc: float, pct: float, tol_pp: float = 0.05) -> bool:
    return abs(calc - pct) <= tol_pp

def validate_test_log(path: Path, text: str, strict: bool) -> ValidationResult:
    reasons, details, artifacts = [], [], []

    if RE_TRACEBACK.search(text) or RE_ANY_ERROR.search(text):
        return ValidationResult(False, "corrupted", ["error_stacktrace"], ["Traceback/Exception detected"], [])

    if not RE_TEST_START.search(text):
        reasons.append("missing_test_start")
    if not RE_MODEL_OK.search(text):
        reasons.append("missing_model_loaded")

    test_total_declared = None
    m = RE_DATA_TEST.search(text)
    if m:
        test_total_declared = int(m.group(1))

    m = RE_TEST_SUM.search(text)
    if not m:
        reasons.append("missing_test_summary")
    else:
        loss, hit, total, pct = float(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
        if test_total_declared is not None and total != test_total_declared:
            reasons.append("total_mismatch")
        calc_pct = 100.0 * hit / total if total > 0 else 0.0
        if not approx_equal_percent(calc_pct, pct):
            reasons.append("pct_inconsistent")
        if strict and loss < 0:
            reasons.append("loss_invalid")

    m = RE_CM_SAVED.search(text)
    if not m:
        reasons.append("missing_confusion_matrix_paths")
    else:
        npy, png = m.group(1).strip(), m.group(2).strip()
        artifacts.extend([Path(npy), Path(png)])

    ok = len(reasons) == 0
    status = "ok" if ok else "corrupted"
    return ValidationResult(ok, status, reasons, details, artifacts)

def validate_train_log(path: Path, text: str, strict: bool) -> ValidationResult:
    reasons, details, artifacts = [], [], []

    if RE_TRACEBACK.search(text) or RE_ANY_ERROR.search(text):
        return ValidationResult(False, "corrupted", ["error_stacktrace"], ["Traceback/Exception detected"], [])

    if not RE_TRAIN_START.search(text):
        reasons.append("missing_train_start")

    epochs_train = [m.groups() for m in RE_EPOCH_TRAIN.finditer(text)]
    if not epochs_train:
        reasons.append("missing_epochs")

    epochs_val = [m.groups() for m in RE_EPOCH_VAL.finditer(text)]

    if epochs_train:
        ep_nums = [int(e[0]) for e in epochs_train]
        if ep_nums != sorted(ep_nums) or len(ep_nums) != len(set(ep_nums)):
            reasons.append("epoch_order")
        if strict and any(float(e[1]) < 0 for e in epochs_train):
            reasons.append("train_loss_invalid")

    val_total_declared = None
    m = RE_DATA_VAL.search(text)
    if m:
        val_total_declared = int(m.group(1))

    for ev in epochs_val:
        ep, vloss, hit, total, pct = int(ev[0]), float(ev[1]), int(ev[2]), int(ev[3]), float(ev[4])
        if val_total_declared is not None and total != val_total_declared:
            reasons.append("val_total_mismatch")
        calc_pct = 100.0 * hit / total if total > 0 else 0.0
        if not approx_equal_percent(calc_pct, pct):
            reasons.append("val_pct_inconsistent")
        if strict and vloss < 0:
            reasons.append("val_loss_invalid")

    if not RE_TRAIN_DONE.search(text):
        reasons.append("missing_training_done")

    ok = len(reasons) == 0
    status = "ok" if ok else "corrupted"
    return ValidationResult(ok, status, reasons, details, artifacts)

def classify_log(path: Path, strict: bool) -> Tuple[ValidationResult, bool]:
    text = read_text(path)
    if text is None:
        return ValidationResult(False, "corrupted", ["encoding_error"], ["Cannot decode UTF-8"], []), False

    is_test = bool(RE_TEST_START.search(text))
    is_train = bool(RE_TRAIN_START.search(text))
    if is_test and not is_train:
        result = validate_test_log(path, text, strict)
    elif is_train and not is_test:
        result = validate_train_log(path, text, strict)
    else:
        result = ValidationResult(False, "corrupted", ["unknown_log_type"], ["Cannot determine log type"], [])

    # Detect benign warnings but DO NOT change status
    has_benign_warnings = bool(RE_BENIGN_WARN.search(text))
    return result, has_benign_warnings

def verify_declared_artifacts(result: ValidationResult) -> List[str]:
    reasons = []
    for art in result.artifacts:
        if not art.exists():
            reasons.append(f"missing_artifact:{art}")
    return reasons

# ---------- Expected/actual sets ----------

def collect_expected_sets(dataset: str) -> Tuple[Set[Path], Set[Path], Set[Path]]:
    dataset = dataset.upper()
    scenario_path = Path(f"train_test_scenarios_{dataset}.json")
    logs_base = Path(f"logs/json_{dataset}")

    train_log_dir = logs_base / "train"
    test_log_dir = logs_base / "test"
    cm_dir = logs_base / "confusion_matrices"

    models = ["cyvgg19", "cyresnet56", "vgg19", "resnet56"]
    activations = ["linearpolar", "logpolar"]

    with open(scenario_path) as f:
        scenario = json.load(f)

    expected_train_logs, expected_test_logs, expected_cm_files = set(), set(), set()

    for model in models:
        for activation in activations:
            prefix = f"{dataset}-{model}-{activation}"
            for train_key, test_keys in scenario.items():
                train_id = normalize_key(train_key)
                train_file = Path(f"logs/json_{dataset}/train/{prefix}_{train_id}_train.txt")
                expected_train_logs.add(train_file.resolve())

                test_subdir = Path(f"logs/json_{dataset}/test/{prefix}_{train_id}")
                cm_model_dir = Path(f"logs/json_{dataset}/confusion_matrices/{prefix}_{train_id}")

                for test_key in test_keys:
                    test_id = normalize_key(test_key)
                    test_file = test_subdir / f"{prefix}_{train_id}_test_on_{test_id}.txt"
                    cm_subfolder = cm_model_dir / f"{train_id}_test_on_{test_id}"

                    cm_npy = cm_subfolder / "confusion_matrix.npy"
                    cm_png = cm_subfolder / "confusion_matrix.png"

                    expected_test_logs.add(test_file.resolve())
                    expected_cm_files.update([cm_npy.resolve(), cm_png.resolve()])

    return expected_train_logs, expected_test_logs, expected_cm_files

def collect_actual_sets(train_dir: Path, test_dir: Path, cm_dir: Path) -> Tuple[Set[Path], Set[Path], Set[Path]]:
    actual_train_logs = set(p.resolve() for p in train_dir.rglob("*.txt"))
    actual_test_logs  = set(p.resolve() for p in test_dir.rglob("*.txt"))
    actual_cm_files   = set(
        p.resolve() for p in cm_dir.rglob("*") if p.name in {"confusion_matrix.npy", "confusion_matrix.png"}
    )
    return actual_train_logs, actual_test_logs, actual_cm_files

# ---------- Quarantine ----------

def quarantine_file(path: Path, quarantine_dir: Path, reason_slug: str, details: List[str]):
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dst = quarantine_dir / path.name
    shutil.move(str(path), str(dst))
    diag = {
        "reason": reason_slug,
        "details": details,
        "original_path": str(path),
    }
    (quarantine_dir / (path.stem + ".json")).write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- Main ----------

def main(
    dataset: str,
    delete_extras: bool = True,
    strict: bool = False,
    quarantine_dir: Optional[Path] = None,
    warn_policy: str = "ignore",
    show_ok: bool = False,
):
    dataset_up = dataset.upper()
    logs_base = Path(f"logs/json_{dataset_up}")
    train_log_dir = logs_base / "train"
    test_log_dir = logs_base / "test"
    cm_dir = logs_base / "confusion_matrices"

    expected_train_logs, expected_test_logs, expected_cm_files = collect_expected_sets(dataset)
    actual_train_logs, actual_test_logs, actual_cm_files = collect_actual_sets(train_log_dir, test_log_dir, cm_dir)

    print_diff("Train Logs", expected_train_logs, actual_train_logs)
    print_diff("Test Logs", expected_test_logs, actual_test_logs)
    print_diff("Confusion Matrices (.npy + .png)", expected_cm_files, actual_cm_files)

    print("\n🧪 Content validation of expected log files:")

    corrupted, ok_files = [], []
    files_with_benign_warnings: List[Path] = []

    for p in sorted((expected_train_logs | expected_test_logs) & (actual_train_logs | actual_test_logs)):
        if not is_nonempty_file(p):
            continue
        result, has_benign = classify_log(p, strict=strict)

        if result.artifacts:
            missing_art_reasons = verify_declared_artifacts(result)
            if missing_art_reasons:
                result.ok = False
                result.status = "corrupted"
                result.reasons.extend(["missing_artifacts"])
                result.details.extend(missing_art_reasons)

        if has_benign:
            files_with_benign_warnings.append(p)

        if result.status == "ok":
            ok_files.append(p)
        else:
            corrupted.append((p, result))

    if show_ok and ok_files:
        print("✅ OK (content):")
        for p in ok_files:
            print(f"   - {p}")

    if corrupted:
        print("\n❌ Corrupted content:")
        for p, r in corrupted:
            print(f"   - {p}")
            if r.reasons:
                print(f"      reason(s): {', '.join(r.reasons)}")

    # Benign warnings reporting policy
    if warn_policy == "summary":
        print(f"\nℹ️ Benign warnings detected in {len(files_with_benign_warnings)} file(s).")
    elif warn_policy == "list":
        print(f"\nℹ️ Benign warnings detected in {len(files_with_benign_warnings)} file(s). Showing up to 10:")
        for p in files_with_benign_warnings[:10]:
            print(f"   - {p}")
        if len(files_with_benign_warnings) > 10:
            print(f"   ... and {len(files_with_benign_warnings) - 10} more")

    if delete_extras:
        for file in actual_train_logs - expected_train_logs:
            print(f"🗑️ Deleting extra train log: {file}")
            file.unlink(missing_ok=True)
        for file in actual_test_logs - expected_test_logs:
            print(f"🗑️ Deleting extra test log: {file}")
            file.unlink(missing_ok=True)
        for file in actual_cm_files - expected_cm_files:
            print(f"🗑️ Deleting extra confusion matrix: {file}")
            try:
                file.unlink()
            except FileNotFoundError:
                pass
            try:
                file.parent.rmdir()
            except OSError:
                pass

    if quarantine_dir and corrupted:
        print(f"\n🧩 Quarantining {len(corrupted)} corrupted logs to: {quarantine_dir}")
        for p, r in corrupted:
            quarantine_file(p, quarantine_dir, "+".join(sorted(set(r.reasons))) or "corrupted", r.details)

    print("\n📊 Summary:")
    print(f"   Train logs expected: {len(expected_train_logs)}")
    print(f"   Test logs expected:  {len(expected_test_logs)}")
    print(f"   Confusion matrices expected: {len(expected_cm_files)}")
    print(f"   OK (content): {len(ok_files)}")
    print(f"   Corrupted: {len(corrupted)}")

# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., MNIST, LEGO, GTSRB_RGB)")
    parser.add_argument("--no-delete", action="store_true", help="Don't delete extra files")
    parser.add_argument("--strict", action="store_true", help="Enable stricter validation")
    parser.add_argument("--quarantine-dir", type=str, default=None, help="Move corrupted logs into this directory")
    parser.add_argument(
        "--warn-policy",
        choices=["ignore", "summary", "list"],
        default="ignore",
        help="How to report benign warnings (FutureWarning/UserWarning)",
    )
    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="Print the list of OK (content) files",
    )
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        delete_extras=not args.no_delete,
        strict=args.strict,
        quarantine_dir=Path(args.quarantine_dir) if args.quarantine_dir else None,
        warn_policy=args.warn_policy,
        show_ok=args.show_ok,
    )
