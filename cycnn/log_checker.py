import os
import json
from pathlib import Path

# === CONFIGURATION ===
json_path = "train_test_scenarios.json"
logs_base = Path("logs/json_4")

train_log_dir = logs_base / "train"
test_log_dir = logs_base / "test"
cm_dir = logs_base / "confusion_matrices"

models = ["cyvgg19", "cyresnet56"]
activations = ["linearpolar", "logpolar"]

# === LOAD JSON SCENARIO ===
with open(json_path) as f:
    scenario = json.load(f)

# === EXPECTED PATHS ===
expected_train_logs = set()
expected_test_logs = set()
expected_cm_files = set()

for model in models:
    for activation in activations:
        prefix = f"mnist-custom-{model}-{activation}"
        for train_set, test_sets in scenario.items():
            train_file = train_log_dir / f"{prefix}_{train_set}_train.txt"
            expected_train_logs.add(train_file.resolve())

            test_subdir = test_log_dir / f"{prefix}_{train_set}"
            cm_model_dir = cm_dir / f"{prefix}_{train_set}"

            for test_set in test_sets:
                test_file = test_subdir / f"{prefix}_{train_set}_test_on_{test_set}.txt"
                cm_subfolder = cm_model_dir / f"{train_set}_test_on_{test_set}"

                cm_npy = cm_subfolder / "confusion_matrix.npy"
                cm_png = cm_subfolder / "confusion_matrix.png"

                expected_test_logs.add(test_file.resolve())
                expected_cm_files.update([
                    cm_npy.resolve(),
                    cm_png.resolve()
                ])

# === ACTUAL FILES ===
actual_train_logs = set(p.resolve() for p in train_log_dir.rglob("*.txt"))
actual_test_logs = set(p.resolve() for p in test_log_dir.rglob("*.txt"))
actual_cm_files = set(
    p.resolve() for p in cm_dir.rglob("*") if p.name in {"confusion_matrix.npy", "confusion_matrix.png"}
)

# === HELPERS ===
def print_diff(title, expected, actual):
    missing = expected - actual
    extra = actual - expected

    print(f"\n📂 {title}")
    if missing:
        print("❌ Missing files:")
        for p in sorted(missing):
            print(f"   - {p}")
    if extra:
        print("⚠️ Extra files:")
        for p in sorted(extra):
            print(f"   - {p}")
    if not missing and not extra:
        print("✅ All expected files are present.")

# === REPORT ===
print_diff("Train Logs", expected_train_logs, actual_train_logs)
print_diff("Test Logs", expected_test_logs, actual_test_logs)
print_diff("Confusion Matrices (.npy + .png)", expected_cm_files, actual_cm_files)

# === OPTIONAL CLEANUP ===
delete_extras = False  # Set True if you want auto cleanup

if delete_extras:
    for file in actual_train_logs - expected_train_logs:
        print(f"🗑️ Deleting train log: {file}")
        file.unlink()

    for file in actual_test_logs - expected_test_logs:
        print(f"🗑️ Deleting test log: {file}")
        file.unlink()

    for file in actual_cm_files - expected_cm_files:
        print(f"🗑️ Deleting confusion matrix: {file}")
        file.unlink()
        try:
            file.parent.rmdir()
        except OSError:
            pass
