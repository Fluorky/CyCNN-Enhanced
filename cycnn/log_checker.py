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

def normalize_key(key: str) -> str:
    return key.replace("/", "_")

# === EXPECTED PATHS ===
expected_train_logs = set()
expected_test_logs = set()
expected_cm_files = set()

for model in models:
    for activation in activations:
        prefix = f"mnist-custom-{model}-{activation}"
        for train_key, test_keys in scenario.items():
            train_id = normalize_key(train_key)
            train_file = train_log_dir / f"{prefix}_{train_id}_train.txt"
            expected_train_logs.add(train_file.resolve())

            test_subdir = test_log_dir / f"{prefix}_{train_id}"
            cm_model_dir = cm_dir / f"{prefix}_{train_id}"

            for test_key in test_keys:
                test_id = normalize_key(test_key)
                test_file = test_subdir / f"{prefix}_{train_id}_test_on_{test_id}.txt"
                cm_subfolder = cm_model_dir / f"{train_id}_test_on_{test_id}"

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
def is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0

def print_diff(title, expected, actual):
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


# === REPORT ===
print_diff("Train Logs", expected_train_logs, actual_train_logs)
print_diff("Test Logs", expected_test_logs, actual_test_logs)
print_diff("Confusion Matrices (.npy + .png)", expected_cm_files, actual_cm_files)

# === SUMMARY ===
print("\n📊 Summary of expected files:")
print(f"   🟡 Train logs: {len(expected_train_logs)}")
print(f"   🔵 Test logs: {len(expected_test_logs)}")
print(f"   🟢 Confusion matrices (.npy + .png): {len(expected_cm_files)}")

# === OPTIONAL CLEANUP ===
delete_extras = True  # Set True if you want auto cleanup

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
