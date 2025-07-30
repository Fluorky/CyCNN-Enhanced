import os
import json
import shutil

# === CONFIG ===
saves_dir = "./saves/MNIST"
json_file = "./train_test_scenarios_MNIST.json"
models = ["cyvgg19", "cyresnet56"]
transforms = ["linearpolar", "logpolar"]
prefix = "mnist-custom"
ext = ".pt"
move_to = os.path.join(saves_dir, "old")
os.makedirs(move_to, exist_ok=True)

# === LOAD JSON ===
with open(json_file) as f:
    train_test_dict = json.load(f)

# === BUILD expected names ===
expected_filenames = set()
for trainset in train_test_dict:
    safe_name = trainset.replace("/", "_")
    for model in models:
        for transform in transforms:
            fname = f"{prefix}-{model}-{transform}_{safe_name}{ext}"
            expected_filenames.add(fname)

# === MOVE unnecessary files ===
all_files = [f for f in os.listdir(saves_dir) if f.endswith(ext) and os.path.isfile(os.path.join(saves_dir, f))]
moved_count = 0

for fname in all_files:
    if fname not in expected_filenames:
        src = os.path.join(saves_dir, fname)
        dst = os.path.join(move_to, fname)
        shutil.move(src, dst)
        moved_count += 1
        print(f"📦 Moved: {fname}")

print(f"\n✅ Moved {moved_count} unexpected model files to: {move_to}")
print(f"📂 Kept {len(expected_filenames)} expected models.")
