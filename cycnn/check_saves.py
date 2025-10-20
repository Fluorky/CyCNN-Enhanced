import os
import json
import shutil

# === CONFIG ===
# datasets = ["MNIST", "LEGO", "GTSRB"]
datasets = ["LEGO"]
models = ["cyvgg19", "cyresnet56", "vgg19", "resnet56"]
transforms = ["linearpolar", "logpolar"]
ext = ".pt"

for dataset in datasets:
    prefix = dataset.lower() + "-custom"
    saves_dir = os.path.join("./saves", dataset)
    move_to = os.path.join(saves_dir, "old")
    os.makedirs(move_to, exist_ok=True)

    json_path = f"./train_test_scenarios_{dataset}.json"
    if not os.path.isfile(json_path):
        print(f"⚠️ Skipping {dataset}: missing {json_path}")
        continue

    with open(json_path) as f:
        train_test_dict = json.load(f)

    # Build expected filenames for this dataset
    expected_filenames = set()
    for trainset in train_test_dict:
        safe_name = trainset.replace("/", "_")
        for model in models:
            for transform in transforms:
                fname = f"{prefix}-{model}-{transform}_{safe_name}{ext}"
                expected_filenames.add(fname)

    # Find all actual files in saves_dir
    all_files = [
        f for f in os.listdir(saves_dir)
        if f.endswith(ext) and os.path.isfile(os.path.join(saves_dir, f))
    ]

    moved_count = 0
    for fname in all_files:
        if fname not in expected_filenames:
            src = os.path.join(saves_dir, fname)
            dst = os.path.join(move_to, fname)
            shutil.move(src, dst)
            moved_count += 1
            print(f"📦 [{dataset}] Moved: {fname}")

    print(f"\n✅ {dataset}: Moved {moved_count} unexpected model files to: {move_to}")
    print(f"📂 {dataset}: Kept {len(expected_filenames)} expected models.\n")
