import os
import json

with open("train_test_scenariosv2.json") as f:
    train_test_dict = json.load(f)

main_script = "main.py"
venv_python = "venv/bin/python"

base_data_dir = "./data/MNIST_WIN"
merged_dir = base_data_dir
dataset_mnist_non_rotated = os.path.join(base_data_dir, "dataset_mnist_non_rotated")

base_save_dir = "./saves/MNIST/"
base_log_dir = "./logs/json_4/"
train_log_dir = os.path.join(base_log_dir, "train")
test_log_dir = os.path.join(base_log_dir, "test")
cm_log_dir = os.path.join(base_log_dir, "confusion_matrices")
model_name = "cyvgg19"
polar_transforms = ["logpolar", "linearpolar"]
overwrite_logs = False
overwrite_models = False

required_files = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]


def dataset_valid(path):
    return all(os.path.exists(os.path.join(path, f)) for f in required_files)


def run_command(cmd, log_file=None):
    if log_file:
        if not overwrite_logs and os.path.exists(log_file):
            print(f"⚠️ Skipping existing log: {log_file}")
            return
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        cmd = f"{cmd} > {log_file} 2>&1"
    print(f"🚀 Running: {cmd}")
    os.system(cmd)


def generate_model_save_path(train_set, polar_transform):
    """Generate model save path based on dataset, model, and polar transform"""
    fname = f"mnist-custom-{model_name}-{polar_transform}_{train_set}.pt"
    return os.path.join(base_save_dir, fname)


def main():
    for polar_transform in polar_transforms:
        for train_set, test_sets in train_test_dict.items():
            train_data_dir = os.path.join(merged_dir, train_set)
            train_set_safe = train_set.replace("/", "_")
            train_log_file = os.path.join(train_log_dir, f"mnist-custom-{model_name}-{polar_transform}_{train_set_safe}_train.txt")
            model_save_path = generate_model_save_path(train_set_safe, polar_transform)
            cm_output_dir = os.path.join(cm_log_dir, train_set_safe)

            if not dataset_valid(train_data_dir):
                print(f"❌ Missing training data files for: {train_set}")
                continue

            print(f"\n=== TRAINING on {train_set} with {polar_transform} ===")

            if os.path.exists(model_save_path) and not overwrite_models:
                print(f"✅ Model already exists and overwrite is disabled: {model_save_path}")
            else:
                train_cmd = (
                    f"{venv_python} {main_script} "
                    f"--train --model={model_name} --dataset=mnist-custom "
                    f"--polar-transform={polar_transform} --data-dir={train_data_dir} "
                    f"--model-save-path={model_save_path} --output-dir={cm_output_dir}"
                )
                run_command(train_cmd, train_log_file)

            for test_set in test_sets:
                test_data_dir = (
                    dataset_mnist_non_rotated
                    if test_set == "dataset_mnist_non_rotated"
                    else os.path.join(merged_dir, test_set)
                )

                if not dataset_valid(test_data_dir):
                    print(f"❌ Missing test data for: {test_set}, skipping.")
                    continue

                print(f"--- TESTING {train_set} model on {test_set} with {polar_transform} ---")
                test_set_safe = test_set.replace("/", "_")
                test_subdir = os.path.join(test_log_dir, f"mnist-custom-{model_name}-{polar_transform}_{train_set_safe}")
                cm_output_dir = os.path.join(cm_log_dir, f"mnist-custom-{model_name}-{polar_transform}_{train_set_safe}/{train_set_safe}_test_on_{test_set_safe}")
                os.makedirs(test_subdir, exist_ok=True)
                test_log_file = os.path.join(test_subdir, f"mnist-custom-{model_name}-{polar_transform}_{train_set_safe}_test_on_{test_set_safe}.txt")

                test_cmd = (
                    f"{venv_python} {main_script} "
                    f"--test --model={model_name} --dataset=mnist-custom "
                    f"--polar-transform={polar_transform} "
                    f"--data-dir={train_data_dir} "
                    f"--test-data-dir={test_data_dir} "
                    f"--output-dir={cm_output_dir} "
                    f"--model-path={model_save_path}"
                )
                run_command(test_cmd, test_log_file)

    print("\n✅ All training and testing complete!")


if __name__ == "__main__":
    main()
