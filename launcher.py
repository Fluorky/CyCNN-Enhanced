import os
import json

# Load training and testing scenarios from the JSON file
with open("train_test_scenariosv2.json") as f:
    train_test_dict = json.load(f)

main_script = "main.py"
venv_python = "venv/bin/python"

base_data_dir = "./data/MNIST_WIN"
base_save_dir = "./saves/MNIST"
base_log_dir = "./logs/json_2/"
train_log_dir = os.path.join(base_log_dir, "train")
test_log_dir = os.path.join(base_log_dir, "test")
model_name = "cyvgg19"
polar_transform = "linearpolar"

overwrite_logs = False
overwrite_models = False

required_files = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]

def dataset_valid(path):
    """Check if the dataset folder contains the necessary files."""
    return all(os.path.exists(os.path.join(path, f)) for f in required_files)

# Create necessary directories
os.makedirs(base_save_dir, exist_ok=True)
os.makedirs(train_log_dir, exist_ok=True)
os.makedirs(test_log_dir, exist_ok=True)

def run_command(cmd, log_file=None):
    """Run a system command with optional logging."""
    if log_file:
        if not overwrite_logs and os.path.exists(log_file):
            print(f"⚠️ Skipping existing log: {log_file}")
            return
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        cmd = f"{cmd} > {log_file} 2>&1"
    print(f"🚀 Running: {cmd}")
    os.system(cmd)

def generate_model_save_path(train_set):
    """Generate a safe path to save the trained model."""
    sanitized_name = train_set.replace("/", "_")
    fname = f"mnist-custom-{model_name}-{polar_transform}_{sanitized_name}.pt"
    return os.path.join(base_save_dir, fname)

def main():
    """Main function to train and test models."""
    for train_set, test_sets in train_test_dict.items():
        train_data_dir = os.path.join(base_data_dir, train_set)
        train_log_file = os.path.join(train_log_dir, f"{train_set.replace('/', '_')}_train.txt")
        model_save_path = generate_model_save_path(train_set)
        print(f"Model save path: {model_save_path}")

        # Validate the training dataset
        if not dataset_valid(train_data_dir):
            print(f"❌ Missing training data files for: {train_set}")
            continue

        print(f"\n=== TRAINING on {train_set} ===")

        # Train the model if not already done
        if os.path.exists(model_save_path) and not overwrite_models:
            print(f"✅ Model already exists: {model_save_path}")
        else:
            train_cmd = (
                f"{venv_python} {main_script} "
                f"--train --model={model_name} --dataset=mnist-custom "
                f"--polar-transform={polar_transform} --data-dir={train_data_dir} "
                f"--model-save-path={model_save_path}"
            )
            run_command(train_cmd, train_log_file)

        # Test the model on all specified test sets
        for test_set in test_sets:
            test_data_dir = os.path.join(base_data_dir, test_set)

            # Validate the test dataset
            if not dataset_valid(test_data_dir):
                print(f"❌ Missing test data for: {test_set}, skipping.")
                continue

            print(f"--- TESTING {train_set} model on {test_set} ---")
            test_subdir = os.path.join(test_log_dir, train_set.replace("/", "_"))
            os.makedirs(test_subdir, exist_ok=True)
            test_log_file = os.path.join(test_subdir, f"{train_set.replace('/', '_')}_test_on_{test_set.replace('/', '_')}.txt")

            test_cmd = (
                f"{venv_python} {main_script} "
                f"--test --model={model_name} --dataset=mnist-custom "
                f"--polar-transform={polar_transform} "
                f"--data-dir={train_data_dir} "
                f"--test-data-dir={test_data_dir}"
            )
            run_command(test_cmd, test_log_file)

    print("\n✅ All training and testing complete!")

if __name__ == "__main__":
    main()
