import os
import json
import subprocess
from pathlib import Path
import csv

VENVPY = "venv/bin/python"
DRIVER = "optuna_driver_universal.py"

DATASETS = [
    {
        "dataset_key": "mnist-custom",
        "baseline_dir": "./data/MNIST_WIN/dataset_mnist_non_rotated",
    },
    {
        "dataset_key": "LEGO",
        "baseline_dir": "./data/LEGO_WIN/dataset_LEGO_non_rotated",
    }
]

MODELS = ["cyvgg19", "cyresnet56", "vgg19", "resnet56"]
POLARS = ["logpolar", "linearpolar"]  
TRIALS = 25
EPOCHS = 10
BATCH_SIZE = 128
RESULTS_ROOT = "./optuna_results"

def run_cmd(cmd):
    print("🚀", cmd)
    return subprocess.run(cmd, shell=True).returncode

def main():
    Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for ds in DATASETS:
        dataset_key = ds["dataset_key"]
        baseline_dir = ds["baseline_dir"]
        for model in MODELS:
            for polar in POLARS:
                out_dir = os.path.join(RESULTS_ROOT, dataset_key)
                Path(out_dir).mkdir(parents=True, exist_ok=True)

                cmd = (
                    f"{VENVPY} {DRIVER} "
                    f"--dataset_key {dataset_key} "
                    f"--dataset_dir {baseline_dir} "
                    f"--model {model} --polar {polar} "
                    f"--trials {TRIALS} --epochs {EPOCHS} --batch_size {BATCH_SIZE} "
                    f"--results_dir {RESULTS_ROOT}"
                )
                rc = run_cmd(cmd)
                if rc != 0:
                    print(f"❌ Failed: {cmd}")
                    continue

                tag = f"{dataset_key}_{model}_{polar}"
                best_json = os.path.join(out_dir, f"{tag}_best.json")
                if not os.path.exists(best_json):
                    print(f"⚠️ Missing best.json: {best_json}")
                    continue
                with open(best_json) as f:
                    best = json.load(f)

                row = {
                    "dataset_key": dataset_key,
                    "baseline_dir": baseline_dir,
                    "model": model,
                    "polar": polar,
                    "best_val_acc": best.get("best_val_acc", None),
                    "lr": best["best_params"].get("lr"),
                    "momentum": best["best_params"].get("momentum"),
                    "weight_decay": best["best_params"].get("weight_decay"),
                    "trials": best.get("trials"),
                    "epochs_per_trial": best.get("epochs_per_trial"),
                }
                summary_rows.append(row)

    summary_csv = os.path.join(RESULTS_ROOT, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset_key","baseline_dir","model","polar",
                        "best_val_acc","lr","momentum","weight_decay",
                        "trials","epochs_per_trial"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n✅ Done. Summary: {summary_csv}")

if __name__ == "__main__":
    main()
