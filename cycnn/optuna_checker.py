# optuna_checker.py
# Example:
#   venv/bin/python optuna_checker.py \
#     --dataset_key mnist-custom \
#     --base_data_dir ./data/MNIST_WIN \
#     --train_set dataset_mnist_non_rotated \
#     --scenarios_json train_test_scenarios_MNIST.json \
#     --optuna_dir ./optuna_results \
#     --results_dir ./optuna_checked \
#     --models cyvgg19 cyresnet56 vgg19 resnet56 \
#     --polars logpolar linearpolar \
#     --epochs 50 --batch_size 128 --use_prerotated_test_set

import re
import json
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from logging import FileHandler

from models.getmodel import get_model
from data import load_data
import image_transforms

# ---- CUDA knobs -------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# ----------------------------------------------------------------------------


# ------------------------------- IO utils -----------------------------------
def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)
# ----------------------------------------------------------------------------


# --------------------------- Logging (scoped tee) ----------------------------
class PrintTee:
    """
    Context manager that tees builtins.print to a *file-only* logger
    (no console handler) and restores the original print on exit.
    Ensures no handler/print stacking across runs.
    """
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.logger: Optional[logging.Logger] = None
        self._prev_print = None

    def __enter__(self):
        ensure_dir(self.log_path.parent)
        logger_name = f"optuna_checker::{self.log_path.as_posix()}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        # fresh single file handler
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        fh = FileHandler(self.log_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        self.logger = logger

        import builtins
        self._prev_print = builtins.print

        def tee_print(*args, **kwargs):
            # first: normal console print
            self._prev_print(*args, **kwargs)
            # then: write a single line to file logger
            try:
                msg = " ".join(str(x) for x in args)
                logger.info(msg)
            except Exception:
                pass

        builtins.print = tee_print
        return self

    def __exit__(self, exc_type, exc, tb):
        # restore previous print and close file handlers
        import builtins
        if self._prev_print is not None:
            builtins.print = self._prev_print
        if self.logger is not None:
            for h in list(self.logger.handlers):
                self.logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
            self.logger = None
# ----------------------------------------------------------------------------


# --------------------- Best-params JSON handling -----------------------------
def load_best_params(json_path: Path) -> Dict[str, Any]:
    data = load_json(json_path)
    params = data.get("params", data)

    def _f(k: str, default: float) -> float:
        v = params.get(k, data.get(k, default))
        try:
            return float(v)
        except Exception:
            return default

    lr = _f("lr", _f("learning_rate", 0.01))
    momentum = _f("momentum", 0.9)
    weight_decay = _f("weight_decay", 1e-5)
    model = params.get("model", data.get("model"))
    polar = params.get("polar", data.get("polar"))
    return {
        "lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "model": model,
        "polar": polar,
    }
# ----------------------------------------------------------------------------


# --------------------------- Filename parsing --------------------------------
def parse_model_polar_from_filename(
    filename: str,
    dataset_key: str,
    known_models: List[str],
    known_polars: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    stem = filename.removesuffix("_best.json").removesuffix(".json")
    norm = stem.replace("-", "_")
    if norm.startswith(dataset_key + "_"):
        norm = norm[len(dataset_key) + 1 :]
    parts = norm.split("_")
    for i in range(len(parts) - 1):
        m, p = parts[i], parts[i + 1]
        if m in known_models and p in known_polars:
            return m, p
    if len(parts) == 2:
        m, p = parts
        return (m if m in known_models else None), (p if p in known_polars else None)
    return None, None
# ----------------------------------------------------------------------------


# ------------------------------ Discovery ------------------------------------
def discover_best_files(
    optuna_dir: Path, dataset_key: str, models: List[str], polars: List[str]
) -> List[Path]:
    base = optuna_dir / dataset_key
    if not base.is_dir():
        return []
    hits: List[Path] = []
    for m in models:
        for p in polars:
            candidates = [
                f"{m}_{p}_best.json",
                f"{m}-{p}_best.json",
                f"{dataset_key}_{m}_{p}_best.json",
                f"{dataset_key}-{m}-{p}_best.json",
            ]
            got = False
            for c in candidates:
                path = base / c
                if path.is_file():
                    hits.append(path)
                    got = True
                    break
            if not got:
                pattern = re.compile(
                    rf".*(?:{re.escape(dataset_key)}[_-])?{re.escape(m)}[_-]{re.escape(p)}_best\.json$"
                )
                for pth in base.iterdir():
                    if pth.is_file() and pattern.match(pth.name):
                        hits.append(pth)
                        break
    # dedup
    seen = set()
    uniq: List[Path] = []
    for h in hits:
        if h.as_posix() not in seen:
            uniq.append(h)
            seen.add(h.as_posix())
    return uniq
# ----------------------------------------------------------------------------


# --------------------------- Data utilities ----------------------------------
def resize_if_needed(images: torch.Tensor, dataset_key: str) -> torch.Tensor:
    if dataset_key in [
        "mnist",
        "mnist-custom",
        "svhn",
        "gtsrb",
        "gtsrb-custom",
        "gtsrb-rgb",
        "gtsrb-rgb-custom",
        "GTSRB-custom",
        "GTSRB-RGB-custom",
    ]:
        return image_transforms.resize_images(images, 32, 32)
    return images


@torch.no_grad()
def eval_basic(
    model,
    device,
    criterion,
    loader,
    dataset_key: str,
    polar_transform: Optional[str],
) -> tuple[float, float, int, int]:
    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    n = 0
    for images, labels in loader:
        if dataset_key == "svhn":
            labels[labels == 9] = 6
        images = resize_if_needed(images, dataset_key)
        if polar_transform:
            images = image_transforms.polar_transform(
                images, transform_type=polar_transform
            )
        images = images.to(device, non_blocking=True).contiguous()
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_sum += loss.detach()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum()
        n += labels.size(0)
    val_loss = (loss_sum / max(1, len(loader))).item()
    acc = (correct.float() * 100.0 / max(1, n)).item()
    return val_loss, acc, int(correct.item()), n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    loader: torch.utils.data.DataLoader,
    dataset_key: str,
    polar_transform: Optional[str],
    use_prerotated_test_set: bool,
    cm_dir: Path,
    tag: str,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    n = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in loader:
        if dataset_key == "svhn":
            labels[labels == 9] = 6
        images = resize_if_needed(images, dataset_key)
        if not use_prerotated_test_set:
            images = image_transforms.random_rotate(images)
        if polar_transform:
            images = image_transforms.polar_transform(
                images, transform_type=polar_transform
            )
        images = images.to(device, non_blocking=True).contiguous()
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_sum += loss.detach()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum()
        n += labels.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    test_loss = (loss_sum / max(1, len(loader))).item()
    acc = (correct.float() * 100.0 / max(1, n)).item()

    # save CM
    safe_tag = tag.replace("/", "_")
    ensure_dir(cm_dir)
    classes = sorted(list(set(all_labels) | set(all_preds)))
    cm = confusion_matrix(all_labels, all_preds, labels=classes)
    np.save(cm_dir / f"confusion_matrix_{safe_tag}.npy", cm)

    plt.figure(figsize=(max(10, len(classes) // 2), max(8, len(classes) // 2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_dir / f"confusion_matrix_{safe_tag}.png")
    plt.close()
    return test_loss, acc


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    loader: torch.utils.data.DataLoader,
    dataset_key: str,
    polar_transform: Optional[str],
) -> float:
    model.train()
    loss_sum = torch.zeros((), device=device)
    for images, labels in loader:
        images = resize_if_needed(images, dataset_key)
        # (train-time augs are off by default)
        if polar_transform:
            images = image_transforms.polar_transform(
                images, transform_type=polar_transform
            )
        images = images.to(device, non_blocking=True).contiguous()
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach()
    return (loss_sum / max(1, len(loader))).item()
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_key", required=True)
    ap.add_argument("--base_data_dir", required=True)
    ap.add_argument("--train_set", required=True)
    ap.add_argument("--scenarios_json", required=True)
    ap.add_argument("--optuna_dir", default="./optuna_results")
    ap.add_argument("--results_dir", default="./optuna_checked")
    ap.add_argument(
        "--models", nargs="+", default=["cyvgg19", "cyresnet56", "vgg19", "resnet56"]
    )
    ap.add_argument("--polars", nargs="+", default=["logpolar", "linearpolar"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--use_prerotated_test_set", action="store_true")
    ap.add_argument(
        "--early_stop_epochs",
        type=int,
        default=15,
        help="Stop if no val acc improvement for N epochs after >=20.",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    save_dir = results_dir / "saves"
    cm_root = results_dir / "confusion_matrices"
    logs_dir = results_dir / "logs"
    ensure_dir(save_dir)
    ensure_dir(cm_root)
    ensure_dir(logs_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scenarios = load_json(Path(args.scenarios_json))
    if args.train_set not in scenarios:
        raise KeyError(f"'{args.train_set}' not found in {args.scenarios_json}")
    test_sets = list(scenarios.get(args.train_set, [])) or [args.train_set]

    best_files = discover_best_files(
        Path(args.optuna_dir), args.dataset_key, args.models, args.polars
    )
    if not best_files:
        print("❌ No *_best.json files found. Check --optuna_dir/--dataset_key.")
        return

    train_path = Path(args.base_data_dir) / args.train_set
    if not train_path.is_dir():
        raise FileNotFoundError(f"Train path not found: {train_path}")
    train_loader, val_loader, _ = load_data(
        dataset=args.dataset_key, data_dir=str(train_path), batch_size=args.batch_size
    )

    print(f"Using device: {device}")
    print(f"Train set: {args.train_set}")
    print(f"Number of test sets (from JSON): {len(test_sets)}")

    # Collect all test results into a single CSV
    summary_rows: List[Dict[str, Any]] = []

    for best_path in best_files:
        tmp = load_best_params(best_path)
        model_name, polar = tmp.get("model"), tmp.get("polar")
        if model_name not in args.models or polar not in args.polars:
            fm, fp = parse_model_polar_from_filename(
                best_path.name, args.dataset_key, args.models, args.polars
            )
            if model_name not in args.models:
                model_name = fm
            if polar not in args.polars:
                polar = fp
        if model_name is None or polar is None:
            print(f"⚠️ Unable to infer model/polar for {best_path.name} — skipping.")
            continue

        run_log_path = (
            logs_dir / f"{args.dataset_key}_{args.train_set}_{model_name}_{polar}.log"
        )

        # Scope logging to this single run; no global monkeypatch leaks.
        with PrintTee(run_log_path):
            print(f"\n=== Processing {best_path.name} ===")
            best = load_best_params(best_path)
            print(
                f"Training {model_name} ({polar}) "
                f"lr={best['lr']:.6g} mom={best['momentum']:.3f} "
                f"wd={best['weight_decay']:.2e}"
            )

            model = get_model(model=model_name, dataset=args.dataset_key).to(device)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=best["lr"],
                momentum=best["momentum"],
                weight_decay=best["weight_decay"],
            )
            criterion = nn.CrossEntropyLoss()

            best_acc = -1.0
            last_saved = -1
            ckpt_name = (
                f"{args.dataset_key}-{model_name}-{polar}_{args.train_set}.pt"
            )
            ckpt_path = save_dir / ckpt_name

            for ep in range(args.epochs):
                t0 = time.time()
                tr_loss = train_one_epoch(
                    model,
                    device,
                    optimizer,
                    criterion,
                    train_loader,
                    args.dataset_key,
                    polar,
                )
                val_loss, val_acc, val_correct, val_total = eval_basic(
                    model, device, criterion, val_loader, args.dataset_key, polar
                )

                saved = ""
                if val_acc > best_acc:
                    best_acc = val_acc
                    last_saved = ep
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "epoch": ep,
                            "hparams": best,
                            "dataset_key": args.dataset_key,
                            "train_set": args.train_set,
                        },
                        ckpt_path,
                    )
                    saved = f"\n💾 Model saved at: {ckpt_path}"

                elapsed = time.time() - t0
                print(
                    f"[Epoch {ep}] Train Loss: {tr_loss:.6f}\n"
                    f"[Epoch {ep}] Validation loss: {val_loss:.4f}, "
                    f"Accuracy: {val_correct}/{val_total} ({val_acc:.2f}%)\n"
                    f"Elapsed time: {elapsed:.1f} sec" + (saved if saved else "")
                )

                if ep >= 20 and ep - last_saved > args.early_stop_epochs:
                    print(
                        f"⏹ Early stopping at epoch {ep}, "
                        f"no improvement for {args.early_stop_epochs} epochs."
                    )
                    break

            # TESTS: exactly the list from JSON for the given key
            for test_set in test_sets:
                test_path = Path(args.base_data_dir) / test_set
                if not test_path.is_dir():
                    print(f"⚠️ Missing test path: {test_set} -> {test_path}")
                    continue
                _, _, test_loader = load_data(
                    dataset=args.dataset_key,
                    data_dir=str(test_path),
                    batch_size=args.batch_size,
                )

                safe_test_set = test_set.replace("/", "_")
                tag = f"{model_name}-{polar}_{args.train_set}_test_on_{safe_test_set}"
                cm_dir = cm_root / args.train_set / safe_test_set
                loss, acc = evaluate(
                    model,
                    device,
                    criterion,
                    test_loader,
                    dataset_key=args.dataset_key,
                    polar_transform=polar,
                    use_prerotated_test_set=args.use_prerotated_test_set,
                    cm_dir=cm_dir,
                    tag=tag,
                )
                print(f"[TEST] {tag}  loss={loss:.4f}  acc={acc:.2f}%")

                summary_rows.append(
                    {
                        "dataset_key": args.dataset_key,
                        "train_set": args.train_set,
                        "test_set": test_set,
                        "model": model_name,
                        "polar": polar,
                        "lr": f"{best['lr']:.8g}",
                        "momentum": f"{best['momentum']:.6g}",
                        "weight_decay": f"{best['weight_decay']:.8g}",
                        "epochs": str(args.epochs),
                        "batch_size": str(args.batch_size),
                        "ckpt_path": str(ckpt_path),
                        "acc_percent": f"{acc:.2f}",
                        "loss": f"{loss:.6f}",
                        "best_file": str(best_path),
                        "log_file": str(run_log_path),
                    }
                )

    # Write summary CSV
    ensure_dir(results_dir)
    summary_csv = results_dir / "summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n✅ Done. Summary: {summary_csv}")
    else:
        print("\n⚠️ Nothing evaluated; no summary generated.")


if __name__ == "__main__":
    main()
