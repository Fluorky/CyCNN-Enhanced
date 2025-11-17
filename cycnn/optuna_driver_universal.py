# -*- coding: utf-8 -*-
"""
Optuna HPO with rotation-aware validation.

Train on --train_dir (e.g., dataset_*_non_rotated),
validate on a mix of angle splits collected from --val_dir subfolders,
and optimize val objective:
  - AUC_theta: area under Acc(Δθ) curve (default, rotation-aware), or
  - acc0: plain accuracy on the first/only val split (fallback).

Usage (example):
  python optuna_driver_universal.py \
    --dataset_key GTSRB-custom \
    --train_dir .\\data\\GTSRB_WIN\\dataset_GTSRB_non_rotated \
    --val_dir   .\\data\\GTSRB_WIN\\merged_datasets\\ \
    --model cyresnet56 --polar logpolar \
    --trials 25 --epochs 10 --batch_size 128 \
    --results_dir .\\optuna_resultsV2 \
    --val_objective auc_theta --theta_step 15
"""

from __future__ import annotations
import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import optuna
from optuna.pruners import MedianPruner

from models.getmodel import get_model
from data import load_data
import image_transforms


# --------------------------- Δθ utils ---------------------------------------

ANGLE_TOKEN = re.compile(
    r'(?i)(?:rotated-(\d+)(?:-(\d+))?)|(?:range[_-](\d+)[_-](\d+))|(?:full[_-]0[_-]360)'
)

def interval_from_token(name: str) -> Optional[Tuple[float, float]]:
    s = (name or "").lower()
    m = ANGLE_TOKEN.search(s)
    if not m:
        return (0.0, 0.0) if "non_rotated" in s else None
    if m.group(1):  # rotated-a or rotated-a-b
        a = float(m.group(1))
        b = float(m.group(2)) if m.group(2) else float(m.group(1))
        return (a, b)
    if m.group(3):  # range_a_b
        return (float(m.group(3)), float(m.group(4)))
    return (0.0, 360.0)

def center_deg(iv: Tuple[float, float]) -> float:
    a, b = iv
    return (a + b) / 2.0 if b >= a else (a + ((b + 360.0 - a) / 2.0)) % 360.0

def delta_deg(train_label: str, test_label: str) -> Optional[float]:
    it = interval_from_token(train_label)
    ie = interval_from_token(test_label)
    if not it or not ie:
        return None
    d = abs(center_deg(it) - center_deg(ie)) % 360.0
    return 360.0 - d if d > 180.0 else d

def bin_delta(d: Optional[float], step: int = 15) -> Optional[int]:
    if d is None:
        return None
    b = int(round(d / step) * step)
    return min(b, 180)


# --------------------------- arch/transform tokens --------------------------

ARCH_TOK = ["cyresnet56", "cyvgg19", "resnet56", "vgg19"]
ACT_TOK  = ["linearpolar", "logpolar"]

def detect_arch_act(label: str) -> Tuple[Optional[str], Optional[str]]:
    L = (label or "").lower()
    arch = next((t for t in ARCH_TOK if t in L), None)
    act  = next((t for t in ACT_TOK if t in L), None)
    return arch, act


# --------------------------- training / eval --------------------------------

@torch.no_grad()
def _move_to_device(images, labels, device):
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return images, labels

# ZAMIANA: sygnatura + użycie optimizer

def run_one_epoch(model, device, loader, criterion, optimizer, args, train=True):
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    n = 0

    for images, labels in loader:
        # (Twoje przekształcenia wejścia – bez zmian)
        if args['dataset'] in ['mnist', 'mnist-custom', 'svhn']:
            images = image_transforms.resize_images(images, 32, 32)
        if args.get('augmentation') is not None and 'scale' in args['augmentation']:
            images = image_transforms.random_scale(images)
        if args.get('augmentation') is not None and 'rot' in args['augmentation']:
            images = image_transforms.random_rotate(images)
        if args.get('augmentation') is not None and 'trans' in args['augmentation']:
            images = image_transforms.random_translate(images)
        if args.get('polar_transform') is not None:
            images = image_transforms.polar_transform(images, transform_type=args['polar_transform'])

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        loss_sum += loss.detach()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum()
        n += labels.size(0)

    avg_loss = (loss_sum / max(1, len(loader))).item()
    acc_pct  = (correct.float() * 100.0 / max(1, n)).item()
    return avg_loss, acc_pct

@torch.no_grad()
def evaluate_loader_acc(model, device, loader, args) -> float:
    """Top-1 accuracy [%] on a single loader (no training)."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        if args['dataset'] in ['mnist', 'mnist-custom', 'svhn']:
            images = image_transforms.resize_images(images, 32, 32)
        if args['polar_transform'] is not None:
            images = image_transforms.polar_transform(images, transform_type=args['polar_transform'])
        images, labels = _move_to_device(images, labels, device)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return (100.0 * correct / total) if total > 0 else 0.0


# --------------------------- val splits discovery ---------------------------

def discover_val_splits(val_dir: str, explicit: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Return mapping {split_name -> split_path}.
    If `explicit` is given, pick those subfolders under val_dir.
    Else, list immediate subfolders of val_dir that look like angle/test splits.
    If val_dir itself looks like a leaf dataset folder, return {basename: val_dir}.
    """
    root = Path(val_dir)
    if not root.exists():
        return {}

    if explicit:
        out = {}
        for name in explicit:
            p = root / name
            if p.exists() and p.is_dir():
                out[name] = str(p)
        if out:
            return out

    # collect subfolders
    subs = [p for p in root.iterdir() if p.is_dir()]
    angleish = []
    for p in subs:
        n = p.name.lower()
        if ("non_rotated" in n) or ANGLE_TOKEN.search(n) or n.startswith("dataset_") or n.startswith("merged"):
            angleish.append(p)

    if angleish:
        return {p.name: str(p) for p in sorted(angleish, key=lambda x: x.name)}

    # fallback: treat val_dir as a leaf
    return {root.name: str(root)}


def build_loader_from_dir(dataset_key: str, data_dir: str, batch_size: int):
    """
    Use your project's load_data API. Prefer test_loader for evaluation;
    if unavailable, fall back to val_loader; if still None, use train_loader with shuffle=False.
    """
    train_loader, val_loader, test_loader = load_data(dataset=dataset_key, data_dir=data_dir, batch_size=batch_size)
    if test_loader is not None:
        return test_loader
    if val_loader is not None:
        return val_loader
    return train_loader  # last resort


def acc_vs_delta(model, device, train_label: str, val_loaders: Dict[str, torch.utils.data.DataLoader],
                 args, theta_step: int = 15) -> Tuple[float, float, float]:
    """
    Evaluate accuracy on each val split, group by Δθ bins, forward-fill gaps,
    return (AUC_theta [0..1], worst_from_curve, avg_from_curve) in [0..1].
    """
    # collect per-split accuracies in [0,1]
    per_split_acc = {}
    for split_name, loader in val_loaders.items():
        acc_pct = evaluate_loader_acc(model, device, loader, args)  # [%]
        per_split_acc[split_name] = acc_pct / 100.0

    # bin by Δθ
    bins = list(range(0, 181, theta_step))
    bucket: Dict[int, List[float]] = {b: [] for b in bins}
    for split_name, acc in per_split_acc.items():
        d = delta_deg(train_label, split_name)
        b = bin_delta(d, theta_step)
        if b is not None:
            bucket[b].append(acc)

    # forward-fill missing bins
    ys = []
    last = None
    for b in bins:
        v = float(np.mean(bucket[b])) if bucket[b] else None
        if v is None:
            v = last
        ys.append(v)
        if v is not None:
            last = v
    first = next((v for v in ys if v is not None), 0.0)
    ys = [first if v is None else v for v in ys]

    # trapezoidal AUC normalized by 180
    auc = 0.0
    for i in range(1, len(bins)):
        auc += (ys[i - 1] + ys[i]) * (bins[i] - bins[i - 1]) / 2.0
    auc /= 180.0

    return float(auc), float(min(ys) if ys else 0.0), float(sum(ys) / len(ys) if ys else 0.0)


# --------------------------- Optuna objective -------------------------------

def make_objective(dataset_key: str,
                   train_dir: str,
                   val_dir: str,
                   model_name: str,
                   polar_transform: Optional[str],
                   n_epochs: int,
                   batch_size: int,
                   val_objective: str,
                   theta_step: int):
    """
    val_objective: "auc_theta" | "acc0"
    """
    def objective(trial: optuna.trial.Trial):
        # Safe search space
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        momentum = trial.suggest_float("momentum", 0.85, 0.99)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training loader (non_rotated or whatever user passed)
        train_loader, _, _ = load_data(dataset=dataset_key, data_dir=train_dir, batch_size=batch_size)

        # Validation splits (a mix of angles)
        # If val_dir has subfolders, each is a split; else treat val_dir as one split.
        val_splits = discover_val_splits(val_dir)
        val_loaders = {name: build_loader_from_dir(dataset_key, path, batch_size)
                       for name, path in val_splits.items()}
        if not val_loaders:
            # Fallback: evaluate on train_dir val/test (acc0)
            _, val_loader_fallback, test_loader_fallback = load_data(dataset=dataset_key, data_dir=train_dir, batch_size=batch_size)
            chosen = test_loader_fallback or val_loader_fallback or train_loader
            val_loaders = {"non_rotated": chosen}

        # Construct model & optimizer
        model = get_model(model=model_name, dataset=dataset_key).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        # common args bag
        args_bag = {
            'dataset': dataset_key,
            'augmentation': None,                  # baseline: no train-time aug (as before)
            'polar_transform': polar_transform,    # 'logpolar' / 'linearpolar' / None
            'model': model_name,
            'optimizer': optimizer
        }

        # Use train_dir basename as "train label" to infer Δθ
        train_label = Path(train_dir).name

        best_score = -1.0
        for epoch in range(n_epochs):
            _ = run_one_epoch(model, device, train_loader, criterion, optimizer, args_bag, train=True)

            if val_objective == "auc_theta":
                auc, worst, avg = acc_vs_delta(model, device, train_label, val_loaders, args_bag, theta_step)
                score = auc  # maximize AUC_theta
            else:
                # "acc0": take the first split's accuracy (single-split validation)
                first_name = next(iter(val_loaders.keys()))
                acc_pct = evaluate_loader_acc(model, device, val_loaders[first_name], args_bag)
                score = acc_pct / 100.0

            # report & prune
            trial.report(score, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if score > best_score:
                best_score = score

        return best_score

    return objective


# --------------------------- CLI & main -------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_key", required=True, help="e.g. mnist-custom / GTSRB-custom / GTSRB-RGB-custom")
    parser.add_argument("--train_dir",  required=True, help="training folder (e.g., dataset_*_non_rotated)")
    parser.add_argument("--val_dir",    required=True, help="validation root (folder with angle splits OR a single split)")
    parser.add_argument("--model", required=True, choices=["resnet56", "vgg19", "cyresnet56", "cyvgg19"])
    parser.add_argument("--polar", required=True, choices=["logpolar", "linearpolar", "none"])
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--results_dir", type=str, default="./optuna_results")

    # Rotation-aware validation controls
    parser.add_argument("--val_objective", choices=["auc_theta", "acc0"], default="auc_theta",
                        help="auc_theta = rotation-aware (recommended), acc0 = plain accuracy on first val split")
    parser.add_argument("--theta_step", type=int, default=15, help="Δθ bin size (degrees)")
    parser.add_argument("--val_splits", nargs="*", default=None,
                        help="Optional explicit subfolder names inside --val_dir to use as validation splits")

    args = parser.parse_args()
    polar = None if args.polar == "none" else args.polar

    # Output paths
    tag = f"{args.dataset_key}_{args.model}_{args.polar}"
    out_dir = os.path.join(args.results_dir, args.dataset_key)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{tag}.csv")
    json_path = os.path.join(out_dir, f"{tag}_best.json")

    # Study
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    objective = make_objective(
        dataset_key=args.dataset_key,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        model_name=args.model,
        polar_transform=polar,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        val_objective=args.val_objective,
        theta_step=args.theta_step
    )

    t0 = time.time()
    study.optimize(objective, n_trials=args.trials)
    dt = time.time() - t0

    # Save best summary
    best = {
        "tag": tag,
        "dataset_key": args.dataset_key,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "model": args.model,
        "polar": args.polar,
        "val_objective": args.val_objective,
        "theta_step": args.theta_step,
        "best_val_score": study.best_value,  # AUC_theta or acc0 in [0..1]
        "best_params": study.best_params,
        "trials": args.trials,
        "epochs_per_trial": args.epochs,
        "elapsed_sec": dt
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    # Dump all trials to CSV
    try:
        df = study.trials_dataframe(attrs=("number", "value", "params", "state", "datetime_start", "datetime_complete"))
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Warning: could not export trials dataframe: {e}")

    print(f"[OK] {tag} -> {json_path}")
    print("Best params:", study.best_params)
    print(f"Best val score ({args.val_objective}): {study.best_value:.4f} | Elapsed: {dt:.1f}s")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
