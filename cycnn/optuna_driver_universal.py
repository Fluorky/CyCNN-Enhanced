import os
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import optuna
from optuna.pruners import MedianPruner

from models.getmodel import get_model
from data import load_data
import image_transforms


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
        # === keep preprocessing consistent with your main.py ===
        if args['dataset'] in ['mnist', 'mnist-custom', 'svhn']:
            images = image_transforms.resize_images(images, 32, 32)

        if args['augmentation'] is not None and 'scale' in args['augmentation']:
            images = image_transforms.random_scale(images)
        if args['augmentation'] is not None and 'rot' in args['augmentation']:
            images = image_transforms.random_rotate(images)
        if args['augmentation'] is not None and 'trans' in args['augmentation']:
            images = image_transforms.random_translate(images)

        if args['polar_transform'] is not None:
            images = image_transforms.polar_transform(images, transform_type=args['polar_transform'])

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        loss_sum += loss.detach()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum()
        n += labels.size(0)

    return (loss_sum / len(loader)).item(), (correct.float() * 100.0 / n).item()


def make_objective(dataset_key, dataset_dir, model_name, polar_transform, n_epochs, batch_size):
    def objective(trial: optuna.trial.Trial):
        # Safe search space (no deprecated APIs)
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        momentum = trial.suggest_float("momentum", 0.85, 0.99)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args = {
            'dataset': dataset_key,
            'augmentation': None,                 # baseline: no aug
            'polar_transform': polar_transform,   # 'logpolar' / 'linearpolar' / None
            'model': model_name
        }

        train_loader, val_loader, _ = load_data(
            dataset=dataset_key, data_dir=dataset_dir, batch_size=batch_size
        )

        model = get_model(model=model_name, dataset=dataset_key).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        best_val = 0.0
        for epoch in range(n_epochs):
            _ = run_one_epoch(model, device, train_loader, criterion, optimizer, args, train=True)
            val_loss, val_acc = run_one_epoch(model, device, val_loader, criterion, optimizer, args, train=False)

            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_acc > best_val:
                best_val = val_acc

        return best_val
    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_key", required=True, help="e.g. mnist-custom / GTSRB-RGB-custom")
    parser.add_argument("--dataset_dir", required=True, help="baseline folder (e.g., dataset_*_non_rotated)")
    parser.add_argument("--model", required=True, choices=["resnet56", "vgg19", "cyresnet56", "cyvgg19"])
    parser.add_argument("--polar", required=True, choices=["logpolar", "linearpolar", "none"])
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--results_dir", type=str, default="./optuna_results")
    args = parser.parse_args()

    polar = None if args.polar == "none" else args.polar

    # Output paths
    tag = f"{args.dataset_key}_{args.model}_{args.polar}"
    out_dir = os.path.join(args.results_dir, args.dataset_key)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{tag}.csv")
    json_path = os.path.join(out_dir, f"{tag}_best.json")

    # Study (version-agnostic)
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    objective = make_objective(
        dataset_key=args.dataset_key,
        dataset_dir=args.dataset_dir,
        model_name=args.model,
        polar_transform=polar,
        n_epochs=args.epochs,
        batch_size=args.batch_size
    )

    t0 = time.time()
    study.optimize(objective, n_trials=args.trials)
    dt = time.time() - t0

    # Save best summary
    best = {
        "tag": tag,
        "dataset_key": args.dataset_key,
        "dataset_dir": args.dataset_dir,
        "model": args.model,
        "polar": args.polar,
        "best_val_acc": study.best_value,
        "best_params": study.best_params,
        "trials": args.trials,
        "epochs_per_trial": args.epochs,
        "elapsed_sec": dt
    }
    with open(json_path, "w") as f:
        json.dump(best, f, indent=2)

    # Dump all trials to CSV (no CSVLogger needed)
    try:
        df = study.trials_dataframe(attrs=("number", "value", "params", "state", "datetime_start", "datetime_complete"))
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Warning: could not export trials dataframe: {e}")

    print(f"[OK] {tag} -> {json_path}")
    print("Best params:", study.best_params)
    print(f"Best val acc: {study.best_value:.3f} | Elapsed: {dt:.1f}s")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
