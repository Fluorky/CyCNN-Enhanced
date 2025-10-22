import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from models.getmodel import get_model
from data import load_data
import csv

def objective(trial):
    # --- Hyperparameters that Optuna will tune ---
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    momentum = trial.suggest_uniform("momentum", 0.8, 0.99)

    # --- Model + data ---
    model = get_model("cyresnet20", dataset="mnist").cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    train_loader, val_loader, _ = load_data(
        dataset="mnist", data_dir="./data", batch_size=128
    )

    # --- Train a few epochs (short run for demo purposes) ---
    model.train()
    for epoch in range(3):
        for imgs, labels in train_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

    # --- Validation ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            outputs = model(imgs)
            pred = outputs.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = correct / total

    with open("optuna_trials_cyresnet20_mnist.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([trial.number, lr, weight_decay, momentum, acc])

    return acc


if __name__ == "__main__":

    with open("optuna_trials_cyresnet20_mnist.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "lr", "weight_decay", "momentum", "accuracy"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    print(study.best_trial.params)

    print(study.best_params)
    print(study.best_value)

