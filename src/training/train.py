import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import DEFAULT_HIDDEN_DIMS, build_model, parameter_count


DEFAULT_BITS = (2, 4, 8)
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train quantized NSL-KDD intrusion detection models."
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=list(DEFAULT_BITS),
        help="Bit-widths to train. Default: 2 4 8",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Maximum epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay used by AdamW.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.02,
        help="Cross-entropy label smoothing.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation split fraction from the training set.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs.",
    )
    return parser.parse_args()


def load_arrays():
    print("Loading preprocessed arrays...")
    X_train = np.load("data/X_train.npy").astype(np.float32)
    y_train = np.load("data/y_train.npy").astype(np.int64)
    X_test = np.load("data/X_test.npy").astype(np.float32)
    y_test = np.load("data/y_test.npy").astype(np.int64)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    return X_train, y_train, X_test, y_test


def make_loaders(X_train, y_train, X_val, y_val, batch_size):
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def evaluate_model(model, X, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        preds = logits.argmax(dim=1).cpu().numpy()

    return {
        "accuracy": accuracy_score(y, preds) * 100.0,
        "f1_weighted": f1_score(y, preds, average="weighted"),
        "f1_macro": f1_score(y, preds, average="macro"),
        "precision_attack": precision_score(y, preds, zero_division=0),
        "recall_attack": recall_score(y, preds, zero_division=0),
        "predictions": preds,
    }


def train_one_model(bits, X_train, y_train, X_val, y_val, X_test, y_test, args, device):
    print(f"\n{'=' * 70}")
    print(f"Training {bits}-bit quantized model")
    print(f"{'=' * 70}")

    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2.0 * np.maximum(class_counts, 1))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    model = build_model(bits).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=args.label_smoothing,
    )

    train_loader, val_loader = make_loaders(
        X_train,
        y_train,
        X_val,
        y_val,
        args.batch_size,
    )

    best_state = None
    best_metrics = None
    best_epoch = 0
    stale_epochs = 0
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size

        train_loss = running_loss / max(seen, 1)

        val_metrics = evaluate_model(model, X_val, y_val, device)
        scheduler.step(val_metrics["accuracy"])

        improved = False
        if best_metrics is None:
            improved = True
        elif val_metrics["accuracy"] > best_metrics["accuracy"] + 1e-6:
            improved = True
        elif (
            abs(val_metrics["accuracy"] - best_metrics["accuracy"]) <= 1e-6
            and val_metrics["f1_weighted"] > best_metrics["f1_weighted"]
        ):
            improved = True

        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        print(
            f"epoch {epoch:02d}/{args.epochs}  "
            f"loss: {train_loss:.4f}  "
            f"val_acc: {val_metrics['accuracy']:.2f}%  "
            f"val_f1: {val_metrics['f1_weighted']:.4f}  "
            f"attack_recall: {val_metrics['recall_attack']:.4f}"
        )

        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, X_test, y_test, device)
    training_seconds = time.time() - started_at
    memory_kb = (parameter_count(model) * bits) / (8 * 1024)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/model_{bits}bit.pt")

    summary = {
        "bits": bits,
        "best_epoch": best_epoch,
        "hidden_dims": list(DEFAULT_HIDDEN_DIMS),
        "parameters": parameter_count(model),
        "memory_kb": memory_kb,
        "train_class_counts": class_counts.tolist(),
        "class_weights": class_weights.tolist(),
        "validation": {k: v for k, v in best_metrics.items() if k != "predictions"},
        "test": {k: v for k, v in test_metrics.items() if k != "predictions"},
        "training_seconds": training_seconds,
    }

    print(
        f"Saved models/model_{bits}bit.pt  "
        f"| test_acc: {test_metrics['accuracy']:.2f}%  "
        f"| test_f1: {test_metrics['f1_weighted']:.4f}"
    )
    return summary


def main():
    args = parse_args()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Shared hidden dims: {DEFAULT_HIDDEN_DIMS}")

    X_train_full, y_train_full, X_test, y_test = load_arrays()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=args.val_size,
        random_state=SEED,
        stratify=y_train_full,
    )

    print(f"Train split: {X_train.shape}  labels: {y_train.shape}")
    print(f"Val split:   {X_val.shape}  labels: {y_val.shape}")
    print(f"Test split:  {X_test.shape}  labels: {y_test.shape}")

    os.makedirs("results", exist_ok=True)

    summaries = []
    for bits in args.bits:
        summaries.append(
            train_one_model(
                bits,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                args,
                device,
            )
        )

    summary_path = "results/training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": SEED,
                "hidden_dims": list(DEFAULT_HIDDEN_DIMS),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "val_size": args.val_size,
                "patience": args.patience,
                "models": summaries,
            },
            handle,
            indent=2,
        )

    print(f"\nSaved summary -> {summary_path}")
    print(f"\n{'Bits':<8}{'Accuracy':>12}{'F1':>12}{'RecallAtk':>14}{'MemoryKB':>12}")
    print("-" * 58)
    for item in summaries:
        print(
            f"{item['bits']:<8}"
            f"{item['test']['accuracy']:>11.2f}%"
            f"{item['test']['f1_weighted']:>12.4f}"
            f"{item['test']['recall_attack']:>14.4f}"
            f"{item['memory_kb']:>12.2f}"
        )


if __name__ == "__main__":
    main()
