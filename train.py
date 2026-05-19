"""
SLRNet Training Script

Trains the CNN-LSTM model on WLASL sign language dataset.
Supports both Kaggle and local environments.

Usage:
    python train.py --data_dir processed --output_dir outputs --epochs 50
    python train.py --data_dir /kaggle/input/.../processed --output_dir /kaggle/working
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import SLRNet


# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR   = "processed"
DEFAULT_OUTPUT_DIR = "outputs"
BATCH_SIZE         = 32
EPOCHS             = 50
LR                 = 1e-3
PATIENCE           = 10
SAVE_EVERY         = 10


def load_data(data_dir):
    """Load preprocessed numpy arrays and class names."""
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_val   = np.load(os.path.join(data_dir, "y_val.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))
    class_names = np.load(os.path.join(data_dir, "class_names.npy"),
                          allow_pickle=True)

    # Convert one-hot labels to class indices if needed
    if y_train.ndim > 1:
        y_train = y_train.argmax(axis=1)
        y_val   = y_val.argmax(axis=1)
        y_test  = y_test.argmax(axis=1)

    y_train = y_train.astype(np.int64)
    y_val   = y_val.astype(np.int64)
    y_test  = y_test.astype(np.int64)

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names


def make_loaders(X_train, X_val, y_train, y_val, batch_size=BATCH_SIZE):
    """Create PyTorch DataLoaders from numpy arrays."""
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss, correct = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        correct    += (output.argmax(1) == y_batch).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            correct    += (output.argmax(1) == y_batch).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def save_label_map(class_names, output_dir):
    """Save label_map.json (class_name → index) for inference."""
    label_map = {str(name): idx for idx, name in enumerate(class_names)}
    path = os.path.join(output_dir, "label_map.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"  → Saved label map: {path}")
    return label_map


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ───────────────────────────────────────────────────────
    print(f"\nLoading data from: {args.data_dir}")
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = \
        load_data(args.data_dir)

    INPUT_SIZE  = X_train.shape[2]
    SEQ_LEN     = X_train.shape[1]
    NUM_CLASSES = len(class_names)

    print(f"X_train : {X_train.shape}")
    print(f"X_val   : {X_val.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train : {y_train.shape} | dtype: {y_train.dtype} "
          f"| range: [{y_train.min()}, {y_train.max()}]")
    print(f"Classes : {NUM_CLASSES} → {list(class_names)}")

    # ── DataLoaders ─────────────────────────────────────────────────────
    train_loader, val_loader = make_loaders(
        X_train, X_val, y_train, y_val, args.batch_size
    )
    print(f"\nTrain batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ───────────────────────────────────────────────────────────
    model = SLRNet(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(device)

    total  = sum(p.numel() for p in model.parameters())
    cnn_p  = sum(p.numel() for p in model.cnn.parameters())
    lstm_p = sum(p.numel() for n, p in model.named_parameters() if "lstm" in n)
    fc_p   = sum(p.numel() for n, p in model.named_parameters() if "fc" in n)
    print(f"\nTotal params : {total:,}")
    print(f"  CNN        : {cnn_p:,}")
    print(f"  LSTM       : {lstm_p:,}")
    print(f"  FC         : {fc_p:,}")

    # ── Training setup ──────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Save label map for inference
    save_label_map(class_names, args.output_dir)

    # Save training config for reproducibility
    train_config = {
        "input_size": int(INPUT_SIZE),
        "seq_len": int(SEQ_LEN),
        "num_classes": int(NUM_CLASSES),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "data_dir": args.data_dir,
        "class_names": list(class_names),
    }
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)
    print(f"  → Saved training config: {config_path}")

    # ── Training loop ───────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} "
          f"| {'Val Loss':>9} | {'Val Acc':>8}")
    print("-" * 58)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} "
              f"| {val_loss:>9.4f} | {val_acc:>8.4f}")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            path = os.path.join(args.output_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), path)
            print(f"         → Saved checkpoint: {path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_path = os.path.join(args.output_dir, "model_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"         → Saved best (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest val_loss: {best_val_loss:.4f}")

    # ── Save training history ───────────────────────────────────────────
    history_path = os.path.join(args.output_dir, "history.npy")
    np.save(history_path, history)
    print(f"Training history saved to: {history_path}")

    # ── Evaluate on test set ────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)

    model.load_state_dict(torch.load(best_path, map_location=device,
                                      weights_only=True))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), args.batch_size):
            batch = torch.tensor(X_test[i:i + args.batch_size],
                                 dtype=torch.float32).to(device)
            preds = model(batch).argmax(1).cpu().numpy()
            all_preds.append(preds)
    y_pred = np.concatenate(all_preds)

    # Classification report
    try:
        from sklearn.metrics import classification_report
        print(classification_report(y_test, y_pred,
                                    target_names=class_names, zero_division=0))
    except ImportError:
        from collections import Counter
        correct = (y_pred == y_test).sum()
        print(f"Test accuracy: {correct / len(y_test):.4f} ({correct}/{len(y_test)})")

    # Save confusion matrix plot
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix — SLRNet")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")
    except ImportError:
        print("(matplotlib/seaborn not available — skipping confusion matrix plot)")

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SLRNet sign language recognition model"
    )
    parser.add_argument("--data_dir",   type=str,   default=DEFAULT_DATA_DIR,
                        help="Directory containing preprocessed .npy files")
    parser.add_argument("--output_dir", type=str,   default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save model checkpoints and outputs")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--patience",   type=int,   default=PATIENCE,
                        help="Early stopping patience")
    parser.add_argument("--save_every", type=int,   default=SAVE_EVERY,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    main(args)
