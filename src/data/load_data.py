import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_and_prepare_data(data_dir="data/processed", output_dir="checkpoints"):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    with open(os.path.join(data_dir, "..", "label_map.json"), "r") as f:
        label_map = json.load(f)
    sorted_items = sorted(label_map.items(), key=lambda item: item[1])
    class_names = np.array([name for name, idx in sorted_items])

    NUM_CLASSES = len(class_names)

    if y_train.ndim > 1:
        y_train = y_train.argmax(axis=1)
        y_val = y_val.argmax(axis=1)
        y_test = y_test.argmax(axis=1)

    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)

    train_data = X_train.reshape(-1, X_train.shape[2])
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0) + 1e-6

    X_train = np.clip((X_train - mean) / std, -5, 5)
    X_val = np.clip((X_val - mean) / std, -5, 5)
    X_test = np.clip((X_test - mean) / std, -5, 5)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "norm_mean.npy"), mean)
    np.save(os.path.join(output_dir, "norm_std.npy"), std)

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names, NUM_CLASSES


def make_loaders(X_train, X_val, y_train, y_val, batch_size=32):
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
