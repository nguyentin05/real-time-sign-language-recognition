import torch

CONFIG = {
    "data_dir": "data/processed",
    "output_dir": "checkpoints",
    "results_dir": "results",
    "batch_size": 32,
    "epochs": 150,
    "lr": 0.001,
    "weight_decay": 5e-4,
    "patience": 25,
    "scheduler_patience": 15,
    "save_every": 20,
    "num_classes": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
