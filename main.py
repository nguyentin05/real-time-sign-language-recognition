# main.py
import torch
import numpy as np
from config import CONFIG
from src.data.load_data import load_and_prepare_data, make_loaders
from src.model.slrnet import SLRNetModel
from src.model.train import run_training
from src.evaluation.evaluate import evaluate_on_test


def main():
    print(f"Device: {CONFIG['device']}")
    device = torch.device(CONFIG['device'])

    X_train, X_val, X_test, y_train, y_val, y_test, class_names, NUM_CLASSES = load_and_prepare_data(
        data_dir=CONFIG["data_dir"],
        output_dir=CONFIG["output_dir"]
    )
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"Classes: {list(class_names)}")

    train_loader, val_loader = make_loaders(X_train, X_val, y_train, y_val, batch_size=CONFIG["batch_size"])
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = SLRNetModel(num_classes=NUM_CLASSES).to(device)

    total = sum(p.numel() for p in model.parameters())
    cnn_p = sum(p.numel() for p in model.cnn.parameters())
    lstm_p = sum(p.numel() for n, p in model.named_parameters() if "lstm" in n)
    fc_p = sum(p.numel() for n, p in model.named_parameters() if "fc" in n)
    print(f"Total params: {total:,} | CNN: {cnn_p:,} | LSTM: {lstm_p:,} | FC: {fc_p:,}")

    model, history = run_training(
        model, train_loader, val_loader, CONFIG, device, output_dir=CONFIG["output_dir"]
    )

    model.load_state_dict(torch.load(f"{CONFIG['output_dir']}/model_best.pth", map_location=device))
    y_pred, cm = evaluate_on_test(
        model, X_test, y_test, class_names, device,
        batch_size=CONFIG["batch_size"], output_dir=CONFIG["results_dir"]
    )

    np.save(f"{CONFIG['results_dir']}/history.npy", history)
    print("Hoàn tất!")


if __name__ == "__main__":
    main()
