import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_on_test(model, X_test, y_test, class_names, device, batch_size=32, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.tensor(X_test[i:i + batch_size], dtype=torch.float32).to(device)
            preds = model(batch).argmax(1).cpu().numpy()
            all_preds.append(preds)
    y_pred = np.concatenate(all_preds)

    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(report)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    return y_pred, cm
