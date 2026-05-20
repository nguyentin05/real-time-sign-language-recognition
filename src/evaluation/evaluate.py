"""
evaluate.py — Đánh giá mô hình SLRNet

Gồm:
  1. Integration tests (kiểm tra architecture, save/load, training loop)
  2. Folder evaluation (đánh giá trên thư mục video)

Usage:
    python src/evaluation/evaluate.py --mode test
    python src/evaluation/evaluate.py --mode eval --data_root data/raw_videos
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Project root path setup ─────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model.slrnet import SLRNet

# ── Configuration ───────────────────────────────────────────────────────────
NUM_CLASSES   = 10
INPUT_SIZE    = 1662
SEQ_LEN       = 30
BATCH_SIZE    = 4
NUM_SAMPLES   = 32


# ═══════════════════════════════════════════════════════════════════════════
#  Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_model_architecture():
    """Test 1: Model architecture and shapes."""
    print("=" * 60)
    print("TEST 1: Model Architecture")
    print("=" * 60)

    model = SLRNet(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)

    total  = sum(p.numel() for p in model.parameters())
    cnn_p  = sum(p.numel() for p in model.cnn.parameters())
    lstm_p = sum(p.numel() for n, p in model.named_parameters() if "lstm" in n)
    fc_p   = sum(p.numel() for n, p in model.named_parameters() if "fc" in n)

    print(f"  Total params : {total:,}")
    print(f"    CNN        : {cnn_p:,}")
    print(f"    LSTM       : {lstm_p:,}")
    print(f"    FC         : {fc_p:,}")

    assert total == 416_682,  f"Expected 416,682 params, got {total:,}"
    assert cnn_p == 35_648,   f"Expected 35,648 CNN params, got {cnn_p:,}"

    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
    out = model(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES), \
        f"Expected output shape ({BATCH_SIZE}, {NUM_CLASSES}), got {out.shape}"

    probs = model.predict_proba(x)
    assert probs.shape == (BATCH_SIZE, NUM_CLASSES)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5)

    print("  [PASS] All architecture checks passed!")
    return True


def test_save_load():
    """Test 2: Model save/load consistency."""
    print("\n" + "=" * 60)
    print("TEST 2: Save / Load Consistency")
    print("=" * 60)

    model = SLRNet(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
    model.eval()
    x = torch.randn(1, SEQ_LEN, INPUT_SIZE)

    with torch.no_grad():
        out_before = model(x)

    tmp = os.path.join(ROOT, "_test_model_tmp.pt")
    torch.save(model.state_dict(), tmp)

    model2 = SLRNet(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
    model2.load_state_dict(torch.load(tmp, map_location="cpu", weights_only=True))
    model2.eval()

    with torch.no_grad():
        out_after = model2(x)

    os.remove(tmp)

    assert torch.allclose(out_before, out_after, atol=1e-6), \
        "Outputs differ after save/load!"
    print("  [PASS] Save/load produces identical outputs!")
    return True


def test_training_loop():
    """Test 3: Simulated training loop (1 epoch on synthetic data)."""
    print("\n" + "=" * 60)
    print("TEST 3: Training Loop (synthetic data)")
    print("=" * 60)

    X = np.random.randn(NUM_SAMPLES, SEQ_LEN, INPUT_SIZE).astype(np.float32)
    y = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES).astype(np.int64)

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SLRNet(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    total_loss, correct = 0.0, 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        correct += (output.argmax(1) == y_batch).sum().item()

    avg_loss = total_loss / NUM_SAMPLES
    acc = correct / NUM_SAMPLES
    print(f"  Epoch 1 — Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    assert avg_loss > 0, "Loss should be positive"
    assert 0 <= acc <= 1, "Accuracy should be in [0, 1]"
    print("  [PASS] Training loop works correctly!")
    return True


def test_label_map():
    """Test 4: Label map generation and loading."""
    print("\n" + "=" * 60)
    print("TEST 4: Label Map Generation")
    print("=" * 60)

    class_names = [
        'FINISH', 'FRIEND', 'MANY', 'NIGHT', 'READ',
        'START', 'WATER', 'WHERE', 'WRITE', 'YOU'
    ]

    label_map = {name: idx for idx, name in enumerate(class_names)}
    idx_to_label = {v: k for k, v in label_map.items()}

    tmp = os.path.join(ROOT, "_test_label_map.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    with open(tmp, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    os.remove(tmp)

    assert loaded == label_map, "Label map mismatch after save/load"
    assert len(idx_to_label) == NUM_CLASSES
    assert idx_to_label[0] == 'FINISH'
    assert idx_to_label[9] == 'YOU'
    print(f"  [PASS] Label map: {NUM_CLASSES} classes, correctly saved/loaded!")
    return True


def test_keypoint_dimensions():
    """Test 5: Verify keypoint extraction output dimensions (no camera)."""
    print("\n" + "=" * 60)
    print("TEST 5: Keypoint Vector Dimensions")
    print("=" * 60)

    pose_size  = 33 * 4    # 132
    face_size  = 468 * 3   # 1404
    lh_size    = 21 * 3    # 63
    rh_size    = 21 * 3    # 63
    total_size = pose_size + face_size + lh_size + rh_size

    print(f"  Pose      : 33 × 4 = {pose_size}")
    print(f"  Face      : 468 × 3 = {face_size}")
    print(f"  Left hand : 21 × 3 = {lh_size}")
    print(f"  Right hand: 21 × 3 = {rh_size}")
    print(f"  Total     : {total_size}")

    assert total_size == 1662, f"Expected 1662, got {total_size}"

    keypoints = np.zeros(total_size)
    assert keypoints.shape == (1662,)

    model = SLRNet(input_size=1662, num_classes=10)
    model.eval()
    sequence = np.stack([keypoints] * SEQ_LEN)  # (30, 1662)
    X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = model.predict_proba(X)
    assert probs.shape == (1, 10)
    print("  [PASS] Keypoint dimensions match model input!")
    return True


def run_tests():
    """Run all integration tests."""
    print("\n[TEST] SLRNet Integration Test Suite")
    print("=" * 60)

    tests = [
        ("Architecture",     test_model_architecture),
        ("Save/Load",        test_save_load),
        ("Training Loop",    test_training_loop),
        ("Label Map",        test_label_map),
        ("Keypoint Dims",    test_keypoint_dimensions),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  [FAIL] FAILED: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} — {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print(">>> All tests passed!")
    else:
        print(">>> Some tests FAILED.")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
#  Folder Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_model_path(path: str) -> str:
    """Try multiple paths/extensions to find the model checkpoint."""
    for p in [path, path.replace(".pth", ".pt"), path.replace(".pt", ".pth"),
              os.path.join(ROOT, "checkpoints", os.path.basename(path))]:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Model file not found: {path}")


def load_label_map(path: str, num_classes: int) -> dict:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            lm = json.load(f)
        idx_map = {v: k for k, v in lm.items()}
        if len(idx_map) != num_classes:
            print(f"  [WARN] Label map has {len(idx_map)} classes but model expects {num_classes}.")
            return {i: f"class_{i}" for i in range(num_classes)}
        return idx_map
    print(f"  [WARN] Label map not found: {path}. Using default.")
    return {i: f"class_{i}" for i in range(num_classes)}


def evaluate_folder(checkpoint_path, label_map_path, data_root):
    """Evaluate model on a folder of videos organized by label."""
    resolved = _resolve_model_path(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(resolved, map_location=device, weights_only=True)
    nc = state["fc.3.weight"].shape[0]
    model = SLRNet(input_size=1662, num_classes=nc).to(device)
    model.load_state_dict(state)
    model.eval()

    idx_to_label = load_label_map(label_map_path, nc)
    print(f"[Evaluate] Model loaded. Device: {device}, Classes: {nc}")

    from src.data.preprocess import load_video_keypoints

    correct, total, per_class = 0, 0, {}

    for label in idx_to_label.values():
        label_dir = os.path.join(data_root, label)
        if not os.path.isdir(label_dir):
            continue
        per_class[label] = {"correct": 0, "total": 0}
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(".mp4"):
                continue
            try:
                kp = load_video_keypoints(os.path.join(label_dir, fname), 30)
                clip = torch.tensor(kp, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_label = idx_to_label[model(clip).argmax(1).item()]
                total += 1
                per_class[label]["total"] += 1
                if pred_label == label:
                    correct += 1
                    per_class[label]["correct"] += 1
            except Exception as e:
                print(f"Error: {e}")

    if total == 0:
        print("No videos found.")
        return
    print(f"\nOverall accuracy: {correct}/{total} = {100*correct/total:.1f}%\n")
    for word, s in sorted(per_class.items()):
        if s["total"] > 0:
            print(f"  {word:>15}: {100*s['correct']/s['total']:5.1f}%  ({s['correct']}/{s['total']})")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLRNet Evaluation & Tests")
    parser.add_argument("--mode", default="test", choices=["test", "eval"],
                        help="test: integration tests | eval: folder evaluation")
    parser.add_argument("--model", default="model_best.pth", help="Model checkpoint path")
    parser.add_argument("--label_map", default=os.path.join("data", "label_map.json"))
    parser.add_argument("--data_root", default=os.path.join("data", "raw_videos"))
    args = parser.parse_args()

    if args.mode == "test":
        run_tests()
    else:
        evaluate_folder(args.model, args.label_map, args.data_root)
