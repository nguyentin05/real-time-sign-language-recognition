"""
demo_webcam.py — Nhận dạng ngôn ngữ ký hiệu real-time từ webcam

Usage:
    python src/demo/demo_webcam.py
    python src/demo/demo_webcam.py --model checkpoints/model_best.pth --threshold 0.4
"""

import os
import sys
import json
import time
import argparse
import collections
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp

# ── Project root path setup ─────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model.slrnet import SLRNet
from src.data.extract_keypoints import mediapipe_detection, draw_landmarks, extract_keypoints

# ─── MediaPipe ──────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic

# ─── Constants ──────────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 30
THRESHOLD        = 0.30
TOP_K            = 3
IDLE_THRESHOLD   = 0.003
PREDICT_INTERVAL = 1
STABILITY_COUNT  = 1
EMA_ALPHA        = 0.65

# Keypoint index ranges (1662-dim)
POSE_START, POSE_END = 0, 132
LH_START,  LH_END   = 1536, 1599
RH_START,  RH_END   = 1599, 1662

# UI Colors
COL_ACCENT = (200, 120, 30)
COL_GREEN  = (50, 200, 50)
COL_RED    = (50, 50, 220)
COL_YELLOW = (30, 200, 220)
COL_WHITE  = (240, 240, 240)
COL_GRAY   = (120, 120, 120)


# ─── Model Loading ──────────────────────────────────────────────────────────

def _resolve_model_path(path: str) -> str:
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
            print(f"  [WARN] Label map: {len(idx_map)} classes vs model {num_classes}.")
            return {i: f"class_{i}" for i in range(num_classes)}
        return idx_map
    print(f"  [WARN] Label map not found: {path}. Using default.")
    return {i: f"class_{i}" for i in range(num_classes)}


class SLRPredictor:
    def __init__(self, checkpoint, label_map="data/label_map.json", device="auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
        resolved = _resolve_model_path(checkpoint)
        state = torch.load(resolved, map_location=self.device, weights_only=True)
        self.num_classes = state["fc.3.weight"].shape[0]
        self.model = SLRNet(input_size=1662, num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.idx_to_label = load_label_map(label_map, self.num_classes)
        print(f"[SLRPredictor] Ready. Device: {self.device}")


# ─── Motion Detection ───────────────────────────────────────────────────────

def detect_motion(sequence, window=10):
    """Hand-focused motion detection. Returns weighted score."""
    if len(sequence) < window:
        return 0.0
    recent = sequence[-window:]
    hand_diffs, pose_diffs = [], []
    for i in range(1, len(recent)):
        prev, curr = recent[i - 1], recent[i]
        lh = np.abs(curr[LH_START:LH_END] - prev[LH_START:LH_END]).mean()
        rh = np.abs(curr[RH_START:RH_END] - prev[RH_START:RH_END]).mean()
        hand_diffs.append(max(lh, rh))
        pose_diffs.append(np.abs(curr[POSE_START:POSE_END] - prev[POSE_START:POSE_END]).mean())
    hand = float(np.mean(hand_diffs)) if hand_diffs else 0.0
    pose = float(np.mean(pose_diffs)) if pose_diffs else 0.0
    return 0.7 * hand + 0.3 * pose


# ─── Keypoint Normalization ─────────────────────────────────────────────────

def normalize_keypoints(keypoints):
    """Per-frame z-score normalization for non-zero keypoints.
    
    Normalizes keypoint values to zero-mean, unit-variance to reduce
    sensitivity to camera distance and position. Only normalizes non-zero
    values to preserve the zero-padding for undetected landmarks.
    """
    kp = keypoints.copy()
    nonzero = kp != 0
    if nonzero.any():
        mean = kp[nonzero].mean()
        std = kp[nonzero].std()
        if std > 1e-6:
            kp[nonzero] = (kp[nonzero] - mean) / std
    return kp


# ─── UI Drawing ─────────────────────────────────────────────────────────────

def draw_ui(frame, predictions, seq_len, fps, threshold, is_idle, motion_level, stable_label=None):
    h, w = frame.shape[:2]
    pw = 320

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (pw, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Header
    cv2.putText(frame, "SLRNet", (10, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1, COL_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, "Sign Language Recognition", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GRAY, 1, cv2.LINE_AA)
    cv2.line(frame, (10, 70), (pw - 10, 70), COL_ACCENT, 1)

    # Buffer bar
    by = 82
    cv2.putText(frame, f"Buffer: {seq_len}/{SEQUENCE_LENGTH}", (10, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GRAY, 1, cv2.LINE_AA)
    filled = int((seq_len / SEQUENCE_LENGTH) * (pw - 20))
    cv2.rectangle(frame, (10, by+6), (pw-10, by+14), (50,50,50), -1)
    cv2.rectangle(frame, (10, by+6), (10+filled, by+14), COL_GREEN if seq_len == SEQUENCE_LENGTH else COL_YELLOW, -1)

    # Motion
    my = by + 26
    m_text = "IDLE - No movement" if is_idle else f"Motion: {motion_level:.4f}"
    cv2.putText(frame, m_text, (10, my), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_RED if is_idle else COL_GREEN, 1, cv2.LINE_AA)

    # Stable label
    if stable_label and not is_idle:
        cv2.putText(frame, "DETECTED:", (10, my+20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, stable_label.upper(), (10, my+48), cv2.FONT_HERSHEY_DUPLEX, 0.9, COL_GREEN, 2, cv2.LINE_AA)
        sy = my + 70
    else:
        sy = my + 25

    # Predictions
    cv2.putText(frame, "TOP PREDICTIONS", (10, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_GRAY, 1, cv2.LINE_AA)
    if is_idle:
        cv2.putText(frame, "Waiting for sign...", (14, sy+30), cv2.FONT_HERSHEY_DUPLEX, 0.6, COL_GRAY, 1, cv2.LINE_AA)
    else:
        for rank, (label, prob) in enumerate(predictions[:TOP_K]):
            yp = sy + rank * 80
            is_top = rank == 0
            if is_top and prob >= threshold:
                cv2.rectangle(frame, (8, yp+2), (pw-8, yp+68), COL_ACCENT, 1)
            fs, ft = (0.75, 2) if is_top else (0.55, 1)
            tc = COL_WHITE if is_top else (180, 180, 180)
            dl = label.upper() if prob >= threshold * 0.5 else "..."
            cv2.putText(frame, f"#{rank+1} {dl}", (14, yp+26), cv2.FONT_HERSHEY_DUPLEX, fs, tc, ft, cv2.LINE_AA)
            cv2.putText(frame, f"{prob*100:.1f}%", (14, yp+48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tc, 1, cv2.LINE_AA)
            bx1, bx2 = 90, pw - 12
            cv2.rectangle(frame, (bx1, yp+52), (bx2, yp+62), (50,50,50), -1)
            bf = int((bx2 - bx1) * prob)
            if bf > 0:
                bc = COL_GREEN if prob >= 0.8 else COL_YELLOW if prob >= 0.5 else COL_RED
                cv2.rectangle(frame, (bx1, yp+52), (bx1+bf, yp+62), bc, -1)

    # Footer
    fy = h - 50
    cv2.line(frame, (10, fy-10), (pw-10, fy-10), COL_ACCENT, 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, fy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, "Q:Quit  R:Reset  S:Screenshot", (10, fy+28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_GRAY, 1, cv2.LINE_AA)
    return frame


# ─── Webcam Demo ────────────────────────────────────────────────────────────

def webcam_demo(model_path, label_map_path, threshold, camera_idx, show_landmarks, save_dir="demo_output"):
    predictor = SLRPredictor(model_path, label_map_path)
    sequence, predictions = [], [("...", 0.0)] * TOP_K
    ema_probs, stable_label = None, None
    stability_buf = collections.deque(maxlen=STABILITY_COUNT)
    frame_count, is_idle, motion_level, idle_cooldown = 0, True, 0.0, 0

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("\n=== Demo đang chạy. Nhấn Q để thoát ===")
    prev_time = time.time()

    with mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)

            if results is not None and show_landmarks:
                draw_landmarks(image, results)

            keypoints = extract_keypoints(results)
            keypoints = normalize_keypoints(keypoints)
            sequence.append(keypoints)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)
            frame_count += 1

            # Motion detection
            motion_level = detect_motion(sequence, window=10)
            was_idle = is_idle
            is_idle = motion_level < IDLE_THRESHOLD

            if was_idle and not is_idle:
                # Reset buffer: discard stale idle frames, keep only recent ones
                keep = min(len(sequence), 10)
                sequence = sequence[-keep:]
                idle_cooldown = 2
            if idle_cooldown > 0:
                idle_cooldown -= 1

            # Prediction
            if len(sequence) == SEQUENCE_LENGTH and frame_count % PREDICT_INTERVAL == 0:
                if is_idle:
                    ema_probs = None
                    stability_buf.clear()
                    stable_label = None
                    predictions = [("...", 0.0)] * TOP_K
                elif idle_cooldown <= 0:
                    X = torch.tensor(np.array([sequence], dtype=np.float32), device=predictor.device)
                    with torch.no_grad():
                        probs = predictor.model.predict_proba(X)[0].cpu().numpy()

                    ema_probs = probs.copy() if ema_probs is None else EMA_ALPHA * probs + (1 - EMA_ALPHA) * ema_probs
                    top_idx = np.argsort(ema_probs)[::-1][:TOP_K]
                    current = [(predictor.idx_to_label[i], float(ema_probs[i])) for i in top_idx]

                    top_label, top_conf = current[0]
                    if top_conf >= threshold:
                        stability_buf.append(top_label)
                    else:
                        stability_buf.clear()

                    stable_label = stability_buf[0] if (len(stability_buf) >= STABILITY_COUNT and
                                                        len(set(stability_buf)) == 1) else None
                    predictions = current if top_conf >= threshold * 0.5 else [("...", 0.0)] * TOP_K

            # Display
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            image = draw_ui(image, predictions, len(sequence), fps, threshold, is_idle, motion_level, stable_label)
            cv2.imshow("SLRNet — Real-Time Sign Language Recognition", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                sequence.clear()
                ema_probs = stable_label = None
                stability_buf.clear()
                predictions = [("...", 0.0)] * TOP_K
                is_idle, frame_count, idle_cooldown = True, 0, 0
                print("  [RESET] Sequence buffer cleared")
            elif key == ord("s"):
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f"screenshot_{int(time.time())}.png")
                cv2.imwrite(fname, image)
                print(f"  [SAVE] {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("Demo đã đóng.")


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLRNet Real-Time Webcam Demo")
    parser.add_argument("--model",        default=os.path.join("checkpoints", "model_best.pth"))
    parser.add_argument("--label_map",    default=os.path.join("data", "label_map.json"))
    parser.add_argument("--threshold",    type=float, default=THRESHOLD)
    parser.add_argument("--camera",       type=int,   default=0)
    parser.add_argument("--no_landmarks", action="store_true")
    args = parser.parse_args()

    webcam_demo(args.model, args.label_map, args.threshold, args.camera, not args.no_landmarks)
