"""
inference.py — Nhận dạng ngôn ngữ ký hiệu từ video / webcam
Gộp tính năng demo webcam và API dự đoán từ các script cũ.
"""
import os
import json
import time
import argparse
import collections
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp

from model import SLRNet
from extract_keypoints import mediapipe_detection, draw_landmarks, extract_keypoints

# Constants
SEQUENCE_LENGTH = 30
THRESHOLD       = 0.55    # Minimum confidence to accept a prediction
TOP_K           = 3       # Number of top predictions to display
SMOOTHING       = 7       # Number of recent predictions for smoothing
IDLE_THRESHOLD  = 0.002   # Motion threshold to detect if user is idle
PREDICT_INTERVAL = 2      # Predict every N frames (when buffer is full)

# Colors
COL_BG          = (20, 20, 20)
COL_ACCENT      = (200, 120, 30)     
COL_GREEN       = (50, 200, 50)
COL_RED         = (50, 50, 220)
COL_YELLOW      = (30, 200, 220)
COL_WHITE       = (240, 240, 240)
COL_GRAY        = (120, 120, 120)

def _resolve_model_path(model_path: str) -> str:
    if os.path.isfile(model_path): return model_path
    base, ext = os.path.splitext(model_path)
    alt_ext = ".pt" if ext == ".pth" else ".pth"
    alt_path = base + alt_ext
    if os.path.isfile(alt_path): return alt_path
    outputs_path = os.path.join("outputs", os.path.basename(model_path))
    if os.path.isfile(outputs_path): return outputs_path
    raise FileNotFoundError(f"Model file not found: {model_path}")

def load_label_map(label_map_path: str, num_classes: int) -> dict:
    if os.path.isfile(label_map_path):
        with open(label_map_path, encoding="utf-8") as f:
            label_map = json.load(f)
        idx_to_label = {v: k for k, v in label_map.items()}
        if len(idx_to_label) != num_classes:
            print(f"  [WARN] Label map has {len(idx_to_label)} classes but model expects {num_classes}.")
            idx_to_label = {i: f"class_{i}" for i in range(num_classes)}
    else:
        print(f"  [WARN] Label map not found: {label_map_path}. Using default.")
        idx_to_label = {i: f"class_{i}" for i in range(num_classes)}
    return idx_to_label

class SLRPredictor:
    def __init__(self, checkpoint_path: str, label_map_path: str = "outputs/label_map.json", device="auto", seq_len=30):
        self.seq_len = seq_len
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        resolved_path = _resolve_model_path(checkpoint_path)
        
        # Load state dict to determine num_classes (dynamically from final layer)
        state_dict = torch.load(resolved_path, map_location=self.device, weights_only=True)
        self.num_classes = state_dict["fc.3.weight"].shape[0]
        
        self.model = SLRNet(input_size=1662, num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.idx_to_label = load_label_map(label_map_path, self.num_classes)
        print(f"[SLRPredictor] Ready. Device: {self.device}")

    @torch.no_grad()
    def _run(self, clip: torch.Tensor) -> dict:
        logits = self.model(clip)
        probs = F.softmax(logits, dim=1)[0]
        k = min(5, self.num_classes)
        top_vals, top_idxs = probs.topk(k)
        top5 = [{"label": self.idx_to_label[i.item()], "confidence": round(v.item(), 4)} for v, i in zip(top_vals, top_idxs)]
        return {"label": top5[0]["label"], "confidence": top5[0]["confidence"], "top5": top5}

    def predict_video(self, video_path: str) -> dict:
        from dataset import load_video_keypoints
        keypoints = load_video_keypoints(video_path, self.seq_len)
        clip = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(self.device)
        return self._run(clip)


def detect_motion(sequence: list, window: int = 10) -> float:
    """
    Detect motion level from recent keypoints.
    Compares the mean absolute difference between recent frames.
    Higher value = more motion = likely signing.
    """
    if len(sequence) < window:
        return 0.0
    recent = sequence[-window:]
    diffs = []
    for i in range(1, len(recent)):
        diff = np.abs(recent[i] - recent[i-1]).mean()
        diffs.append(diff)
    return float(np.mean(diffs))


def draw_ui(frame: np.ndarray, predictions: list, sequence_len: int, fps: float, 
            threshold: float, is_idle: bool, motion_level: float):
    h, w = frame.shape[:2]
    panel_w = 320
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "SLRNet", (10, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1, COL_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, "Sign Language Recognition", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GRAY, 1, cv2.LINE_AA)
    cv2.line(frame, (10, 70), (panel_w - 10, 70), COL_ACCENT, 1)

    bar_y = 82
    cv2.putText(frame, f"Buffer: {sequence_len}/{SEQUENCE_LENGTH}", (10, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GRAY, 1, cv2.LINE_AA)
    bar_filled = int((sequence_len / SEQUENCE_LENGTH) * (panel_w - 20))
    cv2.rectangle(frame, (10, bar_y + 6), (panel_w - 10, bar_y + 14), (50, 50, 50), -1)
    color_buf = COL_GREEN if sequence_len == SEQUENCE_LENGTH else COL_YELLOW
    cv2.rectangle(frame, (10, bar_y + 6), (10 + bar_filled, bar_y + 14), color_buf, -1)

    # Motion indicator
    motion_y = bar_y + 26
    motion_text = "IDLE - No movement" if is_idle else f"Motion: {motion_level:.4f}"
    motion_color = COL_RED if is_idle else COL_GREEN
    cv2.putText(frame, motion_text, (10, motion_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, motion_color, 1, cv2.LINE_AA)

    start_y = motion_y + 25
    cv2.putText(frame, "TOP PREDICTIONS", (10, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_GRAY, 1, cv2.LINE_AA)

    if is_idle:
        # Show idle state
        cv2.putText(frame, "Waiting for sign...", (14, start_y + 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, COL_GRAY, 1, cv2.LINE_AA)
    else:
        for rank, (label, prob) in enumerate(predictions[:TOP_K]):
            y_pos   = start_y + rank * 80
            is_top  = rank == 0

            if is_top and prob >= threshold:
                cv2.rectangle(frame, (8, y_pos + 2), (panel_w - 8, y_pos + 68), COL_ACCENT, 1)

            font_scale = 0.75 if is_top else 0.55
            font_thick = 2    if is_top else 1
            text_color = COL_WHITE if is_top else (180, 180, 180)

            display_label = label.upper() if prob >= threshold * 0.5 else "..."
            cv2.putText(frame, f"#{rank+1} {display_label}", (14, y_pos + 26), cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, font_thick, cv2.LINE_AA)
            cv2.putText(frame, f"{prob*100:.1f}%", (14, y_pos + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

            bar_x1, bar_x2 = 90, panel_w - 12
            bar_y1, bar_y2 = y_pos + 52, y_pos + 62
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)

            filled = int((bar_x2 - bar_x1) * prob)
            if filled > 0:
                if prob >= 0.8: bar_color = COL_GREEN
                elif prob >= 0.5: bar_color = COL_YELLOW
                else: bar_color = COL_RED
                cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + filled, bar_y2), bar_color, -1)

    status_y = h - 50
    cv2.line(frame, (10, status_y - 10), (panel_w - 10, status_y - 10), COL_ACCENT, 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, status_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, "Q:Quit  R:Reset  S:Screenshot", (10, status_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_GRAY, 1, cv2.LINE_AA)

    return frame

def webcam_demo(model_path: str, label_map_path: str, threshold: float, camera_idx: int, show_landmarks: bool, save_dir: str = "screenshots"):
    predictor = SLRPredictor(model_path, label_map_path)
    sequence = []                                  
    predictions = [("...", 0.0)] * TOP_K
    
    # Smoothing: store (label, probability_array) tuples for weighted averaging
    smooth_probs = collections.deque(maxlen=SMOOTHING)
    
    # Frame counter for prediction interval
    frame_count = 0
    is_idle = True
    motion_level = 0.0

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n=== Demo đang chạy. Nhấn Q để thoát ===")
    prev_time = time.time()

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)  
            image, results = mediapipe_detection(frame, holistic)
            if show_landmarks: draw_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)
            
            frame_count += 1

            # Detect motion to determine if user is signing
            motion_level = detect_motion(sequence, window=8)
            is_idle = motion_level < IDLE_THRESHOLD

            if len(sequence) == SEQUENCE_LENGTH and frame_count % PREDICT_INTERVAL == 0:
                if is_idle:
                    # User is idle — clear smoothing buffer, show waiting state
                    smooth_probs.clear()
                    predictions = [("...", 0.0)] * TOP_K
                else:
                    # Run inference
                    X = torch.tensor(np.array([sequence], dtype=np.float32), device=predictor.device)  
                    with torch.no_grad():
                        probs = predictor.model.predict_proba(X)[0].cpu().numpy()
                    
                    # Add to smoothing buffer
                    smooth_probs.append(probs)
                    
                    # Weighted averaging: more recent predictions get higher weight
                    if len(smooth_probs) >= 3:
                        weights = np.linspace(0.5, 1.0, len(smooth_probs))
                        weights /= weights.sum()
                        avg_probs = np.zeros_like(probs)
                        for w, p in zip(weights, smooth_probs):
                            avg_probs += w * p
                    else:
                        avg_probs = probs
                    
                    # Get top-K from smoothed probabilities
                    top_indices = np.argsort(avg_probs)[::-1][:TOP_K]
                    current_preds = [(predictor.idx_to_label[i], float(avg_probs[i])) for i in top_indices]
                    
                    # Only show prediction if top confidence exceeds a minimum
                    if current_preds[0][1] >= threshold * 0.3:
                        predictions = current_preds
                    else:
                        predictions = [("...", 0.0)] * TOP_K

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            image = draw_ui(image, predictions, len(sequence), fps, threshold, is_idle, motion_level)
            cv2.imshow("SLRNet — Real-Time Sign Language Recognition", image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"): break
            elif key == ord("r"):
                sequence.clear()
                smooth_probs.clear()
                predictions = [("...", 0.0)] * TOP_K
                is_idle = True
                frame_count = 0
                print("  [RESET] Sequence buffer cleared")
            elif key == ord("s"):
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f"screenshot_{int(time.time())}.png")
                cv2.imwrite(fname, image)
                print(f"  [SAVE] {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("Demo đã đóng.")

def evaluate_folder(checkpoint_path: str, label_map_path: str, data_root: str):
    predictor = SLRPredictor(checkpoint_path, label_map_path)
    correct, total = 0, 0
    per_class = {}
    
    for label in predictor.idx_to_label.values():
        label_dir = os.path.join(data_root, label)
        if not os.path.isdir(label_dir): continue
        per_class[label] = {"correct": 0, "total": 0}
        
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(".mp4"): continue
            path = os.path.join(label_dir, fname)
            try:
                result = predictor.predict_video(path)
                total += 1
                per_class[label]["total"] += 1
                if result["label"] == label:
                    correct += 1
                    per_class[label]["correct"] += 1
            except Exception as e:
                print(f"Error processing {path}: {e}")

    if total == 0:
        print("No videos found to evaluate.")
        return
        
    print(f"\nOverall accuracy: {correct}/{total} = {100*correct/total:.1f}%\n")
    print("Per-class:")
    for word, stats in sorted(per_class.items()):
        if stats['total'] > 0:
            acc = 100 * stats["correct"] / stats["total"]
            print(f"  {word:>15}: {acc:5.1f}%  ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLRNet Inference & Real-Time Demo")
    parser.add_argument("--mode",         type=str,   default="webcam", choices=["webcam", "eval"], help="Mode: webcam or eval")
    parser.add_argument("--model",        type=str,   default="model_best.pth", help="Path to model checkpoint")
    parser.add_argument("--label_map",    type=str,   default="outputs/label_map.json", help="Path to label map")
    parser.add_argument("--data_root",    type=str,   default="data/raw_videos", help="Root directory for evaluation")
    parser.add_argument("--threshold",    type=float, default=THRESHOLD, help="Confidence threshold")
    parser.add_argument("--camera",       type=int,   default=0, help="Camera index")
    parser.add_argument("--no_landmarks", action="store_true", help="Turn off MediaPipe landmarks")
    args = parser.parse_args()

    if args.mode == "webcam":
        webcam_demo(args.model, args.label_map, args.threshold, args.camera, not args.no_landmarks)
    elif args.mode == "eval":
        evaluate_folder(args.model, args.label_map, args.data_root)
