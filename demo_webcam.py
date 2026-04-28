import os
import json
import time
import argparse
import collections
import numpy as np
import cv2
import torch
import mediapipe as mp

from model import SLRNet
from extract_keypoints import mediapipe_detection, draw_landmarks, extract_keypoints


MODEL_PATH      = "outputs/model_best.pt"
LABEL_MAP_PATH  = "processed/label_map.json"
SEQUENCE_LENGTH = 30
THRESHOLD       = 0.6     
TOP_K           = 3       
SMOOTHING       = 5       

COL_BG          = (20, 20, 20)
COL_ACCENT      = (200, 120, 30)     
COL_GREEN       = (50, 200, 50)
COL_RED         = (50, 50, 220)
COL_YELLOW      = (30, 200, 220)
COL_WHITE       = (240, 240, 240)
COL_GRAY        = (120, 120, 120)


def load_model_and_labels(model_path: str, label_map_path: str):
    with open(label_map_path, encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}
    num_classes  = len(idx_to_label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SLRNet(input_size=1662, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"  → Model loaded: {num_classes} classes | Device: {device}")
    return model, idx_to_label, device


def draw_ui(frame: np.ndarray, predictions: list, sequence_len: int, fps: float, threshold: float):
    h, w = frame.shape[:2]

    panel_w = 320
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "SLRNet", (10, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, COL_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, "Sign Language Recognition", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GRAY, 1, cv2.LINE_AA)

    cv2.line(frame, (10, 70), (panel_w - 10, 70), COL_ACCENT, 1)

    bar_y = 82
    cv2.putText(frame, f"Buffer: {sequence_len}/{SEQUENCE_LENGTH}", (10, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GRAY, 1, cv2.LINE_AA)
    bar_filled = int((sequence_len / SEQUENCE_LENGTH) * (panel_w - 20))
    cv2.rectangle(frame, (10, bar_y + 6), (panel_w - 10, bar_y + 14), (50, 50, 50), -1)
    color_buf = COL_GREEN if sequence_len == SEQUENCE_LENGTH else COL_YELLOW
    cv2.rectangle(frame, (10, bar_y + 6), (10 + bar_filled, bar_y + 14), color_buf, -1)

    start_y = 115
    cv2.putText(frame, "TOP PREDICTIONS", (10, start_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_GRAY, 1, cv2.LINE_AA)

    for rank, (label, prob) in enumerate(predictions[:TOP_K]):
        y_pos   = start_y + rank * 80
        is_top  = rank == 0

        if is_top and prob >= threshold:
            cv2.rectangle(frame, (8, y_pos + 2), (panel_w - 8, y_pos + 68), COL_ACCENT, 1)

        font_scale = 0.75 if is_top else 0.55
        font_thick = 2    if is_top else 1
        text_color = COL_WHITE if is_top else (180, 180, 180)

        cv2.putText(frame, f"#{rank+1} {label.upper()}", (14, y_pos + 26),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, font_thick, cv2.LINE_AA)
        cv2.putText(frame, f"{prob*100:.1f}%", (14, y_pos + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

        bar_x1, bar_x2 = 90, panel_w - 12
        bar_y1, bar_y2 = y_pos + 52, y_pos + 62
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)

        filled = int((bar_x2 - bar_x1) * prob)
        if prob >= 0.8:
            bar_color = COL_GREEN
        elif prob >= 0.5:
            bar_color = COL_YELLOW
        else:
            bar_color = COL_RED

        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + filled, bar_y2), bar_color, -1)

    status_y = h - 50
    cv2.line(frame, (10, status_y - 10), (panel_w - 10, status_y - 10), COL_ACCENT, 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, status_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, "Q:Quit  R:Reset  S:Screenshot", (10, status_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_GRAY, 1, cv2.LINE_AA)

    return frame


def run_demo(
    model_path:     str   = MODEL_PATH,
    label_map_path: str   = LABEL_MAP_PATH,
    threshold:      float = THRESHOLD,
    camera_idx:     int   = 0,
    show_landmarks: bool  = True,
    save_dir:       str   = "screenshots",
):
    model, idx_to_label, device = load_model_and_labels(model_path, label_map_path)

    sequence    = []                                  
    predictions = [(name, 0.0) for name in list(idx_to_label.values())[:TOP_K]]
    smooth_buf  = collections.deque(maxlen=SMOOTHING) 

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n=== Demo đang chạy. Nhấn Q để thoát ===")
    prev_time = time.time()

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  

            image, results = mediapipe_detection(frame, holistic)
            if show_landmarks:
                draw_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            if len(sequence) == SEQUENCE_LENGTH:
                X = torch.tensor(
                    np.array([sequence], dtype=np.float32),
                    device=device
                )  

                with torch.no_grad():
                    probs = model.predict_proba(X)[0].cpu().numpy()  


                top_indices = np.argsort(probs)[::-1][:TOP_K]
                current_preds = [(idx_to_label[i], float(probs[i])) for i in top_indices]

                smooth_buf.append(current_preds[0][0])
                most_common = collections.Counter(smooth_buf).most_common(1)[0][0]
                predictions = current_preds

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            image = draw_ui(image, predictions, len(sequence), fps, threshold)

            cv2.imshow("SLRNet — Real-Time Sign Language Recognition", image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                sequence.clear()
                smooth_buf.clear()
                print("  [RESET] Sequence buffer cleared")
            elif key == ord("s"):
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f"screenshot_{int(time.time())}.png")
                cv2.imwrite(fname, image)
                print(f"  [SAVE] {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("Demo đã đóng.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLRNet Real-Time Demo")
    parser.add_argument("--model",          type=str,   default=MODEL_PATH)
    parser.add_argument("--label_map",      type=str,   default=LABEL_MAP_PATH)
    parser.add_argument("--threshold",      type=float, default=THRESHOLD)
    parser.add_argument("--camera",         type=int,   default=0)
    parser.add_argument("--no_landmarks",   action="store_true",
                        help="Tắt vẽ MediaPipe landmarks")
    args = parser.parse_args()

    run_demo(
        model_path     = args.model,
        label_map_path = args.label_map,
        threshold      = args.threshold,
        camera_idx     = args.camera,
        show_landmarks = not args.no_landmarks,
    )
