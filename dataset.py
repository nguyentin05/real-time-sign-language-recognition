"""
dataset.py -- WLASL Sign Language Recognition
Dataset loader with frame sampling + augmentation

Supports two modes:
  1. Video mode: loads raw .mp4 files, outputs (C, T, H, W) tensors
  2. Keypoint mode: loads raw .mp4 files, extracts MediaPipe keypoints,
     outputs (T, 1662) tensors compatible with SLRNet

Usage:
    # Video mode (for 3D Conv models)
    train_loader, val_loader, test_loader = get_dataloaders("data/raw_videos", mode="video")

    # Keypoint mode (for SLRNet CNN-LSTM)
    train_loader, val_loader, test_loader = get_dataloaders("data/raw_videos", mode="keypoint")
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


# ─── Default labels (loaded dynamically from outputs/label_map.json) ────────
_DEFAULT_LABEL_MAP = os.path.join(os.path.dirname(__file__), "outputs", "label_map.json")

def _load_labels(label_map_path=_DEFAULT_LABEL_MAP):
    """Load labels from label_map.json. Falls back to default if not found."""
    if os.path.isfile(label_map_path):
        with open(label_map_path, encoding="utf-8") as f:
            label_map = json.load(f)
        sorted_items = sorted(label_map.items(), key=lambda x: x[1])
        return [name for name, _ in sorted_items]
    else:
        print(f"  [WARN] Label map not found: {label_map_path}. Using default.")
        return [
            "FINISH", "FRIEND", "MANY", "NIGHT", "READ",
            "START", "WATER", "WHERE", "WRITE", "YOU"
        ]

LABELS = _load_labels()
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# Hyperparameters
SEQ_LEN   = 32     # number of frames after sampling
IMG_SIZE  = 112    # resize to 112x112
MEAN      = [0.485, 0.456, 0.406]
STD       = [0.229, 0.224, 0.225]

# Keypoint mode settings
KP_SEQ_LEN = 30    # SLRNet expects 30 frames
KP_DIM     = 1662  # MediaPipe holistic keypoint dimension


# ─── Utility ────────────────────────────────────────────────────────────────

def uniform_sample(frames: list, n: int) -> list:
    """Sample exactly n frames uniformly from the list."""
    total = len(frames)
    if total == 0:
        return []
    if total <= n:
        # Repeat last frame to fill
        indices = list(range(total))
        while len(indices) < n:
            indices.append(indices[-1])
    else:
        indices = [int(i * total / n) for i in range(n)]
    return [frames[i] for i in indices]


def load_video_frames(path: str, seq_len: int = SEQ_LEN, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Read video -> sample `seq_len` frames uniformly -> resize.
    Returns array (seq_len, H, W, 3) uint8.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        raw_frames.append(frame)
    cap.release()

    if len(raw_frames) == 0:
        return np.zeros((seq_len, img_size, img_size, 3), dtype=np.uint8)

    sampled = uniform_sample(raw_frames, seq_len)
    return np.stack(sampled, axis=0)  # (T, H, W, 3)


def load_video_keypoints(path: str, seq_len: int = KP_SEQ_LEN) -> np.ndarray:
    """
    Read video -> extract MediaPipe holistic keypoints per frame -> sample.
    Returns array (seq_len, 1662) float32.
    """
    try:
        from extract_keypoints import extract_keypoints, mediapipe_detection
        import mediapipe as mp
    except ImportError:
        raise ImportError(
            "Keypoint mode requires mediapipe and extract_keypoints.py. "
            "Install mediapipe: pip install mediapipe"
        )

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    all_keypoints = []

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image, results = mediapipe_detection(frame, holistic)
            kp = extract_keypoints(results)
            all_keypoints.append(kp)
    cap.release()

    if len(all_keypoints) == 0:
        return np.zeros((seq_len, KP_DIM), dtype=np.float32)

    sampled = uniform_sample(all_keypoints, seq_len)
    return np.stack(sampled, axis=0).astype(np.float32)  # (T, 1662)


# ─── Transforms ─────────────────────────────────────────────────────────────

def get_transforms(train: bool):
    """
    Spatial augmentation applied consistently across all frames in a clip.
    """
    if train:
        return T.Compose([
            # NOTE: No RandomHorizontalFlip — flipping changes left/right hand
            # meaning in sign language, which would corrupt training data
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])


# ─── Dataset classes ────────────────────────────────────────────────────────

class WLASLVideoDataset(Dataset):
    """
    Video-frame based dataset.
    Output shape: (C, T, H, W) - suitable for 3D Conv models.

    Directory structure:
        root/
          accident/
            xxxxx.mp4
            ...
          bed/
            ...
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        seq_len: int = SEQ_LEN,
        img_size: int = IMG_SIZE,
        val_ratio: float = 0.15,
        test_ratio: float = 0.10,
        seed: int = 42,
    ):
        self.root     = root
        self.seq_len  = seq_len
        self.img_size = img_size
        self.transform = get_transforms(train=(split == "train"))

        # Collect all samples
        all_samples = []
        for label in LABELS:
            label_dir = os.path.join(root, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in sorted(os.listdir(label_dir)):
                if fname.endswith(".mp4"):
                    all_samples.append((os.path.join(label_dir, fname), LABEL2IDX[label]))

        # Shuffle and split
        rng = np.random.default_rng(seed)
        idxs = rng.permutation(len(all_samples))
        n_test = int(len(all_samples) * test_ratio)
        n_val  = int(len(all_samples) * val_ratio)

        if split == "test":
            chosen = idxs[:n_test]
        elif split == "val":
            chosen = idxs[n_test:n_test + n_val]
        else:  # train
            chosen = idxs[n_test + n_val:]

        self.samples = [all_samples[i] for i in chosen]
        print(f"[WLASLVideoDataset] split={split}: {len(self.samples)} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # (T, H, W, 3) -> apply transform per frame
        frames = load_video_frames(path, self.seq_len, self.img_size)

        # Same spatial augmentation for all frames in the clip
        seed = np.random.randint(0, 2**31)
        tensor_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            torch.manual_seed(seed)
            t = self.transform(img)
            tensor_frames.append(t)

        # Stack -> (T, C, H, W)
        clip = torch.stack(tensor_frames, dim=0)

        # Transpose to (C, T, H, W) for 3D Conv
        clip = clip.permute(1, 0, 2, 3)

        return clip, label


class WLASLKeypointDataset(Dataset):
    """
    Keypoint-based dataset compatible with SLRNet.
    Output shape: (T, 1662) - suitable for SLRNet CNN-LSTM model.

    Directory structure:
        root/
          accident/
            xxxxx.mp4
            ...
          bed/
            ...
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        seq_len: int = KP_SEQ_LEN,
        val_ratio: float = 0.15,
        test_ratio: float = 0.10,
        seed: int = 42,
    ):
        self.root    = root
        self.seq_len = seq_len

        # Collect all samples
        all_samples = []
        for label in LABELS:
            label_dir = os.path.join(root, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in sorted(os.listdir(label_dir)):
                if fname.endswith(".mp4"):
                    all_samples.append((os.path.join(label_dir, fname), LABEL2IDX[label]))

        # Shuffle and split
        rng = np.random.default_rng(seed)
        idxs = rng.permutation(len(all_samples))
        n_test = int(len(all_samples) * test_ratio)
        n_val  = int(len(all_samples) * val_ratio)

        if split == "test":
            chosen = idxs[:n_test]
        elif split == "val":
            chosen = idxs[n_test:n_test + n_val]
        else:  # train
            chosen = idxs[n_test + n_val:]

        self.samples = [all_samples[i] for i in chosen]
        print(f"[WLASLKeypointDataset] split={split}: {len(self.samples)} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        keypoints = load_video_keypoints(path, self.seq_len)
        return torch.tensor(keypoints, dtype=torch.float32), label


# ─── DataLoader factory ─────────────────────────────────────────────────────

def get_dataloaders(
    root: str,
    mode: str = "keypoint",
    batch_size: int = 8,
    num_workers: int = 0,
    seq_len: int = None,
    img_size: int = IMG_SIZE,
):
    """
    Create train/val/test DataLoaders.

    Args:
        root: path to directory with label subfolders containing .mp4 files
        mode: "video" for 3D Conv models, "keypoint" for SLRNet
        batch_size: batch size
        num_workers: DataLoader workers (0 for Windows compatibility)
        seq_len: override default sequence length
        img_size: frame size (video mode only)
    """
    if seq_len is None:
        seq_len = KP_SEQ_LEN if mode == "keypoint" else SEQ_LEN

    if mode == "video":
        DatasetClass = WLASLVideoDataset
        kwargs = dict(seq_len=seq_len, img_size=img_size)
    elif mode == "keypoint":
        DatasetClass = WLASLKeypointDataset
        kwargs = dict(seq_len=seq_len)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'video' or 'keypoint'.")

    train_ds = DatasetClass(root, split="train", **kwargs)
    val_ds   = DatasetClass(root, split="val",   **kwargs)
    test_ds  = DatasetClass(root, split="test",  **kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test WLASL Dataset")
    parser.add_argument("--root", type=str, default="data/raw_videos",
                        help="Path to video directory")
    parser.add_argument("--mode", type=str, default="keypoint",
                        choices=["video", "keypoint"],
                        help="Dataset mode: video or keypoint")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(
        args.root, mode=args.mode, batch_size=args.batch_size
    )

    clip, label = next(iter(train_loader))
    print(f"Clip shape : {clip.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Labels     : {[IDX2LABEL[l.item()] for l in label]}")
