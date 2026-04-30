"""
SLRNet - Sign Language Recognition Network

Architecture: CNN (spatial) + LSTM (temporal) + FC (classification)
- CNN: 1D convolutions extract spatial features from each frame's landmark vector
- LSTM: 3-layer stacked LSTM models temporal dependencies across frames
- FC: Fully connected layers for final classification

Reference: SLRNet paper (sections D, E, F)
"""

import torch
import torch.nn as nn


class SLRNet(nn.Module):
    """
    SLRNet: CNN-LSTM model for sign language recognition.

    Input : (batch, seq_len, input_size)  e.g. (B, 30, 1662)
    Output: (batch, num_classes)          logits (raw, no softmax)

    Args:
        input_size  : Feature dimension per frame (default 1662 for MediaPipe holistic)
        num_classes : Number of sign language classes
    """

    def __init__(self, input_size=1662, num_classes=30):
        super().__init__()

        # ── CNN: spatial feature extraction (Section D) ─────────────────
        # Each frame's landmark vector is treated as a 1D signal
        # in_channels=1 because each frame is a single 1D vector of size input_size
        self.cnn = nn.Sequential(
            # Block 1: learn local landmark group features
            nn.Conv1d(in_channels=1,   out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),          # input_size → input_size//2

            # Block 2: learn more complex features
            nn.Conv1d(in_channels=32,  out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),          # → input_size//4

            # Block 3: compress into feature vector
            nn.Conv1d(in_channels=64,  out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1),  # → 1 (regardless of input_size)
        )
        # CNN output per frame: Ft ∈ R^128

        # ── LSTM: temporal sequence modeling (Section E) ────────────────
        self.lstm1    = nn.LSTM(input_size=128, hidden_size=64,
                                num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2    = nn.LSTM(input_size=64,  hidden_size=128,
                                num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)

        self.lstm3    = nn.LSTM(input_size=128, hidden_size=64,
                                num_layers=1, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)

        # ── FC: classification (Section F) ──────────────────────────────
        self.fc1  = nn.Linear(64, 64)
        self.fc2  = nn.Linear(64, 32)
        self.fc3  = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, T, F) where
               B = batch size, T = sequence length, F = feature dim

        Returns:
            Tensor of shape (B, num_classes) — raw logits (no softmax)
        """
        B, T, F = x.shape

        # ── Step 1: CNN applied in parallel across all frames ───────────
        x = x.reshape(B * T, 1, F)    # (B*T, 1, F)
        x = self.cnn(x)               # (B*T, 128, 1)
        x = x.squeeze(-1)             # (B*T, 128)
        x = x.reshape(B, T, -1)       # (B, T, 128) = [F1,...,FT]

        # ── Step 2: LSTM temporal modeling ──────────────────────────────
        x, _ = self.lstm1(x)          # (B, T, 64)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)          # (B, T, 128)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)          # (B, T, 64)
        x = self.dropout3(x)

        x = x[:, -1, :]               # (B, 64) — take last hidden state

        # ── Step 3: FC classification ───────────────────────────────────
        x = self.relu(self.fc1(x))    # (B, 64)
        x = self.relu(self.fc2(x))    # (B, 32)
        x = self.fc3(x)               # (B, num_classes) — softmax in loss
        return x

    def predict_proba(self, x):
        """
        Predict class probabilities (used by demo_webcam.py).

        Args:
            x: Tensor of shape (B, T, F)

        Returns:
            Tensor of shape (B, num_classes) — softmax probabilities
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLRNet(input_size=1662, num_classes=30).to(device)

    total  = sum(p.numel() for p in model.parameters())
    cnn_p  = sum(p.numel() for p in model.cnn.parameters())
    lstm_p = sum(p.numel() for n, p in model.named_parameters() if "lstm" in n)
    fc_p   = sum(p.numel() for n, p in model.named_parameters() if "fc" in n)

    print(f"Device       : {device}")
    print(f"Total params : {total:,}")
    print(f"  CNN        : {cnn_p:,}")
    print(f"  LSTM       : {lstm_p:,}")
    print(f"  FC         : {fc_p:,}")

    # Test forward pass
    dummy = torch.randn(2, 30, 1662).to(device)
    out = model(dummy)
    print(f"\nInput shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")

    probs = model.predict_proba(dummy)
    print(f"Probs shape  : {probs.shape}")
    print(f"Probs sum    : {probs.sum(dim=-1)}")  # should be [1.0, 1.0]
