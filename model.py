"""
SLRNet - Sign Language Recognition Network

Architecture: CNN (spatial) + Frame FC + LSTM (temporal) + FC (classification)
- CNN: 1D convolutions extract spatial features (128x4 per frame)
- Bridge: FC layer to reduce dimension before LSTM
- LSTM: 2-layer stacked LSTM models temporal dependencies across frames
- FC: Sequential with dropout for final classification
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
        num_classes : Number of sign language classes (default 10)
    """

    def __init__(self, input_size=1662, num_classes=10):
        super().__init__()

        # ── CNN: spatial feature extraction ──────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1,   out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32,  out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64,  out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=4),  # → (B*T, 128, 4)
        )

        # ── Frame-level FC bridge ────────────────────────────────────────
        self.frame_fc = nn.Sequential(
            nn.Linear(128 * 4, 256),  # 512 → 256
        )

        # ── LSTM: temporal sequence modeling ─────────────────────────────
        self.lstm1    = nn.LSTM(input_size=256, hidden_size=128,
                                num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2    = nn.LSTM(input_size=128, hidden_size=64,
                                num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)

        # ── FC: classification with dropout ──────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.
        """
        B, T, F = x.shape

        # ── Step 1: CNN per frame ────────────────────────────────────────
        x = x.reshape(B * T, 1, F)    # (B*T, 1, F)
        x = self.cnn(x)               # (B*T, 128, 4)
        x = x.reshape(B * T, -1)      # (B*T, 512)

        # ── Step 2: Frame-level FC bridge ────────────────────────────────
        x = self.frame_fc(x)          # (B*T, 256)
        x = x.reshape(B, T, -1)       # (B, T, 256)

        # ── Step 3: LSTM temporal modeling ───────────────────────────────
        x, _ = self.lstm1(x)          # (B, T, 128)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)          # (B, T, 64)
        x = self.dropout2(x)

        x = x[:, -1, :]               # (B, 64) — take last hidden state

        # ── Step 4: FC classification ────────────────────────────────────
        x = self.fc(x)                # (B, num_classes)
        return x

    def predict_proba(self, x):
        """
        Predict class probabilities.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLRNet(input_size=1662, num_classes=10).to(device)

    total  = sum(p.numel() for p in model.parameters())
    print(f"Device       : {device}")
    print(f"Total params : {total:,}")

    dummy = torch.randn(2, 30, 1662).to(device)
    out = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")

    probs = model.predict_proba(dummy)
    print(f"Probs shape  : {probs.shape}")
    print(f"Probs sum    : {probs.sum(dim=-1)}")
