import torch.nn as nn


class SLRNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.frame_fc = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.lstm1 = nn.LSTM(256, 128, batch_first=True)
        self.dropout1 = nn.Dropout(0.4)

        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.dropout2 = nn.Dropout(0.4)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        B, T, F = x.shape

        x = x.reshape(B * T, 1, F)
        x = self.cnn(x)
        x = x.view(B * T, -1)
        x = self.frame_fc(x)
        x = x.view(B, T, -1)

        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x = x.mean(dim=1)

        return self.fc(x)
