import torch
import torch.nn as nn

class SLRNetModel(nn.Module):
    def __init__(self, input_size=1629, num_classes=10, hidden_size=64, num_layers=3):
        super(SLRNetModel, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=128, 
                            num_layers=2, 
                            batch_first=True,
                            dropout=0.2)
        
        self.lstm2 = nn.LSTM(input_size=128, 
                             hidden_size=64, 
                             num_layers=1, 
                             batch_first=True)
        
        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)
        
        x = x[:, -1, :]
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = SLRNetModel(num_classes=15)
    dummy_input = torch.randn(4, 30, 1629)
    output = model(dummy_input)