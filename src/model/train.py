import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model.slrnet import SLRNetModel

def dummy_training():
    X_dummy = torch.randn(300, 30, 1629)
    y_dummy = torch.randint(0, 15, (300,))
    
    dataset = TensorDataset(X_dummy, y_dummy)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SLRNetModel(num_classes=15)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    dummy_training()