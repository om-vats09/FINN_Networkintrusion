import torch
import torch.nn as nn
import numpy as np
from model import build_model
import os

torch.manual_seed(42)

device = torch.device("cpu")
print(f"Using device: {device}")

print("Loading preprocessed data...")
X_train = torch.tensor(np.load('data/X_train.npy')).to(device)
y_train = torch.tensor(np.load('data/y_train.npy'), dtype=torch.long).to(device)
X_test  = torch.tensor(np.load('data/X_test.npy')).to(device)
y_test  = torch.tensor(np.load('data/y_test.npy'),  dtype=torch.long).to(device)

print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}   y_test:  {y_test.shape}")

def train_model(bits, epochs=30, batch_size=256, lr=0.001):
    print(f"\n{'='*50}")
    print(f"  Training {bits}-bit quantized model")
    print(f"{'='*50}")

    model     = build_model(bits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n         = X_train.shape[0]

    for epoch in range(epochs):
        model.train()
        perm       = torch.randperm(n)
        total_loss = 0
        batches    = 0

        for i in range(0, n, batch_size):
            idx    = perm[i:i+batch_size]
            out    = model(X_train[idx])
            loss   = criterion(out, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches    += 1

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / batches
            model.eval()
            with torch.no_grad():
                preds    = model(X_test).argmax(dim=1)
                correct  = (preds == y_test).sum().item()
                val_acc  = correct / len(y_test) * 100
            print(f"  epoch {epoch+1:2d}/{epochs}  loss: {avg_loss:.4f}  val_acc: {val_acc:.2f}%")

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/model_{bits}bit.pt')
    print(f"  Saved → models/model_{bits}bit.pt")
    return model

for bits in [2, 4, 8]:
    train_model(bits)

print("\nAll models trained and saved.")