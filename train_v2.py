import torch
import torch.nn as nn
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.combine import SMOTETomek
from collections import Counter
from model import build_model

# ── Config ────────────────────────────────────────────────────────
BITS         = 8
EPOCHS       = 100
BATCH_SIZE   = 256
LR           = 0.001
PATIENCE     = 10       # stop if no improvement for 10 epochs
SAVE_PATH    = f'models/model_{BITS}bit_v2.pt'
DEVICE       = torch.device('cpu')

# ── Load data ─────────────────────────────────────────────────────
print("Loading data...")
X_train = np.load('data/X_train.npy').astype(np.float32)
y_train = np.load('data/y_train.npy').astype(np.int64)
X_test  = np.load('data/X_test.npy').astype(np.float32)
y_test  = np.load('data/y_test.npy').astype(np.int64)

print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
print(f"Class distribution before SMOTE: {Counter(y_train)}")

# ── SMOTETomek ────────────────────────────────────────────────────
print("\nApplying SMOTETomek resampling...")
sampler = SMOTETomek(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
print(f"Class distribution after SMOTETomek: {Counter(y_resampled)}")
print(f"Dataset size: {len(X_train)} → {len(X_resampled)}")

X_train_t = torch.tensor(X_resampled)
y_train_t = torch.tensor(y_resampled)
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test)

# ── Model ─────────────────────────────────────────────────────────
print(f"\nBuilding {BITS}-bit quantized model...")
model     = build_model(BITS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5

)

# ── Training ──────────────────────────────────────────────────────
best_acc   = 0.0
best_epoch = 0
wait       = 0

print(f"\nTraining for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...")
print(f"{'Epoch':>6} {'Loss':>10} {'Val Acc':>10} {'F1':>8} {'LR':>12}")
print("-" * 52)

n_train = len(X_resampled)

for epoch in range(1, EPOCHS + 1):
    # ── Train
    model.train()
    perm  = torch.randperm(n_train)
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, n_train, BATCH_SIZE):
        idx   = perm[i:i+BATCH_SIZE]
        xb    = X_train_t[idx].to(DEVICE)
        yb    = y_train_t[idx].to(DEVICE)
        optimizer.zero_grad()
        loss  = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    avg_loss = total_loss / n_batches

    # ── Validate
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t.to(DEVICE)).argmax(dim=1).cpu().numpy()

    val_acc = accuracy_score(y_test, preds)
    val_f1  = f1_score(y_test, preds, average='weighted')
    cur_lr  = optimizer.param_groups[0]['lr']

    print(f"{epoch:>6} {avg_loss:>10.4f} {val_acc*100:>9.2f}% {val_f1:>8.4f} {cur_lr:>12.6f}")

    scheduler.step(val_acc)

    # ── Early stopping
    if val_acc > best_acc:
        best_acc   = val_acc
        best_epoch = epoch
        wait       = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ New best: {best_acc*100:.2f}% — saved to {SAVE_PATH}")
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} — no improvement for {PATIENCE} epochs.")
            break

# ── Final evaluation ──────────────────────────────────────────────
print(f"\n{'='*52}")
print(f"Best accuracy : {best_acc*100:.2f}% at epoch {best_epoch}")
print(f"Model saved   : {SAVE_PATH}")
print(f"{'='*52}")

model.load_state_dict(torch.load(SAVE_PATH, map_location='cpu'))
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1).numpy()

print("\nFull classification report:")
print(classification_report(y_test, preds, target_names=['Normal', 'Attack']))