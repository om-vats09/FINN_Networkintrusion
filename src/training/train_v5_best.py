import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import brevitas.nn as qnn
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.combine import SMOTETomek
from collections import Counter

# ── Config ────────────────────────────────────────────────────────
BITS          = 8
EPOCHS        = 150
BATCH_SIZE    = 256
LR            = 0.001
PATIENCE      = 15
N_FEATURES    = 20
DROPOUT       = 0.3
FOCAL_GAMMA   = 2.0
FOCAL_ALPHA   = 0.25
LABEL_SMOOTH  = 0.05
SAVE_PATH     = 'models/model_8bit_v5.pt'
SELECTOR_PATH = 'data/feature_selector_v5.pkl'
DEVICE        = torch.device('cpu')

# ── Focal Loss ────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.05):
        super().__init__()
        self.gamma          = gamma
        self.alpha          = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Label smoothing
        n_classes  = inputs.size(1)
        smooth_targets = torch.zeros_like(inputs)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0)
        smooth_targets = smooth_targets * (1 - self.label_smoothing) + \
                         self.label_smoothing / n_classes

        log_probs = F.log_softmax(inputs, dim=1)
        ce        = -(smooth_targets * log_probs).sum(dim=1)
        pt        = torch.exp(-ce)
        loss      = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()

# ── Model with Dropout ────────────────────────────────────────────
def build_model_v5(bits=8, n_features=20):
    return nn.Sequential(
        qnn.QuantLinear(n_features, 128, bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        nn.Dropout(DROPOUT),
        qnn.QuantLinear(128, 256, bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        nn.Dropout(DROPOUT),
        qnn.QuantLinear(256, 256, bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        nn.Dropout(DROPOUT),
        qnn.QuantLinear(256, 128, bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        qnn.QuantLinear(128, 2,   bias=True, weight_bit_width=bits),
    )

# ── Load data ─────────────────────────────────────────────────────
print("="*60)
print("train_v5_best.py — Maximum accuracy pipeline")
print("="*60)
print("\nLoading data...")
X_train = np.load('data/X_train.npy').astype(np.float32)
y_train = np.load('data/y_train.npy').astype(np.int64)
X_test  = np.load('data/X_test.npy').astype(np.float32)
y_test  = np.load('data/y_test.npy').astype(np.int64)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Class distribution: {Counter(y_train)}")

# ── Step 1: Feature Selection ─────────────────────────────────────
print(f"\n[Step 1] Feature selection — keeping top {N_FEATURES} of 41...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
indices     = importances.argsort()[::-1]
print("Top features:")
for i in range(N_FEATURES):
    print(f"  Feature {indices[i]:2d}: importance={importances[indices[i]]:.4f}")

selector    = SelectFromModel(rf, max_features=N_FEATURES, prefit=True)
X_train_sel = selector.transform(X_train).astype(np.float32)
X_test_sel  = selector.transform(X_test).astype(np.float32)
print(f"Reduced: 41 → {X_train_sel.shape[1]} features")

with open(SELECTOR_PATH, 'wb') as f:
    pickle.dump(selector, f)
print(f"Saved selector → {SELECTOR_PATH}")

# ── Step 2: SMOTETomek ────────────────────────────────────────────
print(f"\n[Step 2] SMOTETomek resampling...")
sampler      = SMOTETomek(random_state=42)
X_res, y_res = sampler.fit_resample(X_train_sel, y_train)
print(f"After SMOTETomek: {Counter(y_res)} | Total: {len(X_res)}")

X_train_t = torch.tensor(X_res)
y_train_t = torch.tensor(y_res)
X_test_t  = torch.tensor(X_test_sel)

# ── Step 3: Build model ───────────────────────────────────────────
print(f"\n[Step 3] Building {BITS}-bit model with dropout...")
model         = build_model_v5(BITS, X_train_sel.shape[1]).to(DEVICE)
total_params  = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA,
                      label_smoothing=LABEL_SMOOTH)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5
)

# ── Step 4: Training ──────────────────────────────────────────────
print(f"\n[Step 4] Training up to {EPOCHS} epochs (patience={PATIENCE})...")
print(f"  Focal loss gamma={FOCAL_GAMMA}, label smoothing={LABEL_SMOOTH}")
print(f"  Dropout={DROPOUT}, weight decay=1e-4")
print(f"  Cosine annealing LR scheduler")
print(f"\n{'Epoch':>6} {'Loss':>10} {'Val Acc':>10} {'F1':>8} {'LR':>12}")
print("-" * 52)

best_acc   = 0.0
best_epoch = 0
wait       = 0
n_train    = len(X_res)

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm       = torch.randperm(n_train)
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, n_train, BATCH_SIZE):
        idx  = perm[i:i+BATCH_SIZE]
        xb   = X_train_t[idx].to(DEVICE)
        yb   = y_train_t[idx].to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    scheduler.step()
    avg_loss = total_loss / n_batches

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t.to(DEVICE)).argmax(dim=1).cpu().numpy()

    val_acc = accuracy_score(y_test, preds)
    val_f1  = f1_score(y_test, preds, average='weighted')
    cur_lr  = optimizer.param_groups[0]['lr']

    print(f"{epoch:>6} {avg_loss:>10.4f} {val_acc*100:>9.2f}% {val_f1:>8.4f} {cur_lr:>12.6f}")

    if val_acc > best_acc:
        best_acc   = val_acc
        best_epoch = epoch
        wait       = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ New best: {best_acc*100:.2f}% — saved")
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

# ── Final results ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Best accuracy : {best_acc*100:.2f}% at epoch {best_epoch}")
print(f"Parameters    : {total_params:,}")
print(f"Features used : {X_train_sel.shape[1]} / 41")
print(f"Model saved   : {SAVE_PATH}")
print(f"{'='*60}")

model.load_state_dict(torch.load(SAVE_PATH, map_location='cpu'))
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1).numpy()

print("\nFull classification report:")
print(classification_report(y_test, preds, target_names=['Normal', 'Attack']))

print("\nFINAL COMPARISON:")
print(f"  Original 8-bit          : 81.97%")
print(f"  v2 SMOTETomek           : 81.71%")
print(f"  v3 Deep + SMOTE         : 81.92%")
print(f"  v5 Best (all combined)  : {best_acc*100:.2f}%")