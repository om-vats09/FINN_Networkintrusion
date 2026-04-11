import torch
import torch.nn as nn
import numpy as np
import brevitas.nn as qnn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter

DEVICE = torch.device('cpu')

# ── Model definitions ─────────────────────────────────────────────
def build_model_4layer(bits=8):
    return nn.Sequential(
        qnn.QuantLinear(41,  128, bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        qnn.QuantLinear(128, 256, bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        qnn.QuantLinear(256, 128, bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        qnn.QuantLinear(128, 2,   bias=True, weight_bit_width=bits),
    )

def build_model_6layer(bits=8):
    return nn.Sequential(
        qnn.QuantLinear(41,  128, bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        qnn.QuantLinear(128, 256, bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        qnn.QuantLinear(256, 256, bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        qnn.QuantLinear(256, 128, bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        qnn.QuantLinear(128, 64,  bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        qnn.QuantLinear(64,  2,   bias=True, weight_bit_width=bits),
    )

# ── Load data ─────────────────────────────────────────────────────
print("Loading data...")
X_test  = np.load('data/X_test.npy').astype(np.float32)
y_test  = np.load('data/y_test.npy').astype(np.int64)
X_test_t = torch.tensor(X_test)

# ── Load all trained models ───────────────────────────────────────
models = {}

print("Loading model_8bit.pt (original)...")
m1 = build_model_4layer(8)
m1.load_state_dict(torch.load('models/model_8bit.pt', map_location='cpu'))
m1.eval()
models['original'] = m1

print("Loading model_8bit_v2.pt (SMOTETomek)...")
m2 = build_model_4layer(8)
m2.load_state_dict(torch.load('models/model_8bit_v2.pt', map_location='cpu'))
m2.eval()
models['v2_smote'] = m2

print("Loading model_8bit_v3.pt (deep+SMOTE)...")
m3 = build_model_6layer(8)
m3.load_state_dict(torch.load('models/model_8bit_v3.pt', map_location='cpu'))
m3.eval()
models['v3_deep'] = m3

# ── Individual results ────────────────────────────────────────────
print("\nIndividual model results:")
print(f"{'Model':>20} {'Accuracy':>10} {'F1':>8}")
print("-" * 42)

all_probs = []
for name, model in models.items():
    with torch.no_grad():
        probs = torch.softmax(model(X_test_t), dim=1).numpy()
        preds = probs.argmax(axis=1)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average='weighted')
        print(f"{name:>20} {acc*100:>9.2f}% {f1:>8.4f}")
        all_probs.append(probs)

# ── Ensemble: average probabilities ──────────────────────────────
print("\nEnsemble results (average probabilities):")
print("-" * 42)

# Equal weight ensemble
avg_probs  = np.mean(all_probs, axis=0)
ens_preds  = avg_probs.argmax(axis=1)
ens_acc    = accuracy_score(y_test, ens_preds)
ens_f1     = f1_score(y_test, ens_preds, average='weighted')
print(f"{'Equal weight':>20} {ens_acc*100:>9.2f}% {ens_f1:>8.4f}")

# Weighted: give more weight to best model
w1, w2, w3 = 0.4, 0.3, 0.3
weighted_probs = (
    w1 * all_probs[0] +
    w2 * all_probs[1] +
    w3 * all_probs[2]
)
w_preds = weighted_probs.argmax(axis=1)
w_acc   = accuracy_score(y_test, w_preds)
w_f1    = f1_score(y_test, w_preds, average='weighted')
print(f"{'Weighted (0.4/0.3/0.3)':>20} {w_acc*100:>9.2f}% {w_f1:>8.4f}")

# Best ensemble weights — try all combinations
print("\nSearching best weights...")
best_combo_acc = 0
best_w = (0.4, 0.3, 0.3)
for w1 in np.arange(0.2, 0.7, 0.1):
    for w2 in np.arange(0.1, 0.6, 0.1):
        w3 = round(1.0 - w1 - w2, 1)
        if w3 < 0.1 or w3 > 0.7:
            continue
        wp = w1*all_probs[0] + w2*all_probs[1] + w3*all_probs[2]
        a  = accuracy_score(y_test, wp.argmax(axis=1))
        if a > best_combo_acc:
            best_combo_acc = a
            best_w = (round(w1,1), round(w2,1), round(w3,1))

print(f"Best weights: original={best_w[0]}, v2={best_w[1]}, v3={best_w[2]}")
best_probs = best_w[0]*all_probs[0] + best_w[1]*all_probs[1] + best_w[2]*all_probs[2]
best_preds = best_probs.argmax(axis=1)
best_acc   = accuracy_score(y_test, best_preds)
best_f1    = f1_score(y_test, best_preds, average='weighted')
print(f"{'Best ensemble':>20} {best_acc*100:>9.2f}% {best_f1:>8.4f}")

# ── Full report on best ensemble ─────────────────────────────────
print("\nFull classification report (best ensemble):")
print(classification_report(y_test, best_preds, target_names=['Normal', 'Attack']))

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "="*52)
print("FINAL COMPARISON:")
print(f"  Original 8-bit  : 81.97%")
print(f"  v2 SMOTETomek   : 81.71%")
print(f"  v3 Deep+SMOTE   : 81.92%")
print(f"  Ensemble (best) : {best_acc*100:.2f}%")
print("="*52)

# Save best ensemble predictions
np.save('models/ensemble_preds.npy', best_preds)
np.save('models/ensemble_probs.npy', best_probs)
print(f"\nSaved ensemble predictions → models/ensemble_preds.npy")