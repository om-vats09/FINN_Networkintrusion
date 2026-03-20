import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from model import build_model
import os

device = torch.device("cpu")

X_test = torch.tensor(np.load('data/X_test.npy'))
y_test = np.load('data/y_test.npy')

os.makedirs('results', exist_ok=True)

all_results = []

for bits in [2, 4, 8]:
    print(f"\n{'='*50}")
    print(f"  Evaluating {bits}-bit model")
    print(f"{'='*50}")

    model = build_model(bits)
    model.load_state_dict(torch.load(f'models/model_{bits}bit.pt'))
    model.eval()

    with torch.no_grad():
        preds = model(X_test).argmax(dim=1).numpy()

    acc    = accuracy_score(y_test, preds) * 100
    f1     = f1_score(y_test, preds, average='weighted')
    mem_kb = (sum(p.numel() for p in model.parameters()) * bits) / (8 * 1024)

    print(f"  Accuracy : {acc:.2f}%")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Memory   : {mem_kb:.2f} KB")
    print(f"\n{classification_report(y_test, preds, target_names=['Normal','Attack'])}")

    all_results.append({'bits': bits, 'acc': acc, 'f1': f1, 'mem': mem_kb})

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal','Attack'],
                yticklabels=['Normal','Attack'])
    plt.title(f'{bits}-bit Quantized Model — Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{bits}bit.png')
    plt.close()
    print(f"  Confusion matrix saved → results/confusion_matrix_{bits}bit.png")

print(f"\n{'='*50}")
print(f"  Summary")
print(f"{'='*50}")
print(f"{'Bits':<8} {'Accuracy':>10} {'F1 Score':>10} {'Memory KB':>10}")
print(f"{'-'*40}")
for r in all_results:
    print(f"{r['bits']:<8} {r['acc']:>9.2f}% {r['f1']:>10.4f} {r['mem']:>9.2f}")

plt.figure(figsize=(8, 4))
bits_list = [r['bits'] for r in all_results]
acc_list  = [r['acc']  for r in all_results]
f1_list   = [r['f1']   for r in all_results]

plt.subplot(1, 2, 1)
plt.bar([str(b)+'-bit' for b in bits_list], acc_list, color=['#534AB7','#1D9E75','#185FA5'])
plt.title('Accuracy by bit-width')
plt.ylabel('Accuracy (%)')
plt.ylim([70, 85])

plt.subplot(1, 2, 2)
plt.bar([str(b)+'-bit' for b in bits_list], f1_list, color=['#534AB7','#1D9E75','#185FA5'])
plt.title('F1 Score by bit-width')
plt.ylabel('F1 Score')
plt.ylim([0.70, 0.85])

plt.tight_layout()
plt.savefig('results/accuracy_f1_comparison.png')
plt.close()
print("\nComparison chart saved → results/accuracy_f1_comparison.png")
print("\nDone.")