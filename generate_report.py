import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import torch
from model import build_model
import pickle
import os
from datetime import datetime

os.makedirs("results", exist_ok=True)

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

bits_list   = [2, 4, 8]
results     = []
all_preds   = {}

print("Evaluating all models...")
for bits in bits_list:
    model = build_model(bits)
    model.load_state_dict(torch.load(f"models/model_{bits}bit.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).argmax(dim=1).numpy()
    acc   = accuracy_score(y_test, preds) * 100
    f1    = f1_score(y_test, preds, average="weighted")
    mem   = (sum(p.numel() for p in model.parameters()) * bits) / (8 * 1024)
    cm    = confusion_matrix(y_test, preds)
    rep   = classification_report(y_test, preds, target_names=["Normal", "Attack"], output_dict=True)
    results.append({"bits": bits, "acc": acc, "f1": f1, "mem": mem, "cm": cm, "rep": rep})
    all_preds[bits] = preds
    print(f"  {bits}-bit: {acc:.2f}% accuracy")

print("Generating charts...")

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.bar([f"{r['bits']}-bit" for r in results],
        [r["acc"] for r in results],
        color=["#534AB7", "#1D9E75", "#185FA5"], width=0.5)
ax1.set_title("Accuracy by bit-width", fontsize=12)
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim([70, 85])
for i, r in enumerate(results):
    ax1.text(i, r["acc"] + 0.2, f"{r['acc']:.2f}%", ha="center", fontsize=10)

ax2 = fig.add_subplot(gs[0, 1])
ax2.bar([f"{r['bits']}-bit" for r in results],
        [r["f1"] for r in results],
        color=["#534AB7", "#1D9E75", "#185FA5"], width=0.5)
ax2.set_title("F1 Score by bit-width", fontsize=12)
ax2.set_ylabel("F1 Score")
ax2.set_ylim([0.70, 0.85])
for i, r in enumerate(results):
    ax2.text(i, r["f1"] + 0.002, f"{r['f1']:.4f}", ha="center", fontsize=10)

ax3 = fig.add_subplot(gs[0, 2])
ax3.bar([f"{r['bits']}-bit" for r in results],
        [r["mem"] for r in results],
        color=["#534AB7", "#1D9E75", "#185FA5"], width=0.5)
ax3.set_title("Memory usage by bit-width", fontsize=12)
ax3.set_ylabel("Memory (KB)")
for i, r in enumerate(results):
    ax3.text(i, r["mem"] + 0.1, f"{r['mem']:.2f} KB", ha="center", fontsize=10)

import seaborn as sns
for i, r in enumerate(results):
    ax = fig.add_subplot(gs[1, i])
    sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"],
                cbar=False)
    ax.set_title(f"{r['bits']}-bit Confusion Matrix", fontsize=11)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

ax7 = fig.add_subplot(gs[2, :])
metrics   = ["Precision", "Recall", "F1"]
x         = np.arange(len(metrics))
width     = 0.2
colors    = ["#534AB7", "#1D9E75", "#185FA5"]
for i, r in enumerate(results):
    vals = [
        r["rep"]["weighted avg"]["precision"],
        r["rep"]["weighted avg"]["recall"],
        r["rep"]["weighted avg"]["f1-score"]
    ]
    ax7.bar(x + i * width, vals, width, label=f"{r['bits']}-bit", color=colors[i])
ax7.set_title("Precision / Recall / F1 comparison", fontsize=12)
ax7.set_xticks(x + width)
ax7.set_xticklabels(metrics)
ax7.set_ylim([0.70, 0.90])
ax7.legend()
ax7.set_ylabel("Score")

fig.suptitle("FPGA-Based NIDS — Quantized MLP Results on NSL-KDD",
             fontsize=14, fontweight="bold", y=0.98)

plt.savefig("results/full_report_charts.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Charts saved → results/full_report_charts.png")

print("Generating HTML report...")
date = datetime.now().strftime("%B %d, %Y")

rows = ""
for r in results:
    tp = r["cm"][1][1]
    tn = r["cm"][0][0]
    fp = r["cm"][0][1]
    fn = r["cm"][1][0]
    rows += f"""
    <tr>
        <td><b>{r['bits']}-bit</b></td>
        <td>{r['acc']:.2f}%</td>
        <td>{r['f1']:.4f}</td>
        <td>{r['rep']['weighted avg']['precision']:.4f}</td>
        <td>{r['rep']['weighted avg']['recall']:.4f}</td>
        <td>{r['mem']:.2f} KB</td>
        <td>{tp}</td><td>{tn}</td><td>{fp}</td><td>{fn}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>IDS FINN Project Report</title>
<style>
  body      {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; color: #222; }}
  h1        {{ color: #1a1a2e; border-bottom: 2px solid #185FA5; padding-bottom: 8px; }}
  h2        {{ color: #185FA5; margin-top: 36px; }}
  table     {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th        {{ background: #185FA5; color: white; padding: 10px 12px; text-align: left; }}
  td        {{ padding: 9px 12px; border-bottom: 1px solid #ddd; }}
  tr:hover  {{ background: #f0f4ff; }}
  .badge    {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
               background: #e6f1fb; color: #0C447C; font-size: 13px; }}
  .summary  {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 20px 0; }}
  .card     {{ background: #f8f9ff; border: 1px solid #dde3f5; border-radius: 8px;
               padding: 16px; text-align: center; }}
  .card h3  {{ margin: 0 0 6px; font-size: 13px; color: #666; }}
  .card p   {{ margin: 0; font-size: 22px; font-weight: bold; color: #185FA5; }}
  img       {{ width: 100%; border: 1px solid #ddd; border-radius: 8px; margin: 16px 0; }}
  .arch     {{ background: #f4f4f4; border-left: 4px solid #185FA5;
               padding: 12px 16px; font-family: monospace; margin: 12px 0; }}
</style>
</head>
<body>

<h1>FPGA-Based Network Intrusion Detection System</h1>
<p>Using Quantized Neural Networks and the FINN Framework &nbsp;|&nbsp;
<span class="badge">NSL-KDD Dataset</span> &nbsp;
<span class="badge">Brevitas QAT</span> &nbsp;
<span class="badge">QONNX Pipeline</span></p>
<p style="color:#888">Generated: {date}</p>

<h2>Summary</h2>
<div class="summary">
  <div class="card"><h3>Best Accuracy (8-bit)</h3><p>{results[2]['acc']:.2f}%</p></div>
  <div class="card"><h3>Best F1 Score (8-bit)</h3><p>{results[2]['f1']:.4f}</p></div>
  <div class="card"><h3>Min Memory (2-bit)</h3><p>{results[0]['mem']:.2f} KB</p></div>
</div>

<h2>Model Architecture</h2>
<div class="arch">
Input (41 features)
  → QuantLinear(41 → 64)  + QuantReLU
  → QuantLinear(64 → 128) + QuantReLU
  → QuantLinear(128 → 64) + QuantReLU
  → QuantLinear(64 → 2)
Output: Normal / Attack
Total parameters: 19,397
</div>

<h2>Dataset — NSL-KDD</h2>
<table>
  <tr><th>Split</th><th>Samples</th><th>Normal</th><th>Attack</th></tr>
  <tr><td>Train</td><td>125,973</td><td>67,343</td><td>58,630</td></tr>
  <tr><td>Test</td><td>22,544</td><td>9,711</td><td>12,833</td></tr>
</table>

<h2>Results — All Quantization Levels</h2>
<table>
  <tr>
    <th>Model</th><th>Accuracy</th><th>F1</th>
    <th>Precision</th><th>Recall</th><th>Memory</th>
    <th>TP</th><th>TN</th><th>FP</th><th>FN</th>
  </tr>
  {rows}
</table>

<h2>Charts</h2>
<img src="full_report_charts.png" alt="Results Charts">

<h2>Pipeline Summary</h2>
<table>
  <tr><th>Stage</th><th>Tool</th><th>Status</th></tr>
  <tr><td>Data preprocessing</td><td>scikit-learn, pandas</td><td>✅ Complete</td></tr>
  <tr><td>QAT Training</td><td>Brevitas + PyTorch</td><td>✅ Complete</td></tr>
  <tr><td>ONNX Export</td><td>torch.onnx</td><td>✅ Complete</td></tr>
  <tr><td>Graph cleanup</td><td>QONNX</td><td>✅ Complete</td></tr>
  <tr><td>ONNX Verification</td><td>onnxruntime</td><td>✅ Complete</td></tr>
  <tr><td>HLS Compilation</td><td>FINN + Vivado</td><td>⏳ Pending hardware</td></tr>
  <tr><td>FPGA Deployment</td><td>PYNQ-Z2</td><td>⏳ Pending hardware</td></tr>
</table>

</body>
</html>"""

with open("results/report.html", "w") as f:
    f.write(html)

print("  HTML report saved → results/report.html")
print("\nDone. Open results/report.html in the Codespaces file panel to view.")