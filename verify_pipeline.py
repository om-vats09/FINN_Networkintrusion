import onnxruntime as ort
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

print("Loading ONNX model with onnxruntime...")
sess        = ort.InferenceSession("models/nids_8bit.onnx")
input_name  = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print(f"Input  : {input_name}")
print(f"Output : {output_name}")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

print(f"Running inference on {len(X_test)} samples...")

all_preds = []
for i, sample in enumerate(X_test):
    inp  = sample.reshape(1, 41)
    out  = sess.run([output_name], {input_name: inp})
    pred = np.argmax(out[0], axis=1)[0]
    all_preds.append(pred)
    if i % 5000 == 0:
        print(f"  processed {i}/{len(X_test)}")

all_preds = np.array(all_preds)
acc = accuracy_score(y_test, all_preds) * 100
f1  = f1_score(y_test, all_preds, average='weighted')

print(f"\nONNX inference results:")
print(f"  Accuracy : {acc:.2f}%")
print(f"  F1 Score : {f1:.4f}")
print(f"\nMatches original PyTorch model: {abs(acc - 81.97) < 1.0}")