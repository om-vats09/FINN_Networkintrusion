import torch
import numpy as np
from model import build_model

print("Loading 8-bit model...")
model = build_model(8)
model.load_state_dict(torch.load('models/model_8bit.pt', map_location='cpu'))
model.eval()

X_test = np.load('data/X_test.npy')
dummy_input = torch.tensor(X_test[:1])

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    "models/nids_8bit.onnx",
    input_names   = ['input'],
    output_names  = ['output'],
    dynamic_axes  = {'input': {0: 'batch_size'}},
    opset_version = 18
)
print("Saved → models/nids_8bit.onnx")

print("\nValidating ONNX model...")
import onnxruntime as ort
sess   = ort.InferenceSession("models/nids_8bit.onnx")
sample = X_test[:10]
preds  = sess.run(None, {'input': sample})[0].argmax(axis=1)
labels = np.load('data/y_test.npy')[:10]
print(f"Predictions  : {preds}")
print(f"Ground truth : {labels}")
print("\nDone. ONNX model is ready for FINN compiler.")