import torch
import numpy as np
from model import build_model

print("Loading 8-bit model...")
model = build_model(8)
model.load_state_dict(torch.load('models/model_8bit.pt', map_location='cpu'))
model.eval()

X_test      = np.load('data/X_test.npy')
dummy_input = torch.tensor(X_test[:1])

print("Exporting to ONNX with batch size 1...")
torch.onnx.export(
    model,
    dummy_input,
    "models/nids_8bit.onnx",
    input_names   = ['input'],
    output_names  = ['output'],
    opset_version = 18
)
print("Saved → models/nids_8bit.onnx")