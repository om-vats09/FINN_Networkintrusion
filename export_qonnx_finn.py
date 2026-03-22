from brevitas.export import export_qonnx
from model import build_model
import torch
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames, RemoveStaticGraphInputs

model = build_model(4)
model.load_state_dict(torch.load('models/model_4bit.pt', map_location='cpu'))
model.eval()

dummy = torch.tensor(np.load('data/X_test.npy')[:1], dtype=torch.float32)

print("Exporting with QONNX opset 11...")
export_qonnx(model, input_t=dummy, export_path='models/nids_4bit_qonnx.onnx', opset_version=11)

print("Cleaning graph...")
m = ModelWrapper('models/nids_4bit_qonnx.onnx')
m = m.transform(InferShapes())
m = m.transform(FoldConstants())
m = m.transform(GiveUniqueNodeNames())
m = m.transform(GiveReadableTensorNames())
m = m.transform(RemoveStaticGraphInputs())
m.save('models/nids_4bit_qonnx_clean.onnx')

print("Nodes in cleaned model:")
for i, node in enumerate(m.graph.node):
    print(f"  [{i:02d}] {node.op_type}")

print("\nDone — saved to models/nids_4bit_qonnx_clean.onnx")
