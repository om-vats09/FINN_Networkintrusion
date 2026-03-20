from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs
)
from qonnx.util.cleanup import cleanup_model
import numpy as np
import os

print("=" * 50)
print("  QONNX Software Pipeline")
print("=" * 50)

onnx_path = "models/nids_8bit.onnx"

print(f"\nLoading ONNX model...")
model = ModelWrapper(onnx_path)
print(f"  Nodes   : {len(model.graph.node)}")
print(f"  Inputs  : {model.graph.input[0].name}")
print(f"  Outputs : {model.graph.output[0].name}")

print("\nStep 1 — Inferring shapes...")
model = model.transform(InferShapes())
print("  Done")

print("\nStep 2 — Folding constants...")
model = model.transform(FoldConstants())
print("  Done")

print("\nStep 3 — Cleaning up graph...")
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(RemoveStaticGraphInputs())
print("  Done")

os.makedirs("models/finn", exist_ok=True)
model.save("models/finn/nids_clean.onnx")
print("\nCleaned model saved → models/finn/nids_clean.onnx")

print("\nModel graph summary:")
for i, node in enumerate(model.graph.node):
    inputs  = list(node.input)
    outputs = list(node.output)
    print(f"  [{i:02d}] {node.op_type:25s} in={inputs} out={outputs}")

print("\nTensor shape summary:")
for info in model.graph.value_info:
    shape = [d.dim_value for d in info.type.tensor_type.shape.dim]
    print(f"  {info.name:40s} shape={shape}")

print("\nDone.")