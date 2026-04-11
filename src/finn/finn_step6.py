from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferQuantizedMatrixVectorActivation,
    InferThresholdingLayer,
)
import os

os.makedirs('models/finn_hw', exist_ok=True)

print("Loading streamlined model...")
model = ModelWrapper("models/finn_hw/nids_optimised.onnx")
print(f"  Nodes before: {len(model.graph.node)}")

print("\nConverting to HW layers...")
model = model.transform(InferQuantizedMatrixVectorActivation())
print(f"  After InferQuantizedMatrixVectorActivation: {len(model.graph.node)} nodes")

model = model.transform(InferThresholdingLayer())
print(f"  After InferThresholdingLayer: {len(model.graph.node)} nodes")

model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model.save("models/finn_hw/nids_hw_layers.onnx")

print("\nHW layer graph:")
for i, node in enumerate(model.graph.node):
    print(f"  [{i:02d}] {node.op_type:40s}")

print("\nDone.")
